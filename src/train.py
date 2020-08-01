import datetime as dt
import hydra
import logging
import numpy as np
import os
import time
import torch
import torch.distributed as dist

from hydra.utils import instantiate
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, TerminateOnNan
from ignite.metrics import Metric, RunningAverage
from omegaconf import DictConfig
from timm.scheduler.scheduler import Scheduler
from torch.utils.data import DistributedSampler
from typing import Dict, List, Optional

from data.sampler import CustomSampler
from data.utils import create_train_loader, create_val_loader
from utils.train import build_optimizer, build_process_batch_func, setup_checkpoints, setup_ema
from utils.typings import Batch, Device, FloatDict
from utils.visualize import setup_visualizations

Metrics = Dict[str, Metric]


def humanize_time(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')


def on_epoch_start(engine: Engine):
    engine.state.t0 = time.time()
    engine.state.lr = 0.0


def log_iter(engine, pbar, interval_it=100, name="stage"):
    # type: (Engine, ProgressBar, Optional[int], Optional[str]) -> None
    epoch = engine.state.epoch
    iteration = engine.state.iteration
    metrics = engine.state.metrics
    stats = ", ".join(["%s: %.4f" % k_v for k_v in metrics.items()])
    stats += ", lr: %.4f" % engine.state.lr
    t0 = engine.state.t0
    t1 = time.time()
    it_time = (t1 - t0) / interval_it
    cur_time = humanize_time(t1)
    pbar.log_message("[{}][{:.3f} s] {} | ep: {:2d}, it: {:3d}, {}".format(
        cur_time, it_time, name, epoch, iteration, stats))
    engine.state.t0 = t1


def log_epoch(engine: Engine, name="stage") -> None:
    epoch = engine.state.epoch
    metrics = engine.state.metrics
    stats = ", ".join(["%s: %.3f" % k_v for k_v in metrics.items()])
    logging.info("{} | ep: {}, {}".format(name, epoch, stats))


def build_engine(loop_func, metrics=None):
    trainer = Engine(loop_func)
    if metrics:
        for name, metric in metrics.items():
            metric.attach(trainer, name)
    return trainer


def create_metrics(keys: List[str], device: Device = None) -> Metrics:
    def _out_transform(kek: str):
        return lambda out: out[kek]

    metrics = {key: RunningAverage(output_transform=_out_transform(key),
                                   device=device)
               for key in keys}
    return metrics


def _upd_pbar_iter_from_cp(engine: Engine, pbar: ProgressBar) -> None:
    pbar.n = engine.state.iteration


def run(conf: DictConfig, local_rank=0, distributed=False):
    epochs = conf.train.epochs
    epoch_length = conf.train.epoch_length
    torch.manual_seed(conf.seed)

    if distributed:
        rank = dist.get_rank()
        num_replicas = dist.get_world_size()
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        num_replicas = 1
        torch.cuda.set_device(conf.gpu)
    device = torch.device('cuda')
    loader_args = dict(mean=conf.data.mean, std=conf.data.std)
    master_node = rank == 0

    if master_node:
        print(conf.pretty())
    if num_replicas > 1:
        epoch_length = epoch_length // num_replicas
        loader_args["rank"] = rank
        loader_args["num_replicas"] = num_replicas

    train_dl = create_train_loader(conf.data.train, **loader_args)
    valid_dl = create_val_loader(conf.data.val, **loader_args)

    if epoch_length < 1:
        epoch_length = len(train_dl)

    model = instantiate(conf.model).to(device)
    model_ema, update_ema = setup_ema(conf, model, device=device, master_node=master_node)
    optim = build_optimizer(conf.optim, model)

    scheduler_kwargs = dict()
    if "schedule.OneCyclePolicy" in conf.lr_scheduler["class"]:
        scheduler_kwargs["cycle_steps"] = epoch_length
    lr_scheduler: Scheduler = instantiate(conf.lr_scheduler, optim, **scheduler_kwargs)

    use_amp = False
    if conf.use_apex:
        import apex
        from apex import amp
        logging.debug("Nvidia's Apex package is available")

        model, optim = amp.initialize(model, optim, **conf.amp)
        use_amp = True
        if master_node:
            logging.info("Using AMP with opt_level={}".format(conf.amp.opt_level))
    else:
        apex, amp = None, None

    to_save = dict(model=model, optim=optim)
    if use_amp:
        to_save["amp"] = amp
    if model_ema is not None:
        to_save["model_ema"] = model_ema

    if master_node and conf.logging.model:
        logging.info(model)

    if distributed:
        sync_bn = conf.distributed.sync_bn
        if apex is not None:
            if sync_bn:
                model = apex.parallel.convert_syncbn_model(model)
            model = apex.parallel.distributed.DistributedDataParallel(
                model, delay_allreduce=True)
        else:
            if sync_bn:
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank, ], output_device=local_rank)

    upd_interval = conf.optim.step_interval
    ema_interval = conf.smoothing.interval_it * upd_interval
    clip_grad = conf.optim.clip_grad

    _handle_batch_train = build_process_batch_func(conf.data, stage="train", device=device)
    _handle_batch_val = build_process_batch_func(conf.data, stage="val", device=device)

    def _update(eng: Engine, batch: Batch) -> FloatDict:
        model.train()
        batch = _handle_batch_train(batch)
        losses: Dict = model(*batch)
        stats = {k: v.item() for k, v in losses.items()}
        loss = losses["loss"]
        del losses

        if use_amp:
            with amp.scale_loss(loss, optim) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        it = eng.state.iteration
        if not it % upd_interval:
            if clip_grad > 0:
                params = amp.master_params(optim) if use_amp else model.parameters()
                torch.nn.utils.clip_grad_norm_(params, clip_grad)
            optim.step()
            optim.zero_grad()
            lr_scheduler.step_update(it)

            if not it % ema_interval:
                update_ema()

            eng.state.lr = optim.param_groups[0]["lr"]

        return stats

    calc_map = conf.validate.calc_map
    min_score = conf.validate.get("min_score", -1)

    model_val = model
    if conf.train.skip and model_ema is not None:
        model_val = model_ema.to(device)

    def _validate(eng: Engine, batch: Batch) -> FloatDict:
        model_val.eval()
        images, targets = _handle_batch_val(batch)

        with torch.no_grad():
            out: Dict = model_val(images, targets)

        pred_boxes = out.pop("detections")
        stats = {k: v.item() for k, v in out.items()}

        if calc_map:
            pred_boxes = pred_boxes.detach().cpu().numpy()
            true_boxes = targets['bbox'].cpu().numpy()
            img_scale = targets['img_scale'].cpu().numpy()
            # yxyx -> xyxy
            true_boxes = true_boxes[:, :, [1, 0, 3, 2]]
            # xyxy -> xywh
            true_boxes[:, :, [2, 3]] -= true_boxes[:, :, [0, 1]]
            # scale downsized boxes to match predictions on a full-sized image
            true_boxes *= img_scale[:, None, None]

            scores = []
            for i in range(len(images)):
                mask = pred_boxes[i, :, 4] >= min_score
                s = calculate_image_precision(true_boxes[i], pred_boxes[i, mask, :4],
                                              thresholds=IOU_THRESHOLDS,
                                              form='coco')
                scores.append(s)
            stats['map'] = np.mean(scores)

        return stats

    train_metric_names = list(conf.logging.out.train)
    train_metrics = create_metrics(train_metric_names, device if distributed else None)

    val_metric_names = list(conf.logging.out.val)
    if calc_map:
        from utils.metric import calculate_image_precision, IOU_THRESHOLDS
        val_metric_names.append('map')
    val_metrics = create_metrics(val_metric_names, device if distributed else None)

    trainer = build_engine(_update, train_metrics)
    evaluator = build_engine(_validate, val_metrics)
    to_save['trainer'] = trainer

    every_iteration = Events.ITERATION_COMPLETED
    trainer.add_event_handler(every_iteration, TerminateOnNan())

    if distributed:
        dist_bn = conf.distributed.dist_bn
        if dist_bn in ["reduce", "broadcast"]:
            from timm.utils import distribute_bn

            @trainer.on(Events.EPOCH_COMPLETED)
            def _distribute_bn_stats(eng: Engine):
                reduce = dist_bn == "reduce"
                if master_node:
                    logging.info("Distributing BN stats...")
                distribute_bn(model, num_replicas, reduce)

    sampler = train_dl.sampler
    if isinstance(sampler, (CustomSampler, DistributedSampler)):
        @trainer.on(Events.EPOCH_STARTED)
        def _set_epoch(eng: Engine):
            sampler.set_epoch(eng.state.epoch - 1)

    @trainer.on(Events.EPOCH_COMPLETED)
    def _scheduler_step(eng: Engine):
        # it starts from 1, so we don't need to add 1 here
        ep = eng.state.epoch
        lr_scheduler.step(ep)

    cp = conf.checkpoints
    pbar, pbar_vis = None, None

    if master_node:
        log_interval = conf.logging.interval_it
        log_event = Events.ITERATION_COMPLETED(every=log_interval)
        pbar = ProgressBar(persist=False)
        pbar.attach(trainer, metric_names=train_metric_names)
        pbar.attach(evaluator, metric_names=val_metric_names)

        for engine, name in zip([trainer, evaluator], ['train', 'val']):
            engine.add_event_handler(Events.EPOCH_STARTED, on_epoch_start)
            engine.add_event_handler(log_event, log_iter, pbar, interval_it=log_interval, name=name)
            engine.add_event_handler(Events.EPOCH_COMPLETED, log_epoch, name=name)

        setup_checkpoints(trainer, to_save, epoch_length, conf)

    if 'load' in cp.keys() and cp.load is not None:
        if master_node:
            logging.info("Resume from a checkpoint: {}".format(cp.load))
            trainer.add_event_handler(Events.STARTED, _upd_pbar_iter_from_cp, pbar)
        to_load = {k: v for k, v in to_save.items() if v is not None}
        if cp.drop_state:
            # we might want to swap optimizer or to reset it state
            drop_keys = set(cp.drop_state)
            to_load = {k: v for k, v in to_load.items() if k not in drop_keys}
        Checkpoint.load_objects(to_load=to_load,
                                checkpoint=torch.load(cp.load, map_location=device))
        state = trainer.state
        # epoch counter start from 1
        lr_scheduler.step(state.epoch - 1)
        state.max_epochs = epochs

    @trainer.on(Events.EPOCH_COMPLETED(every=conf.validate.interval_ep))
    def _run_validation(eng: Engine):
        if distributed:
            torch.cuda.synchronize(device)
        evaluator.run(valid_dl)

    skip_train = conf.train.skip
    if master_node and conf.visualize.enabled:
        vis_eng = evaluator if skip_train else trainer
        setup_visualizations(vis_eng, model, valid_dl, device, conf, force_run=skip_train)

    try:
        if skip_train:
            evaluator.run(valid_dl)
        else:
            trainer.run(train_dl, max_epochs=epochs, epoch_length=epoch_length)
    except Exception as e:
        import traceback
        logging.error(traceback.format_exc())

    for pb in [pbar, pbar_vis]:
        if pb is not None:
            pbar.close()


@hydra.main(config_path="../config/train.yaml")
def main(conf: DictConfig):
    env = os.environ.copy()
    world_size = int(env.get('WORLD_SIZE', -1))
    local_rank = int(env.get('LOCAL_RANK', -1))
    dist_conf: DictConfig = conf.distributed
    distributed = world_size > 1 and local_rank >= 0

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Unable to find any CUDA device")

        torch.backends.cudnn.benchmark = True
        dist.init_process_group(dist_conf.backend, init_method=dist_conf.url)
        if local_rank == 0:
            print("\nDistributed setting:")
            print("\tbackend: {}".format(dist.get_backend()))
            print("\tworld size: {}".format(dist.get_world_size()))
            print("\trank: {}\n".format(dist.get_rank()))

    try:
        run(conf, local_rank, distributed)
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        if distributed:
            dist.destroy_process_group()
        raise e

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
