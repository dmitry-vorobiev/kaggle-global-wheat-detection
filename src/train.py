import numpy as np
import albumentations as A
import datetime as dt
import hydra
import itertools
import logging
import math
import os
import time
import torch
import torch.distributed as dist

from hydra.utils import instantiate
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Metric, RunningAverage
from omegaconf import DictConfig
from timm.optim.optim_factory import add_weight_decay
from timm.scheduler.scheduler import Scheduler
from torch import nn, Tensor
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple

from data.dataset import ExtendedWheatDataset
from data.sampler import CustomSampler
from data.loader import fast_collate, PrefetchLoader
from utils.typings import Batch, Device, FloatDict
from utils.visualize import draw_bboxes, save_image

Metrics = Dict[str, Metric]
float_3 = Tuple[float, float, float]


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


def _build_process_batch_func(conf: DictConfig, stage="train", device=None):
    # conf -> root.data
    assert stage in ["train", "val"]
    is_val = stage == "val"
    c = getattr(conf, stage)  # root.data.train
    prefetch = c.loader.prefetch

    def _filter_targets(target):
        return {k: v for k, v in target.items() if k in ['bbox', 'cls']}

    if prefetch and is_val:
        def _handle(batch: Batch) -> Batch:
            return batch

    elif prefetch and not is_val:
        def _handle(batch: Batch) -> Batch:
            return batch[0], _filter_targets(batch[1])

    else:
        mean = torch.tensor(list(conf.mean)).to(device).view(1, 3, 1, 1).mul_(255)
        std = torch.tensor(list(conf.std)).to(device).view(1, 3, 1, 1).mul_(255)

        def _handle(batch: Batch) -> Batch:
            images, target = batch
            images = images.to(device).float().sub_(mean).div_(std)
            if not is_val:
                target = _filter_targets(target)
            target = {k: v.to(device) for k, v in target.items()}
            return images, target

    return _handle


def create_metrics(keys: List[str], device: Device = None) -> Metrics:
    def _out_transform(kek: str):
        return lambda out: out[kek]

    metrics = {key: RunningAverage(output_transform=_out_transform(key),
                                   device=device)
               for key in keys}
    return metrics


def _upd_pbar_iter_from_cp(engine: Engine, pbar: ProgressBar) -> None:
    pbar.n = engine.state.iteration


def create_dataset(conf, show_progress=False, name="train"):
    # type: (DictConfig, Optional[bool], Optional[str]) -> Dataset
    def _build_tfm(conf_tfm: DictConfig):
        tfm = [instantiate(v) for k, v in conf_tfm.items()]
        tfm = A.Compose(tfm, bbox_params=A.BboxParams(**conf.bbox_params, label_fields=["cls"]))
        return tfm

    if show_progress:
        print("Loading {} data...".format(name))
    ds = instantiate(conf, show_progress=show_progress)
    # hydra will raise "ValueError: key transforms: Compose is not a primitive type",
    # if you try to pass transforms directly to instantiate(...)
    for field in ["transforms", "affine_tfm", "affine_tfm_mosaic"]:
        if field in conf:
            c = getattr(conf, field)
            setattr(ds, field, _build_tfm(c))
    if show_progress:
        print("{}: {} images".format(name, len(ds)))
    return ds


def create_train_loader(conf, rank=None, num_replicas=None, mean=None, std=None):
    # type: (DictConfig, Optional[int], Optional[int], Optional[float_3], Optional[float_3]) -> DataLoader
    show_progress = rank is None or rank == 0
    data = create_dataset(conf, show_progress=show_progress, name="train")

    sampler = None
    if isinstance(data, ExtendedWheatDataset):
        sampler = CustomSampler(data,
                                orig_images_ratio=conf.get("orig_images_ratio", 0.5),
                                num_replicas=num_replicas,
                                rank=rank)
    elif num_replicas is not None:
        sampler = DistributedSampler(data, num_replicas=num_replicas, rank=rank)

    loader = DataLoader(data,
                        sampler=sampler,
                        batch_size=conf.loader.batch_size,
                        num_workers=conf.get('loader.workers', 0),
                        collate_fn=fast_collate,
                        drop_last=True,
                        shuffle=not sampler)
    if conf.loader.prefetch:
        loader = PrefetchLoader(loader, mean=mean, std=std)
    return loader


def create_val_loader(conf, rank=None, num_replicas=None, mean=None, std=None):
    # type: (DictConfig, Optional[int], Optional[int], Optional[float_3], Optional[float_3]) -> DataLoader
    show_progress = rank is None or rank == 0
    data = create_dataset(conf, show_progress=show_progress, name="val")

    sampler = None
    if num_replicas is not None:
        sampler = DistributedSampler(data, num_replicas=num_replicas, rank=rank, shuffle=False)

    loader = DataLoader(data,
                        sampler=sampler,
                        batch_size=conf.loader.batch_size,
                        num_workers=conf.get('loader.workers', 0),
                        collate_fn=fast_collate,
                        drop_last=False,
                        shuffle=not sampler)
    if conf.loader.prefetch:
        loader = PrefetchLoader(loader, mean=mean, std=std)
    return loader


def build_optimizer(conf: DictConfig, model: nn.Module) -> Optimizer:
    parameters = model.parameters()
    p = conf.params
    if 'weight_decay' in p and p.weight_decay > 0:
        parameters = add_weight_decay(model, p.weight_decay)
        p.weight_decay = 0.0
    return instantiate(conf, parameters)


def setup_checkpoints(trainer, obj_to_save, epoch_length, conf):
    # type: (Engine, Dict[str, Any], int, DictConfig) -> None
    cp = conf.checkpoints
    save_path = cp.get('save_dir', os.getcwd())
    logging.info("Saving checkpoints to {}".format(save_path))
    max_cp = max(int(cp.get('max_checkpoints', 1)), 1)
    save = DiskSaver(save_path, create_dir=True, require_empty=True)
    make_checkpoint = Checkpoint(obj_to_save, save, n_saved=max_cp)
    cp_iter = cp.interval_it
    cp_epoch = cp.interval_ep
    if cp_iter > 0:
        save_event = Events.ITERATION_COMPLETED(every=cp_iter)
        trainer.add_event_handler(save_event, make_checkpoint)
    if cp_epoch > 0:
        if cp_iter < 1 or epoch_length % cp_iter:
            save_event = Events.EPOCH_COMPLETED(every=cp_epoch)
            trainer.add_event_handler(save_event, make_checkpoint)


def setup_visualizations(engine, model, dl, device, conf, force_run=True):
    # type: (Engine, nn.Module, DataLoader, Device, DictConfig, Optional[bool]) -> None
    vis_conf = conf.visualize
    save_dir = vis_conf.get("save_dir", os.path.join(os.getcwd(), 'images'))
    min_score = vis_conf.get("min_score", -1)
    num_images = vis_conf.num_images
    interval_ep = 1 if force_run else vis_conf.interval_ep
    target_yxyx = conf.data.train.params.box_format == 'yxyx'
    bs = dl.loader.batch_size if hasattr(dl, "loader") else dl.batch_size

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    elif os.path.isfile(save_dir):
        raise AttributeError("Unable to save images, not a valid directory: {}")

    logging.info("Saving visualizations to {}".format(save_dir))

    mean = torch.tensor(list(conf.data.mean)).to(device).view(1, 3, 1, 1).mul_(255)
    std = torch.tensor(list(conf.data.std)).to(device).view(1, 3, 1, 1).mul_(255)

    @engine.on(Events.EPOCH_COMPLETED(every=interval_ep))
    def _make_visualizations(eng: Engine):
        iterations = int(math.ceil(num_images / bs))
        iterations = min(iterations, len(dl))
        epoch = engine.state.epoch

        data = iter(dl)
        model.eval()
        _handle_batch_val = _build_process_batch_func(conf.data, stage="val", device=device)

        for i_batch in tqdm(range(iterations), desc="Saving visualizations"):
            batch = next(data)
            images, targets = _handle_batch_val(batch)

            with torch.no_grad():
                out: Dict = model(images, targets)

            predictions = out["detections"]
            predictions[:, :, :4] /= targets['img_scale'][:, None, None]
            predictions = predictions.cpu().numpy()

            target_boxes = targets['bbox'].cpu().numpy()

            images = (images * std + mean).clamp(0, 255).permute(0, 2, 3, 1)
            images = images.cpu().numpy().astype(np.uint8)

            done = i_batch * bs
            to_do = min(num_images - done, len(predictions))

            for i in range(to_do):
                image = images[i]
                scores_i = predictions[i, :, 4]
                pred_i = predictions[i, scores_i >= min_score, :4]

                draw_bboxes(image, target_boxes[i], (255, 0, 0), box_format='pascal_voc',
                            yxyx=target_yxyx)
                draw_bboxes(image, pred_i, (0, 255, 0), box_format='coco')
                path = os.path.join(save_dir, '%02d_%03d.png' % (epoch, done + i))
                save_image(image, path)
                del image

            del images, targets, predictions, out


def setup_ema(conf: DictConfig, model: nn.Module, device=None, master_node=False):
    ema = conf.smoothing
    model_ema = None
    def _update(): pass

    if master_node and ema.enabled:
        model_ema = instantiate(conf.model)
        if not ema.use_cpu:
            model_ema = model_ema.to(device)
        model_ema.load_state_dict(model.state_dict())
        model_ema.requires_grad_(False)

        beta = 1 - ema.alpha ** ema.interval_it

        def _update():
            states = itertools.chain(
                zip(model_ema.parameters(), model.parameters()),
                zip(model_ema.buffers(), model.buffers()))

            with torch.no_grad():
                for t_ema, t in states:
                    # filter out *.bn1.num_batches_tracked
                    if t.dtype != torch.int64:
                        t = t.to(dtype=t_ema.dtype, device=t_ema.device)
                        t_ema.lerp_(t, beta)

    return model_ema, _update


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

    to_save = dict(model=model, model_ema=model_ema, optim=optim)
    if use_amp:
        to_save["amp"] = amp

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

    _handle_batch_train = _build_process_batch_func(conf.data, stage="train", device=device)
    _handle_batch_val = _build_process_batch_func(conf.data, stage="val", device=device)

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
