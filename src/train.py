import albumentations as A
import datetime as dt
import hydra
import logging
import math
import os
import time
import torch
import torchvision
import torch.distributed as dist
import torchvision.transforms as T

from hydra.utils import instantiate
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, TerminateOnNan
from ignite.metrics import Metric, RunningAverage
from ignite.utils import convert_tensor
from omegaconf import DictConfig
from torch import nn, Tensor
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple, Sized

from data.dataset import WheatDataset
from data.utils import basic_collate
from utils.typings import Batch, Device, FloatDict
from utils.visualize import visualize_detections

Metrics = Dict[str, Metric]


def humanize_time(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')


def on_epoch_start(engine: Engine):
    engine.state.t0 = time.time()


def log_iter(engine, pbar, interval_it=100, name="stage"):
    # type: (Engine, ProgressBar, Optional[int], Optional[str]) -> None
    epoch = engine.state.epoch
    iteration = engine.state.iteration
    metrics = engine.state.metrics
    stats = ", ".join(["%s: %.3f" % k_v for k_v in metrics.items()])
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


def _prepare_batch(batch: Batch, device: torch.device, non_blocking: bool) -> Batch:
    kwargs = dict(device=device, non_blocking=non_blocking)
    x, y = batch
    return convert_tensor(x, **kwargs), convert_tensor(y, **kwargs)


def _prepare_batch_efficientdet(batch: Batch, device: Device, is_val=False):
    images, bboxes = batch
    images = images.to(device).float()
    # TODO: add GPU transforms here
    boxes, cls = [], []

    for b in bboxes:
        c = torch.ones(len(b), device=device)
        b = b.to(device).float()
        boxes.append(b)
        cls.append(c)

    target = dict(bbox=boxes, cls=cls)
    if is_val:
        N, C, H, W = images.shape
        target["img_scale"] = torch.ones(N, dtype=torch.float32, device=device)
        target["img_size"] = torch.tensor([W, H], dtype=torch.int64, device=device).expand(N, 2)
    return images, target


def create_metrics(keys: List[str], device: Device = None) -> Metrics:
    def _out_transform(kek: str):
        return lambda out: out[kek]

    metrics = {key: RunningAverage(output_transform=_out_transform(key),
                                   device=device)
               for key in keys}
    return metrics


def _upd_pbar_iter_from_cp(engine: Engine, pbar: ProgressBar) -> None:
    pbar.n = engine.state.iteration


def create_dataset(conf, transforms, show_progress=False, name="train"):
    # type: (DictConfig, DictConfig, Optional[bool], Optional[str]) -> WheatDataset
    transforms = [instantiate(v) for k, v in transforms.items()]
    compose = T.Compose
    compose_kwargs = dict()
    if all(isinstance(t, A.BasicTransform) for t in transforms):
        compose = A.Compose
        compose_kwargs["bbox_params"] = A.BboxParams('pascal_voc')
    transforms = compose(transforms, **compose_kwargs)

    if show_progress:
        print("Loading {} data...".format(name))
    ds = WheatDataset(**conf.params, transforms=transforms, show_progress=show_progress)
    if show_progress:
        print("{}: {} images".format(name, len(ds)))
    return ds


def create_train_loader(conf, rank=None, num_replicas=None):
    # type: (DictConfig, Optional[int], Optional[int]) -> DataLoader
    show_progress = rank is None or rank == 0
    data = create_dataset(conf, conf.transforms, show_progress=show_progress, name="train")

    sampler = None
    if num_replicas is not None:
        sampler = DistributedSampler(data, num_replicas=num_replicas, rank=rank)

    loader = DataLoader(data,
                        sampler=sampler,
                        batch_size=conf.loader.batch_size,
                        num_workers=conf.get('loader.workers', 0),
                        collate_fn=basic_collate,
                        drop_last=True,
                        shuffle=True)
    return loader


def create_val_loader(conf, rank=None, num_replicas=None):
    # type: (DictConfig, Optional[int], Optional[int]) -> DataLoader
    show_progress = rank is None or rank == 0
    data = create_dataset(conf, conf.transforms, show_progress=show_progress, name="val")

    sampler = None
    if num_replicas is not None:
        sampler = DistributedSampler(data, num_replicas=num_replicas, rank=rank, shuffle=False)

    loader = DataLoader(data,
                        sampler=sampler,
                        batch_size=conf.loader.batch_size,
                        num_workers=conf.get('loader.workers', 0),
                        collate_fn=basic_collate,
                        drop_last=False,
                        shuffle=False)
    return loader


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


def setup_visualizations(trainer, model, dl, device, conf):
    # type: (Engine, nn.Module, DataLoader, Device, DictConfig) -> None
    save_dir = conf.get('save_dir', os.path.join(os.getcwd(), 'images'))

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    elif os.path.isfile(save_dir):
        raise AttributeError("Unable to save images, not a valid directory: {}")

    logging.info("Saving visualizations to {}".format(save_dir))

    @trainer.on(Events.EPOCH_COMPLETED(every=conf.interval_ep))
    def _make_visualizations(eng: Engine):
        bs = dl.batch_size
        iterations = int(math.ceil(conf.num_images / bs))
        iterations = min(iterations, len(dl))
        epoch = trainer.state.epoch

        data = iter(dl)
        model.eval()

        for i_batch in tqdm(range(iterations), desc="Saving visualizations"):
            batch = next(data)
            images, targets = _prepare_batch_efficientdet(batch, device, is_val=True)

            with torch.no_grad():
                out: Dict = model(images, targets)
            predictions = out["detections"]

            done = i_batch * bs
            to_do = min(conf.num_images - done, len(predictions))

            for i in range(to_do):
                image = images[i]
                visualize_detections(image, targets['bbox'][i], predictions[i][:, :4])
                path = os.path.join(save_dir, '%02d_%03d.png' % (epoch, done + i))
                torchvision.utils.save_image(image, path, normalize=True)
                del image

            del images, targets, predictions, out


def run(conf: DictConfig, local_rank=0, distributed=False):
    epochs = conf.train.epochs
    epoch_length = conf.train.epoch_length
    torch.manual_seed(conf.general.seed)

    if distributed:
        rank = dist.get_rank()
        num_replicas = dist.get_world_size()
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        num_replicas = 1
        torch.cuda.set_device(conf.general.gpu)
    device = torch.device('cuda')
    loader_args = dict()
    master_node = rank == 0

    if master_node:
        print(conf.pretty())
    if num_replicas > 1:
        epoch_length = epoch_length // num_replicas
        loader_args = dict(rank=rank, num_replicas=num_replicas)

    train_dl = create_train_loader(conf.data.train, **loader_args)
    valid_dl = create_val_loader(conf.data.val, **loader_args)

    if epoch_length < 1:
        epoch_length = len(train_dl)

    model = instantiate(conf.model).to(device)
    optim = instantiate(conf.optim, model.parameters())

    to_save = dict(model=model, optim=optim)

    if master_node and conf.logging.model:
        logging.info(model)

    if distributed:
        ddp_kwargs = dict(device_ids=[local_rank, ], output_device=local_rank)
        model = torch.nn.parallel.DistributedDataParallel(model, **ddp_kwargs)

    update_freq = conf.optim.step_interval

    def _update(eng: Engine, batch: Batch) -> FloatDict:
        model.train()
        batch = _prepare_batch_efficientdet(batch, device)
        losses: Dict = model(*batch)
        stats = {k: v.item() for k, v in losses.items()}
        loss = losses["loss"]
        del losses

        it = eng.state.iteration
        if not it % update_freq:
            optim.zero_grad()
        loss.backward()
        if not (it + 1) % update_freq:
            optim.step()

        return stats

    def _validate(eng: Engine, batch: Batch) -> FloatDict:
        model.eval()
        images, targets = _prepare_batch_efficientdet(batch, device, is_val=True)

        with torch.no_grad():
            out: Dict = model(images, targets)

        _ = out.pop("detections")
        stats = {k: v.item() for k, v in out.items()}
        return stats

    metric_names = list(conf.logging.out)
    metrics = create_metrics(metric_names, device if distributed else None)

    trainer = build_engine(_update, metrics)
    evaluator = build_engine(_validate, metrics)
    to_save['trainer'] = trainer

    every_iteration = Events.ITERATION_COMPLETED
    trainer.add_event_handler(every_iteration, TerminateOnNan())

    sampler = train_dl.sampler
    if isinstance(sampler, DistributedSampler):
        @trainer.on(Events.EPOCH_STARTED)
        def _set_epoch(eng: Engine):
            sampler.set_epoch(eng.state.epoch - 1)

    cp = conf.checkpoints
    pbar, pbar_vis = None, None

    if master_node:
        log_interval = conf.logging.interval_it
        log_event = Events.ITERATION_COMPLETED(every=log_interval)
        pbar = ProgressBar(persist=False)

        for engine, name in zip([trainer, evaluator], ['train', 'val']):
            engine.add_event_handler(Events.EPOCH_STARTED, on_epoch_start)
            engine.add_event_handler(log_event, log_iter, pbar, interval_it=log_interval, name=name)
            engine.add_event_handler(Events.EPOCH_COMPLETED, log_epoch, name=name)
            pbar.attach(engine, metric_names=metric_names)

        setup_checkpoints(trainer, to_save, epoch_length, conf)

    if 'load' in cp.keys() and cp.load is not None:
        if master_node:
            logging.info("Resume from a checkpoint: {}".format(cp.load))
            trainer.add_event_handler(Events.STARTED, _upd_pbar_iter_from_cp, pbar)
        Checkpoint.load_objects(to_load=to_save,
                                checkpoint=torch.load(cp.load, map_location=device))

    @trainer.on(Events.EPOCH_COMPLETED(every=conf.validate.interval_ep))
    def _run_validation(eng: Engine):
        if distributed:
            torch.cuda.synchronize(device)
        evaluator.run(valid_dl)

    if master_node and conf.visualize.enabled:
        setup_visualizations(trainer, model, valid_dl, device, conf.visualize)

    try:
        if conf.train.skip:
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
