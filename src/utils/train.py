import datetime as dt
import itertools
import logging
import os
import torch
from hydra.utils import instantiate
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver
from omegaconf import DictConfig
from timm.optim.optim_factory import add_weight_decay
from torch import nn
from torch.optim.optimizer import Optimizer
from typing import Any, Dict

from .common import mean_std_tensors
from .typings import Batch, Device


def humanize_time(timestamp: float) -> str:
    return dt.datetime.fromtimestamp(timestamp).strftime('%H:%M:%S')


def build_process_batch_func(conf: DictConfig, stage="train", device=None):
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
        mean, std = mean_std_tensors(conf, device)

        def _handle(batch: Batch) -> Batch:
            images, target = batch
            images = images.to(device).float().sub_(mean).div_(std)
            if not is_val:
                target = _filter_targets(target)
            target = {k: v.to(device) for k, v in target.items()}
            return images, target

    return _handle


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


def resume_from_checkpoint(to_save, conf, device=None):
    # type: (Dict[str, Any], DictConfig, Device) -> None
    to_load = {k: v for k, v in to_save.items() if v is not None}

    if conf.drop_state:
        # we might want to swap optimizer or to reset it state
        drop_keys = set(conf.drop_state)
        to_load = {k: v for k, v in to_load.items() if k not in drop_keys}

    checkpoint = torch.load(conf.load, map_location=device)
    ema_key = "model_ema"
    if ema_key in to_load and ema_key not in checkpoint:
        checkpoint[ema_key] = checkpoint["model"]
        logging.warning("There are no EMA weights in the checkpoint. "
                        "Using saved model weights as a starting point for the EMA.")

    Checkpoint.load_objects(to_load=to_load, checkpoint=checkpoint)


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
