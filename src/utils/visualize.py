import numpy as np
import cv2
import logging
import math
import os
import torch

from ignite.engine import Engine, Events
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, Optional

from .train import build_process_batch_func
from .typings import Device


def draw_bboxes(image, bboxes, color, box_format='coco', yxyx=False):
    if yxyx:
        bboxes = bboxes[:, [1, 0, 3, 2]]

    for box in bboxes:
        pt1 = tuple(box[:2])
        if box_format == 'coco':
            pt2 = tuple(box[:2] + box[2:])
        elif box_format == 'pascal_voc':
            pt2 = tuple(box[2:])
        else:
            raise AttributeError("Not supported: {}".format(box_format))
        cv2.rectangle(image, pt1, pt2, color, 1)


def save_image(image, path):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, image)


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
        _handle_batch_val = build_process_batch_func(conf.data, stage="val", device=device)

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
