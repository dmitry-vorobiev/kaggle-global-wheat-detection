import os
import torch
import torchvision

from torch import Tensor


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


def draw_bbox(image: Tensor, bbox: Tensor, channel=0) -> None:
    """Edits image inplace, draws red rectangle

    Args:
        image: tensor of shape (channels, height, width)
        bbox: tensor of shape (x0, y0, x1, y1)
        channel: which channel to use to color boxes
    """
    C, H, W = image.shape
    bbox = bbox.to(torch.int64)
    bbox[[0, 2]] = bbox[[0, 2]].clamp(0, W - 1)
    bbox[[1, 3]] = bbox[[1, 3]].clamp(0, H - 1)
    x0, y0, x1, y1 = list(map(int, bbox))

    C_max = [image[i].max().item() for i in range(C)]
    C_min = [image[i].min().item() for i in range(C)]

    c = [max(1.0, C_max[i]) if channel == i else min(-1.0, C_min[i])
         for i in range(C)]
    c = torch.tensor(c)[:, None]

    image[:, y0, x0:x1 + 1] = c
    image[:, y1, x0:x1 + 1] = c
    image[:, y0:y1 + 1, x0] = c
    image[:, y0:y1 + 1, x1] = c


def visualize_detections(image, targets, predictions) -> None:
    for box in predictions:
        draw_bbox(image, box, channel=0)

    for box in targets:
        draw_bbox(image, box, channel=1)

    device = image.device
    image *= torch.tensor(IMAGENET_DEFAULT_STD, device=device)[:, None, None]
    image += torch.tensor(IMAGENET_DEFAULT_MEAN, device=device)[:, None, None]
    return image.clamp(0, 1)
