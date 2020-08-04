import numpy as np
import torch
from ensemble_boxes.ensemble_boxes_wbf import weighted_boxes_fusion
from itertools import combinations
from torch import Tensor
from typing import List


class BaseTTA:
    """ author: @shonenkov """
    def __init__(self, image_size: int):
        self.image_size = image_size

    def __call__(self, images):
        raise NotImplementedError

    def decode(self, boxes):
        raise NotImplementedError

    @staticmethod
    def check_images(images):
        assert isinstance(images, torch.Tensor)
        N, C, H, W = images.shape
        assert C == 3
        assert H == W

    @staticmethod
    def prepare_boxes(boxes, box_format):
        assert box_format in ["xywh", "xyxy", "yxyx", "coco", "pascal_voc"]
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        boxes = boxes.copy()
        if box_format in ["xywh", "coco"]:
            boxes[:, [2, 3]] += boxes[:, [0, 1]]
        elif box_format == "yxyx":
            boxes[:] = boxes[:, [1, 0, 3, 2]]
        return boxes


class HorizontalFlipTTA(BaseTTA):
    """ author: @shonenkov """

    def __call__(self, images):
        self.check_images(images)
        return images.flip(3)

    def decode(self, boxes):
        boxes = boxes.copy()
        # xyxy
        boxes[:, [0, 2]] = self.image_size - boxes[:, [2, 0]]
        return boxes


class VerticalFlipTTA(BaseTTA):
    """ author: @shonenkov """

    def __call__(self, images):
        self.check_images(images)
        return images.flip(2)

    def decode(self, boxes):
        boxes = boxes.copy()
        # xyxy
        boxes[:, [1, 3]] = self.image_size - boxes[:, [3, 1]]
        return boxes


class Rotate90TTA(BaseTTA):
    """ author: @shonenkov """

    def __call__(self, images):
        self.check_images(images)
        return torch.rot90(images, 1, (2, 3))

    def decode(self, boxes):
        res_boxes = boxes.copy()
        res_boxes[:, [0, 2]] = self.image_size - boxes[:, [1, 3]]
        res_boxes[:, [1, 3]] = boxes[:, [2, 0]]
        return res_boxes


class ComposeTTA(BaseTTA):
    """ author: @shonenkov """

    def __init__(self, transforms: List[BaseTTA]):
        super(ComposeTTA, self).__init__(0)
        self.transforms = transforms

    def __call__(self, images):
        for transform in self.transforms:
            images = transform(images)
        return images

    def sanitize_boxes(self, boxes):
        result_boxes = boxes.copy()
        result_boxes[:, 0] = np.min(boxes[:, [0, 2]], axis=1)
        result_boxes[:, 2] = np.max(boxes[:, [0, 2]], axis=1)
        result_boxes[:, 1] = np.min(boxes[:, [1, 3]], axis=1)
        result_boxes[:, 3] = np.max(boxes[:, [1, 3]], axis=1)
        return np.clip(result_boxes, 0, self.transforms[0].image_size)

    def decode(self, boxes):
        for transform in reversed(self.transforms):
            boxes = transform.decode(boxes)
        return self.sanitize_boxes(boxes)


def combine_tta(image_size: int):
    tta_pipes = []
    ttas = [HorizontalFlipTTA(image_size), VerticalFlipTTA(image_size), Rotate90TTA(image_size)]
    for tta_comb in combinations(ttas, 2):
        tta_comb = ComposeTTA(list(filter(bool, tta_comb)))
        tta_pipes.append(tta_comb)
    return tta_pipes


def ensemble_predictions(predictions, image_size=1024, iou_threshold=0.5, skip_box_threshold=0.0,
                         box_format="coco"):
    assert predictions[0].ndim == 3
    assert predictions[0].size(2) == 6
    fused_predictions = []
    norm = image_size - 1
    num_images = len(predictions[0])

    for i in range(num_images):
        boxes = [(p[i, :, :4] / norm).tolist() for p in predictions]
        scores = [p[i, :, 4] for p in predictions]
        labels = [p[i, :, 5] for p in predictions]
        boxes, scores, labels = weighted_boxes_fusion(
            boxes, scores, labels, weights=None,
            iou_thr=iou_threshold,
            skip_box_thr=skip_box_threshold)

        boxes *= norm
        scores = scores[:, None]
        labels = labels[:, None]

        if box_format == "coco":
            boxes[:, [2, 3]] -= boxes[:, [0, 1]]

        fused_i = list(map(torch.from_numpy, [boxes, scores, labels]))
        fused_i = torch.cat(fused_i, dim=1)
        fused_predictions.append(fused_i)

    return fused_predictions
