import torch

from torch import Tensor
# noinspection PyUnresolvedReferences
from torch.utils.data.dataloader import default_collate
from typing import Any, Dict, Iterable, List, Tuple, Union

Batch = Iterable[Tuple[Any, Any]]


def basic_collate(batch: Batch) -> Tuple[Tensor, List[Tensor]]:
    images, targets = tuple(zip(*batch))
    images = default_collate(images)
    targets = list(map(default_collate, targets))
    return images, targets


def collate_dict(batch: Batch) -> Tuple[Tensor, Dict[str, Union[Tensor, List[Tensor]]]]:
    images, bboxes = basic_collate(batch)
    cls = [torch.ones(len(b)) for b in bboxes]
    target = dict(bbox=bboxes, cls=cls)
    return images, target
