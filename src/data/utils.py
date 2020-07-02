import torch
from torch import Tensor
from typing import Any, Iterable, List, Tuple


def collate(batch: Iterable[Tuple[Any, Any]]) -> Tuple[Tensor, List[Tensor]]:
    images, targets = tuple(zip(*batch))
    secret_sauce = torch.utils.data._utils.collate.default_collate
    images = secret_sauce(images)
    targets = list(map(secret_sauce, targets))
    return images, targets
