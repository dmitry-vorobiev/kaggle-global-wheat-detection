from torch import Tensor
# noinspection PyUnresolvedReferences
from torch.utils.data.dataloader import default_collate
from typing import Any, Iterable, List, Tuple


def collate(batch: Iterable[Tuple[Any, Any]]) -> Tuple[Tensor, List[Tensor]]:
    images, targets = tuple(zip(*batch))
    images = default_collate(images)
    targets = list(map(default_collate, targets))
    return images, targets
