import torch
from torch import Tensor
from typing import Tuple

FloatInterval = Tuple[float, float]


def interval_to_tensor(batch_size: int, interval: FloatInterval, device=None) -> Tensor:
    rnd = torch.rand(batch_size, 1, device=device)
    start, end = interval
    return start + (end - start) * rnd
