import torch
from torch import Tensor
from typing import Tuple

FloatInterval = Tuple[float, float]


def interval_to_tensor(batch_size: int, interval: FloatInterval, device=None) -> Tensor:
    if len(interval) != 2:
        raise AttributeError("interval must have two entries: (min, max)")
    start, end = interval
    if end <= start or any(filter(lambda x: x < 0. or 1. < x, interval)):
        raise AttributeError("interval values must satisfy: 0.0 <= `min_val` < `max_val` <= 1.0")
    rnd = torch.rand(batch_size, 1, device=device)
    return start + (end - start) * rnd
