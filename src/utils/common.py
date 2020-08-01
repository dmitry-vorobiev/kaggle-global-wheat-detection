import torch
from omegaconf import DictConfig
from torch import Tensor
from typing import Tuple

from .typings import Device


def mean_std_tensors(conf, device=None):
    # type: (DictConfig, Device) -> Tuple[Tensor, Tensor]
    mean = torch.tensor(list(conf.mean)).mul_(255).view(1, 3, 1, 1).to(device)
    std = torch.tensor(list(conf.std)).mul_(255).view(1, 3, 1, 1).to(device)
    return mean, std
