import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple

Batch = Tuple[Tensor, Dict[str, Tensor]]
Device = Optional[torch.device]

FloatDict = Dict[str, float]

LossWithStats = Tuple[Tensor, FloatDict]
