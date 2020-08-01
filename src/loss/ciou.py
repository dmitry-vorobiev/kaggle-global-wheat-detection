import math
import torch
from torch import Tensor
from torch.jit import trace
from typing import Tuple


EPS = torch.tensor(1e-8)


def _yxyx_to_c_hw_yx0_yx1(box: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    center = (box[:, [0, 1]] + box[:, [2, 3]]) / 2
    center = torch.sigmoid(center)

    hw = box[:, [2, 3]] - box[:, [0, 1]]
    hw = torch.sigmoid(hw).exp()

    yx0 = center - hw / 2
    yx1 = center + hw / 2
    return center, hw, yx0, yx1


def _calc_iou(hw_a: Tensor, hw_b: Tensor, inter_hw: Tensor, eps=EPS) -> Tensor:
    area_a = hw_a[:, 0] * hw_a[:, 1]
    area_b = hw_b[:, 0] * hw_b[:, 1]
    inter_area = inter_hw[:, 0] * inter_hw[:, 1]
    union_area = area_a + area_b - inter_area
    return inter_area / (union_area + eps)


def _calc_closing_diag(yx0_a, yx1_a, yx0_b, yx1_b) -> Tensor:
    # Enclosing box
    c_yx0 = torch.min(yx0_a, yx0_b)
    c_yx1 = torch.max(yx1_a, yx1_b)
    c_hw = torch.clamp_min(c_yx1 - c_yx0, 0)
    return torch.pow(c_hw, 2).sum(dim=1)


def _calc_shape_consistency(hw_a: Tensor, hw_b: Tensor, iou: Tensor, eps=EPS) -> Tensor:
    # Shape consistency term
    v = torch.atan(hw_a[:, 0] / hw_a[:, 1]) - torch.atan(hw_b[:, 0] / hw_b[:, 1])
    v = (4 / math.pi ** 2) * torch.pow(v, 2)

    with torch.no_grad():
        s = iou > 0.5
        alpha = s * v / ((1 - iou + v) + eps)

    return alpha * v


yxyx_to_c_hw_yx0_yx1 = trace(_yxyx_to_c_hw_yx0_yx1, (torch.randn(10, 4),))
calc_iou = trace(_calc_iou,
                 (torch.randn(10, 2),
                  torch.randn(10, 2),
                  torch.randn(10, 2),
                  EPS))
calc_closing_diag = trace(_calc_closing_diag, tuple(torch.randn(10, 2) for _ in range(4)))
calc_shape_consistency = trace(_calc_shape_consistency,
                               (torch.randn(10, 2),
                                torch.randn(10, 2),
                                torch.randn(10),
                                EPS))


def ciou(box_a: Tensor, box_b: Tensor, eps=1e-8):
    """
    Args:
        box_a: tensor of shape [boxes, 4]
        box_b: tensor of shape [boxes, 4]
        eps: float

    Original:
    https://github.com/Zzh-tju/CIoU/blob/8995056b1e93b86d03c384f042514391b70e58e0/layers/modules/multibox_loss.py#L11
    """
    assert box_a.shape == box_b.shape
    if box_a.size(0) < 1:
        return torch.zeros(0)

    c_a, hw_a, yx0_a, yx1_a = yxyx_to_c_hw_yx0_yx1(box_a)
    c_b, hw_b, yx0_b, yx1_b = yxyx_to_c_hw_yx0_yx1(box_b)

    # Intersection
    inter_yx0 = torch.max(yx0_a, yx0_b)
    inter_yx1 = torch.min(yx1_a, yx1_b)
    inter_hw = torch.clamp_min(inter_yx1 - inter_yx0, 0)

    iou = calc_iou(hw_a, hw_b, inter_hw, eps=torch.tensor(eps))
    del inter_yx0, inter_yx1, inter_hw

    c_diag = calc_closing_diag(yx0_a, yx1_a, yx0_b, yx1_b)
    inter_diag = torch.pow(c_b - c_a, 2).sum(dim=1)
    u = inter_diag / (c_diag + eps)
    del inter_diag, c_a, c_b, yx0_a, yx1_a, yx0_b, yx1_b

    v = calc_shape_consistency(hw_a, hw_b, iou, eps=torch.tensor(eps))
    ciou = iou - u - v
    return torch.clamp(ciou, -1, 1)
