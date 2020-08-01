import math
import torch
from torch import Tensor


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

    hw_a = box_a[:, [2, 3]] - box_a[:, [0, 1]]
    hw_a = torch.sigmoid(hw_a).exp()

    hw_b = box_b[:, [2, 3]] - box_b[:, [0, 1]]
    hw_b = torch.sigmoid(hw_b).exp()

    # Center points (y,x)
    c_a = (box_a[:, [0, 1]] + box_a[:, [2, 3]]) / 2
    c_a = torch.sigmoid(c_a)

    c_b = (box_b[:, [0, 1]] + box_b[:, [2, 3]]) / 2
    c_b = torch.sigmoid(c_b)

    yx0_a = c_a - hw_a / 2
    yx1_a = c_a + hw_a / 2
    yx0_b = c_b - hw_b / 2
    yx1_b = c_b + hw_b / 2

    # Intersection
    inter_yx0 = torch.max(yx0_a, yx0_b)
    inter_yx1 = torch.min(yx1_a, yx1_b)
    inter_hw = torch.clamp_min(inter_yx1 - inter_yx0, 0)

    area_a = hw_a[:, 0] * hw_a[:, 1]
    area_b = hw_b[:, 0] * hw_b[:, 1]
    inter_area = inter_hw[:, 0] * inter_hw[:, 1]
    union_area = area_a + area_b - inter_area
    iou = inter_area / (union_area + eps)

    # Enclosing box
    clos_yx0 = torch.min(yx0_a, yx0_b)
    clos_yx1 = torch.max(yx1_a, yx1_b)
    clos_hw = torch.clamp_min(clos_yx1 - clos_yx0, 0)

    # Distance term
    clos_diag = torch.pow(clos_hw, 2).sum(dim=1)
    inter_diag = torch.pow(c_b - c_a, 2).sum(dim=1)
    u = inter_diag / (clos_diag + eps)

    # Shape consistency term
    v = torch.atan(hw_a[:, 0] / hw_a[:, 1]) - torch.atan(hw_b[:, 0] / hw_b[:, 1])
    v = (4 / math.pi ** 2) * torch.pow(v, 2)

    with torch.no_grad():
        s = iou > 0.5
        alpha = s * v / ((1 - iou + v) + eps)

    ciou = iou - u - alpha * v
    return torch.clamp(ciou, -1, 1)
