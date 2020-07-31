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

    # Normalize?
    box_a = torch.sigmoid(box_a)
    box_b = torch.sigmoid(box_b)

    # Expecting yxyx format, but might work for xyxy (not tested)
    H_a = torch.exp(box_a[:, 2] - box_a[:, 0])  # y1 - y0
    W_a = torch.exp(box_a[:, 3] - box_a[:, 1])  # x1 - x0
    H_b = torch.exp(box_b[:, 2] - box_b[:, 0])
    W_b = torch.exp(box_b[:, 3] - box_b[:, 1])

    # Center points
    yc_a = box_a[:, 0] + H_a / 2  # y0 + h/2
    xc_a = box_a[:, 1] + W_a / 2  # x0 + w/2
    yc_b = box_b[:, 0] + H_b / 2
    xc_b = box_b[:, 1] + W_b / 2

    # Intersection
    inter_yx0 = torch.max(box_a[:, [0, 1]], box_b[:, [0, 1]])
    inter_yx1 = torch.min(box_a[:, [2, 3]], box_b[:, [2, 3]])
    H_i = torch.clamp_min(inter_yx1[:, 0] - inter_yx0[:, 0], 0)
    W_i = torch.clamp_min(inter_yx1[:, 1] - inter_yx0[:, 1], 0)

    inter_area = W_i * H_i
    union_area = (W_a * H_a) + (W_b * H_b) - inter_area
    iou = inter_area / (union_area + eps)

    # Enclosing box
    clos_yx0 = torch.min(box_a[:, [0, 1]], box_b[:, [0, 1]])
    clos_yx1 = torch.max(box_a[:, [2, 3]], box_b[:, [2, 3]])
    H_c = torch.clamp_min(clos_yx1[:, 0] - clos_yx0[:, 0], 0)
    W_c = torch.clamp_min(clos_yx1[:, 1] - clos_yx0[:, 1], 0)

    # Distance term
    clos_diag = torch.pow(H_c, 2) + torch.pow(W_c, 2)
    inter_diag = torch.pow(yc_b - yc_a, 2) + torch.pow(xc_b - xc_a, 2)
    u = inter_diag / (clos_diag + eps)

    # Shape consistency term
    v = torch.atan(W_b / H_b) - torch.atan(W_a / H_a)
    v = (4 / math.pi ** 2) * torch.pow(v, 2)

    with torch.no_grad():
        s = (iou > 0.5).to(dtype=iou.dtype)
        alpha = s * v / ((1 + eps) - iou + v)

    ciou = iou - u - alpha * v
    return torch.clamp(ciou, -1, 1)
