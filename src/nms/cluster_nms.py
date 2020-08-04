import torch
from torch import Tensor

EPS = torch.tensor(1e-8)


@torch.jit.script
def dist_iou_ab(box_a: Tensor, box_b: Tensor, eps=EPS):
    """
    Args:
        box_a: tensor of shape [batch_size, boxes_a, 4]
        box_b: tensor of shape [batch_size, boxes_b, 4]
        gamma: float
        eps: float

    Original:
    https://github.com/Zzh-tju/CIoU/blob/8995056b1e93b86d03c384f042514391b70e58e0/layers/functions/detection.py#L162
    https://github.com/Zzh-tju/CIoU/blob/8995056b1e93b86d03c384f042514391b70e58e0/layers/box_utils.py#L82
    """
    assert box_a.dim() == 3
    assert box_b.dim() == 3
    assert box_a.size(0) == box_b.size(0)

    A, B = box_a.size(1), box_b.size(1)
    box_a = box_a.unsqueeze(2).expand(-1, -1, A, -1)
    box_b = box_b.unsqueeze(1).expand(-1, B, -1, -1)

    inter_yx0 = torch.max(box_a[..., :2], box_b[..., :2])
    inter_yx1 = torch.min(box_a[..., 2:4], box_b[..., 2:4])

    inter_hw = torch.clamp_min_(inter_yx1 - inter_yx0, 0)
    inter_area = torch.prod(inter_hw, dim=-1)
    # del inter_hw, inter_yx0, inter_yx1

    hw_a = box_a[..., 2:4] - box_a[..., :2]
    hw_b = box_b[..., 2:4] - box_b[..., :2]

    area_a = torch.prod(hw_a, dim=-1)
    area_b = torch.prod(hw_b, dim=-1)

    union_area = area_a + area_b - inter_area
    iou = inter_area / (union_area + eps)
    # del inter_area, union_area, area_a, area_b, hw_a, hw_b

    center_a = (box_a[..., :2] + box_a[..., 2:4]) / 2
    center_b = (box_b[..., :2] + box_b[..., 2:4]) / 2
    inter_diag = torch.pow(center_b - center_a, 2).sum(dim=-1)

    clos_yx0 = torch.min(box_a[..., :2], box_b[..., :2])
    clos_yx1 = torch.max(box_a[..., 2:4], box_b[..., 2:4])
    clos_hw = torch.clamp_min_(clos_yx1 - clos_yx0, 0)
    clos_diag = torch.pow(clos_hw, 2).sum(dim=-1)
    # del clos_yx0, clos_yx1, clos_hw, center_a, center_b

    dist = inter_diag / (clos_diag + eps)
    return iou - dist ** 0.9


def cluster_nms_dist_iou(boxes: Tensor, scores: Tensor, iou_threshold=0.5, top_k=200):
    assert boxes.dim() == 2
    assert scores.dim() == 2
    assert boxes.size(0) == scores.size(0)

    scores, classes = torch.max(scores, dim=1)
    # scores: [detections]
    _, idx = scores.sort(descending=True)
    idx = idx[:top_k]
    # add batch dim
    top_k_boxes = boxes[idx][None, ...]

    # [1, top_k, top_k] -> [top_k, top_k]
    iou = dist_iou_ab(top_k_boxes, top_k_boxes)[0]
    iou = iou.triu_(diagonal=1)
    best_iou = torch.zeros_like(idx)

    iou_b = iou
    for i in range(top_k):
        iou_a = iou_b
        best_iou, _ = torch.max(iou_b, dim=0)
        # keep far away boxes
        keep = (best_iou <= iou_threshold)[:, None].expand_as(iou_b)
        iou_b = torch.where(keep, iou, torch.zeros_like(iou_b))
        if iou_b.eq(iou_a).all():
            break

    idx = idx[best_iou <= iou_threshold]
    return boxes[idx], scores[idx], classes[idx]
