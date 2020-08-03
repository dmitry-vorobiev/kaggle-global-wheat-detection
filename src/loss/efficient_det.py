"""
Copyright 2020 Ross Wightman

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Original:
https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/loss.py
Add a new option to use CIoU loss instead of Huber loss for box regression
"""
import torch
import torch.nn.functional as F
from effdet.loss import _box_loss, _classification_loss
from torch import nn, Tensor
from typing import List, Optional

from .ciou import ciou


def ciou_loss(inp, target, weights=None, size_average=True):
    # type: (Tensor, Tensor, Optional[Tensor], Optional[bool]) -> Tensor
    assert inp.shape == target.shape
    N, H, W, A = inp.shape
    A = A // 4
    # TODO: Why the inp is float32 and the target is float16 when the amp mode is on?
    inp = inp.view(-1, 4)
    target = target.reshape(-1, 4).to(inp.dtype)
    loss = 1 - ciou(inp, target).view(N, H, W, A)
    if weights is not None:
        loss *= weights.reshape(N, H, W, A, 4).any(dim=-1)
    return loss.mean() if size_average else loss.sum()


def _box_loss_ciou(box_outputs, box_targets, num_positives, delta: float = 0.1):
    """Computes box regression loss."""
    # delta is typically around the mean value of regression target.
    # for instances, the regression targets of 512x512 input with 6 anchors on
    # P3-P7 pyramid is about [0.1, 0.1, 0.2, 0.2].
    normalizer = num_positives * 4.0
    mask = box_targets != 0.0
    box_loss = ciou_loss(box_targets, box_outputs, weights=mask, size_average=False)
    box_loss /= normalizer
    return box_loss


class DetectionLoss(nn.Module):
    def __init__(self, config):
        super(DetectionLoss, self).__init__()
        self.config = config
        self.num_classes = config.num_classes
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.delta = config.delta
        self.box_loss_weight = config.box_loss_weight

        box_loss = config.get("box_loss", "huber")
        if box_loss == "huber":
            self.box_loss_fn = _box_loss
        elif box_loss == "ciou":
            self.box_loss_fn = _box_loss_ciou
        else:
            raise ValueError("Not supported: {}".format(box_loss))

    def forward(
            self, cls_outputs: List[torch.Tensor], box_outputs: List[torch.Tensor],
            cls_targets: List[torch.Tensor], box_targets: List[torch.Tensor], num_positives: torch.Tensor):
        """Computes total detection loss.
        Computes total detection loss including box and class loss from all levels.
        Args:
            cls_outputs: a List with values representing logits in [batch_size, height, width, num_anchors].
                at each feature level (index)

            box_outputs: a List with values representing box regression targets in
                [batch_size, height, width, num_anchors * 4] at each feature level (index)

            cls_targets: groundtruth class targets.

            box_targets: groundtrusth box targets.

            num_positives: num positive grountruth anchors

        Returns:
            total_loss: an integer tensor representing total loss reducing from class and box losses from all levels.

            cls_loss: an integer tensor representing total class loss.

            box_loss: an integer tensor representing total box regression loss.
        """
        # Sum all positives in a batch for normalization and avoid zero
        # num_positives_sum, which would lead to inf loss during training
        num_positives_sum = num_positives.sum() + 1.0
        levels = len(cls_outputs)

        cls_losses = []
        box_losses = []
        for l in range(levels):
            cls_targets_at_level = cls_targets[l]
            box_targets_at_level = box_targets[l]

            # Onehot encoding for classification labels.
            # NOTE: PyTorch one-hot does not handle -ve entries (no hot) like Tensorflow, so mask them out
            cls_targets_non_neg = cls_targets_at_level >= 0
            cls_targets_at_level_oh = F.one_hot(cls_targets_at_level * cls_targets_non_neg, self.num_classes)
            cls_targets_at_level_oh = torch.where(
               cls_targets_non_neg.unsqueeze(-1), cls_targets_at_level_oh, torch.zeros_like(cls_targets_at_level_oh))

            bs, height, width, _, _ = cls_targets_at_level_oh.shape
            cls_targets_at_level_oh = cls_targets_at_level_oh.view(bs, height, width, -1)
            cls_loss = _classification_loss(
                cls_outputs[l].permute(0, 2, 3, 1),
                cls_targets_at_level_oh,
                num_positives_sum,
                alpha=self.alpha, gamma=self.gamma)
            cls_loss = cls_loss.view(bs, height, width, -1, self.num_classes)
            cls_loss *= (cls_targets_at_level != -2).unsqueeze(-1).float()
            cls_losses.append(cls_loss.sum())

            box_losses.append(self.box_loss_fn(
                box_outputs[l].permute(0, 2, 3, 1),
                box_targets_at_level,
                num_positives_sum,
                delta=self.delta))

        # Sum per level losses to total loss.
        cls_loss = torch.sum(torch.stack(cls_losses, dim=-1), dim=-1)
        box_loss = torch.sum(torch.stack(box_losses, dim=-1), dim=-1)
        total_loss = cls_loss + self.box_loss_weight * box_loss
        return total_loss, cls_loss, box_loss
