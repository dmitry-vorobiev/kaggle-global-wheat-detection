# Model factory for EfficientDet by Ross Wightman
# https://github.com/rwightman/efficientdet-pytorch
#
# Original:
# https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/factory.py
#
# Copyright 2020 Ross Wightman
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
import torch
import torch.nn as nn
from effdet import DetBenchPredict, EfficientDet, load_checkpoint, load_pretrained
from effdet.anchors import Anchors, AnchorLabeler, MAX_DETECTIONS_PER_IMAGE, MAX_DETECTION_POINTS, \
    decode_box_outputs, clip_boxes_xyxy
from effdet.bench import _post_process, _batch_detection
from effdet.efficientdet import HeadNet

from loss.efficient_det import DetectionLoss
from nms.cluster_nms import cluster_nms_dist_iou


def create_model_from_config(config, bench_name='', pretrained=False, checkpoint_path='', **kwargs):
    model = EfficientDet(config, **kwargs)

    # FIXME handle different head classes / anchors and re-init of necessary layers w/ pretrained load

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path)
    elif pretrained:
        load_pretrained(model, config.url)

    config = copy.deepcopy(config)
    # override num classes
    config.num_classes = 1
    model.class_net = HeadNet(config, num_outputs=1, norm_kwargs=dict(eps=0.001, momentum=0.01))

    # wrap model in task specific bench if set
    if bench_name == 'train':
        model = DetBenchTrain(model, config)
        model.loss_fn = DetectionLoss(config)
    elif bench_name == 'predict':
        model = DetBenchPredict(model, config)
    return model


def generate_detections_cluster_nms(
        cls_outputs, box_outputs, anchor_boxes, indices, img_scale, img_size, min_score,
        iou_threshold, max_det_per_image: int = 200):
    """Generates detections with RetinaNet model outputs and anchors.

    Args:
        cls_outputs: a torch tensor with shape [N, 1], which has the highest class
            scores on all feature levels. The N is the number of selected
            top-K total anchors on all levels.  (k being MAX_DETECTION_POINTS)

        box_outputs: a torch tensor with shape [N, 4], which stacks box regression
            outputs on all feature levels. The N is the number of selected top-k
            total anchors on all levels. (k being MAX_DETECTION_POINTS)

        anchor_boxes: a torch tensor with shape [N, 4], which stacks anchors on all
            feature levels. The N is the number of selected top-k total anchors on all levels.

        indices: a torch tensor with shape [N], which is the indices from top-k selection.

        classes: a torch tensor with shape [N], which represents the class
            prediction on all selected anchors from top-k selection.

        img_scale: a float tensor representing the scale between original image
            and input image for the detector. It is used to rescale detections for
            evaluating with the original groundtruth annotations.

        max_det_per_image: an int constant, added as argument to make torchscript happy

    Returns:
        detections: detection results in a tensor with shape [MAX_DETECTION_POINTS, 6],
            each row representing [x, y, width, height, score, class]
    """
    anchor_boxes = anchor_boxes[indices, :]

    # apply bounding box regression to anchors
    boxes = decode_box_outputs(box_outputs.float(), anchor_boxes, output_xyxy=True)
    boxes = clip_boxes_xyxy(boxes, img_size / img_scale)  # clip before NMS better?

    scores = cls_outputs.sigmoid().float()
    confidence, _ = torch.max(scores, dim=-1)
    keep = confidence >= min_score
    scores = scores[keep]
    boxes = boxes[keep, :]

    boxes, scores, classes = cluster_nms_dist_iou(
        boxes, scores, iou_threshold=iou_threshold, top_k=MAX_DETECTION_POINTS)
    boxes = boxes[:max_det_per_image]
    scores = scores[:max_det_per_image, None]
    classes = classes[:max_det_per_image, None]

    # xyxy to xywh & rescale to original image
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]
    boxes *= img_scale

    classes += 1  # back to class idx with background class = 0

    # stack em and pad out to MAX_DETECTIONS_PER_IMAGE if necessary
    detections = torch.cat([boxes, scores, classes.float()], dim=1)
    num_pad = max_det_per_image - len(detections)
    if num_pad > 0:
        detections = torch.cat([
            detections,
            torch.zeros((num_pad, 6), device=detections.device, dtype=detections.dtype)
        ], dim=0)
    return detections


def _batch_detection_cluster_nms(batch_size: int, class_out, box_out, anchor_boxes, indices,
                                 img_scale, img_size, min_score, iou_threshold):
    batch_detections = []
    # FIXME we may be able to do this as a batch with some tensor reshaping/indexing, PR welcome
    for i in range(batch_size):
        detections = generate_detections_cluster_nms(
            class_out[i], box_out[i], anchor_boxes, indices[i], img_scale[i], img_size[i],
            min_score, iou_threshold)
        batch_detections.append(detections)
    return torch.stack(batch_detections, dim=0)


class DetBenchTrain(nn.Module):
    def __init__(self, model, config):
        super(DetBenchTrain, self).__init__()
        self.config = config
        self.model = model
        self.anchors = Anchors(
            config.min_level, config.max_level,
            config.num_scales, config.aspect_ratios,
            config.anchor_scale, config.image_size)
        self.anchor_labeler = AnchorLabeler(self.anchors, config.num_classes, match_threshold=0.5)
        self.loss_fn = DetectionLoss(self.config)

    def forward(self, x, target):
        class_out, box_out = self.model(x)
        cls_targets, box_targets, num_positives = self.anchor_labeler.batch_label_anchors(
            x.shape[0], target['bbox'], target['cls'])
        loss, class_loss, box_loss = self.loss_fn(class_out, box_out, cls_targets, box_targets, num_positives)
        output = dict(loss=loss, class_loss=class_loss, box_loss=box_loss)
        if not self.training:
            # if eval mode, output detections for evaluation
            class_out, box_out, indices, classes = _post_process(self.config, class_out, box_out)
            if self.config.custom_nms:
                min_score = self.config.nms_min_score
                iou_threshold = self.config.nms_max_iou
                detections = _batch_detection_cluster_nms(
                    x.shape[0], class_out, box_out, self.anchors.boxes, indices,
                    target['img_scale'], target['img_size'], min_score, iou_threshold)
            else:
                detections = _batch_detection(
                    x.shape[0], class_out, box_out, self.anchors.boxes, indices, classes,
                    target['img_scale'], target['img_size'])
            output['detections'] = detections
        return output
