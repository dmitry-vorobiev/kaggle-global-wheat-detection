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

from effdet import DetBenchTrain, DetBenchPredict, EfficientDet, load_checkpoint, load_pretrained
from effdet.efficientdet import HeadNet

from loss.efficient_det import DetectionLoss


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
