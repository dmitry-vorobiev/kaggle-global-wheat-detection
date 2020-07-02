"""
Overrides for some factory functions in rwightman/efficientdet-pytorch
Changed get_efficientdet_config method, because h.update gives me error

Original:
https://github.com/rwightman/efficientdet-pytorch/blob/master/effdet/factory.py
"""


from effdet import default_detection_model_configs, load_checkpoint, load_pretrained, \
    EfficientDet, DetBenchTrain, DetBenchPredict
from effdet.config.model_config import efficientdet_model_param_dict
from omegaconf import OmegaConf


def get_efficientdet_config(model_name='tf_efficientdet_d1'):
    """Get the default config for EfficientDet based on model name."""
    config = default_detection_model_configs()
    config.num_classes = 1
    model_config = efficientdet_model_param_dict[model_name]
    return OmegaConf.merge(config, OmegaConf.create(model_config))


def create_model(model_name, bench_task='', pretrained=False, checkpoint_path='',
                 checkpoint_ema=False, **kwargs):
    config = get_efficientdet_config(model_name)

    pretrained_backbone = kwargs.pop('pretrained_backbone', True)
    if pretrained or checkpoint_path:
        pretrained_backbone = False  # no point in loading backbone weights

    redundant_bias = kwargs.pop('redundant_bias', None)
    if redundant_bias is not None:
        # override config if set to something
        config.redundant_bias = redundant_bias

    model = EfficientDet(config, pretrained_backbone=pretrained_backbone, **kwargs)

    # FIXME handle different head classes / anchors and re-init of necessary layers w/ pretrained load

    if checkpoint_path:
        load_checkpoint(model, checkpoint_path, use_ema=checkpoint_ema)
    elif pretrained:
        load_pretrained(model, config.url)

    # wrap model in task specific bench if set
    if bench_task == 'train':
        model = DetBenchTrain(model, config)
    elif bench_task == 'predict':
        model = DetBenchPredict(model, config)
    return model
