# EfficientDet

## Installation
To install EffDet PyTorch impl from 
[Ross Wightman](https://github.com/rwightman/efficientdet-pytorch) run these commands:

```shell script
git clone https://github.com/rwightman/efficientdet-pytorch
cd efficientdet-pytorch
python setup.py install
```

Install optional packages if needed.

```shell script
conda install -c conda-forge pycocotools
```

## Usage
For example: 

```python
from effdet import create_model

model = create_model(
    'tf_efficientdet_d1', 
    bench_task='',
    pretrained=False,
    pretrained_backbone=True,
    redundant_bias=None,
    checkpoint_path=''
)
```

One could use `pretrained=True` to get fully-pretrained on the MS COCO dataset EffDet models. In that case the classifier head 
would contain 90 classes and should be discarded.

For reference:
[train.py](https://github.com/rwightman/efficientdet-pytorch/blob/master/train.py),
[example notebook](../nbs/effdet_rwightman.ipynb)
