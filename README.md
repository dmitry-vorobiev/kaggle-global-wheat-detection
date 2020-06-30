# kaggle-global-wheat-detection
Code for Global Wheat Detection competition, hosted on Kaggle

## Installation

### EfficientDet

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

then you can use it similar to this:

```python
from effdet import create_model

model = create_model(
    'tf_efficientdet_d1', 
    bench_task='',
    pretrained=True,
    pretrained_backbone=True,
    redundant_bias=None,
    checkpoint_path='path/to/checkpoint'
)
```

For reference:
[train.py](https://github.com/rwightman/efficientdet-pytorch/blob/master/train.py)
