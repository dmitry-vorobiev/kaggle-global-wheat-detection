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

### Transfer images to GPU faster

The idea is to send images using **uint8** data type, which takes only 1 byte per pixel,
whereas **float32** occupies 4 bytes per each pixel. 
So essentially you send 4 times less data using **uint8**.

```python
import albumentations as A
import cv2
import torch
from pathlib import Path

from data.dataset import WheatDataset
# https://github.com/rwightman/efficientdet-pytorch/blob/master/data/loader.py
from data.loader import fast_collate

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

DATA_DIR = Path('/path/to/your/data')
image_dir = DATA_DIR/'train'
csv_path = DATA_DIR/'train.csv'

tfms = [
    A.Flip(),
    A.RandomRotate90(),
    A.Resize(640, 640, interpolation=cv2.INTER_AREA),
]
tfms = A.Compose(tfms, bbox_params=A.BboxParams('pascal_voc'))

ds = WheatDataset(image_dir, csv_path, transforms=tfms, source=['usask_1'])
dl = torch.utils.data.DataLoader(ds, batch_size=4, shuffle=False, collate_fn=fast_collate)

device = torch.device('cuda')
mean = torch.tensor(IMAGENET_DEFAULT_MEAN).to(device).view(1, 3, 1, 1).mul_(255)
std = torch.tensor(IMAGENET_DEFAULT_STD).to(device).view(1, 3, 1, 1).mul_(255)

for x, y in dl:
    x = x.permute(0, 3, 1, 2).to(device).float().sub_(mean).div_(std)
    y = {k: v.to(device) for k, v in y.items()}
    out = model(x, y)
    # then do what you must to do...
```
