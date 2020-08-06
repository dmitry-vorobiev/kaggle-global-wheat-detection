# Style augmentation

1. Download model weights first
2. Generate new images with *style augment*. 
Use either runtime augmentation or generate more images upfront.
3. (Optional) Use `ExtendedWheatDataset` with `CustomSampler` to load both original and synthetic data

## Pretrained weights

Weights were ported from the [official repo](https://github.com/philipjackson/style-augmentation) using [this notebook](../nbs/style-aug-impl.ipynb):

[Google drive](https://drive.google.com/file/d/1t_OZZfkfCf8Z26F-jea1tcw9u6SmtWxY/view?usp=sharing)

## Apply styles

### In your code

```python
import torch
from models.style_augment import StyleAugmentNet

style_aug = StyleAugmentNet().cuda()
weights = torch.load('weights.pth')
style_aug.load_state_dict(weights)
style_aug.requires_grad_(False)

styled_images = style_aug(images, alpha=(0.1, 0.33))
```

See [this notebook](../nbs/style-aug.ipynb) with complete example on how to use style augmentation.

### Run script

It will produce `out.num_images` new images and save them in `out.dir`. 
Each generated image should have filename such as `0c3d9007c_1.jpg`, 
where the part before `_` is the original filename.

```shell script
python src/generate_data.py model.alpha=(0.1, 0.33) \
 model.weights=/path/to/weights \
 out.dir=/path/to/save/dir \
 out.num_images=3000
```

See other available options in [generate_data.yaml](../config/generate_data.yaml)

## Load generated images

```python
import albumentations as A
import cv2
import torch

from pathlib import Path

from data.dataset import ExtendedWheatDataset
from data.sampler import CustomSampler
from data.utils import basic_collate

DATA_DIR = Path('/path/to/data')
image_dir = DATA_DIR/'train'
csv_path = DATA_DIR/'train.csv'
gen_image_dirs = ['/path/to/gen/images']

tfms = [
    A.Flip(),
    A.RandomRotate90(),
    A.Resize(512, 512, interpolation=cv2.INTER_AREA),
    A.Normalize(),
    A.pytorch.transforms.ToTensorV2()
]
tfms = A.Compose(tfms, bbox_params=A.BboxParams('pascal_voc'))

ds = ExtendedWheatDataset(image_dir, csv_path, 
                          gen_image_dirs=gen_image_dirs, 
                          transforms=tfms, 
                          source=['usask_1'])

sampler = CustomSampler(ds, orig_images_ratio=0.5)
dl = torch.utils.data.DataLoader(ds, sampler=sampler, batch_size=8, collate_fn=basic_collate)
```

## Citation

```bibtex
@inproceedings{jackson2019style,
  title={Style Augmentation: Data Augmentation via Style Randomization},
  author={Jackson, Philip T and Atapour-Abarghouei, Amir and Bonner, Stephen and Breckon, Toby P and Obara, Boguslaw},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={83--92},
  year={2019}
}
```