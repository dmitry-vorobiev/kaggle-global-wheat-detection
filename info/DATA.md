# Style augmentation

## Pretrained weights

Weights were ported from the [official repo](https://github.com/philipjackson/style-augmentation) using [this notebook](../nbs/style-aug-impl.ipynb):

[Google drive](https://drive.google.com/file/d/1t_OZZfkfCf8Z26F-jea1tcw9u6SmtWxY/view?usp=sharing)

## How to use

1. Download model weights first
2. Generate new images with *style augment*.
3. Use `ExtendedWheatDataset` to load both original and synthetic data

### From code

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