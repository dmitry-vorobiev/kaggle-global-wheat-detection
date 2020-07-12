# Style augmentation

## Pretrained weights

Weights were ported from the [official repo](https://github.com/philipjackson/style-augmentation) using [this notebook](../nbs/style-aug-impl.ipynb):

[Google drive](https://drive.google.com/file/d/1t_OZZfkfCf8Z26F-jea1tcw9u6SmtWxY/view?usp=sharing)

## How to use

```python
import torch
from models.style_augment import StyleAugmentNet

style_aug = StyleAugmentNet().cuda()
weights = torch.load('weights.pth')
style_aug.load_state_dict(weights)
style_aug.requires_grad_(False)

# you may also use float value here
alpha = torch.rand(8) * 0.33 + 0.33  # 0.33 ... 0.66
styled_images = style_aug(images, alpha=alpha)
```

See [this notebook](../nbs/style-aug.ipynb) with complete example on how to use style augmentation.