# Simple augmentations

My pipeline. Keep in mind, that docs can be outdated, 
so lease check the [current config](../config/train.yaml).

```python
import albumentations as A
import cv2

# efficient_det_d3
height, width = 896, 896

resizes = A.OneOf([
    A.Resize(height, width, interpolation=cv2.INTER_NEAREST, p=1.0),
    A.Resize(height, width, interpolation=cv2.INTER_LINEAR, p=1.0),
    A.Resize(height, width, interpolation=cv2.INTER_CUBIC, p=1.0),
    A.Resize(height, width, interpolation=cv2.INTER_AREA, p=1.0),
])

tfms = [
    # crop or resize
    A.OneOf([resizes, A.RandomCrop(height, width, p=1.0)], p=1.0),
    A.Flip(p=0.75),
    A.RandomRotate90(p=0.75),
    A.ShiftScaleRotate(
        shift_limit=0,
        scale_limit=(0.05, 0.15),
        rotate_limit=20,
        interpolation=cv2.INTER_CUBIC,
        p=0.33),
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
        A.RGBShift(r_shift_limit=20, b_shift_limit=20, g_shift_limit=20, p=1.0),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=10, p=1.0)
    ], p=0.75),
    A.OneOf([
        A.CLAHE(p=1.0),
        A.GaussianBlur(blur_limit=5, p=0.5),
        A.GaussNoise(p=1.0)
    ], p=0.33),
]

tfms = A.Compose(tfms, bbox_params=A.BboxParams('pascal_voc', min_visibility=0.05))
```