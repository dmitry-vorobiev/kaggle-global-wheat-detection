"""
Wrappers for ease of use in the hydra configs
"""
import albumentations as A
import cv2


def as_tuple(v):
    return tuple(v) if hasattr(v, '__getitem__') else v


def resize(height: int, width: int, interpolation=cv2.INTER_LINEAR):
    def _build(method):
        return A.Resize(height, width, interpolation=method, p=1.0)

    if hasattr(interpolation, '__iter__'):
        tfm = A.OneOf(list(map(_build, interpolation)), p=1.0)
    else:
        tfm = _build(interpolation)
    return tfm


def resize_or_crop(height: int, width: int, interpolation=cv2.INTER_LINEAR):
    return A.OneOf([
        resize(height, width, interpolation),
        A.RandomCrop(height, width, p=1.0),
    ], p=1.0)


def affine(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=cv2.INTER_LINEAR,
           border_mode=cv2.BORDER_REFLECT_101, value=None, p=0.5):
    return A.ShiftScaleRotate(shift_limit=as_tuple(shift_limit),
                              scale_limit=as_tuple(scale_limit),
                              rotate_limit=as_tuple(rotate_limit),
                              interpolation=interpolation,
                              border_mode=border_mode,
                              value=as_tuple(value),
                              p=p)


def color_jitter(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True,
                 r_shift_limit=20, g_shift_limit=20, b_shift_limit=20,
                 hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5):
    return A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=brightness_limit,
                                   contrast_limit=contrast_limit,
                                   brightness_by_max=brightness_by_max,
                                   p=1.0),
        A.RGBShift(r_shift_limit=r_shift_limit,
                   b_shift_limit=b_shift_limit,
                   g_shift_limit=g_shift_limit,
                   p=1.0),
        A.HueSaturationValue(hue_shift_limit=hue_shift_limit,
                             sat_shift_limit=sat_shift_limit,
                             val_shift_limit=val_shift_limit,
                             p=1.0)
    ], p=p)


def enhancer(clip_limit=4.0, tile_grid_size=(8, 8), noise_var_limit=(10.0, 50.0),
             noise_mean=0, p=0.5):
    return A.OneOf([
        A.CLAHE(clip_limit=clip_limit, tile_grid_size=as_tuple(tile_grid_size), p=1.0),
        A.GaussNoise(var_limit=as_tuple(noise_var_limit), mean=noise_mean, p=1.0),
        # A.IAASharpen()
    ], p=p)
