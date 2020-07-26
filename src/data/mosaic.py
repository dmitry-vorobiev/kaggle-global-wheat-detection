import numpy as np
from typing import List, Tuple

int_2 = Tuple[int, int]
int_3 = Tuple[int, int, int]
ImageWithBoxes = Tuple[np.ndarray, np.ndarray]


def mosaic_yxyx(mos_size: int_2, img_size: np.ndarray):
    g1 = np.array([
        [-1, -1, 0, 0],
        [0, -1, 1, 0],
        [-1, 0, 0, 1],
        [0, 0, 1, 1]
    ])

    g2 = np.array([
        [1, 1, 1, 1],
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [0, 0, 1, 1]
    ])

    H, W = mos_size
    mos_hw = np.ones_like(g1) * np.array([H, W, H, W], dtype=np.int)
    orig_hw = np.concatenate([img_size, img_size], axis=-1).astype(np.int)

    mos_yx = g1 * orig_hw + mos_hw // 2
    mos_yx[:, [0, 2]] = np.clip(mos_yx[:, [0, 2]], 0, H)
    mos_yx[:, [1, 3]] = np.clip(mos_yx[:, [1, 3]], 0, W)

    slice_hw = mos_yx[:, 2:] - mos_yx[:, :2]
    slice_hw = np.concatenate([slice_hw] * 2, axis=1)

    inp_yx = (g1 * slice_hw) + (g2 * orig_hw)
    inp_yx[:, 2] = np.clip(inp_yx[:, 2], 0, inp_yx[:, 0] + slice_hw[:, 2])
    inp_yx[:, 3] = np.clip(inp_yx[:, 3], 0, inp_yx[:, 1] + slice_hw[:, 3])

    offset_yx = mos_yx[:, :2] - inp_yx[:, :2]

    return mos_yx, inp_yx, offset_yx


def build_mosaic(samples: List[ImageWithBoxes], shape: int_3, box_yxyx=False) -> ImageWithBoxes:
    assert len(samples) == 4

    H, W, C = shape
    img_sizes = np.stack([img.shape[:2] for img, tg in samples])
    mos_yx, img_yx, off_yx = mosaic_yxyx((H, W), img_sizes)

    mosaic = np.zeros(shape, dtype=np.uint8)
    mos_boxes = []

    for i in range(4):
        image, boxes = samples[i]
        y0m, x0m, y1m, x1m = mos_yx[i]
        y0i, x0i, y1i, x1i = img_yx[i]
        mosaic[y0m:y1m, x0m:x1m] = image[y0i:y1i, x0i:x1i]

        offset = off_yx[i]
        if not box_yxyx:
            offset = offset[::-1]
        boxes = boxes + np.concatenate([offset] * 2)
        mos_boxes.append(boxes)

    return mosaic, np.concatenate(mos_boxes)
