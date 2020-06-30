import numpy as np

import cv2
import os
import pandas as pd

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union


def make_dataset(image_dir: str, files: Iterable[str]) -> List[str]:
    image_dir = os.path.expanduser(image_dir)

    if not os.path.isdir(image_dir):
        raise RuntimeError("Unable to read folder {}".format(image_dir))

    images = [os.path.join(image_dir, f) for f in files]
    return images


def cv2_imread(path: Union[str, Path]) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_bbox(bbox: str, bbox_format='pascal_voc') -> Tuple[int]:
    bb = map(float, bbox[1:-1].split(','))
    bb = list(map(int, bb))

    if len(bb) < 4:
        raise ValueError("Dumb bbox: {}".format(bbox))

    # 0 is a label
    if bbox_format == 'coco':
        bb = tuple(bb + [0])
    elif bbox_format == 'pascal_voc':
        x0, y0, w, h = bb
        bb = (x0, y0, x0 + w, y0 + h, 0)
    else:
        raise NotImplementedError(bbox_format)

    return bb


def bbox_str_to_numpy(bbox: str) -> np.ndarray:
    return np.array(read_bbox(bbox), dtype=np.uint16)


Transforms = Callable[[Any], Any]


class WheatDataset(Dataset):
    def __init__(self, image_dir, csv, transforms=None):
        # type: (str, str, Optional[Transforms]) -> WheatDataset
        super(WheatDataset, self).__init__()
        self.transforms = transforms

        df = pd.read_csv(csv)
        ids = df['image_id'].unique()
        files = map(lambda x: x + '.jpg', ids)
        self.images = make_dataset(image_dir, files)

        bboxes = []
        for image_id in tqdm(ids, desc="Parsing bboxes..."):
            image_bb = df.loc[df['image_id'] == image_id, 'bbox']
            image_bb = np.stack(list(map(bbox_str_to_numpy, image_bb)))
            bboxes.append(image_bb)

        self.bboxes = bboxes

    def __getitem__(self, index):
        path = self.images[index]
        bboxes = self.bboxes[index]
        image = cv2_imread(path)

        if self.transforms is not None:
            out = self.transforms(image=image, bboxes=bboxes)
            image, bboxes = out['image'], out['bboxes']

        return image, bboxes

    def __len__(self):
        return len(self.images)
