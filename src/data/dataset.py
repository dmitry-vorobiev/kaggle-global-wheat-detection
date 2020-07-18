import numpy as np

import cv2
import logging
import os
import pandas as pd
import torch

from pathlib import Path
from tqdm import tqdm
from torch.utils.data import Dataset
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union


log = logging.getLogger(__name__)


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
    bb = [int(float(b)) for b in bbox[1:-1].split(',')]

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
DataSource = Union[str, Sequence[str]]


def filter_by_source(df: pd.DataFrame, source: DataSource) -> pd.DataFrame:
    if isinstance(source, str):
        source = [source]
    elif isinstance(source, Iterable):
        source = list(source)
    elif not (isinstance(source, Sequence) or hasattr(source, '__getitem__')):
        raise AttributeError("source should be of type Sequence or str")

    mask = df['source'].isin(source)
    return df[mask]


def parse_bboxes(df, ids, show_progress=True):
    # type: (pd.DataFrame, Iterable[str], Optional[bool]) -> Tuple[List[np.ndarray], Dict[str, int]]
    if show_progress:
        ids = tqdm(ids, desc="Parsing bboxes...")

    id_to_ordinal = dict()
    bboxes = []
    for i, image_id in enumerate(ids):
        image_bb = df.loc[df['image_id'] == image_id, 'bbox']
        image_bb = np.stack(list(map(bbox_str_to_numpy, image_bb)))
        id_to_ordinal[image_id] = i
        bboxes.append(image_bb)

    assert len(bboxes) == len(id_to_ordinal)
    return bboxes, id_to_ordinal


class WheatDataset(Dataset):
    def __init__(self, image_dir, csv, transforms=None, show_progress=True, source=None):
        # type: (str, str, Optional[Transforms], Optional[bool], Optional[DataSource]) -> None
        super(WheatDataset, self).__init__()
        self.transforms = transforms

        df = pd.read_csv(csv)
        if source is not None:
            df = filter_by_source(df, source)

        ids = df['image_id'].unique()
        files = map(lambda x: x + '.jpg', ids)
        self.images = make_dataset(image_dir, files)
        self.bboxes = parse_bboxes(df, ids, show_progress=show_progress)[0]
        assert len(self.bboxes) == len(self.images)

    def __getitem__(self, index):
        path = self.images[index]

        if not os.path.exists(path):
            log.warning("Unable to read from {}".format(path))
            index = np.random.randint(len(self))
            # Bad luck :) Lets make another dice roll...
            return self[index]

        image = cv2_imread(path)
        bboxes = self.bboxes[index]

        if self.transforms is not None:
            out = self.transforms(image=image, bboxes=bboxes)
            image, bboxes = out['image'], out['bboxes']
            bboxes = np.stack(bboxes)
        else:
            image = torch.from_numpy(image)
            bboxes = torch.from_numpy(bboxes)

        # Remove class label and downcast from float64 to int16
        # to send less data to GPU. Some boxes have fractional part of .5,
        # but for high res images this shouldn't be an issue
        bboxes = bboxes[:, :4].astype(np.int16)
        return image, bboxes

    def __len__(self):
        return len(self.images)


class ExtendedWheatDataset(Dataset):
    def __init__(self, image_dir, csv, gen_image_dirs=None, transforms=None, show_progress=True,
                 source=None):
        # type: (str, str, Optional[Iterable[str]], Optional[Transforms], Optional[bool], Optional[DataSource]) -> None
        super(ExtendedWheatDataset, self).__init__()
        self.transforms = transforms

        df = pd.read_csv(csv)
        ids = df['image_id'].unique()
        bboxes, id_to_ordinal = parse_bboxes(df, ids, show_progress=show_progress)
        self.bboxes = bboxes
        self.id_to_ordinal = id_to_ordinal

        if source is not None:
            df = filter_by_source(df, source)
            ids = df['image_id'].unique()

        files = map(lambda x: x + '.jpg', ids)
        self.images = make_dataset(image_dir, files)
        self.num_orig_images = len(self.images)

        if gen_image_dirs is not None:
            ids = set(ids.tolist())
            for img_dir in gen_image_dirs:
                for file in os.listdir(img_dir):
                    image_id = self._parse_image_id(file)
                    # filter synthetic images the same way as the original ones
                    # to avoid leakage on validation
                    if image_id in ids:
                        path = os.path.join(img_dir, file)
                        self.images.append(path)

    def __getitem__(self, index):
        path = self.images[index]
        bboxes = self._find_bboxes(path)

        if not os.path.exists(path):
            log.warning("Unable to read from {}".format(path))
            index = np.random.randint(len(self))
            # Bad luck :) Lets make another dice roll...
            return self[index]

        image = cv2_imread(path)

        if self.transforms is not None:
            out = self.transforms(image=image, bboxes=bboxes)
            image, bboxes = out['image'], out['bboxes']
            bboxes = np.stack(bboxes)
        else:
            image = torch.from_numpy(image)
            bboxes = torch.from_numpy(bboxes)

        # Remove class label and downcast from float64 to int16
        # to send less data to GPU. Some boxes have fractional part of .5,
        # but for high res images this shouldn't be an issue
        bboxes = bboxes[:, :4].astype(np.int16)
        return image, bboxes

    @staticmethod
    def _parse_image_id(filename: str) -> str:
        image_id, ext = os.path.splitext(filename)
        # Gets original image_id. Removes indexes from synthetic images if there are any
        image_id = str(image_id).split('_')[0]
        return image_id

    def _find_bboxes(self, path: str) -> np.ndarray:
        file = os.path.split(path)[-1]
        image_id = self._parse_image_id(file)
        index = self.id_to_ordinal[image_id]
        return self.bboxes[index]

    def __len__(self):
        return len(self.images)
