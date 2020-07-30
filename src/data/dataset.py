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

from .mosaic import build_mosaic

Sample = Tuple[np.ndarray, Dict[str, np.ndarray]]


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
        bb = tuple(bb)
    elif bbox_format == 'pascal_voc':
        x0, y0, w, h = bb
        bb = (x0, y0, x0 + w, y0 + h)
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


def extract_bboxes(df, ids):
    # type: (pd.DataFrame, Iterable[str]) -> Tuple[List[str], Dict[str, int]]
    id_to_ordinal = dict()
    bboxes = []
    for i, image_id in enumerate(ids):
        image_bb = df.loc[df['image_id'] == image_id, 'bbox']
        id_to_ordinal[image_id] = i
        bboxes.append(image_bb)

    assert len(bboxes) == len(id_to_ordinal)
    return bboxes, id_to_ordinal


def _apply_transforms(image, bboxes, transforms=None):
    # type: (np.ndarray, np.ndarray, Optional[Transforms]) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]
    H0, W0 = image.shape[:2]

    if transforms is not None:
        cls = np.zeros(bboxes.shape[0], dtype=bboxes.dtype)
        out = transforms(image=image, bboxes=bboxes, cls=cls)
        image, bboxes = out['image'], out['bboxes']

        if len(bboxes) > 0:
            bboxes = np.stack(bboxes)[:, :4]
        else:
            bboxes = np.empty((0, 4), dtype=np.float32)
    else:
        image = torch.from_numpy(image)
        bboxes = torch.from_numpy(bboxes)[:, :4]

    return image, bboxes, (H0, W0)


def check_box_format(box_format: str) -> None:
    if box_format not in ["xyxy", "yxyx"]:
        raise AttributeError("Box format '{}' is unsupported".format(box_format))


class WheatDataset(Dataset):
    def __init__(self, image_dir, csv, transforms=None, show_progress=True, source=None,
                 box_format="xyxy", box_lazy=True):
        # type: (str, str, Optional[Transforms], Optional[bool], Optional[DataSource], Optional[str], Optional[bool]) -> None
        super(WheatDataset, self).__init__()
        check_box_format(box_format)
        self.transforms = transforms
        self.box_format = box_format
        self.box_lazy = box_lazy

        df = pd.read_csv(csv)
        if source is not None:
            df = filter_by_source(df, source)
        ids = df['image_id'].unique()

        files = map(lambda x: x + '.jpg', ids)
        self.images = make_dataset(image_dir, files)
        if box_lazy:
            self.bboxes = extract_bboxes(df, ids)[0]
        else:
            self.bboxes = parse_bboxes(df, ids, show_progress=show_progress)[0]
        assert len(self.bboxes) == len(self.images)

    def __getitem__(self, index: int) -> Sample:
        path = self.images[index]
        bboxes = self.bboxes[index]

        if not os.path.exists(path):
            log.warning("Unable to read from {}".format(path))
            index = np.random.randint(len(self))
            # Bad luck :) Lets make another dice roll...
            return self[index]

        image = cv2_imread(path)
        if self.box_lazy:
            bboxes = np.stack(list(map(bbox_str_to_numpy, bboxes)))
        image, bboxes, (H0, W0) = _apply_transforms(image, bboxes, self.transforms)

        # we don't use ToTensor() pipe, so H, W, C -> C, H, W
        image = image.transpose(2, 0, 1)

        if self.box_format == "yxyx":
            bboxes = bboxes[:, [1, 0, 3, 2]]

        H1, W1 = image.shape[:2]
        target = dict(bbox=bboxes,
                      cls=np.array([1], dtype=np.int64),
                      img_scale=min(H0 / H1, W0 / W1),
                      img_size=(W0, H0))
        return image, target

    def __len__(self):
        return len(self.images)


class ExtendedWheatDataset(Dataset):
    def __init__(self, image_dir, csv, gen_image_dirs=None, transforms=None, affine_tfm=None,
                 affine_tfm_mosaic=None, p_mosaic=0.5, mosaic_num_orig=2, show_progress=True,
                 source=None, box_format="xyxy", box_lazy=True):
        super(ExtendedWheatDataset, self).__init__()
        check_box_format(box_format)
        self.transforms = transforms
        self.affine_tfm = affine_tfm
        self.affine_tfm_mosaic = affine_tfm_mosaic
        self.p_mosaic = p_mosaic
        self.mosaic_num_orig = mosaic_num_orig
        self.box_format = box_format
        self.box_lazy = box_lazy

        df = pd.read_csv(csv)
        if source is not None:
            df = filter_by_source(df, source)
        ids = df['image_id'].unique()

        files = map(lambda x: x + '.jpg', ids)
        self.images = make_dataset(image_dir, files)
        self.num_orig_images = len(self.images)
        if gen_image_dirs is not None:
            self._add_gen_images(gen_image_dirs, ids)

        if box_lazy:
            self.bboxes, self.id_to_ordinal = extract_bboxes(df, ids)
        else:
            self.bboxes, self.id_to_ordinal = parse_bboxes(df, ids, show_progress=show_progress)
        assert self.num_orig_images == len(self.bboxes)

    def _add_gen_images(self, dirs: Iterable[str], ids: Union[np.ndarray, Iterable[str]]):
        assert hasattr(self, "images")
        ids = set(list(ids))

        for img_dir in dirs:
            for file in os.listdir(img_dir):
                image_id = self._parse_image_id(file)
                # filter synthetic images the same way as the original ones
                # to avoid leakage on validation
                if image_id in ids:
                    path = os.path.join(img_dir, file)
                    self.images.append(path)

    def __getitem__(self, index: int) -> Sample:
        H0, W0 = 1, 1

        if np.random.rand() > self.p_mosaic:
            image, bboxes = self._read_image_with_boxes(index)
            image, bboxes, (H0, W0) = _apply_transforms(image, bboxes, self.transforms)
            image, bboxes, _ = _apply_transforms(image, bboxes, self.affine_tfm)
        else:
            indices = self._sample_mosaic_indices(index, sample_orig=self.mosaic_num_orig)
            samples = []

            for idx in indices:
                image, bboxes = self._read_image_with_boxes(idx)
                image, bboxes, (H0, W0) = _apply_transforms(image, bboxes, self.transforms)
                samples.append((image, bboxes))

            H, W, C = samples[0][0].shape  # first image
            affine = self.affine_tfm_mosaic or self.affine_tfm
            image, bboxes = build_mosaic(samples, (H * 2, W * 2, C), box_yxyx=False)
            image, bboxes, _ = _apply_transforms(image, bboxes, affine)

        # we don't use ToTensor() pipe, so H, W, C -> C, H, W
        image = image.transpose(2, 0, 1)

        if self.box_format == "yxyx":
            bboxes = bboxes[:, [1, 0, 3, 2]]

        H1, W1 = image.shape[:2]
        target = dict(bbox=bboxes,
                      cls=np.array([1], dtype=np.int64),
                      img_scale=min(H0 / H1, W0 / W1),
                      img_size=(W0, H0))
        return image, target

    def _sample_mosaic_indices(self, index, sample_orig):
        num_orig = self.num_orig_images
        orig_indices = np.random.randint(0, num_orig, sample_orig)
        gen_indices = np.random.randint(num_orig, len(self), 4 - sample_orig)

        # use already sampled index
        if index < num_orig and sample_orig > 0:
            orig_indices[0] = index
        elif index >= num_orig and sample_orig < 4:
            gen_indices[0] = index

        indices = np.concatenate([orig_indices, gen_indices])
        indices = np.random.permutation(indices)
        return indices

    def _read_image_with_boxes(self, index):
        path = self.images[index]

        if index < len(self.bboxes):
            bboxes = self.bboxes[index]
        else:
            bboxes = self._find_orig_bboxes(path)

        if not os.path.exists(path):
            log.warning("Unable to read from {}".format(path))
            index = np.random.randint(len(self))
            # Bad luck :) Lets make another dice roll...
            return self[index]
        image = cv2_imread(path)
        if self.box_lazy:
            bboxes = np.stack(list(map(bbox_str_to_numpy, bboxes)))
        return image, bboxes

    @staticmethod
    def _parse_image_id(filename: str) -> str:
        image_id, ext = os.path.splitext(filename)
        # Gets original image_id. Removes indexes from synthetic images if there are any
        image_id = str(image_id).split('_')[0]
        return image_id

    def _find_orig_bboxes(self, path: str) -> np.ndarray:
        file = os.path.split(path)[-1]
        image_id = self._parse_image_id(file)
        index = self.id_to_ordinal[image_id]
        return self.bboxes[index]

    def __len__(self):
        return len(self.images)
