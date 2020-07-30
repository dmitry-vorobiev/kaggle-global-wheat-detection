import numpy as np

import logging
import os
import pandas as pd
import torch

from torch.utils.data import Dataset
from typing import Optional

from .dataset import Transforms, DataSource, cv2_imread, filter_by_source, make_dataset

log = logging.getLogger(__name__)


class ImagesWithFileNames(Dataset):
    def __init__(self, image_dir, csv=None, transforms=None, source=None, metadata=False):
        # type: (str, str, Optional[Transforms], Optional[DataSource], Optional[bool]) -> None
        super(ImagesWithFileNames, self).__init__()
        self.transforms = transforms
        self.metadata = metadata

        if csv is None:
            if not os.path.isdir(image_dir):
                raise ValueError("Not a valid directory: {}".format(image_dir))
            files = os.listdir(image_dir)
            ids = [os.path.splitext(file)[0] for file in files]
        else:
            df = pd.read_csv(csv)
            if source is not None:
                df = filter_by_source(df, source)

            ids = df['image_id'].unique()
            files = map(lambda x: x + '.jpg', ids)

        self.names = ids
        self.images = make_dataset(image_dir, files)

    def __getitem__(self, index):
        name = self.names[index]
        path = self.images[index]

        if not os.path.exists(path):
            log.warning("Unable to read from {}".format(path))
            index = np.random.randint(len(self))
            # Bad luck :) Lets make another dice roll...
            return self[index]

        image = cv2_imread(path)
        H0, W0 = image.shape[:2]

        if self.transforms is not None:
            out = self.transforms(image=image)
            image = out['image']

        H1, W1 = image.shape[:2]
        # we don't use ToTensor() pipe, so H, W, C -> C, H, W
        image = image.transpose(2, 0, 1)

        if self.metadata:
            meta = dict(img_scale=min(H0 / H1, W0 / W1),
                        img_size=(W0, H0))
            return image, str(name), meta
        else:
            return image, str(name)

    def __len__(self):
        return len(self.images)
