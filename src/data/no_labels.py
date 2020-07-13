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
    def __init__(self, image_dir, csv, transforms=None, source=None):
        # type: (str, str, Optional[Transforms], Optional[DataSource]) -> None
        super(ImagesWithFileNames, self).__init__()
        self.transforms = transforms

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

        if self.transforms is not None:
            out = self.transforms(image=image)
            image = out['image']
        else:
            image = torch.from_numpy(image)

        return image, str(name)

    def __len__(self):
        return len(self.images)
