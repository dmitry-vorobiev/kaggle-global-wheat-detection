import albumentations as A
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, DistributedSampler
# noinspection PyUnresolvedReferences
from torch.utils.data.dataloader import default_collate
from typing import Any, Iterable, List, Optional, Tuple

from .dataset import ExtendedWheatDataset
from .loader import PrefetchLoader, fast_collate
from .sampler import CustomSampler

Batch = Iterable[Tuple[Any, Any]]
float_3 = Tuple[float, float, float]


def basic_collate(batch: Batch) -> Tuple[Tensor, List[Tensor]]:
    images, targets = tuple(zip(*batch))
    images = default_collate(images)
    targets = list(map(default_collate, targets))
    return images, targets


def create_dataset(conf, show_progress=False, name="train"):
    # type: (DictConfig, Optional[bool], Optional[str]) -> Dataset
    def _build_tfm(conf_tfm: DictConfig):
        tfm = [instantiate(v) for k, v in conf_tfm.items()]
        tfm = A.Compose(tfm, bbox_params=A.BboxParams(**conf.bbox_params, label_fields=["cls"]))
        return tfm

    if show_progress:
        print("Loading {} data...".format(name))
    ds = instantiate(conf, show_progress=show_progress)
    # hydra will raise "ValueError: key transforms: Compose is not a primitive type",
    # if you try to pass transforms directly to instantiate(...)
    for field in ["transforms", "affine_tfm", "affine_tfm_mosaic"]:
        if field in conf:
            c = getattr(conf, field)
            setattr(ds, field, _build_tfm(c))
    if show_progress:
        print("{}: {} images".format(name, len(ds)))
    return ds


def create_train_loader(conf, rank=None, num_replicas=None, mean=None, std=None):
    # type: (DictConfig, Optional[int], Optional[int], Optional[float_3], Optional[float_3]) -> DataLoader
    show_progress = rank is None or rank == 0
    data = create_dataset(conf, show_progress=show_progress, name="train")

    sampler = None
    if isinstance(data, ExtendedWheatDataset):
        sampler = CustomSampler(data,
                                orig_images_ratio=conf.get("orig_images_ratio", 0.5),
                                num_replicas=num_replicas,
                                rank=rank)
    elif num_replicas is not None:
        sampler = DistributedSampler(data, num_replicas=num_replicas, rank=rank)

    loader = DataLoader(data,
                        sampler=sampler,
                        batch_size=conf.loader.batch_size,
                        num_workers=conf.get('loader.workers', 0),
                        collate_fn=fast_collate,
                        drop_last=True,
                        shuffle=not sampler)
    if conf.loader.prefetch:
        loader = PrefetchLoader(loader, mean=mean, std=std)
    return loader


def create_val_loader(conf, rank=None, num_replicas=None, mean=None, std=None):
    # type: (DictConfig, Optional[int], Optional[int], Optional[float_3], Optional[float_3]) -> DataLoader
    show_progress = rank is None or rank == 0
    data = create_dataset(conf, show_progress=show_progress, name="val")

    sampler = None
    if num_replicas is not None:
        sampler = DistributedSampler(data, num_replicas=num_replicas, rank=rank, shuffle=False)

    loader = DataLoader(data,
                        sampler=sampler,
                        batch_size=conf.loader.batch_size,
                        num_workers=conf.get('loader.workers', 0),
                        collate_fn=fast_collate,
                        drop_last=False,
                        shuffle=not sampler)
    if conf.loader.prefetch:
        loader = PrefetchLoader(loader, mean=mean, std=std)
    return loader
