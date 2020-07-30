import albumentations as A
import hydra
import logging
import os
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from data.loader import PrefetchLoader
from data.no_labels import ImagesWithFileNames


def make_dir(dir_path: str) -> None:
    if os.path.isfile(dir_path) or os.path.splitext(dir_path)[-1]:
        raise AttributeError('{} is not a directory.'.format(dir_path))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def create_dataset(conf: DictConfig) -> ImagesWithFileNames:
    transforms = None
    if conf.transforms:
        transforms = [instantiate(v) for k, v in conf.transforms.items()]
        compose = T.Compose
        if all(isinstance(t, A.BasicTransform) for t in transforms):
            compose = A.Compose
        transforms = compose(transforms)

    ds = ImagesWithFileNames(**conf.params, transforms=transforms)
    print("Found {} images".format(len(ds)))
    return ds


def create_data_loader(conf: DictConfig, mean=None, std=None) -> DataLoader:
    data = create_dataset(conf)
    loader = DataLoader(data,
                        batch_size=conf.loader.batch_size,
                        num_workers=conf.get('loader.workers', 0),
                        shuffle=False)
    if conf.loader.prefetch:
        loader = PrefetchLoader(loader, mean=conf.mean, std=conf.std, ignore_target=True)
    return loader


def _build_process_batch_func(conf: DictConfig, device=None, dtype=torch.float):
    prefetch = conf.loader.prefetch
    scale = conf.upsample.factor
    mode = conf.upsample.method

    def _upsample(images):
        align_corners = None
        if mode in ["bilinear", "bicubic"]:
            align_corners = False
        return F.interpolate(images, scale_factor=scale, mode=mode, align_corners=align_corners)

    kwargs = dict(device=device, dtype=dtype)
    mean = torch.tensor(list(conf.mean)).to(**kwargs).view(1, 3, 1, 1).mul_(255)
    std = torch.tensor(list(conf.std)).to(**kwargs).view(1, 3, 1, 1).mul_(255)

    def _handle(batch):
        images, files = batch
        images = images.to(**kwargs).sub_(mean).div_(std)
        if scale > 1:
            images = _upsample(images)
        return images, files

    return _handle


def _build_postproc_func(conf: DictConfig):
    scale = 1 / max(conf.downsample.factor, 1)
    mode = conf.downsample.method

    if scale < 1:
        def _downsample(images):
            align_corners = None
            if mode in ["bilinear", "bicubic"]:
                align_corners = False
            return F.interpolate(images, scale_factor=scale, mode=mode,
                                 align_corners=align_corners)
    else:
        def _downsample(images):
            return images

    return _downsample


@hydra.main(config_path="../config/generate_data.yaml")
def main(conf: DictConfig):
    torch.backends.cudnn.benchmark = True

    if 'seed' in conf and conf.seed:
        torch.manual_seed(conf.seed)
    if 'gpu' in conf:
        torch.cuda.set_device(conf.gpu)

    device = torch.device('cuda')
    dtype = torch.half if conf.fp16 else torch.float
    num_images = conf.out.num_images

    alpha = conf.model.alpha
    if isinstance(alpha, ListConfig):
        assert len(alpha) == 2
        alpha = tuple(alpha)

    out_dir = conf.out.get('dir', os.path.join(os.getcwd(), 'generated_images'))
    extension = conf.out.ext
    make_dir(out_dir)
    logging.info("Saving {} images to {}".format(extension.upper(), out_dir))

    dl = create_data_loader(conf.data)
    _handle_batch = _build_process_batch_func(conf.data, device=device, dtype=dtype)
    _postproc = _build_postproc_func(conf.data)

    model = instantiate(conf.model)
    model = model.to(device=device, dtype=dtype)
    weights = conf.model.weights
    logging.info("Loading weights from {}".format(weights))
    state_dict = torch.load(weights)
    model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(False)

    pbar = tqdm(desc="Generating images", total=num_images, unit=' img')
    i_img = 0
    i_data = 0
    data = iter(dl)

    while i_img < num_images:
        try:
            batch = next(data)
        except StopIteration:
            data = iter(dl)
            batch = next(data)
            i_data += 1

        images, names = _handle_batch(batch)
        batch[0] = None
        images = model(images, alpha=alpha)
        images = _postproc(images).cpu().float()

        for image, name in zip(images, names):
            file = '{}_{}.{}'.format(name, i_data, extension)
            path = os.path.join(out_dir, file)
            save_image(image, path, nrow=1, normalize=True)
            pbar.update(1)
            i_img += 1

    pbar.close()
    print("DONE")


if __name__ == '__main__':
    main()
