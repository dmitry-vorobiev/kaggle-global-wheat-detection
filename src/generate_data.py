import albumentations as A
import hydra
import logging
import os
import torch
import torchvision.transforms as T

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from data.no_labels import ImagesWithFileNames


def make_dir(dir_path: str) -> None:
    if os.path.isfile(dir_path) or os.path.splitext(dir_path)[-1]:
        raise AttributeError('{} is not a directory.'.format(dir_path))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def create_dataset(conf: DictConfig, transforms: DictConfig) -> ImagesWithFileNames:
    transforms = [instantiate(v) for k, v in transforms.items()]
    compose = T.Compose
    if all(isinstance(t, A.BasicTransform) for t in transforms):
        compose = A.Compose
    transforms = compose(transforms)
    ds = ImagesWithFileNames(**conf.params, transforms=transforms)
    print("Found {} images".format(len(ds)))
    return ds


def create_data_loader(conf: DictConfig) -> DataLoader:
    data = create_dataset(conf, conf.transforms)
    loader = DataLoader(data,
                        batch_size=conf.loader.batch_size,
                        num_workers=conf.get('loader.workers', 0),
                        shuffle=False)
    return loader


@hydra.main(config_path="../config/generate_data.yaml")
def main(conf: DictConfig):
    torch.manual_seed(conf.seed)
    torch.cuda.set_device(conf.gpu)
    device = torch.device('cuda')
    num_images = conf.out.num_images
    alpha = conf.model.alpha

    out_dir = conf.out.get('dir', os.path.join(os.getcwd(), 'generated_images'))
    extension = conf.out.ext
    make_dir(out_dir)
    logging.info("Saving {} images to {}".format(extension.upper(), out_dir))

    dl = create_data_loader(conf.data)
    model = instantiate(conf.model).to(device)

    weights = conf.model.weights
    logging.info("Loading weights from {}".format(weights))
    state_dict = torch.load(weights)
    model.load_state_dict(state_dict)
    model.requires_grad_(False)

    pbar = tqdm(desc="Generating images", total=num_images, unit=' img')
    i = 0
    data = iter(dl)

    while i < num_images:
        try:
            images, names = next(data)
        except StopIteration:
            data = iter(dl)
            images, names = next(data)

        images = images.to(device)
        images = model(images, alpha=alpha)

        for image, name in zip(images, names):
            file = '{}_{}.{}'.format(name, i, extension)
            path = os.path.join(out_dir, file)
            save_image(image, path, nrow=1, normalize=True)
            pbar.update(1)
            i += 1

    pbar.close()
    print("DONE")


if __name__ == '__main__':
    main()
