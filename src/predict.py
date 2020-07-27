import numpy as np
import albumentations as A
import hydra
import logging
import os
import pandas as pd
import torch

from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def make_dir(dir_path: str) -> None:
    if os.path.isfile(dir_path) or os.path.splitext(dir_path)[-1]:
        raise AttributeError('{} is not a directory.'.format(dir_path))
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def create_dataset(conf):
    # type: (DictConfig) -> Dataset
    def _build_tfm(conf_tfm: DictConfig):
        tfm = [instantiate(v) for k, v in conf_tfm.items()]
        return A.Compose(tfm)

    ds = instantiate(conf)
    # hydra will raise "ValueError: key transforms: Compose is not a primitive type",
    # if you try to pass transforms directly to instantiate(...)
    for field in ["transforms", "affine_tfm", "affine_tfm_mosaic"]:
        if field in conf:
            c = getattr(conf, field)
            setattr(ds, field, _build_tfm(c))
    print("Found {} images".format(len(ds)))
    return ds


def create_data_loader(conf: DictConfig) -> DataLoader:
    data = create_dataset(conf)
    loader = DataLoader(data,
                        batch_size=conf.loader.batch_size,
                        num_workers=conf.get('loader.workers', 0),
                        drop_last=False,
                        shuffle=False)
    return loader


def stringify(predictions):
    assert predictions.ndim == 2
    assert predictions.size(1) == 6
    parts = []

    for p in predictions:
        parts.append(float(p[4]))  # score
        parts += [int(round(c)) for c in p[:4].tolist()]

    return " ".join(map(str, parts))


@hydra.main(config_path="../config/predict.yaml")
def main(conf: DictConfig):
    if 'seed' in conf and conf.seed:
        torch.manual_seed(conf.seed)

    torch.cuda.set_device(conf.gpu)
    device = torch.device('cuda')

    out_dir = conf.out.get('dir', os.path.join(os.getcwd(), 'test_prediction'))
    make_dir(out_dir)

    dl = create_data_loader(conf.data)
    model = instantiate(conf.model).to(device)
    model.eval()
    model.requires_grad_(False)

    min_score = conf.get("min_score", -1)
    mean = torch.tensor(list(conf.data.mean)).to(device).view(1, 3, 1, 1).mul_(255)
    std = torch.tensor(list(conf.data.std)).to(device).view(1, 3, 1, 1).mul_(255)

    files = os.listdir(conf.data.params.image_dir)
    df = pd.DataFrame(np.empty((len(files), 2)), columns=["image_id", "PredictionString"])
    i_image = 0

    for images, image_ids, metadata in tqdm(dl, desc="Predict"):
        images = images.permute(0, 3, 1, 2).to(device).float().sub_(mean).div_(std)
        img_scale = metadata['img_scale'].to(dtype=torch.float, device=device)
        assert len(metadata['img_size']) == 2
        img_size = torch.stack(metadata['img_size'], dim=1).to(dtype=torch.float, device=device)

        predictions = model(images, img_scales=img_scale, img_size=img_size)
        predictions = predictions.cpu()
        assert len(image_ids) == len(predictions)

        for j, image_id in enumerate(image_ids):
            # filter by min score
            mask = predictions[j, :, 4] >= min_score
            pred_i = predictions[j, mask]

            df.iloc[i_image, 0] = image_id
            df.iloc[i_image, 1] = stringify(pred_i)
            i_image += 1

    logging.info("Saving {} to {}".format(conf.out.file, out_dir))
    path = os.path.join(out_dir, conf.out.file)
    df.to_csv(path, index=False)
    print("DONE")


if __name__ == '__main__':
    main()
