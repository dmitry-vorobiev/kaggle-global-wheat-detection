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

from utils.common import mean_std_tensors
from utils.visualize import draw_bboxes, save_image


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
    assert predictions.shape[1] == 6
    parts = []

    for p in predictions:
        parts.append(float(p[4]))  # score
        parts += [int(round(c)) for c in p[:4].tolist()]

    return " ".join(map(str, parts))


@hydra.main(config_path="../config/predict.yaml")
def main(conf: DictConfig):
    print(conf.pretty())

    if 'seed' in conf and conf.seed:
        torch.manual_seed(conf.seed)

    torch.cuda.set_device(conf.gpu)
    device = torch.device('cuda')

    out_dir = conf.out.get('dir', os.path.join(os.getcwd(), 'test_prediction'))
    make_dir(out_dir)

    dl = create_data_loader(conf.data)

    model = instantiate(conf.model).to(device)
    assert conf.model.params.bench_name == "predict"
    state_dict = torch.load(conf.model.weights)
    model.model.load_state_dict(state_dict)
    model.eval()
    model.requires_grad_(False)

    num_images_to_save = conf.out.num_images
    image_dir = conf.out.get('image_dir', os.path.join(os.getcwd(), 'images'))
    save_images = num_images_to_save > 0
    if save_images:
        if not os.path.isabs(image_dir):
            image_dir = os.path.join(out_dir, image_dir)
            make_dir(image_dir)
        logging.info("Saving images to {}".format(image_dir))

    min_score = conf.get("min_score", -1)
    mean, std = mean_std_tensors(conf.data, device)

    files = os.listdir(conf.data.params.image_dir)
    df = pd.DataFrame(np.empty((len(files), 2)), columns=["image_id", "PredictionString"])
    i_image = 0
    s_image = 0

    for images, image_ids, metadata in tqdm(dl, desc="Predict"):
        images = images.to(device).float().sub_(mean).div_(std)
        img_scale = metadata['img_scale'].to(dtype=torch.float, device=device)
        assert len(metadata['img_size']) == 2
        img_size = torch.stack(metadata['img_size'], dim=1).to(dtype=torch.float, device=device)

        predictions = model(images, img_scale, img_size).cpu()
        assert len(image_ids) == len(predictions)

        if save_images:
            images = (images * std + mean).clamp(0, 255).permute(0, 2, 3, 1)
            images = images.cpu().numpy().astype(np.uint8)
        else:
            images = None
        del img_scale, img_size

        for j, image_id in enumerate(image_ids):
            scores_i = predictions[j, :, 4]
            pred_i = predictions[j, scores_i >= min_score]

            df.iloc[i_image, 0] = image_id
            df.iloc[i_image, 1] = stringify(pred_i)
            i_image += 1

            if save_images and s_image < num_images_to_save:
                image = images[j]
                draw_bboxes(image, pred_i[:, :4], (0, 255, 0), box_format='coco')
                path = os.path.join(image_dir, '%s.png' % image_id)
                save_image(image, path)
                s_image += 1
                del image

            del pred_i, scores_i

        del predictions

    logging.info("Saving {} to {}".format(conf.out.file, out_dir))
    path = os.path.join(out_dir, conf.out.file)
    df.to_csv(path, index=False)
    print("DONE")


if __name__ == '__main__':
    main()
