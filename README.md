# Global Wheat Detection
Code for Global Wheat Detection competition, hosted on 
[kaggle.com](https://www.kaggle.com/c/global-wheat-detection).

## Notes
1. [Generate data with style augmentation](./info/DATA.md)
2. [EffDet from Ross Wightman](./info/EFFDET.md)

## Configuration

Train script uses [Hydra framework](https://github.com/facebookresearch/hydra) 
 from Facebook Research.

To change various settings you can either edit *.yaml* files 
in the `config` folder or pass corresponding params to the command line.
The second option is useful for quick testing. For example:

```shell script
python src/train.py train.epoch_length=20 logging.iter_freq=10
```

For more information please visit [Hydra docs](https://hydra.cc/).

## Usage

### Multi-GPU training
Launch distributed training on GPUs:

```shell script
python -m torch.distributed.launch --nproc_per_node=2 --use_env src/train.py
```

It's important to run `torch.distributed.launch` with `--use_env`, 
otherwise [hydra](https://github.com/facebookresearch/hydra) will yell 
at you for passing unrecognized arguments.

## Requirements

* OS: Ubuntu 18.04.4 LTS (5.0.0-37-generic)
* CUDA 10.1.243, driver 435.21
* Conda 4.8.3
* Python 3.7.7
* PyTorch 1.4.0

## Citations

```bibtex
@inproceedings{jackson2019style,
  title={Style Augmentation: Data Augmentation via Style Randomization},
  author={Jackson, Philip T and Atapour-Abarghouei, Amir and Bonner, Stephen and Breckon, Toby P and Obara, Boguslaw},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops},
  pages={83--92},
  year={2019}
}

@Inproceedings{zheng2020distance,
  author    = {Zhaohui Zheng, Ping Wang, Wei Liu, Jinze Li, Rongguang Ye, Dongwei Ren},
  title     = {Distance-IoU Loss: Faster and Better Learning for Bounding Box Regression},
  booktitle = {The AAAI Conference on Artificial Intelligence (AAAI)},
   year      = {2020},
}

@Inproceedings{zheng2020distance,
  author={Roman Solovyev, Weimin Wang},
  title={Weighted Boxes Fusion: ensembling boxes for object detection models},
  journal={ArXiv e-prints},
  archivePrefix={arXiv},
  eprint={1910.13302},
  primaryClass={abs},
  year={2019},
}
```
