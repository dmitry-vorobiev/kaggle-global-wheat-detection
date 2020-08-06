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
