# Global Wheat Detection
Code for Global Wheat Detection competition, hosted on 
[kaggle.com](https://www.kaggle.com/c/global-wheat-detection).

## Notes
[Generate data with style augmentation](./info/DATA.md)

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

## Installation

My environment:
* OS: Ubuntu 18.04.4 LTS (5.0.0-37-generic)
* CUDA 10.1.243, driver 435.21
* Conda 4.8.3
* Python 3.7.7
* PyTorch 1.4.0

### EfficientDet

To install EffDet PyTorch impl from 
[Ross Wightman](https://github.com/rwightman/efficientdet-pytorch) run these commands:

```shell script
git clone https://github.com/rwightman/efficientdet-pytorch
cd efficientdet-pytorch
python setup.py install
```

Install optional packages if needed.

```shell script
conda install -c conda-forge pycocotools
```

then you can use it similar to this:

```python
from effdet import create_model

model = create_model(
    'tf_efficientdet_d1', 
    bench_task='',
    pretrained=False,
    pretrained_backbone=True,
    redundant_bias=None,
    checkpoint_path=''
)
```

One could use `pretrained=True` to get fully-pretrained on the MS COCO dataset EffDet models. In that case the classifier head 
would contain 90 classes and should be discarded.

For reference:
[train.py](https://github.com/rwightman/efficientdet-pytorch/blob/master/train.py),
[example notebook](./nbs/effdet_rwightman.ipynb)
