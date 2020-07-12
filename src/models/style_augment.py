import torch

from collections import OrderedDict
from torch import nn, Tensor
from typing import Optional


class EncoderLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        # type: (int, int, int, Optional[int]) -> None
        layers = OrderedDict()

        padding = kernel_size // 2
        if padding:
            layers['pad'] = nn.ReflectionPad2d(padding)

        layers['conv'] = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        layers['norm'] = nn.InstanceNorm2d(out_channels)
        layers['act'] = nn.ReLU(inplace=True)
        super(EncoderLayer, self).__init__(layers)


class StyledLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, style_dim=100,
                 upsample=0, act_fn=nn.ReLU):
        # type: (int, int, int, Optional[int], Optional[int], Optional[int], Optional[nn.Module]) -> None
        super(StyledLayer, self).__init__()
        layers = OrderedDict()

        if upsample > 1:
            layers['upsample'] = nn.Upsample(scale_factor=float(upsample), mode="nearest")

        padding = kernel_size // 2
        if padding:
            layers['pad'] = nn.ReflectionPad2d(padding)

        layers['conv'] = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=0)
        layers['norm'] = nn.InstanceNorm2d(out_channels)
        self.layers = nn.Sequential(layers)

        self.beta = nn.Linear(style_dim, out_channels)
        self.gamma = nn.Linear(style_dim, out_channels)

        if act_fn is not None:
            self.act = act_fn(inplace=True)
        else:
            self.act = None

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        # maybe try cuda streams?
        beta, gamma = self.beta(style), self.gamma(style)
        x = self.layers(x)

        x = x * gamma[:, :, None, None]
        x += beta[:, :, None, None]

        if self.act is not None:
            x = self.act(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResBlock, self).__init__()
        self.conv1 = StyledLayer(channels, channels, 3)
        self.conv2 = StyledLayer(channels, channels, 3, act_fn=None)

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        y = self.conv1(x, style)
        y = self.conv2(y, style)
        return x + y


class TransformNet(nn.Module):
    def __init__(self, img_channels=3):
        # type: (Optional[int]) -> None
        super(TransformNet, self).__init__()

        self.encoder = nn.Sequential(
            EncoderLayer(img_channels, 32, kernel_size=9, stride=1),
            EncoderLayer(32, 64, kernel_size=3, stride=2),
            EncoderLayer(64, 128, kernel_size=3, stride=2))

        bottleneck = [ResBlock(128) for _ in range(5)]
        decoder = [StyledLayer(128, 64, kernel_size=3, upsample=2),
                   StyledLayer(64, 32, kernel_size=3, upsample=2),
                   StyledLayer(32, img_channels, kernel_size=9, act_fn=None)]
        self.layers = nn.ModuleList(bottleneck + decoder)

    def forward(self, x: Tensor, style: Tensor) -> Tensor:
        x = self.encoder(x)

        for layer in self.layers:
            x = layer(x, style)

        return torch.sigmoid_(x)
