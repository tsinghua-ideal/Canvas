import torch
from torch import nn


class Identity(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x


class Conv2D(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        super().__init__()
        self.conv_impl = nn.Conv2d(c, c, 1, bias=False)

    def forward(self, x: torch.Tensor):
        return self.conv_impl(x)
