import torch
from torch import nn


class Conv7x7(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        super().__init__()
        self.conv_impl = nn.Conv2d(c, c, 7, padding=3, bias=False)

    def forward(self, x: torch.Tensor):
        return self.conv_impl(x)
