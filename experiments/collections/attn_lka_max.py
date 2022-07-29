import torch
from torch import nn


class KernelAttnLKA(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        super().__init__()

        self.proj_1 = nn.Conv2d(c, c, 1)
        self.activation = nn.GELU()
        self.lka_conv0 = nn.Conv2d(c, c, 5, padding=2, groups=c)
        self.lka_conv_spatial = nn.Conv2d(c, c, 7, stride=1, padding=9, groups=c, dilation=3)
        self.lka_conv1 = nn.Conv2d(c, c, 1)
        self.proj_2 = nn.Conv2d(c, c, 1)

    def forward(self, x: torch.Tensor):
        shortcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        # LKA.
        u = x.clone()
        x = self.lka_conv0(x)
        x = self.lka_conv_spatial(x)
        x = self.lka_conv1(x)
        x = torch.maximum(x, u)
        # End of LKA.
        x = self.proj_2(x)
        x = x + shortcut
        return x
