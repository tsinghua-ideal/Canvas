import torch
from torch import nn


class KernelIdentity(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        super().__init__()

    def forward(self, x: torch.Tensor):
        return x
