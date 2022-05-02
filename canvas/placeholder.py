import math
import torch
from torch import nn


class Placeholder(nn.Module):
    def __init__(self, ic: int, oc: int, k: int, s: int):
        super().__init__()
        assert math.gcd(ic, oc) == min(ic, oc), f'Input channel {ic} and output {oc} should be aligned'
        self.ic, self.oc, self.k, self.s = ic, oc, k, s
        self.h, self.w = 0, 0
        self.conv = nn.Conv2d(ic, oc, k, s, padding=(k - 1) // 2, bias=False)
        self.id = None

    def clear(self):
        self.h = self.w = 0

    def reload(self, kernel_cls, x: [int]):
        self.conv = kernel_cls(self.ic, self.oc, self.k, self.s, self.h, self.w, x)

    def forward(self, x: torch.Tensor):
        assert self.conv is not None
        if self.h == 0 or self.w == 0:
            self.h, self.w = x.size()[2], x.size()[3]
            assert self.h > 0 and self.w > 0
        else:
            assert self.h == x.size()[2], self.w == x.size()[3]
        return self.conv(x)


def replaceable_filter(conv: nn.Conv2d) -> bool:
    if conv.groups > 1:
        return False
    if conv.kernel_size not in [(1, 1), (3, 3), (5, 5), (7, 7)]:
        return False
    if conv.kernel_size == (1, 1) and conv.padding != (0, 0):
        return False
    if conv.kernel_size == (3, 3) and conv.padding != (1, 1):
        return False
    if conv.kernel_size == (5, 5) and conv.padding != (2, 2):
        return False
    if conv.kernel_size == (7, 7) and conv.padding != (3, 3):
        return False
    width = math.gcd(conv.in_channels, conv.out_channels)
    if width != min(conv.in_channels, conv.out_channels):
        return False
    return True


def replace_with_placeholders(m: nn.Module):
    if isinstance(m, Placeholder):
        return
    for name, child in m.named_children():
        if isinstance(child, nn.Conv2d) and replaceable_filter(child):
            setattr(m, name,
                    Placeholder(child.in_channels, child.out_channels,
                                child.kernel_size[0], child.stride[0]))
        elif len(list(child.named_children())) > 0:
            replace_with_placeholders(child)


def get_placeholders(m: nn.Module):
    r"""Get all placeholders of a `torch.nn.Module`, replace all
        available convolutions if not analyzed before.

        Parameters
        ----------
        m: torch.nn.Module
            The module to analyze.

        Returns
        -------
        placeholders: [canvas.Placeholder]
            All placeholders collected in the input module.

        Example
        -------
        >>> m = torchvision.models.resnet18()
        >>> placeholders = canvas.get_placeholders(m)
    """
    if not hasattr(m, 'canvas_cached_placeholders'):
        replace_with_placeholders(m)
        setattr(m, 'canvas_cached_placeholders', [])
        for kernel in m.modules():
            if isinstance(kernel, Placeholder):
                kernel.id = len(m.canvas_cached_placeholders)
                m.canvas_cached_placeholders.append(kernel)
    return m.canvas_cached_placeholders
