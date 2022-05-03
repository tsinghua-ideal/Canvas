import math
import torch
from torch import nn


class Placeholder(nn.Module):
    r"""Placeholder for replaced kernels.

        Attributes
        ----------
        ic: int
            Input channel numbers.

        oc: int
            Output channel numbers.

        k: int
            Kernel size (height and width).

        s: int
            Striding number.

        h: int
            Input feature map height.

        w: int
            Output feature map width.

        conv: torch.nn.Module
            The replaced kernel instance.

        id: int
            The index of current kernel in the network.

    """

    def __init__(self, ic: int, oc: int, k: int, s: int):
        r"""Construct a kernel placeholder.

            Parameters
            ----------
            ic: int
                Input channel numbers.

            oc: int
                Output channel numbers.

            k: int
                Kernel size (height and width).

            s: int
                Striding number.
        """

        super().__init__()
        assert math.gcd(ic, oc) == min(ic, oc), f'Input channel {ic} and output {oc} should be aligned'
        self.ic, self.oc, self.k, self.s = ic, oc, k, s
        self.h, self.w = 0, 0
        self.conv = nn.Conv2d(ic, oc, k, s, padding=(k - 1) // 2, bias=False)
        self.id = None

    def clear(self):
        r"""Reset the information of `h` and `w`, which is
            inferred during analysis.
        """

        self.h = self.w = 0

    def reload(self, kernel_cls, x: [int]):
        r"""Reload the internal kernel implement.

            Parameters
            ----------
            kernel_cls: type
                The Python class of the kernel to replace.

            x: [int]
                Dynamic variables in the kernel.
        """

        self.conv = kernel_cls(self.ic, self.oc, self.k, self.s, self.h, self.w, x)

    def forward(self, x: torch.Tensor):
        r"""Forward propagation of the kernel.

            Parameters
            ----------
            x: torch.Tensor
                The input tensor.

            Returns
            -------
            x: torch.Tensor
                The calculation result of this module.
        """

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


def get_placeholders(m: nn.Module,
                     example_input: torch.Tensor = None,
                     check_shapes: bool = False):
    r"""Get all placeholders of a `torch.nn.Module`, replace all
        available convolutions if not analyzed before.

        Parameters
        ----------
        m: torch.nn.Module
            The module to analyze.
        example_input: torch.Tensor
            An example input tensor, for static shape inference and
            analysis if set. For the first analysis or changing to
            different shapes, you must not set it into `None`.
        check_shapes: bool

        Returns
        -------
        placeholders: [canvas.Placeholder]
            All placeholders collected in the input module.

        Example
        -------
        >>> m = torchvision.models.resnet18()
        >>> placeholders = canvas.get_placeholders(m, torch.zeros(1, 3, 224, 224))
    """
    if not hasattr(m, 'canvas_cached_placeholders'):
        replace_with_placeholders(m)
        setattr(m, 'canvas_cached_placeholders', [])
        for kernel in m.modules():
            if isinstance(kernel, Placeholder):
                kernel.id = len(m.canvas_cached_placeholders)
                m.canvas_cached_placeholders.append(kernel)

    # Analyze shapes
    if example_input is not None:
        if not isinstance(example_input, torch.Tensor):
            raise ValueError('The example tensor `example_input` should be '
                             'an instance of `torch.Tensor`.')
        for kernel in m.canvas_cached_placeholders:
            kernel.clear()
        setattr(m, 'canvas_cached_example_input_shape', example_input.shape)
        m(example_input)

    # Check shapes
    if check_shapes:
        failure = False
        if not hasattr(m, 'canvas_cached_example_input_shape'):
            failure = True
        for kernel in m.canvas_cached_placeholders:
            if kernel.h == 0 or kernel.w == 0:
                failure = True
        if failure:
            raise AttributeError('Failed to analyze shape information, '
                                 'please set `example_input` as not `None`.')

    return m.canvas_cached_placeholders
