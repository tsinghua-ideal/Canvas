import torch
from torch import nn


class Placeholder(nn.Module):
    r"""Placeholder for replaced kernels.

        Attributes
        ----------
        c: int
            Input/output channel numbers.

        h: int
            Input feature map height.

        w: int
            Output feature map width.

        kernel: torch.nn.Module
            The replaced kernel instance.

        id: int
            The index of current kernel in the network.

    """

    def __init__(self, c: int):
        r"""Construct a kernel placeholder.

            Parameters
            ----------
            c: int
                Input/output channel numbers.
        """

        super().__init__()
        self.id = None
        self.c, self.h, self.w = c, 0, 0
        self.kernel = nn.Conv2d(c, c, 1, bias=False)

    def clear(self):
        r"""Reset the information of `h` and `w`, which is
            inferred during analysis.
        """

        self.h = self.w = 0

    def reload(self, kernel_cls):
        r"""Reload the internal kernel implement.

            Parameters
            ----------
            kernel_cls: type
                The Python class of the kernel to replace.
        """

        self.kernel = kernel_cls(self.c, self.h, self.w)

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

        assert self.kernel is not None
        if self.h == 0 or self.w == 0:
            self.h, self.w = x.size()[2], x.size()[3]
            assert self.h > 0 and self.w > 0
        else:
            assert self.h == x.size()[2], self.w == x.size()[3]
        return self.kernel(x)


def get_placeholders(m: nn.Module,
                     example_input: torch.Tensor = None,
                     check_shapes: bool = False):
    r"""Get all placeholders of a `torch.nn.Module`.

        Parameters
        ----------
        m: torch.nn.Module
            The module to analyze.
        example_input: torch.Tensor
            An example input tensor, for static shape inference and
            analysis if set. For the first analysis or changing to
            different shapes, you must not set it into `None`.
        check_shapes: bool
            With this option, the function will throw errors
            if the shapes are not analyzed.

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
        setattr(m, 'canvas_cached_placeholders', [])
        for kernel in m.modules():
            if isinstance(kernel, Placeholder):
                kernel.id = len(m.canvas_cached_placeholders)
                m.canvas_cached_placeholders.append(kernel)

    # Analyze shapes.
    if example_input is not None:
        if not isinstance(example_input, torch.Tensor):
            raise ValueError('The example tensor `example_input` should be '
                             'an instance of `torch.Tensor`.')
        for kernel in m.canvas_cached_placeholders:
            kernel.clear()
        setattr(m, 'canvas_cached_example_input_shape', example_input.shape)
        m(example_input)

    # Check shapes.
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
