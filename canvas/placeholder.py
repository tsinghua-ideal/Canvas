import torch
from torch import nn

from .modules import Identity
from .utils import init_weights


class Placeholder(nn.Module):
    r"""Placeholder for replaced kernels.

        Attributes
        ----------
        initialized: bool
            Indicating whether the module is initialized.

        spatial_dims: int
            The number of spatial dimensions.

        c: int
            Input/output channel numbers.

        h: int
            Input feature map height.

        w: int
            Output feature map width.

        canvas_placeholder_kernel: torch.nn.Module
            The replaced kernel instance.

        id: int
            The index of current kernel in the network.
    """

    def __init__(self):
        r"""Construct a kernel placeholder.
        """

        super().__init__()
        self.id = None
        self.initialized = False
        self.spatial_dims, self.c, self.h, self.w = 0, 0, 0, 0
        self.canvas_placeholder_kernel = Identity()

    def get_kernel(self):
        return self.canvas_placeholder_kernel

    def clear(self):
        r"""Reset all information, which is inferred during analysis.
        """

        self.initialized = False
        self.spatial_dims, self.c, self.h, self.w = 0, 0, 0, 0

    def reload(self, kernel_cls, device: str, init_weights_func=init_weights):
        r"""Reload the internal kernel implement.

            Parameters
            ----------
            kernel_cls: type
                The Python class of the kernel to replace.
            device: str
                Reload this module to which device.
            init_weights_func:
                Function for initializing weights.
        """

        args = {'c': self.c}
        if self.spatial_dims > 0:
            args['h'] = self.h
        if self.spatial_dims > 1:
            args['w'] = self.w
        self.canvas_placeholder_kernel = kernel_cls(**args).to(device)
        if init_weights_func is not None:
            self.canvas_placeholder_kernel.apply(init_weights_func)

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

        assert self.canvas_placeholder_kernel is not None
        # [N, C], or [N, C, H], or [N, C, H, W]
        assert 2 <= len(x.size()) <= 4
        if not self.initialized:
            # Inference
            self.initialized = True
            self.spatial_dims = len(x.size()) - 2
            self.c = x.size()[1]
            # H and W may be not used if not guaranteed with spatial invariance
            self.h = 1 if self.spatial_dims < 1 else x.size()[2]
            self.w = 1 if self.spatial_dims < 2 else x.size()[3]
            assert self.c > 0 and self.h > 0 and self.w > 0
        else:
            # Check
            assert self.spatial_dims == len(x.size()) - 2
            assert self.c == x.size()[1]
            # Check spatial dimensions in kernel implementation
        return self.canvas_placeholder_kernel(x)


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
        if len(m.canvas_cached_placeholders) == 0:
            print('No `canvas.Placeholder` found in the module, '
              'you may use `function `replace_module_with_placeholder` to refactor your model with placeholders.')
            delattr(m, 'canvas_cached_placeholders')
            return []
        
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
            if not kernel.initialized:
                failure = True
        if failure:
            raise AttributeError('Failed to analyze shape information, '
                                 'please set `example_input` as not `None`.')

        input_spatial_dims = [kernel.spatial_dims for kernel in m.canvas_cached_placeholders]
        if len(set(input_spatial_dims)) > 1:
            raise AttributeError('The input spatial dims in all placeholders should be the same')

    return m.canvas_cached_placeholders
