import torch
from torch import nn
from typing import Tuple

import cpp_canvas
from . import utils
from .kernel_pack import KernelPack
from .placeholder import Placeholder, get_placeholders


def remove_cache():
    r"""Remove the cached kernel code directory.

        Example
        -------
        >>> canvas.remove_cache()
    """
    kernel_pack.remove_cache_dir()


def seed(value: int):
    r"""Set global seed for the C++ random engine.

        Parameters
        ----------
        value: int
            The seed to set for the C++ random engine.

        Example
        -------
        >>> canvas.seed(1998)

        """
    if not (type(value) == int and 0 <= value <= 0xffffffff):
        raise ValueError('The seed should be typed as an int, with a range'
                         'of C++ uint32_t, i.e. [0, 0xffffffff].')
    cpp_canvas.seed(value)


def sample(m: nn.Module,
           example_input: torch.Tensor = None,
           flops_range: Tuple[float, float] = (0, 1.0),
           params_range: Tuple[float, float] = (0, 1.0),
           allow_dynamic: bool = True,
           force_irregular: bool = False,
           add_relu_bn_after_fc: bool = True,
           num_primitive_range: Tuple[int, int] = (3, 12),
           num_fc_range: Tuple[int, int] = (1, 4),
           timeout: int = 0):
    r"""Sample an available kernel for a module from the search space.
        This function will find all convolutions in the module and
        replace them with `canvas.Placeholder`. Later, you could use
        `canvas.replace` function to substitute the kernels in the
        placeholders.

        Parameters
        ----------
        m: torch.nn.Module
            The module to be optimized, all possible replacements for
            convolutions will occur recursively in this module.
        example_input: torch.Tensor
            An example input tensor, for static shape inference and
            analysis if set. For the first sample or different shape
            sample, you must not set it into `None`.
        flops_range: Tuple[float, float]
            The budget ratio range of FLOPs (FLoating point OPerations) compared
            to all the convolutions in the original module.
        params_range: Tuple[float, float]
            The budget ratio range of parameters compared to all the convolutions
            in the original module.
        allow_dynamic: bool
            Whether allow dynamic variables to occur in the search space.
        force_irregular: bool
            Whether force the sampled kernel to be in an irregular pattern,
            which could not be covered by traditional NAS methods.
        add_relu_bn_after_fc: bool
            Whether add `nn.ReLU` and `nn.BatchNorm2d` primitive after every FC
            primitive. It may lead to a worse performance but worth for
            accuracy improvements.
        num_primitive_range: Tuple[int, int]
            The range limitation of the primitive count.
        num_fc_range: Tuple[int, int]
            The range limitation of the FC (Fully-Connected) primitive count.
        timeout: int
            The sampling timeout in seconds, zero for no timeout.

        Returns
        -------
        kernel: canvas.KernelPack
            The generated kernel class sampled from the search space.

        Example
        -------
        >>> conv = canvas.sample(m, torch.zeros((1, 3, 224, 224)))
        >>> print(conv.module)  # Show generated torch.nn.Module class
        >>> print(conv.fills)  # Show dynamic fills for every replaced kernel
        >>> print(conv.graphviz)   # Show generated GraphViz code
        >>> canvas.replace(m, conv)  # Replace all convolution kernels

        """
    # Check module type
    if not isinstance(m, nn.Module):
        raise ValueError('The input module `m` should be an instance of nn.Module.')

    # Check budget types
    if not utils.float_range_check(flops_range, 0, 10):
        raise ValueError('The budget of FLOPs `flops_budget` should be a float, '
                         'which is in the range of (0, 10].')
    if not utils.float_range_check(params_range, 0, 10):
        raise ValueError('The budget of parameters `params_budget` should be a float, '
                         'which is in the range of (0, 10].')

    # Check option types
    if type(allow_dynamic) != bool:
        raise ValueError('The variable `allow_dynamic` should be a bool.')
    if type(force_irregular) != bool:
        raise ValueError('The variable `force_irregular` should be a bool.')
    if type(add_relu_bn_after_fc) != bool:
        raise ValueError('The variable `add_relu_bn_after_fc` should be a bool.')

    if not utils.int_range_check(num_primitive_range):
        raise ValueError('Tuple `num_primitive_range` should be a tuple of'
                         'two positive ints [L, R], where L <= R.')
    if not utils.int_range_check(num_fc_range):
        raise ValueError('Tuple `num_fc_range` should be a tuple of'
                         'two positive ints [L, R], where L <= R.')

    # Check timeout type
    if not (type(timeout) == int and timeout >= 0):
        raise ValueError('Timeout value `timeout` should be an int '
                         'greater than zero.')

    # Replace convolutions with placeholders and analyze shapes
    kernels = placeholder.get_placeholders(m)

    # Analyze shapes
    if example_input is not None:
        if not isinstance(example_input, torch.Tensor):
            raise ValueError('The example tensor `example_input` should be '
                             'an instance of `torch.Tensor`.')
        for kernel in kernels:
            kernel.clear()
        m(example_input)

    # Check shapes
    for kernel in kernels:
        if kernel.h == 0 or kernel.w == 0:
            raise AttributeError('Failed to analyze shape information, '
                                 'please set `example_input` as not `None`.')

    # Sample
    kernel_specs = [cpp_canvas.KernelSpecs(ker.ic, ker.oc, ker.k, ker.h, ker.w, ker.s) for ker in kernels]
    # noinspection PyArgumentList
    pack = cpp_canvas.sample(kernel_specs,
                             flops_range[0], flops_range[1],
                             params_range[0], params_range[1],
                             allow_dynamic, force_irregular,
                             add_relu_bn_after_fc,
                             num_primitive_range[0], num_primitive_range[1],
                             num_fc_range[0], num_fc_range[1],
                             timeout)

    # Load generated code into Python class
    return kernel_pack.KernelPack(pack)


def replace(m: nn.Module, pack: kernel_pack.KernelPack):
    r"""Replace all kernel placeholders of n with sample kernels in pack.

        Parameters
        ----------
        m: torch.nn.Module
            The module to be replaced, all kernel placeholders in this
            will be reloaded with the corresponding kernels.

        pack: canvas.KernelPack
            Sampled kernel solution to replace.

        Returns
        -------
        m: torch.nn.Module
            The original reference of the input module m, but with kernels
            replaced.

        Example
        -------
        >>> m = torchvision.models.resnet18()
        >>> conv = canvas.sample(m, torch.zeros((1, 3, 224, 224)))
        >>> print(conv.module)  # Show generated torch.nn.Module class
        >>> print(conv.fills)  # Show dynamic fills for every replaced kernel
        >>> print(conv.graphviz)   # Show generated GraphViz code
        >>> canvas.replace(m, conv)  # Replace all convolution kernels
    """

    if not hasattr(m, 'canvas_cached_placeholders'):
        raise AttributeError('The module has been not initialized with `canvas.Placeholder`, '
                             'you may run `canvas.sample` or `canvas.get_placeholders` '
                             'before using `canvas.replace`.')

    # Reload all kernels
    kernels = m.canvas_cached_placeholders
    assert len(pack.fills) == len(kernels)
    for (i, kernel) in enumerate(kernels):
        kernel.reload(pack.module, pack.fills[i])

    # Re-initialization
    def reset_weights(module):
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
    for kernel in kernels:
        kernel.apply(reset_weights)

    return m
