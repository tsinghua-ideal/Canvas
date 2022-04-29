import torch
from torch import nn
from typing import Tuple

import cpp_canvas
from . import placeholder
from .utils import *


def seed(value: int):
    # noinspection PyUnresolvedReferences
    r"""Set global seed for the C++ random engine:

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
           example_input: torch.Tensor,
           flops_range: Tuple[float, float] = (0, 1.0),
           params_range: Tuple[float, float] = (0, 1.0),
           allow_dynamic: bool = True,
           force_irregular: bool = False,
           add_relu_bn_after_fc: bool = True,
           num_primitive_range: Tuple[int, int] = (3, 12),
           num_fc_range: Tuple[int, int] = (1, 4),
           timeout: int = 0):
    # noinspection PyUnresolvedReferences
    r"""Sample an available kernel for a module from the search space.
        This function will find all convolutions in the module and
        replace them with `canvas.Placeholder`. Later, you could use
        `canvas.replace` function to substitute the kernels in the
        placeholders.

        Parameters
        ----------
        m: nn.Module
            The module to be optimized, all possible replacements for
            convolutions will occur recursively in this module.
        example_input: torch.Tensor
            An example input tensor, for static shape inference and analysis.
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
            Whether add `ReLU` and `BatchNorm` primitive after every FC
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
        >>> kernel_pack = canvas.sample(m)
        >>> print(kernel_pack.module)  # Show generated torch.nn.Module class
        >>> print(kernel_pack.specs)  # Show specifications for every replaced kernel
        >>> print(kernel_pack.graphviz)   # Show generated GraphViz code
        >>> m.replace(m, kernel_pack)  # Replace all convolution kernels

        """
    # Check module and input type
    if not isinstance(m, nn.Module):
        raise ValueError('The input module \'m\' should be an instance of nn.Module.')
    if not isinstance(example_input, torch.Tensor):
        raise ValueError('The example tensor \'example_input\' should an instance of nn.Tensor.')

    # Check budget types
    if not float_range_check(flops_range, 0, 10):
        raise ValueError('The budget of FLOPs \'flops_budget\' should be a float, '
                         'which is in the range of (0, 10].')
    if not float_range_check(params_range, 0, 10):
        raise ValueError('The budget of parameters \'params_budget\' should be a float, '
                         'which is in the range of (0, 10].')

    # Check option types
    if type(allow_dynamic) != bool:
        raise ValueError('The variable `allow_dynamic` should be a bool.')
    if type(force_irregular) != bool:
        raise ValueError('The variable `force_irregular` should be a bool.')
    if type(add_relu_bn_after_fc) != bool:
        raise ValueError('The variable `add_relu_bn_after_fc` should be a bool.')

    if not int_range_check(num_primitive_range):
        raise ValueError('Tuple \'num_primitive_range\' should be a tuple of'
                         'two positive ints [L, R], where L <= R.')
    if not int_range_check(num_fc_range):
        raise ValueError('Tuple \'num_fc_range\' should be a tuple of'
                         'two positive ints [L, R], where L <= R.')

    # Check timeout type
    if not (type(timeout) == int and timeout >= 0):
        raise ValueError('Timeout value \'timeout\' should be an int '
                         'greater than zero.')

    # Replace convolutions with placeholders and analyze shapes
    kernels = placeholder.get_placeholders(m)
    m(example_input)

    # Sample
    kernel_specs = [cpp_canvas.KernelSpecs(ker.ic, ker.oc, ker.k, ker.h, ker.w, ker.s) for ker in kernels]
    # noinspection PyArgumentList
    return cpp_canvas.sample(kernel_specs,
                             flops_range[0], flops_range[1],
                             params_range[0], params_range[1],
                             allow_dynamic, force_irregular,
                             add_relu_bn_after_fc,
                             num_primitive_range[0], num_primitive_range[1],
                             num_fc_range[0], num_fc_range[1],
                             timeout)
