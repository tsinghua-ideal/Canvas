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


# TODO: fix options of FLOPs and params.
def sample(m: nn.Module,
           example_input: torch.Tensor = None,
           allow_dynamic: bool = True,
           force_irregular: bool = False,
           add_relu_bn_after_fc: bool = False,
           num_primitive_range: Tuple[int, int] = (3, 25),
           num_fc_range: Tuple[int, int] = (1, 8),
           timeout: int = 0):
    r"""Sample an available kernel for a module from the search space.
        This function will find all placeholders in the module, and sample
        an available to substitute the originals.
        You may use `canvas.replace` function to substitute the kernels in the
        placeholders.

        Parameters
        ----------
        m: torch.nn.Module
            The module to be searched, all placeholders will be analyzed to
            create a new kernel.
        example_input: torch.Tensor
            An example input tensor, for static shape inference and
            analysis if set. For the first analysis or changing to
            different shapes, you must not set it into `None`.
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
        >>> kernel = canvas.sample(m, torch.zeros((1, 3, 224, 224)))
        >>> print(kernel.module)        # Show generated torch.nn.Module class.
        >>> print(kernel.graphviz)      # Show generated GraphViz code.
        >>> canvas.replace(m, kernel)   # Replace all kernels.

        """
    # Check module type.
    if not isinstance(m, nn.Module):
        raise ValueError('The input module `m` should be an instance of nn.Module.')

    # Check option types.
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

    # Check timeout type.
    if not (type(timeout) == int and timeout >= 0):
        raise ValueError('Timeout value `timeout` should be an int '
                         'greater than zero.')

    # Get placeholders in the module.
    kernels = placeholder.get_placeholders(m, example_input, check_shapes=True)
    if len(kernels) == 0:
        print('No `canvas.Placeholder` found in the module, '
              'you may re-design your model with placeholders.')
        return None

    # Sample a kernel design.
    kernel_specs = [cpp_canvas.KernelSpecs(ker.c, ker.h, ker.w) for ker in kernels]
    # noinspection PyArgumentList
    pack = cpp_canvas.sample(kernel_specs,
                             allow_dynamic, force_irregular,
                             add_relu_bn_after_fc,
                             num_primitive_range[0], num_primitive_range[1],
                             num_fc_range[0], num_fc_range[1],
                             timeout)

    # Load generated code into Python class.
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
        >>> kernel = canvas.sample(m, torch.zeros((1, 3, 224, 224)))
        >>> print(conv.module)          # Show generated torch.nn.Module class.
        >>> print(conv.graphviz)        # Show generated GraphViz code.
        >>> canvas.replace(m, kernel)   # Replace all kernels.
    """

    # Check placeholders.
    if not hasattr(m, 'canvas_cached_placeholders') or len(m.canvas_cached_placeholders) == 0:
        raise AttributeError('The module has been not initialized with `canvas.Placeholder`, '
                             'you may run `canvas.sample` or `canvas.get_placeholders` '
                             'before using `canvas.replace`.')

    # Reload all kernels.
    kernels = m.canvas_cached_placeholders
    for kernel in kernels:
        kernel.reload(pack.module)

    # TODO: add parameter initialization/reset.
    return m
