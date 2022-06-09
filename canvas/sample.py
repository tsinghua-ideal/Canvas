import torch
from torch import nn
from typing import Tuple

import cpp_canvas
from . import kernel_pack, placeholder, utils


# TODO: fix options of FLOPs and params.
def sample(m: nn.Module,
           example_input: torch.Tensor = None,
           add_relu_bn_after_fc: bool = False,
           allowed_filter: str = '',
           forbidden_filter: str = '',
           kernel_sizes: [int] = [3, 5, 7],
           dilated_sizes: [int] = [1, 2, 3],
           num_primitive_range: Tuple[int, int] = (3, 25),
           num_max_width_range: Tuple[int, int] = (2, 8),
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
        allowed_filter: str
            The filter for allowed primitives, in a comma-seperated format.
        forbidden_filter: str
            The filter for forbidden primitives, in a comma-seperated format.
        kernel_sizes: [int]
            The candidates for kernel sizes.
        dilated_sizes: [int]
            The candidates for dilated sizes.
        add_relu_bn_after_fc: bool
            Whether add `nn.ReLU` and `nn.BatchNorm2d` primitive after every FC
            primitive. It may lead to a worse performance but worth for
            accuracy improvements.
        num_primitive_range: Tuple[int, int]
            The range limitation of the primitive count.
        num_max_width_range: Tuple[int, int]
            The range limitation of the graph width during a search.
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
    if type(add_relu_bn_after_fc) != bool:
        raise ValueError('The variable `add_relu_bn_after_fc` should be a bool.')

    if not utils.int_range_check(num_primitive_range):
        raise ValueError('Tuple `num_primitive_range` should be a tuple of'
                         'two positive ints [L, R], where L <= R.')
    if not utils.int_range_check(num_max_width_range):
        raise ValueError('Tuple `num_max_width_range` should be a tuple of'
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
    options = cpp_canvas.SampleOptions(allowed_filter, forbidden_filter,
                                       kernel_sizes, dilated_sizes,
                                       add_relu_bn_after_fc,
                                       num_primitive_range[0], num_primitive_range[1],
                                       num_max_width_range[0], num_max_width_range[1],
                                       num_fc_range[0], num_fc_range[1],
                                       timeout)
    pack = cpp_canvas.sample(kernel_specs, options)

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

    return m
