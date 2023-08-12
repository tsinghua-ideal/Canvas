import torch
from torch import nn
from typing import Callable, Tuple

import cpp_canvas
from . import kernel_pack, placeholder, utils


def build_sample_options(allowed_filter: str = '',
                         forbidden_filter: str = '',
                         necessary_filter: str = '',
                         kernel_sizes: [int] = (3, 5, 7),
                         dilated_sizes: [int] = (1, 2, 3),
                         shift_sizes: [int] = (1, 2, 3),
                         num_primitive_range: Tuple[int, int] = (5, 25),
                         num_max_width_range: Tuple[int, int] = (2, 8),
                         num_weighted_range: Tuple[int, int] = (1, 8),
                         max_weighted_ratio: float = 0.6,
                         force_bmm_possibility: float = 0.0,
                         min_receptive_size: int = 1,
                         ensure_spatial_invariance: bool = True,
                         workers: int = 1,
                         timeout: int = 0):
    # Check option types.
    if type(allowed_filter) != str:
        raise ValueError('The variable `allowed_filter` should be a string.')
    if type(forbidden_filter) != str:
        raise ValueError('The variable `forbidden_filter` should be a string.')
    if type(necessary_filter) != str:
        raise ValueError('The variable `necessary_filter` should be a string.')
    if not utils.is_type_range(kernel_sizes, int):
        raise ValueError('`kernel_sizes` should be a tuple of int.')
    if not utils.is_type_range(dilated_sizes, int):
        raise ValueError('`dilated_sizes` should be a tuple of int.')
    if not utils.is_type_range(shift_sizes, int):
        raise ValueError('`shift_sizes` should be a tuple of int.')
    if not utils.int_range_check(num_primitive_range):
        raise ValueError('Tuple `num_primitive_range` should be a tuple of'
                         'two positive ints [L, R], where L <= R.')
    if not utils.int_range_check(num_max_width_range):
        raise ValueError('Tuple `num_max_width_range` should be a tuple of'
                         'two positive ints [L, R], where L <= R.')
    if not utils.int_range_check(num_weighted_range):
        raise ValueError('Tuple `num_weighted_range` should be a tuple of'
                         'two positive ints [L, R], where L <= R.')
    if type(max_weighted_ratio) != float or not (0 <= max_weighted_ratio <= 1):
        raise ValueError('The variable `max_weighted_ratio` should be a float between [0.0, 1.0].')
    if type(force_bmm_possibility) != float or not (0 <= force_bmm_possibility <= 1):
        raise ValueError('The variable `force_bmm_possibility` should be a float between [0.0, 1.0].')
    if type(min_receptive_size) != int or min_receptive_size < 0:
        raise ValueError('The variable `min_receptive_size` should be an integer greater than 0.')
    if type(ensure_spatial_invariance) != bool:
        raise ValueError('The variable `ensure_spatial_invariance` should be boolean.')
    if type(workers) != int or workers < 0:
        raise ValueError('The variable `workers` should be an integer greater than 0.')
    if not (type(timeout) == int and timeout >= 0):
        raise ValueError('Timeout value `timeout` should be an int '
                         'greater than zero.')
    # Build options.
    return cpp_canvas.SampleOptions(allowed_filter, forbidden_filter, necessary_filter,
                                    kernel_sizes, dilated_sizes, shift_sizes,
                                    num_primitive_range[0], num_primitive_range[1],
                                    num_max_width_range[0], num_max_width_range[1],
                                    num_weighted_range[0], num_weighted_range[1],
                                    max_weighted_ratio,
                                    force_bmm_possibility,
                                    min_receptive_size,
                                    ensure_spatial_invariance,
                                    workers,
                                    timeout)


def empty_sample(allowed_filter: str = '',
                 forbidden_filter: str = '',
                 necessary_filter: str = '',
                 kernel_sizes: [int] = (3, 5, 7),
                 dilated_sizes: [int] = (1, 2, 3),
                 shift_sizes: [int] = (1, 2, 3),
                 num_primitive_range: Tuple[int, int] = (5, 25),
                 num_max_width_range: Tuple[int, int] = (2, 8),
                 num_weighted_range: Tuple[int, int] = (1, 8),
                 max_weighted_ratio: float = 0.6,
                 force_bmm_possibility: float = 0.0,
                 min_receptive_size: int = 1,
                 ensure_spatial_invariance: bool = True,
                 workers: int = 1,
                 timeout: int = 0):
    r"""Sample an available kernel from the search space, without network reference.

        Parameters
        ----------
        allowed_filter: str
            The filter for allowed primitives, in a comma-seperated format.
        forbidden_filter: str
            The filter for forbidden primitives, in a comma-seperated format.
        necessary_filter: str
            Necessary primitives to be contained.
        kernel_sizes: [int]
            The candidates for kernel sizes.
        dilated_sizes: [int]
            The candidates for dilated sizes.
        shift_sizes: [int]
            The candidates for shifting sizes.
        num_primitive_range: Tuple[int, int]
            The range limitation of the primitive count.
        num_max_width_range: Tuple[int, int]
            The range limitation of the graph width during a search.
        num_weighted_range: Tuple[int, int]
            The range limitation of the weighted primitive count.
        max_weighted_ratio: float
            Maximum FC primitive ratio out of all primitives.
        force_bmm_possibility: float
            The possibility to forcibly contain BMM (attention like) primitive.
        min_receptive_size: int
            A filter for minimum receptive size.
        ensure_spatial_invariance: bool
            Whether we can ensure the spatial dimensions will not change.
            If we can, then we can generate weights with H and W.
        workers: int
            Workers for sampling.
        timeout: int
            The sampling timeout in seconds, zero for no timeout.

        Returns
        -------
        kernel: canvas.KernelPack
            The generated kernel class sampled from the search space.

        Example
        -------
        >>> kernel = canvas.empty_sample()
        >>> print(kernel.module)        # Show generated torch.nn.Module class.
        >>> print(kernel.graphviz)      # Show generated GraphViz code.
    """
    # Sample a kernel design.
    options = build_sample_options(allowed_filter, forbidden_filter, necessary_filter,
                                   kernel_sizes, dilated_sizes, shift_sizes,
                                   num_primitive_range,
                                   num_max_width_range,
                                   num_weighted_range,
                                   max_weighted_ratio,
                                   force_bmm_possibility,
                                   min_receptive_size,
                                   ensure_spatial_invariance,
                                   workers,
                                   timeout)
    pack = cpp_canvas.sample([], options)

    # Load generated code into Python class.
    return kernel_pack.KernelPack.load_from_cpp(pack)


def sample(m: nn.Module,
           example_input: torch.Tensor = None,
           allowed_filter: str = '',
           forbidden_filter: str = '',
           necessary_filter: str = '',
           kernel_sizes: [int] = (3, 5, 7),
           dilated_sizes: [int] = (1, 2, 3),
           shift_sizes: [int] = (1, 2, 3),
           num_primitive_range: Tuple[int, int] = (5, 25),
           num_max_width_range: Tuple[int, int] = (2, 8),
           num_weighted_range: Tuple[int, int] = (1, 8),
           max_weighted_ratio: float = 0.6,
           force_bmm_possibility: float = 0.0,
           min_receptive_size: int = 1,
           ensure_spatial_invariance: bool = True,
           workers: int = 1,
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
        necessary_filter: str
            Necessary primitives to be contained.
        kernel_sizes: [int]
            The candidates for kernel sizes.
        dilated_sizes: [int]
            The candidates for dilated sizes.
        shift_sizes: [int]
            The candidates for shifting sizes.
        num_primitive_range: Tuple[int, int]
            The range limitation of the primitive count.
        num_max_width_range: Tuple[int, int]
            The range limitation of the graph width during a search.
        num_weighted_range: Tuple[int, int]
            The range limitation of the weighted primitive count.
        max_weighted_ratio: float
            Maximum FC primitive ratio out of all primitives.
        force_bmm_possibility: float
            The possibility to forcibly contain BMM (attention like) primitive.
        min_receptive_size: int
            A filter for minimum receptive size.
        ensure_spatial_invariance: bool
            Whether we can ensure the spatial dimensions will not change.
            If we can, then we can generate weights with H and W.
        workers: int
            Workers for sampling.
        timeout: int
            The sampling timeout in seconds, zero for no timeout.

        Returns
        -------
        kernel: canvas.KernelPack
            The generated kernel class sampled from the search space.

        Example
        -------
        >>> kernel = canvas.sample(m, torch.zeros((1, 3, 224, 224)).cuda())
        >>> print(kernel.module)        # Show generated torch.nn.Module class.
        >>> print(kernel.graphviz)      # Show generated GraphViz code.
        >>> canvas.replace(m, kernel.module)   # Replace all kernels.

        """
    # Check module type.
    if not isinstance(m, nn.Module):
        raise ValueError('The input module `m` should be an instance of nn.Module.')

    # Get placeholders in the module.
    kernels = placeholder.get_placeholders(m, example_input, check_shapes=True)
    if len(kernels) == 0:
        print('No `canvas.Placeholder` found in the module, '
              'you may re-design your model with placeholders.')
        return None

    # Sample a kernel design.
    kernel_specs = [cpp_canvas.KernelSpecs(kernel.c, kernel.h, kernel.w, kernel.spatial_dims) for kernel in kernels]
    options = build_sample_options(allowed_filter, forbidden_filter, necessary_filter,
                                   kernel_sizes, dilated_sizes, shift_sizes,
                                   num_primitive_range,
                                   num_max_width_range,
                                   num_weighted_range,
                                   max_weighted_ratio,
                                   force_bmm_possibility,
                                   min_receptive_size,
                                   ensure_spatial_invariance,
                                   workers,
                                   timeout)
    pack = cpp_canvas.sample(kernel_specs, options)

    # Load generated code into Python class.
    return kernel_pack.KernelPack.load_from_cpp(pack)


def replace(m: nn.Module, module: Callable, device: str = 'cuda:0', init_weights_func=utils.init_weights):
    r"""Replace all kernel placeholders of n with sample kernels in pack.

        Parameters
        ----------
        m: torch.nn.Module
            The module to be replaced, all kernel placeholders in this
            will be reloaded with the corresponding kernels.
        module: nn.Module
            Torch module to replace.
        device: str
            Reload kernel to which device.
        init_weights_func:
            Function for initializing weights.

        Returns
        -------
        m: torch.nn.Module
            The original reference of the input module m, but with kernels
            replaced.

        Example
        -------
        >>> kernel = canvas.sample(m, torch.zeros((1, 3, 224, 224)).cuda(), 'cuda:0')
        >>> print(kernel.module)          # Show generated torch.nn.Module class.
        >>> print(kernel.graphviz)        # Show generated GraphViz code.
        >>> canvas.replace(m, kernel.module)   # Replace all kernels.
    """
    assert module is not None, 'Module to replace should not be None.'

    # Check placeholders.
    if not hasattr(m, 'canvas_cached_placeholders'):
        raise AttributeError('The module has been not initialized with `canvas.Placeholder`, '
                             'you may run `canvas.sample` or `canvas.get_placeholders` '
                             'before using `canvas.replace`.')

    # Reload all kernels.
    kernels = m.canvas_cached_placeholders
    for kernel in kernels:
        kernel.reload(module, device, init_weights_func)

    return m


def debug_sample():
    r"""An API for debugging.

    Returns
    -------
    kernel: canvas.KernelPack
        The generated kernel class specified by the debugger.

"""
    return kernel_pack.KernelPack.load_from_cpp(cpp_canvas.debug_sample())
