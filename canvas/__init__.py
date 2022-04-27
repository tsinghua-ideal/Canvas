from torch import nn
from typing import Tuple

import cpp_canvas


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
        raise ValueError('The seed should be typed as `int`, with a range as `uint32_t`, '
                         'i.e. [0, 0xffffffff].')
    cpp_canvas.seed(value)


def sample(m: nn.Module,
           flops_budget: float = 1.0,
           params_budget: float = 1.0,
           allow_dynamic: bool = True,
           force_irregular: bool = False,
           add_relu_bn_after_fc: bool = True,
           num_primitive_range: Tuple[int, int] = (3, 12),
           num_fc_range: Tuple[int, int] = (1, 4),
           timeout: int = 20):
    # noinspection PyUnresolvedReferences
    r"""Sample an available kernel from the search space:

        Parameters
        ----------
        m: nn.Module
            The module to be optimized, all possible replacements for
            convolutions will occur recursively in this module.
        flops_budget: float
            The budget ratio of FLOPs (FLoating point OPerations) compared
            to all the convolutions in the original module.
        params_budget: float
            The budget ratio of parameters compared to all the convolutions
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
            The sampling timeout in seconds.

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
    pass
