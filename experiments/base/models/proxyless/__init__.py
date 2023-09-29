import torch
import torch.nn.functional as F
from torch import nn

import canvas
from .parallel_kernels import ParallelKernels


def sample_and_binarize(model):
    parallels = [p.get_kernel() for p in canvas.get_placeholders(model)]
    for parallel in parallels:
        assert isinstance(parallel, ParallelKernels)

    probs = F.softmax(torch.sum(torch.stack(
        # TODO: may change to probs
        [parallel.alphas for parallel in parallels]
    ), dim=0), dim=0).cpu()

    # TODO: support `ParallelKernels.active_only`
    # TODO: implement GPU version
    active_idx, inactive_idx = torch.multinomial(probs, 2)
    probs_slice = F.softmax(torch.Tensor([probs[active_idx], probs[inactive_idx]]))
    if torch.multinomial(probs_slice, 1)[0] == 1:
        active_idx, inactive_idx = inactive_idx, active_idx
    for parallel in parallels:
        parallel.set_indices(active_idx, inactive_idx)


def set_active_only(model: nn.Module, value: bool = True):
    for parallel in [p.get_kernel() for p in canvas.get_placeholders(model)]:
        parallel.active_only = value
