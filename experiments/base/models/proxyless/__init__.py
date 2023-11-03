import logging
from typing import List
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn

import canvas
from .parallel_kernels import ParallelKernels


def get_parallel_kernels(model: nn.Module) -> List[ParallelKernels]:
    return [p.get_kernel() for p in canvas.get_placeholders(model)]


def sample_and_binarize(model: torch.nn.Module, active_only, valid):
    parallels = get_parallel_kernels(model)
    for parallel in parallels:
        assert isinstance(parallel, ParallelKernels)
    probs = F.softmax(torch.sum(torch.stack(
        [parallel.kernel_alphas for parallel in parallels]
    ), dim=0), dim=0).detach()
    if active_only and valid:
        active_idx, inactive_idx = torch.argmax(probs), None
    elif active_only:
        active_idx, inactive_idx = torch.multinomial(probs, 1), None
    else:
        active_idx, inactive_idx = torch.multinomial(probs, 2)
        probs_slice = F.softmax(torch.tensor([probs[active_idx], probs[inactive_idx]], device='cuda'), 0)
        if torch.multinomial(probs_slice, 1)[0] == 1:
            active_idx, inactive_idx = inactive_idx, active_idx
    for parallel in parallels:
        parallel.set_indices(active_idx=active_idx, inactive_idx=inactive_idx, active_only=active_only)
     
     
def restore_modules(model: nn.Module):
    for parallel in get_parallel_kernels(model):
        parallel.restore_all()


def set_alpha_grad(model: nn.Module):
    for parallel in get_parallel_kernels(model):
        parallel.set_alpha_grad()


def rescale_kernel_alphas(model: nn.Module):
    for parallel in get_parallel_kernels(model):
        parallel.rescale_kernel_alphas()

 
def get_parameters(model: nn.Module, keys=None, mode='include'):
    if keys is None:
        for name, param in model.named_parameters():
            yield param
    elif mode == 'include':
        for name, param in model.named_parameters():
            if any([key in name for key in keys]):
                yield param
    elif mode == 'exclude':
        for name, param in model.named_parameters():
            if not any([key in name for key in keys]):
                yield param
    else:
        raise ValueError(f'{mode} is not supported')


def print_parameters(model: nn.Module):
    logger = logging.getLogger()
    for i, parallel in enumerate(get_parallel_kernels(model)):
        logger.info(f'Layer {i}: ')
        logger.info(f' > kernel_alphas: {parallel.get_softmaxed_kernel_alphas()}')


def get_sum_of_magnitude_probs_with_1D(model: nn.Module):
    return torch.sum(torch.stack([parallel.get_softmaxed_kernel_alphas().detach() for parallel in get_parallel_kernels(model)]), dim=0)


def get_sum_of_magnitude_scores_with_1D(model: nn.Module):
    return torch.sum(torch.stack([parallel.kernel_alphas.detach() for parallel in get_parallel_kernels(model)]), dim=0)


def get_multiplication_of_magnitude_probs_with_1D(model: nn.Module):
    return torch.softmax(torch.log(torch.prod(torch.stack([parallel.get_softmaxed_kernel_alphas().detach() for parallel in get_parallel_kernels(model)]), dim=0)), dim=0)


def sort_and_prune(alpha_list, kernel_list):
    sorted_indices = np.argsort(alpha_list)[::-1]
    return [alpha_list[i] for i in sorted_indices], [kernel_list[i] for i in sorted_indices]
