import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import gc

from . import log
from .utils import *

class ProxylessParallelKernels(nn.Module):
    
    """
    A module representing a parallel combination of multiple kernels with weighted outputs.

    Args:
        kernel_cls_list (list): List of kernel class types to be instantiated.
        i (int): Index of the placeholder.
        **kwargs: Keyword arguments passed to kernel constructors(c, h, w).

    Attributes:
        i (int): Rank of this placeholder in the model.
        module_list (nn.ModuleList): List of instantiated kernel modules.
        alphas (nn.Parameter): Learnable architecture parameter representing weights of kernels.

    Methods:
        forward(x): Forward pass through the module.
        get_max_weight_kernel(): Get the kernel with the maximum weight.
        print_parameters(j): Print the parameters of the module when it's trained.
        get_alphas(): Get the alpha values.
    
    """
    MODE = None  # full, two, None, full_v2
    
    def __init__(self, kernel_cls_list, **kwargs):
        super().__init__()
        assert len(kernel_cls_list) >= 1
        self.module_list = nn.ModuleList([kernel_cls(*kwargs.values()) for kernel_cls in kernel_cls_list])
        self.length = len(self.module_list)
        self.AP_path_alpha = nn.Parameter(torch.ones(self.length) * (1 / self.length))  # architecture parameters
        self.AP_path_wb = nn.Parameter(torch.Tensor(self.length))  # binary gates
        self.active_index = [0]
        self.inactive_index = None

        
    def forward(self, x: torch.Tensor):
        if ProxylessParallelKernels.MODE == 'two':
            return self.AP_path_wb[self.active_index[0]] * self.module_list[self.active_index[0]](x) + self.AP_path_wb[self.inactive_index[0]] * self.module_list[self.inactive_index[0]](x).detach()
        else:
            return self.active_module(x)
    
    def get_max_weight_kernel(self):
        return self.module_list[torch.argmax(self.probs_over_ops)]
    
    @property
    def active_module(self):
        """ assume only one path is active """
        return self.module_list[self.active_index[0]]

    def set_chosen_module_active(self):
        chosen_idx = self.chosen_index
        self.active_index = [chosen_idx]
        self.inactive_index = [_i for _i in range(0, chosen_idx)] + \
                              [_i for _i in range(chosen_idx + 1, self.length)]

    def set_arch_param_grad(self):
        binary_grads = self.AP_path_wb.grad.data
        if self.AP_path_alpha.grad is None:
            self.AP_path_alpha.grad = torch.zeros_like(self.AP_path_alpha.data)
        if ProxylessParallelKernels.MODE == 'two':
            involved_idx = self.active_index + self.inactive_index
            probs_slice = F.softmax(torch.stack([
                self.AP_path_alpha[idx] for idx in involved_idx
            ]), dim=0).data
            for i in range(2):
                for j in range(2):
                    origin_i = involved_idx[i]
                    origin_j = involved_idx[j]
                    self.AP_path_alpha.grad.data[origin_i] += \
                        binary_grads[origin_j] * probs_slice[j] * (delta_ij(i, j) - probs_slice[i])
            for _i, idx in enumerate(self.active_index):
                self.active_index[_i] = (idx, self.AP_path_alpha.data[idx].item())
            for _i, idx in enumerate(self.inactive_index):
                self.inactive_index[_i] = (idx, self.AP_path_alpha.data[idx].item())
        else:
            NotImplementedError
        return

    def rescale_updated_arch_param(self):
        if not isinstance(self.active_index[0], tuple):
            return
        involved_idx = [idx for idx, _ in (self.active_index + self.inactive_index)]
        old_alphas = [alpha for _, alpha in (self.active_index + self.inactive_index)]
        new_alphas = [self.AP_path_alpha.data[idx] for idx in involved_idx]
        offset = math.log(
            sum([math.exp(alpha) for alpha in new_alphas]) / sum([math.exp(alpha) for alpha in old_alphas])
        )
        for idx in involved_idx:
            self.AP_path_alpha.data[idx] -= offset
            
    
                
    @property
    def probs_over_ops(self):
        return F.softmax(self.AP_path_alpha, dim=0)  # softmax to probability

    @property
    def chosen_index(self):
        return torch.argmax(self.probs_over_ops)

    def print_parameters(self, i, j):
        logger = log.get_logger()
        
        # Print parameters in each placeholder  
        logger.info(f'In {i}th Placeholder')  
        
        # Alpha
        logger.info(f'####### ALPHA After {j}th epoch #######')
        logger.info('# Alphas')
        logger.info(self.probs_over_ops)
        logger.info('#####################')
        
         
def get_sum_of_magnitude_scores_with_1D(model):
    return torch.sum(torch.stack([F.softmax(placeholder.canvas_placeholder_kernel.AP_path_alpha.detach().cpu(), dim=0) for placeholder in model.canvas_cached_placeholders]), dim=0)

def get_multiplication_of_magnitude_scores_with_1D(model):
    return torch.prod(torch.stack([F.softmax(placeholder.canvas_placeholder_kernel.AP_path_alpha.detach().cpu(), dim=0) for placeholder in model.canvas_cached_placeholders]),dim=0) * 1e18


def get_one_hot_scores(model):
    """
    Calculate and return the one-hot scores.

    Args:
        model: The model instance.

    Returns:
        The sum of one-hot scores.
    """
    one_hot_scores = torch.stack([torch.eye(len(placeholder.canvas_placeholder_kernel.AP_path_alpha.detach().cpu()))[torch.argmax(F.softmax(placeholder.canvas_placeholder_kernel.AP_path_alpha.detach().cpu(), dim=0))] 
                for placeholder in model.canvas_cached_placeholders])
    return torch.sum(one_hot_scores, dim=0)


""" architecture parameters related methods """

         
def reset_binary_gates_test(model):
            
    # reset binary gates
    for placeholder in model.canvas_cached_placeholders:
        placeholder.canvas_placeholder_kernel.AP_path_wb.data.zero_()

    # binarize according to probs
    probs = torch.sum(torch.stack([placeholder.canvas_placeholder_kernel.probs_over_ops for placeholder in model.canvas_cached_placeholders]), dim=0)
    if ProxylessParallelKernels.MODE == 'two':
        
        # sample two ops according to `probs`
        sample_op = torch.multinomial(probs.data, 2, replacement=False)
        probs_slice = F.softmax(torch.stack([
            probs[idx] for idx in sample_op
        ]), dim=0)
            
        # chose one to be active and the other to be inactive according to probs_slice
        c = torch.multinomial(probs_slice.data, 1)[0]  # 0 or 1
        active_op = sample_op[c].item()
        inactive_op = sample_op[1 - c].item()
        for placeholder in model.canvas_cached_placeholders:
            placeholder.canvas_placeholder_kernel.active_index = [active_op]
            placeholder.canvas_placeholder_kernel.inactive_index = [inactive_op]
            placeholder.canvas_placeholder_kernel.AP_path_wb.data[active_op] = 1.0
    else:
        sample = torch.multinomial(probs.data, 1)[0].item()
        for placeholder in model.canvas_cached_placeholders:
            placeholder.canvas_placeholder_kernel.active_index = [sample]
            placeholder.canvas_placeholder_kernel.inactive_index = [_i for _i in range(0, sample)] + \
                                                [_i for _i in range(sample + 1, placeholder.canvas_placeholder_kernel.length)]
            placeholder.canvas_placeholder_kernel.AP_path_wb.data[sample] = 1.0

def set_arch_param_grad(model):
    for placeholder in model.canvas_cached_placeholders:
        try:
            placeholder.canvas_placeholder_kernel.set_arch_param_grad()
        except AttributeError:
            print(type(placeholder), ' do not support `set_arch_param_grad()`')

def rescale_updated_arch_param(model):
    for placeholder in model.canvas_cached_placeholders:
        try:
            placeholder.canvas_placeholder_kernel.rescale_updated_arch_param()
        except AttributeError:
            print(type(placeholder), ' do not support `rescale_updated_arch_param()`')
            
def sort_and_prune(alpha_list, kernel_list, percentage_to_keep=0.5, n=8):
    # num_keep = len(kernel_list) * percentage_to_keep

    # # 使用argsort获取排序后的alphas的索引
    # sorted_indices = torch.argsort(alpha_list, descending=True)

    # # 选择排序后的索引中的前一半索引
    # num_indices_keep = sorted_indices[:int(num_keep)]
    
    step = len(kernel_list) // n
    sorted_indices = torch.argsort(alpha_list, descending=True)
    num_indices_keep = sorted_indices[:2].tolist() + sorted_indices[2::int(step)].tolist()
    alpha_list = alpha_list.tolist()
    
    # 根据索引获取对应的module_list子数组
    return [kernel_list[i] for i in num_indices_keep], [alpha_list[i] for i in num_indices_keep]

""" training related methods """

#这里要设置一个属性，用来存储未使用的模块
def unused_modules_off(model):
    model._unused_modules = []
    flag = True if ProxylessParallelKernels.MODE in ['full', 'two', 'full_v2'] else False
    for placeholder in model.canvas_cached_placeholders:
        unused = {}
        if flag:
            involved_index = placeholder.canvas_placeholder_kernel.active_index + placeholder.canvas_placeholder_kernel.inactive_index
        else:
            involved_index = placeholder.canvas_placeholder_kernel.active_index
        for i in range(placeholder.canvas_placeholder_kernel.length):
            if i not in involved_index:
                unused[i] = placeholder.canvas_placeholder_kernel.module_list[i]
                placeholder.canvas_placeholder_kernel.module_list[i] = None
        model._unused_modules.append(unused)


def unused_modules_back(model):
    if model._unused_modules is None:
        return
    for placeholder, unused in zip(model.canvas_cached_placeholders, model._unused_modules):
        for i in unused:
            placeholder.canvas_placeholder_kernel.module_list[i] = unused[i]
    model._unused_modules = None

def set_chosen_module_active(model):
    for placeholder in model.canvas_cached_placeholders:
        try:
            placeholder.canvas_placeholder_kernel.set_chosen_module_active()
        except AttributeError:
            print(type(placeholder), ' do not support `set_chosen_module_active()`')

