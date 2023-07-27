
import gc
import itertools
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
import canvas
import random
from canvas import placeholder
from typing import Union, Callable, List, Dict
from copy import deepcopy
def init_ws(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight.data, std=.1)
        if m.bias is not None:
            m.bias.data.zero_()
def get_final_model(model, wsharing):
    kernel_dict = {}
    # Helper function to recursively search for conv layers
    def find(module):
        for name, child in module.named_children():
            if isinstance(child, replaced_module):
                kernel_dict[name] = child
            elif isinstance(child, nn.Module):
                find(child)

    find(model)
    for name, ensembled_kernel in kernel_dict.items():
        final_kernel = ensembled_kernel.get_max_weight_kernel()
        setattr(model, name, final_kernel)
            
class replaced_module(nn.Module):
    def __init__(self, kernel_list):
        super(replaced_module, self).__init__()
        self.kernel_list = kernel_list
        self.module_list = []
        self.ws = nn.Parameter(1e-3*torch.randn(len(kernel_list)))
    def initialize_kernels(self, args):
        for kernel in self.kernel_list:
            
            self.module_list.append(kernel.module(**args))
        
    def forward(self, x):
        outputs = []
        for i, module in enumerate(self.module_list):
            print("self.ws 的形状：", self.ws.shape)
            weight = F.softmax(self.ws, dim=0)[i]
            print("Parameters:")
            sub_out = weight * module(x)
            outputs.append(sub_out)

        # Sum all the weighted kernels to get the final output
        result = sum(outputs)
        return result
    def get_max_weight_kernel(self):
        max_weight_idx = torch.argmax(self.ws)
        return self.kernel_list[max_weight_idx]
    
class in_gt_out(nn.Module):
    def __init__(self, factor):
        super(in_gt_out, self).__init__()
        self.factor = factor 
        self.layer = canvas.Placeholder()
    def forward(self, x):
        split_outputs = []
        # print(f"old channel = {print(x.shape)}, factor = {self.factor}")
        tensors = torch.split(x, x.shape[1] // self.factor, dim = 1)
        for tensor in tensors:
            output = self.layer(tensor)
            split_outputs.append(output)
        output = torch.sum(torch.stack(split_outputs), dim=0)
        return output
class out_gt_in(nn.Module):
    def __init__(self, factor):
        super(out_gt_in, self).__init__()
        self.factor = factor
        self.layer = canvas.Placeholder()
    def forward(self, x):
        outputs = []
        for i in range(self.factor):
            outputs.append(self.layer(x))
        concatenated_tensor = torch.cat(outputs, dim=1)
        # print(f"old channel = {x.shape[1]}, new channel = {concatenated_tensor.shape[1]}")
        return concatenated_tensor
    
def filter(module: nn.Module, type: str = "conv", max_count: int = 0) -> bool:
    match type:
        case "conv":
            if module.groups > 1:
                return False
            if module.kernel_size not in [(1, 1), (3, 3), (5, 5), (7, 7)]:
                return False
            if module.kernel_size == (1, 1) and module.padding != (0, 0):
                return False
            if module.kernel_size == (3, 3) and module.padding != (1, 1):
                return False
            if module.kernel_size == (5, 5) and module.padding != (2, 2):
                return False
            if module.kernel_size == (7, 7) and module.padding != (3, 3):
                return False
            width = math.gcd(module.in_channels, module.out_channels)
            if width != min(module.in_channels, module.out_channels):
                return False
            count = max(module.in_channels, module.out_channels) // width
            if count > max_count != 0:
                return False
            if module.in_channels * module.out_channels < 256 * 256 - 1:
                return False
            return True
        case "resblock":
            return True
def replace_module_with_placeholder(module: nn.Module, old_module_types: Dict[nn.Module, str], filter: Callable = filter):
    if isinstance(module, canvas.Placeholder):
            return 0, 1
    # assert old_module_type == nn.Conv2d or old_module_type == SqueezeExcitation
    
    replaced, not_replaced = 0, 0
    for name, child in module.named_children():
        if type(child) in old_module_types:
            # print("gotcha")
            string_name = old_module_types[type(child)]
            match string_name:
                case "conv":
                    if filter(child, string_name):
                        replaced += 1
                        
                        # print(f"cin = {child.in_channels}, cout = {child.out_channels}")
                        if (child.in_channels > child.out_channels):
                            factor = child.in_channels // child.out_channels
                            # print(f"factor = {factor}")
                            setattr(module, name,
                                    in_gt_out(factor
                                                    ))
                        elif (child.in_channels <= child.out_channels):
                            factor = child.out_channels // child.in_channels
                            # print(f"factor = {factor}")
                            setattr(module, name,
                                    out_gt_in(factor
                                                    ))
                    else:
                        not_replaced += 1
                case "resblock":
                    if filter(child, string_name):
                        replaced += 1
                        if (child.downsample is None):
                            setattr(module, name,
                                    canvas.Placeholder())
                        else:
                            factor = 2
                            # print(f"factor = {factor}")
                            setattr(module, name,
                                    out_gt_in(factor
                                                    ))
                    else:
                        not_replaced += 1       
                                 
        elif len(list(child.named_children())) > 0:
            count = replace_module_with_placeholder(child, old_module_types, filter)
            replaced += count[0]
            not_replaced += count[1]
    return replaced, not_replaced
   