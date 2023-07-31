
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
class TEST(nn.Module):
    def __init__(self, module_list, i):
        super(TEST, self).__init__()
        self.module_list=nn.ModuleList()
        # self.module = module_list[0]
        for module in module_list:      
            self.module_list.append(module)
    def forward(self, x):
        out = torch.zeros(x.shape, device=x.device)
        # print(f"x.device = {x.device}")
        for i in range(len(self.module_list)):
            # weight = F.softmax(self.weights, dim=0)[i]
            out +=  self.module_list[i](x) 
        return out
        
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
    def __init__(self, module_list, i):
        super(replaced_module, self).__init__()
        # print(f"The {i}th placeholder has been initialized")
        # self.n = len(module_list)
        # for i in range(self.n):
        #     self.add_module('module_{}'.format(i), module_list[i])
        
        # self.weights = nn.Parameter(1e-3*torch.randn(self.n))
        # self.module_list=nn.ModuleList()
        self.module_list = module_list
        # self.module = module_list[0]
        # for module in module_list:      
        #     self.module_list.append(module)
        self.weights = nn.Parameter(1e-3*torch.randn(len(module_list)))
    def forward(self, x):
        out = torch.zeros(x.shape, device=x.device)
        # # print(f"x.device = {x.device}")
        # for i in range(self.n):
        #     weight = F.softmax(self.weights, dim=0)[i]
        #     out += weight * getattr(self, 'module_{}'.format(i))(x)
        # return out
        # out = torch.zeros(x.shape, device=x.device)
        # out = torch.zeros(x.shape, device=x.device)
        # # print(f"x.device = {x.device}")
        # for i in range(len(self.module_list)):
        #     weight = F.softmax(self.weights, dim=0)[i]
        #     out += weight * self.module_list[i](x) 
        # return out
        weights_softmax = F.softmax(self.weights, dim=0)  # 预先计算softmax，避免在循环内重复计算
        for i in range(len(self.module_list)):
            # weight = F.softmax(self.weights, dim=0)[i]
            out += weights_softmax[i] * self.module_list[i](x) 
        return out
        # out = torch.cat([weight * module(x) for weight, module in zip(weights_softmax, self.module_list)], dim=0)
        # # 使用向量化操作，避免使用for循环
        
        # return out
        # outputs = []
        # # print(f"x = {x}")
        # for i, module in enumerate(self.module_list):
        #     # print("self.ws 的形状：", self.weights.shape)
        #     # print("i = ", i)
            
        #     weight = F.softmax(self.weights, dim=0)[i]
        #     # print(f"weight{i} = {weight}, module{i} = {module}")
        #     # module = module.to("cuda")
        #     module_out = module(x)
        #     # print(f"module_out{i} = {module_out}")
        #     sub_out = weight * module_out
        #     outputs.append(sub_out)

        # # Sum all the weighted kernels to get the final output
        # result = sum(outputs)
        # # print(f"result = {result}")
        # return result
    def get_max_weight_kernel(self):
        max_weight_idx = torch.argmax(self.weights)
        return self.module_list(max_weight_idx)
    
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
        # outputs = []
        # for i in range(self.factor):
        #     outputs.append(self.layer(x))
        # concatenated_tensor = torch.cat(outputs, dim=1)
        # del outputs
        # # print(f"old channel = {x.shape[1]}, new channel = {concatenated_tensor.shape[1]}")
        # return concatenated_tensor
        output = self.layer(x)  # 避免重复计算

        # 使用torch.cat代替循环和in-place操作
        concatenated_tensor = torch.cat([output.clone() for _ in range(self.factor)], dim=1)
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
                        if (child.in_channels == child.out_channels):
                            setattr(module, name,
                                    placeholder())
                            
                        elif (child.in_channels < child.out_channels):
                            factor = child.out_channels // child.in_channels
                            # print(f"factor = {factor}")
                            setattr(module, name,
                                    out_gt_in(factor
                                                    ))
                        else:
                            factor = child.in_channels // child.out_channels
                            # print(f"factor = {factor}")
                            setattr(module, name,
                                    in_gt_out(factor
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
   