import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable, Dict
import canvas

def get_final_model(model, wsharing):
    kernel_dict = {}
    # Helper function to recursively search for conv layers
    def find(module):
        for name, child in module.named_children():
            if isinstance(child, ReplacedModule):
                kernel_dict[name] = child
            elif isinstance(child, nn.Module):
                find(child)

    find(model)
    for name, ensembled_kernel in kernel_dict.items():
        final_kernel = ensembled_kernel.get_max_weight_kernel()
        setattr(model, name, final_kernel)
            
class ReplacedModule(nn.Module):
    def __init__(self, module_list, i):
        super(ReplacedModule, self).__init__()
        self.module_list = module_list
        self.weights = nn.Parameter(1e-3*torch.randn(len(module_list)))
    def forward(self, x):
        out = torch.zeros(x.shape, device=x.device)
        weights_softmax = F.softmax(self.weights, dim=0)  
        for i in range(len(self.module_list)):
            out.add(weights_softmax[i] * self.module_list[i](x))
        return out
    def get_max_weight_kernel(self):
        max_weight_idx = torch.argmax(self.weights)
        return self.module_list(max_weight_idx)
    
class InGtOut(nn.Module):
    def __init__(self, factor):
        super(InGtOut, self).__init__()
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
class OutGtIn(nn.Module):
    def __init__(self, factor):
        super(OutGtIn, self).__init__()
        self.factor = factor
        self.layer = canvas.Placeholder()
    def forward(self, x):
        output = self.layer(x) 

        concatenated_tensor = torch.cat([output for _ in range(self.factor)], dim=1)
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
                        if (child.in_channels == child.out_channels):
                            setattr(module, name, canvas.Placeholder())
                            
                        elif (child.in_channels < child.out_channels):
                            factor = child.out_channels // child.in_channels
                            setattr(module, name, OutGtIn(factor))
                        else:
                            factor = child.in_channels // child.out_channels
                            setattr(module, name, InGtOut(factor))
                    else:
                        not_replaced += 1
                case "resblock":
                    if filter(child, string_name):
                        replaced += 1
                        if (child.downsample is None):
                            setattr(module, name, canvas.Placeholder())
                        else:
                            factor = 2
                            # print(f"factor = {factor}")
                            setattr(module, name, OutGtIn(factor))
                    else:
                        not_replaced += 1       
                                 
        elif len(list(child.named_children())) > 0:
            count = replace_module_with_placeholder(child, old_module_types, filter)
            replaced += count[0]
            not_replaced += count[1]
    return replaced, not_replaced
   