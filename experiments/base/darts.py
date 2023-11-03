import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable, Dict
import canvas
from . import log
  

class ParallelKernels(nn.Module):
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
    def __init__(self, kernel_cls_list, **kwargs):
        super().__init__()
        assert len(kernel_cls_list) >= 1
        self.kwargs = kwargs
        self.module_list = nn.ModuleList([kernel_cls(*kwargs.values()) for kernel_cls in kernel_cls_list])
        self.alphas = nn.Parameter((1e-3) * torch.randn(len(kernel_cls_list)))       
        
    def forward(self, x: torch.Tensor):
        softmax_alphas = torch.softmax(self.alphas, dim=0)
        stacked_outs = torch.stack([kernel(x) for kernel in self.module_list], dim=0)
        return torch.einsum('i,i...->...', softmax_alphas, stacked_outs)
    
    @property
    def best_kernel_index(self):
        return torch.argmax(self.softmax_alphas)
    
    @property
    def softmax_alphas(self):
        return F.softmax(self.alphas, dim=0)
    
    def get_max_weight_kernel(self):
        max_weight_idx = torch.argmax(self.alphas)
        return self.module_list[max_weight_idx]
    
    def print_parameters(self, i, j):
        logger = log.get_logger()
        
        # Print parameters in each placeholder  
        logger.info(f'In {i}th Placeholder')  
        
        # Alpha
        logger.info(f'####### ALPHA After {j}th epoch #######')
        logger.info('# Alphas')
        logger.info(F.softmax(self.alphas, dim=0))
        logger.info('#####################')    

    
class InGtOut(nn.Module):
    """
    A custom module that applies when Input channels greater than Output channels

    Args:
        factor (int): Split factor for the input tensor.
    """
    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor 
        self.layer = canvas.Placeholder()
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Aggregated output tensor.
        """
        assert x.shape[1] % self.factor == 0, "Input dimension x.shape[1] must be divisible by self.factor"
        tensors = torch.split(x, x.shape[1] // self.factor, dim = 1)
        aggregated_output = torch.sum(torch.stack([self.layer(tensor) for tensor in tensors]), dim=0)
        return aggregated_output
    
    
class OutGtIn(nn.Module):
    """
    A custom module that applies when Output channels greater than Input channels

    Args:
        factor (int): Split factor for the input tensor.
    """
    def __init__(self, factor: int):
        super().__init__()
        self.factor = factor
        self.layer = canvas.Placeholder()
        
    def forward(self, x: torch.Tensor):
        """
        Forward pass of the module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Concatenated output tensor.
        """
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
            return True
        case "resblock":
            return True
        
        
def replace_module_with_placeholder(module: nn.Module, old_module_types: Dict[nn.Module, str], filter: Callable = filter):
    if isinstance(module, canvas.Placeholder):
        return 0, 1
    replaced, not_replaced = 0, 0
    for name, child in module.named_children():
        if type(child) in old_module_types:
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
                            setattr(module, name, OutGtIn(factor))
                    else:
                        not_replaced += 1           
        elif len(list(child.named_children())) > 0:
            count = replace_module_with_placeholder(child, old_module_types, filter)
            replaced += count[0]
            not_replaced += count[1]
    return replaced, not_replaced


