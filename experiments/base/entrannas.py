import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Callable, Dict
import canvas
from . import log
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
        self.alphas = nn.Parameter(torch.randn(len(module_list))) 
        self.temperature = nn.Parameter(torch.tensor(5.0))
        
    def forward(self, x):
        alphas = F.softmax(self.alphas/self.temperature, dim=0)
        out = torch.zeros(x.shape, device=x.device)
        for i in range(len(self.module_list)):
            out.add(alphas[i] * self.module_list[i](x))
        return out
    
    def temperature_anneal(self):
        self.temperature.data = self.temperature * 0.9235

    def get_max_weight_kernel(self):
        max_weight_idx = torch.argmax(self.alphas)
        return self.module_list(max_weight_idx)
    def print_parameters(self, j):
        logger = log.get_logger()
        # Print parameters in each placeholder  
        logger.info(f'In {self.i}th Placeholder')  
        # Alpha
        logger.info(f'####### ALPHA After {j}th epoch #######')
        logger.info('# Alphas')
        logger.info(F.softmax(self.alphas, dim=0))
        logger.info('#####################')
        # Temperature TO DO
        
