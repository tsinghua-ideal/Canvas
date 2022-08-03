import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_16331621907596408686(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_16331621907596408686, self).__init__()
        self.g = 1
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldH_K3_D3: p_2
        pass
        # Convolution_3_3_2_2_DW0: p_3
        self.p_3 = nn.Conv2d(self.c, self.c * 3, (3, 3), dilation=(2, 2), padding=(2, 2), groups=self.g)
        # BMul: p_4
        pass
        # FC: p_7
        self.p_7 = nn.Conv2d(self.c * 3, self.c, 1, padding=0, groups=self.g)
        # Output: p_9
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldH_K3_D3: p_2
        t_2 = F.unfold(t_0, (3, 1), dilation=(3, 1), padding=(3, 0))
        t_2 = t_2.view(self.n, self.c, 3, self.h, self.w)
        # Convolution_3_3_2_2_DW0: p_3
        t_3 = self.p_3(t_0)
        # BMul: p_4
        t_4_lhs = t_2.view(self.n, 1, self.c * 3, self.h, self.w)
        t_4_rhs = t_3.view(self.n, 1, self.c * 3, self.h, self.w)
        t_4 = t_4_lhs * t_4_rhs
        t_4 = t_4.view(self.n, self.c * 3, self.h, self.w)
        # FC: p_7
        t_7 = self.p_7(t_4)
        return t_0 + t_7
