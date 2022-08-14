import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_17164586276825551330(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_17164586276825551330, self).__init__()
        self.g = 8
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Convolution_5x1_1x1_DW0: p_1
        self.p_1 = nn.Conv2d(self.c, self.c, (5, 1), dilation=(1, 1), padding=(2, 0), groups=1, bias=False)
        # UnfoldW_K3_D2: p_2
        pass
        # Convolution_3x1_1x1_DW1: p_3
        self.p_3 = nn.Conv2d(self.c, self.c, (3, 1), dilation=(1, 1), padding=(1, 0), groups=self.c, bias=False)
        # Mix: p_4
        self.p_4_w = nn.Parameter(torch.ones((self.c, 3, 1, self.g, )), requires_grad=True)
        bound = math.sqrt(3.0 / (self.c * 3))
        nn.init.uniform_(self.p_4_w, a=-bound, b=bound)
        # BMul: p_5
        pass
        # Output: p_6
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Convolution_5x1_1x1_DW0: p_1
        t_1 = self.p_1(t_0)
        t_1 = t_1.view(self.n, self.c, self.h, self.w)
        # UnfoldW_K3_D2: p_2
        t_2 = F.unfold(t_0, (1, 3), dilation=(1, 2), padding=(0, 2))
        t_2 = t_2.view(self.n, self.c, 3, self.h, self.w)
        # Convolution_3x1_1x1_DW1: p_3
        t_3 = self.p_3(t_1)
        t_3 = t_3.view(self.n, self.c, self.h, self.w)
        # Mix: p_4
        t_4 = torch.einsum('abcde,bcfg->afgde', [t_2, self.p_4_w]).view(self.n, self.g, self.h, self.w).contiguous()
        # BMul: p_5
        t_5_lhs = t_4.view(self.n, 1, self.g, self.h, self.w)
        t_5_rhs = t_3.view(self.n, self.c // self.g, self.g, self.h, self.w)
        t_5 = t_5_lhs * t_5_rhs
        t_5 = t_5.view(self.n, self.c, self.h, self.w)
        # Output: p_6
        return t_5.view(self.n, self.c, self.h, self.w)