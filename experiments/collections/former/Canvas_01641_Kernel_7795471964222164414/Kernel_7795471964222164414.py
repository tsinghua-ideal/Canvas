import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_7795471964222164414(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_7795471964222164414, self).__init__()
        self.g = 16
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Convolution_1x3_1x3_DW0: p_1
        self.p_1 = nn.Conv2d(self.c, self.c, (1, 3), dilation=(1, 3), padding=(0, 3), groups=1, bias=False)
        # UnfoldH_K5_D2: p_2
        pass
        # Group_0_C/G: p_3
        pass
        # Shift_1/0/H_K1: p_4
        self.p_4_1_0 = random.randint(-1, 1)
        # FC: p_5
        self.p_5 = nn.Conv2d(self.c * 5, self.c, 1, padding=0, groups=1, bias=False)
        # BMul: p_6
        pass
        # Output: p_7
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Convolution_1x3_1x3_DW0: p_1
        t_1 = self.p_1(t_0)
        t_1 = t_1.view(self.n, self.c, self.h, self.w)
        # UnfoldH_K5_D2: p_2
        t_2 = F.unfold(t_1, (5, 1), dilation=(2, 1), padding=(4, 0))
        t_2 = t_2.view(self.n, self.c, 5, self.h, self.w)
        # Group_0_C/G: p_3
        t_3 = t_0.view(self.n, self.c // self.g, self.g, self.h, self.w)
        # Shift_1/0/H_K1: p_4
        t_4 = torch.roll(t_3, self.p_4_1_0, 3)
        # FC: p_5
        t_5 = t_2.view(self.n, self.c * 5, self.h, self.w)
        t_5 = self.p_5(t_5)
        t_5 = t_5.view(self.n, self.c, self.h, self.w)
        # BMul: p_6
        t_6_lhs = t_5.view(self.n, 1, self.c, self.h, self.w)
        t_6_rhs = t_4.view(self.n, 1, self.c, self.h, self.w)
        t_6 = t_6_lhs * t_6_rhs
        t_6 = t_6.view(self.n, self.c // self.g, self.g, self.h, self.w)
        # Output: p_7
        return t_6.view(self.n, self.c, self.h, self.w)