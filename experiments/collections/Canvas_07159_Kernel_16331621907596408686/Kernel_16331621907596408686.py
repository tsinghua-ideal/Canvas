import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_16331621907596408686(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_16331621907596408686, self).__init__()
        self.g = 2
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Convolution_5_1_1_1_DW1: p_1
        self.p_1 = nn.Conv2d(self.c, self.c, (5, 1), dilation=(1, 1), padding=(2, 0), groups=self.c, bias=False)
        # UnfoldH_K3_D3: p_2
        pass
        # Convolution_3_3_2_2_DW0: p_3
        self.p_3 = nn.Conv2d(self.c, self.c * 3, (3, 3), dilation=(2, 2), padding=(2, 2), groups=8, bias=False)
        # BMul: p_4
        pass
        # Scale_0/1/C_1/0/H: p_5
        self.p_5_w = nn.Parameter(torch.ones((1, self.c, self.h, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_5_w, std=.02)
        # Convolution_3_1_3_1_DW1: p_6
        self.p_6 = nn.Conv2d(self.c * 3, self.c * 3, (3, 1), dilation=(3, 1), padding=(3, 0), groups=self.c, bias=False)
        # FC: p_7
        self.p_7 = nn.Conv2d(self.c * 3, self.c, 1, padding=0, groups=1, bias=False)
        # BSub: p_8
        pass
        # Output: p_9
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Convolution_5_1_1_1_DW1: p_1
        t_1 = self.p_1(t_0)
        t_1 = t_1.view(self.n, self.c, self.h, self.w)
        # UnfoldH_K3_D3: p_2
        t_2 = F.unfold(t_0, (3, 1), dilation=(3, 1), padding=(3, 0))
        t_2 = t_2.view(self.n, self.c, 3, self.h, self.w)
        # Convolution_3_3_2_2_DW0: p_3
        t_3 = self.p_3(t_0)
        t_3 = t_3.view(self.n, self.c * 3, self.h, self.w)
        # BMul: p_4
        t_4_lhs = t_2.view(self.n, 1, self.c * 3, self.h, self.w)
        t_4_rhs = t_3.view(self.n, 1, self.c * 3, self.h, self.w)
        t_4 = t_4_lhs * t_4_rhs
        t_4 = t_4.view(self.n, self.c * 3, self.h, self.w)
        # Scale_0/1/C_1/0/H: p_5
        t_5 = self.p_5_w * t_1
        # Convolution_3_1_3_1_DW1: p_6
        t_6 = self.p_6(t_4)
        t_6 = t_6.view(self.n, self.c * 3, self.h, self.w)
        # FC: p_7
        t_7 = self.p_7(t_6)
        t_7 = t_7.view(self.n, self.c, self.h, self.w)
        # BSub: p_8
        t_8_lhs = t_5.view(self.n, 1, self.c, self.h, self.w)
        t_8_rhs = t_7.view(self.n, 1, self.c, self.h, self.w)
        t_8 = t_8_lhs - t_8_rhs
        t_8 = t_8.view(self.n, self.c, self.h, self.w)
        # Output: p_9
        return t_8.view(self.n, self.c, self.h, self.w)