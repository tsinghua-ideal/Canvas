import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_5552027089989569678(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_5552027089989569678, self).__init__()
        self.g = 2
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Convolution_5_1_1_1_DW1: p_1
        self.p_1 = nn.Conv2d(self.c, self.c, (5, 5), dilation=(1, 1), padding=(2, 2), groups=self.c, bias=False)
        # BMM_0_1: p_2
        pass
        # UnfoldH_K7_D3: p_3
        pass
        # BAdd: p_4
        pass
        # Scale_0/1/C_0/2/KH_1/0/H_1/1/W: p_5
        self.p_5_w = nn.Parameter(torch.ones((1, self.c, 7, 1, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_5_w, std=.02)
        # FC: p_6
        self.p_6 = nn.Conv2d(self.c * 7, self.c, 1, padding=0, groups=2, bias=False)
        # GeLU: p_7
        pass
        # Convolution_1_7_1_3_DW1: p_8
        self.p_8 = nn.Conv2d(self.c, self.c, (7, 7), dilation=(3, 3), padding=(9, 9), groups=self.c, bias=False)
        # BMM_1_0: p_9
        pass
        # Output: p_10
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Convolution_5_1_1_1_DW1: p_1
        t_1 = self.p_1(t_0)
        t_1 = t_1.view(self.n, self.c, self.h, self.w)
        # BMM_0_1: p_2
        t_0_lhs = t_0.view(self.n, self.c, self.h * self.w)        
        t_0_rhs = t_0.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_2 = torch.bmm(t_0_lhs, t_0_rhs).view(self.n, self.c, self.c) / math.sqrt(self.h * self.w)
        # UnfoldH_K7_D3: p_3
        t_3 = F.unfold(t_1, (7, 1), dilation=(3, 1), padding=(9, 0))
        t_3 = t_3.view(self.n, self.c, 7, self.h, self.w)
        # BAdd: p_4
        t_4_lhs = t_0.view(self.n, self.c, 1, self.h, self.w)
        t_4_rhs = t_3.view(self.n, self.c, 7, self.h, self.w)
        t_4 = t_4_lhs + t_4_rhs
        t_4 = t_4.view(self.n, self.c, 7, self.h, self.w)
        # Scale_0/1/C_0/2/KH_1/0/H_1/1/W: p_5
        t_5 = self.p_5_w * t_4
        # FC: p_6
        t_6 = t_5.view(self.n, self.c * 7, self.h, self.w)
        t_6 = self.p_6(t_6)
        t_6 = t_6.view(self.n, self.c, self.h, self.w)
        # GeLU: p_7
        t_7 = F.gelu(t_6)
        # Convolution_1_7_1_3_DW1: p_8
        t_8 = self.p_8(t_7)
        t_8 = t_8.view(self.n, self.c, self.h, self.w)
        # BMM_1_0: p_9
        t_2_lhs = t_2.view(self.n, self.c, self.c).transpose(1, 2)        
        t_8_rhs = t_8.view(self.n, self.c, self.h * self.w)        
        t_9 = torch.bmm(t_2_lhs, t_8_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c)
        # Output: p_10
        return t_9.view(self.n, self.c, self.h, self.w)