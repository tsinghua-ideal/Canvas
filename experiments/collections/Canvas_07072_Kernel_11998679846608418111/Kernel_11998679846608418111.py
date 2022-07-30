import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_11998679846608418111(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_11998679846608418111, self).__init__()
        self.g = 2
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldH_K3_D2: p_1
        pass
        # Convolution_5_5_1_1_DW1: p_2
        self.p_2 = nn.Conv2d(self.c, self.c, (5, 5), dilation=(1, 1), padding=(2, 2), groups=self.c, bias=False)
        # FC: p_3
        self.p_3 = nn.Conv2d(self.c * 3, self.c, 1, padding=0, groups=self.c // 4, bias=False)
        # Group_0_G: p_4
        pass
        # Scale_0/1/C_1/0/H_1/1/W: p_5
        self.p_5_w = nn.Parameter(torch.ones((1, self.c, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_5_w, std=.02)
        # UnfoldH_K3_D3: p_6
        pass
        # BMax: p_7
        pass
        # FC: p_8
        self.p_8 = nn.Conv2d(self.c * 3, self.c, 1, padding=0, groups=self.c // 4, bias=False)
        # Convolution_3_3_1_1_DW0: p_9
        self.p_9 = nn.Conv2d(self.c, self.c, (3, 3), dilation=(1, 1), padding=(1, 1), groups=self.c // 4, bias=False)
        # BMul: p_10
        pass
        # Fold_0/1/C_Avg: p_11
        pass
        # BAdd: p_12
        pass
        # BAdd: p_13
        pass
        # BMul: p_14
        pass
        # Output: p_15
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldH_K3_D2: p_1
        t_1 = F.unfold(t_0, (3, 1), dilation=(2, 1), padding=(2, 0))
        t_1 = t_1.view(self.n, self.c, 3, self.h, self.w)
        # Convolution_5_5_1_1_DW1: p_2
        t_2 = self.p_2(t_0)
        t_2 = t_2.view(self.n, self.c, self.h, self.w)
        # FC: p_3
        t_3 = t_1.view(self.n, self.c * 3, self.h, self.w)
        t_3 = self.p_3(t_3)
        t_3 = t_3.view(self.n, self.c, self.h, self.w)
        # Group_0_G: p_4
        t_4 = t_0.view(self.n, self.g, self.c // self.g, self.h, self.w)
        # Scale_0/1/C_1/0/H_1/1/W: p_5
        t_5 = self.p_5_w * t_3
        # UnfoldH_K3_D3: p_6
        t_6 = F.unfold(t_5, (3, 1), dilation=(3, 1), padding=(3, 0))
        t_6 = t_6.view(self.n, self.c, 3, self.h, self.w)
        # BMax: p_7
        t_7 = torch.maximum(t_5, t_2)
        # FC: p_8
        t_8 = t_6.view(self.n, self.c * 3, self.h, self.w)
        t_8 = self.p_8(t_8)
        t_8 = t_8.view(self.n, self.c, self.h, self.w)
        # Convolution_3_3_1_1_DW0: p_9
        t_9 = self.p_9(t_7)
        t_9 = t_9.view(self.n, self.c, self.h, self.w)
        # BMul: p_10
        t_10 = t_7 * t_8
        # Fold_0/1/C_Avg: p_11
        t_11 = t_4.mean(2)
        # BAdd: p_12
        t_12 = t_10 + t_3
        # BAdd: p_13
        t_13_lhs = t_11.view(self.n, 1, self.g, self.h, self.w)
        t_13_rhs = t_12.view(self.n, self.c // self.g, self.g, self.h, self.w)
        t_13 = t_13_lhs + t_13_rhs
        t_13 = t_13.view(self.n, self.c, self.h, self.w)
        # BMul: p_14
        t_14 = t_9 * t_13
        # Output: p_15
        return t_14.view(self.n, self.c, self.h, self.w)