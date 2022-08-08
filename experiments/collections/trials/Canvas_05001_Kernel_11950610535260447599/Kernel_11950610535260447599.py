import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_11950610535260447599(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_11950610535260447599, self).__init__()
        self.g = 4
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldHW_K7_D1: p_1
        pass
        # Neg: p_2
        pass
        # Shift_1/0/H_K3: p_3
        self.p_3_1_0 = random.randint(-3, 3)
        # FC: p_4
        self.p_4 = nn.Conv2d(self.c * 49, self.c, 1, padding=0, groups=1, bias=False)
        # Convolution_1_3_1_2_DW1: p_5
        self.p_5 = nn.Conv2d(self.c, self.c, (1, 3), dilation=(1, 2), padding=(0, 2), groups=self.c, bias=False)
        # BMul: p_6
        pass
        # Shift_1/0/H_K3: p_7
        self.p_7_1_0 = random.randint(-3, 3)
        # FC: p_8
        self.p_8 = nn.Conv2d(self.c, self.c, 1, padding=0, groups=1, bias=False)
        # BAdd: p_9
        pass
        # Output: p_10
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldHW_K7_D1: p_1
        t_1 = F.unfold(t_0, (7, 7), dilation=(1, 1), padding=(3, 3))
        t_1 = t_1.view(self.n, self.c, 7, 7, self.h, self.w)
        # Neg: p_2
        t_2 = -t_1
        # Shift_1/0/H_K3: p_3
        t_3 = torch.roll(t_0, self.p_3_1_0, 2)
        # FC: p_4
        t_4 = t_2.view(self.n, self.c * 49, self.h, self.w)
        t_4 = self.p_4(t_4)
        t_4 = t_4.view(self.n, self.c, self.h, self.w)
        # Convolution_1_3_1_2_DW1: p_5
        t_5 = self.p_5(t_4)
        t_5 = t_5.view(self.n, self.c, self.h, self.w)
        # BMul: p_6
        t_6_lhs = t_0.view(self.n, 1, self.c, self.h, self.w)
        t_6_rhs = t_5.view(self.n, 1, self.c, self.h, self.w)
        t_6 = t_6_lhs * t_6_rhs
        t_6 = t_6.view(self.n, self.c, self.h, self.w)
        # Shift_1/0/H_K3: p_7
        t_7 = torch.roll(t_3, self.p_7_1_0, 2)
        # FC: p_8
        t_8 = self.p_8(t_7)
        t_8 = t_8.view(self.n, self.c, self.h, self.w)
        # BAdd: p_9
        t_9_lhs = t_6.view(self.n, 1, self.c, self.h, self.w)
        t_9_rhs = t_8.view(self.n, 1, self.c, self.h, self.w)
        t_9 = t_9_lhs + t_9_rhs
        t_9 = t_9.view(self.n, self.c, self.h, self.w)
        # Output: p_10
        return t_9.view(self.n, self.c, self.h, self.w)