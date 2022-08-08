import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_8374174466554965253(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_8374174466554965253, self).__init__()
        self.g = 4
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Sigmoid: p_1
        pass
        # BMul: p_2
        pass
        # Convolution_1_3_1_2_DW0: p_3
        self.p_3 = nn.Conv2d(self.c, self.c, (1, 3), dilation=(1, 2), padding=(0, 2), groups=1, bias=False)
        # Convolution_7_1_3_1_DW1: p_4
        self.p_4 = nn.Conv2d(self.c, self.c, (7, 1), dilation=(3, 1), padding=(9, 0), groups=self.c, bias=False)
        # BMM_0_1: p_5
        pass
        # UnfoldH_K7_D2: p_6
        pass
        # Shift_0/1/C_K3: p_7
        self.p_7_0_1 = random.randint(-3, 3)
        # Shift_1/0/H_K3: p_8
        self.p_8_1_0 = random.randint(-3, 3)
        # Scale_0/1/C_1/1/C: p_9
        self.p_9_w = nn.Parameter(torch.ones((1, self.c, self.c,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_9_w, std=.02)
        # Scale_0/1/C_0/2/KH_1/0/H_1/1/W: p_10
        self.p_10_w = nn.Parameter(torch.ones((1, self.c, 7, 1, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_10_w, std=.02)
        # Shift_1/1/W_K1: p_11
        self.p_11_1_1 = random.randint(-1, 1)
        # Shift_1/0/H_1/1/W_K1: p_12
        self.p_12_1_0 = random.randint(-1, 1)
        self.p_12_1_1 = random.randint(-1, 1)
        # Shift_1/1/W_K1: p_13
        self.p_13_1_1 = random.randint(-1, 1)
        # BMM_1_0: p_14
        pass
        # FC: p_15
        self.p_15 = nn.Conv2d(self.c * 7, self.c, 1, padding=0, groups=4, bias=False)
        # BAdd: p_16
        pass
        # Output: p_17
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Sigmoid: p_1
        t_1 = torch.sigmoid(t_0)
        # BMul: p_2
        t_2 = t_1 * t_0
        # Convolution_1_3_1_2_DW0: p_3
        t_3 = self.p_3(t_2)
        t_3 = t_3.view(self.n, self.c, self.h, self.w)
        # Convolution_7_1_3_1_DW1: p_4
        t_4 = self.p_4(t_3)
        t_4 = t_4.view(self.n, self.c, self.h, self.w)
        # BMM_0_1: p_5
        t_1_lhs = t_1.view(self.n, self.c, self.h * self.w)        
        t_0_rhs = t_0.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_5 = torch.bmm(t_1_lhs, t_0_rhs).view(self.n, self.c, self.c) / math.sqrt(self.h * self.w)
        # UnfoldH_K7_D2: p_6
        t_6 = F.unfold(t_4, (7, 1), dilation=(2, 1), padding=(6, 0))
        t_6 = t_6.view(self.n, self.c, 7, self.h, self.w)
        # Shift_0/1/C_K3: p_7
        t_7 = torch.roll(t_6, self.p_7_0_1, 1)
        # Shift_1/0/H_K3: p_8
        t_8 = torch.roll(t_7, self.p_8_1_0, 3)
        # Scale_0/1/C_1/1/C: p_9
        t_9 = self.p_9_w * t_5
        # Scale_0/1/C_0/2/KH_1/0/H_1/1/W: p_10
        t_10 = self.p_10_w * t_8
        # Shift_1/1/W_K1: p_11
        t_11 = torch.roll(t_10, self.p_11_1_1, 4)
        # Shift_1/0/H_1/1/W_K1: p_12
        t_12 = torch.roll(t_11, self.p_12_1_0, 3)
        t_12 = torch.roll(t_12, self.p_12_1_1, 4)
        # Shift_1/1/W_K1: p_13
        t_13 = torch.roll(t_12, self.p_13_1_1, 4)
        # BMM_1_0: p_14
        t_9_lhs = t_9.view(self.n, self.c, self.c).transpose(1, 2)        
        t_2_rhs = t_2.view(self.n, self.c, self.h * self.w)        
        t_14 = torch.bmm(t_9_lhs, t_2_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c)
        # FC: p_15
        t_15 = t_13.view(self.n, self.c * 7, self.h, self.w)
        t_15 = self.p_15(t_15)
        t_15 = t_15.view(self.n, self.c, self.h, self.w)
        # BAdd: p_16
        t_16 = t_15 + t_14
        # Output: p_17
        return t_16.view(self.n, self.c, self.h, self.w)