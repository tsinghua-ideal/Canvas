import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_6199962355332830965(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_6199962355332830965, self).__init__()
        self.g = 2
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Fold_0/1/C_Max: p_1
        pass
        # Neg: p_2
        pass
        # Shift_1/1/W_K3: p_3
        self.p_3_1_1 = random.randint(-3, 3)
        # BMax: p_4
        pass
        # Scale_0/1/C_1/0/H_1/1/W: p_5
        self.p_5_w = nn.Parameter(torch.ones((1, self.c, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_5_w, std=.02)
        # FC: p_6
        self.p_6 = nn.Conv2d(self.c, self.c, 1, padding=0, groups=1, bias=False)
        # BMM_1_0: p_7
        pass
        # Convolution_1x5_1x3_DW0: p_8
        self.p_8 = nn.Conv2d(1, self.c // 2, (1, 5), dilation=(1, 3), padding=(0, 6), groups=1, bias=False)
        # UnfoldH_K3_D2: p_9
        pass
        # FC: p_10
        self.p_10 = nn.Conv2d(self.c * 3 // 2, self.c, 1, padding=0, groups=1, bias=False)
        # Shift_0/1/C_K3: p_11
        self.p_11_0_1 = random.randint(-3, 3)
        # BMax: p_12
        pass
        # BMM_1_1: p_13
        pass
        # Output: p_14
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Fold_0/1/C_Max: p_1
        t_1 = t_0.max(1)[0]
        # Neg: p_2
        t_2 = -t_0
        # Shift_1/1/W_K3: p_3
        t_3 = torch.roll(t_2, self.p_3_1_1, 3)
        # BMax: p_4
        t_4 = torch.maximum(t_2, t_3)
        # Scale_0/1/C_1/0/H_1/1/W: p_5
        t_5 = self.p_5_w * t_4
        # FC: p_6
        t_6 = self.p_6(t_5)
        t_6 = t_6.view(self.n, self.c, self.h, self.w)
        # BMM_1_0: p_7
        t_6_lhs = t_6.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_0_rhs = t_0.view(self.n, self.c, self.h * self.w)        
        t_7 = torch.bmm(t_6_lhs, t_0_rhs).view(self.n, self.h, self.w, self.h, self.w) / math.sqrt(self.c)
        # Convolution_1x5_1x3_DW0: p_8
        t_8 = t_1.view(self.n, 1, self.h, self.w)
        t_8 = self.p_8(t_8)
        t_8 = t_8.view(self.n, self.c // 2, self.h, self.w)
        # UnfoldH_K3_D2: p_9
        t_9 = F.unfold(t_8, (3, 1), dilation=(2, 1), padding=(2, 0))
        t_9 = t_9.view(self.n, self.c // 2, 3, self.h, self.w)
        # FC: p_10
        t_10 = t_9.view(self.n, self.c * 3 // 2, self.h, self.w)
        t_10 = self.p_10(t_10)
        t_10 = t_10.view(self.n, self.c, self.h, self.w)
        # Shift_0/1/C_K3: p_11
        t_11 = torch.roll(t_10, self.p_11_0_1, 1)
        # BMax: p_12
        t_12 = torch.maximum(t_0, t_11)
        # BMM_1_1: p_13
        t_7_lhs = t_7.view(self.n, self.h * self.w, self.h * self.w).transpose(1, 2)        
        t_12_rhs = t_12.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_13 = torch.bmm(t_7_lhs, t_12_rhs).view(self.n, self.h, self.w, self.c) / math.sqrt(self.h * self.w)
        # Output: p_14
        return t_13.permute(0, 3, 1, 2).contiguous().view(self.n, self.c, self.h, self.w)