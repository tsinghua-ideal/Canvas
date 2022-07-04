import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_2066129463177083751(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_2066129463177083751, self).__init__()
        self.g = 2
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # ReLU: p_1
        pass
        # FC: p_2
        self.p_2 = nn.Conv2d(self.c, self.c, 1, padding=0, groups=1, bias=False)
        # Fold_1/0/H_1/1/W_Max: p_3
        pass
        # Scale_1/0/H_1/1/W: p_4
        self.p_4_w = nn.Parameter(torch.ones((1, 1, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_4_w, std=.02)
        # Fold_0/1/C_Avg: p_5
        pass
        # UnfoldHW_K5_D3: p_6
        pass
        # Fold_1/0/H_Max: p_7
        pass
        # FC: p_8
        self.p_8 = nn.Conv2d(25, self.c, 1, padding=0, groups=1, bias=False)
        # BMM_1_0: p_9
        pass
        # FC: p_10
        self.p_10 = nn.Conv2d(self.c, self.h * 25, 1, padding=0, groups=1, bias=False)
        # GeLU: p_11
        pass
        # BAdd: p_12
        pass
        # Scale_1/0/H_1/1/W: p_13
        self.p_13_w = nn.Parameter(torch.ones((1, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_13_w, std=.02)
        # Exp: p_14
        pass
        # UnfoldW_K7_D3: p_15
        pass
        # Fold_0/1/C_Max: p_16
        pass
        # Fold_0/3/KW_Avg: p_17
        pass
        # FC: p_18
        self.p_18 = nn.Conv2d(1, self.c * self.h, 1, padding=0, groups=1, bias=False)
        # BSub: p_19
        pass
        # Output: p_20
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # ReLU: p_1
        t_1 = torch.relu(t_0)
        # FC: p_2
        t_2 = self.p_2(t_1)
        t_2 = t_2.view(self.n, self.c, self.h, self.w)
        # Fold_1/0/H_1/1/W_Max: p_3
        t_3 = t_1.max(2)[0].max(2)[0]
        # Scale_1/0/H_1/1/W: p_4
        t_4 = self.p_4_w * t_2
        # Fold_0/1/C_Avg: p_5
        t_5 = t_4.mean(1)
        # UnfoldHW_K5_D3: p_6
        t_6 = t_5.view(self.n, 1, self.h, self.w)
        t_6 = F.unfold(t_6, (5, 5), dilation=(3, 3), padding=(6, 6))
        t_6 = t_6.view(self.n, 5, 5, self.h, self.w)
        # Fold_1/0/H_Max: p_7
        t_7 = t_6.max(3)[0]
        # FC: p_8
        t_8 = t_7.view(self.n, 25, 1, self.w)
        t_8 = self.p_8(t_8)
        t_8 = t_8.view(self.n, self.c, self.w)
        # BMM_1_0: p_9
        t_3_lhs = t_3.view(self.n, self.c, 1).transpose(1, 2)        
        t_0_rhs = t_0.view(self.n, self.c, self.h * self.w)        
        t_9 = torch.bmm(t_3_lhs, t_0_rhs).view(self.n, self.h, self.w)
        # FC: p_10
        t_10 = t_8.view(self.n, self.c, 1, self.w)
        t_10 = self.p_10(t_10)
        t_10 = t_10.view(self.n, self.h * 25, self.w)
        # GeLU: p_11
        t_11 = F.gelu(t_10)
        # BAdd: p_12
        t_12_lhs = t_6.view(self.n, 1, self.h * 25, self.w)
        t_12_rhs = t_11.view(self.n, 1, self.h * 25, self.w)
        t_12 = t_12_lhs + t_12_rhs
        t_12 = t_12.view(self.n, self.h * 25, self.w)
        # Scale_1/0/H_1/1/W: p_13
        t_13 = self.p_13_w * t_9
        # Exp: p_14
        t_14 = torch.exp(t_12)
        # UnfoldW_K7_D3: p_15
        t_15 = t_14.view(self.n, self.h * 25, 1, self.w)
        t_15 = F.unfold(t_15, (1, 7), dilation=(1, 3), padding=(0, 9))
        t_15 = t_15.view(self.n, self.h * 25, 7, self.w)
        # Fold_0/1/C_Max: p_16
        t_16 = t_15.max(1)[0]
        # Fold_0/3/KW_Avg: p_17
        t_17 = t_16.mean(1)
        # FC: p_18
        t_18 = t_17.view(self.n, 1, 1, self.w)
        t_18 = self.p_18(t_18)
        t_18 = t_18.view(self.n, self.c * self.h, self.w)
        # BSub: p_19
        t_19_lhs = t_13.view(self.n, 1, self.h, self.w)
        t_19_rhs = t_18.view(self.n, self.c, self.h, self.w)
        t_19 = t_19_lhs - t_19_rhs
        t_19 = t_19.view(self.n, self.c * self.h, self.w)
        # Output: p_20
        return t_19.view(self.n, self.c, self.h, self.w)