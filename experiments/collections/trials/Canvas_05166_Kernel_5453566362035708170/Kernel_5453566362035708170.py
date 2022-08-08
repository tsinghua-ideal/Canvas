import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_5453566362035708170(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_5453566362035708170, self).__init__()
        self.g = 1
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Scale_0/1/C_1/0/H: p_1
        self.p_1_w = nn.Parameter(torch.ones((1, self.c, self.h, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_1_w, std=.1)
        # BAdd: p_2
        pass
        # ReLU: p_4
        pass
        # UnfoldH_K3_D1: p_5
        pass
        # FC: p_7
        self.p_7 = nn.Conv2d(self.c, self.c, 1, padding=0, groups=1, bias=False)
        # UnfoldW_K5_D3: p_8
        pass
        # BSub: p_9
        pass
        # FC: p_10
        self.p_10 = nn.Conv2d(self.c, self.c * 21, 1, padding=0, groups=4, bias=False)
        # BMM_0_1: p_11
        pass
        # Fold_0/3/KW_Avg: p_12
        pass
        # BMM_0_0: p_13
        pass
        # BMax: p_14
        pass
        # Output: p_15
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Scale_0/1/C_1/0/H: p_1
        t_1 = self.p_1_w * t_0
        # BAdd: p_2
        t_2 = t_0 + t_1
        # ReLU: p_4
        t_4 = torch.relu(t_0)
        # UnfoldH_K3_D1: p_5
        t_5 = F.unfold(t_0, (3, 7), dilation=(1, 1), padding=(1, 3))
        t_5 = t_5.view(self.n, self.c, 3, 7, self.h, self.w)
        # FC: p_7
        t_7 = self.p_7(t_4)
        t_7 = t_7.view(self.n, self.c, self.h, self.w)
        # UnfoldW_K5_D3: p_8
        t_8 = F.unfold(t_7, (1, 5), dilation=(1, 3), padding=(0, 6))
        t_8 = t_8.view(self.n, self.c, 5, self.h, self.w)
        # BSub: p_9
        t_9_lhs = t_0.view(self.n, self.c, 1, self.h, self.w)
        t_9_rhs = t_8.view(self.n, self.c, 5, self.h, self.w)
        t_9 = t_9_lhs - t_9_rhs
        t_9 = t_9.view(self.n, self.c, 5, self.h, self.w)
        # FC: p_10
        t_10 = self.p_10(t_2)
        t_10 = t_10.view(self.n, self.c * 21, self.h, self.w)
        # BMM_0_1: p_11
        t_1_lhs = t_1.view(self.n, self.c, self.h * self.w)        
        t_6_rhs = t_5.view(self.n, self.c * 21, self.h * self.w).transpose(1, 2)        
        t_11 = torch.bmm(t_1_lhs, t_6_rhs).view(self.n, self.c, self.c, 3, 7) / math.sqrt(self.h * self.w)
        # Fold_0/3/KW_Avg: p_12
        t_12 = t_9.mean(2)
        # BMM_0_0: p_13
        t_11_lhs = t_11.view(self.n, self.c, self.c * 21)        
        t_10_rhs = t_10.view(self.n, self.c * 21, self.h * self.w)        
        t_13 = torch.bmm(t_11_lhs, t_10_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c * 21)
        # BMax: p_14
        t_14 = torch.maximum(t_13, t_12)
        # Output: p_15
        return t_14.view(self.n, self.c, self.h, self.w)