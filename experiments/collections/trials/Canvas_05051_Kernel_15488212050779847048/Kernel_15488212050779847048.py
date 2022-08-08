import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_15488212050779847048(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_15488212050779847048, self).__init__()
        self.g = 32
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldW_K3_D2: p_1
        pass
        # Fold_0/3/KW_Max: p_2
        pass
        # FC: p_3
        self.p_3 = nn.Conv2d(self.c * 3, self.c, 1, padding=0, groups=1, bias=False)
        # BMax: p_4
        pass
        # Scale_1/0/H_1/1/W: p_5
        self.p_5_w = nn.Parameter(torch.ones((1, 1, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_5_w, std=.02)
        # Convolution_3_1_3_1_DW0: p_6
        self.p_6 = nn.Conv2d(self.c, self.c, (3, 1), dilation=(3, 1), padding=(3, 0), groups=1, bias=False)
        # BMax: p_7
        pass
        # Scale_0/1/C: p_8
        self.p_8_w = nn.Parameter(torch.ones((1, self.c, 1, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_8_w, std=.02)
        # BMM_0_1: p_9
        pass
        # BAdd: p_10
        pass
        # BMax: p_11
        pass
        # Scale_0/3/KW_1/0/H: p_12
        self.p_12_w = nn.Parameter(torch.ones((1, 1, 3, self.h, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_12_w, std=.02)
        # BMM_1_1: p_13
        pass
        # Output: p_14
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldW_K3_D2: p_1
        t_1 = F.unfold(t_0, (1, 3), dilation=(1, 2), padding=(0, 2))
        t_1 = t_1.view(self.n, self.c, 3, self.h, self.w)
        # Fold_0/3/KW_Max: p_2
        t_2 = t_1.max(2)[0]
        # FC: p_3
        t_3 = t_1.view(self.n, self.c * 3, self.h, self.w)
        t_3 = self.p_3(t_3)
        t_3 = t_3.view(self.n, self.c, self.h, self.w)
        # BMax: p_4
        t_4_lhs = t_3.view(self.n, self.c, 1, self.h, self.w)
        t_4_rhs = t_1.view(self.n, self.c, 3, self.h, self.w)
        t_4 = torch.maximum(t_4_lhs, t_4_rhs)
        t_4 = t_4.view(self.n, self.c, 3, self.h, self.w)
        # Scale_1/0/H_1/1/W: p_5
        t_5 = self.p_5_w * t_2
        # Convolution_3_1_3_1_DW0: p_6
        t_6 = self.p_6(t_5)
        t_6 = t_6.view(self.n, self.c, self.h, self.w)
        # BMax: p_7
        t_7 = torch.maximum(t_0, t_6)
        # Scale_0/1/C: p_8
        t_8 = self.p_8_w * t_7
        # BMM_0_1: p_9
        t_8_lhs = t_8.view(self.n, self.c, self.h * self.w)        
        t_1_rhs = t_1.view(self.n, self.c * 3, self.h * self.w).transpose(1, 2)        
        t_9 = torch.bmm(t_8_lhs, t_1_rhs).view(self.n, self.c, self.c, 3) / math.sqrt(self.h * self.w)
        # BAdd: p_10
        t_10_lhs = t_7.view(self.n, self.c, 1, self.h, self.w)
        t_10_rhs = t_4.view(self.n, self.c, 3, self.h, self.w)
        t_10 = t_10_lhs + t_10_rhs
        t_10 = t_10.view(self.n, self.c, 3, self.h, self.w)
        # BMax: p_11
        t_11_lhs = t_6.view(self.n, self.c, 1, self.h, self.w)
        t_11_rhs = t_10.view(self.n, self.c, 3, self.h, self.w)
        t_11 = torch.maximum(t_11_lhs, t_11_rhs)
        t_11 = t_11.view(self.n, self.c, 3, self.h, self.w)
        # Scale_0/3/KW_1/0/H: p_12
        t_12 = self.p_12_w * t_11
        # BMM_1_1: p_13
        t_12_lhs = t_12.view(self.n, self.c * 3, self.h * self.w).transpose(1, 2)        
        t_9_rhs = t_9.view(self.n, self.c, self.c * 3).transpose(1, 2)        
        t_13 = torch.bmm(t_12_lhs, t_9_rhs).view(self.n, self.h, self.w, self.c) / math.sqrt(self.c * 3)
        # Output: p_14
        return t_13.permute(0, 3, 1, 2).contiguous().view(self.n, self.c, self.h, self.w)