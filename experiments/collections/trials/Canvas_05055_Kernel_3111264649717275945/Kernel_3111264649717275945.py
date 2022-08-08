import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_3111264649717275945(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_3111264649717275945, self).__init__()
        self.g = 2
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Fold_0/1/C_Avg: p_1
        pass
        # UnfoldW_K7_D1: p_2
        pass
        # Mix: p_3
        self.p_3_w = nn.Parameter(torch.ones((7, self.c, )), requires_grad=True)
        nn.init.trunc_normal_(self.p_3_w, std=.1)
        # Convolution_3x1_1x1_DW1: p_4
        self.p_4 = nn.Conv2d(self.c, self.c, (3, 3), dilation=(1, 1), padding=(1, 1), groups=self.c, bias=False)
        # Shift_1/1/W_K1: p_5
        self.p_5_1_1 = random.randint(-1, 1)
        # Fold_1/0/H_Max: p_6
        pass
        # FC: p_7
        self.p_7 = nn.Conv2d(1, self.c, 1, padding=0, groups=1, bias=False)
        # Convolution_5x5_2x2_DW1: p_8
        self.p_8 = nn.Conv2d(self.c, self.c, (5, 5), dilation=(2, 2), padding=(4, 4), groups=self.c, bias=False)
        # BMM_0_1: p_9
        pass
        # BMM_0_1: p_10
        pass
        # FC: p_11
        self.p_11 = nn.Conv2d(self.c, self.c, 1, padding=0, groups=1, bias=False)
        # Group_1_x_3: p_12
        pass
        # Convolution_3x3_2x2_DW0: p_13
        self.p_13 = nn.Conv2d(self.c, self.c // self.g, (3, 3), dilation=(2, 2), padding=(2, 2), groups=1, bias=False)
        # BAdd: p_14
        pass
        # BMM_1_0: p_15
        pass
        # BAdd: p_16
        pass
        # Output: p_17
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Fold_0/1/C_Avg: p_1
        t_1 = t_0.mean(1)
        # UnfoldW_K7_D1: p_2
        t_2 = t_1.view(self.n, 1, self.h, self.w)
        t_2 = F.unfold(t_2, (1, 7), dilation=(1, 1), padding=(0, 3))
        t_2 = t_2.view(self.n, 7, self.h, self.w)
        # Mix: p_3
        t_3 = torch.einsum('abcd,be->aecd', [t_2, self.p_3_w]).view(self.n, self.c, self.h, self.w).contiguous()
        # Convolution_3x1_1x1_DW1: p_4
        t_4 = self.p_4(t_0)
        t_4 = t_4.view(self.n, self.c, self.h, self.w)
        # Shift_1/1/W_K1: p_5
        t_5 = torch.roll(t_4, self.p_5_1_1, 3)
        # Fold_1/0/H_Max: p_6
        t_6 = t_1.max(1)[0]
        # FC: p_7
        t_7 = t_6.view(self.n, 1, 1, self.w)
        t_7 = self.p_7(t_7)
        t_7 = t_7.view(self.n, self.c, self.w)
        # Convolution_5x5_2x2_DW1: p_8
        t_8 = self.p_8(t_0)
        t_8 = t_8.view(self.n, self.c, self.h, self.w)
        # BMM_0_1: p_9
        t_5_lhs = t_5.view(self.n, self.c, self.h * self.w)        
        t_0_rhs = t_0.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_9 = torch.bmm(t_5_lhs, t_0_rhs).view(self.n, self.c, self.c) / math.sqrt(self.h * self.w)
        # BMM_0_1: p_10
        t_3_lhs = t_3.view(self.n, self.c, self.h * self.w)        
        t_0_rhs = t_0.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_10 = torch.bmm(t_3_lhs, t_0_rhs).view(self.n, self.c, self.c) / math.sqrt(self.h * self.w)
        # FC: p_11
        t_11 = self.p_11(t_8)
        t_11 = t_11.view(self.n, self.c, self.h, self.w)
        # Group_1_x_3: p_12
        t_12 = t_9
        # Convolution_3x3_2x2_DW0: p_13
        t_13 = t_7.view(self.n, self.c, 1, self.w)
        t_13 = self.p_13(t_13)
        t_13 = t_13.view(self.n, self.c // self.g, self.w)
        # BAdd: p_14
        t_14_lhs = t_12.view(self.n, 1, self.c * self.c)
        t_14_rhs = t_10.view(self.n, 1, self.c * self.c)
        t_14 = t_14_lhs + t_14_rhs
        t_14 = t_14.view(self.n, self.c, self.c)
        # BMM_1_0: p_15
        t_11_lhs = t_11.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_14_rhs = t_14.view(self.n, self.c, self.c)        
        t_15 = torch.bmm(t_11_lhs, t_14_rhs).view(self.n, self.h, self.w, self.c) / math.sqrt(self.c)
        # BAdd: p_16
        t_16_lhs = t_13.view(self.n, 1, self.c * self.w // self.g)
        t_16_rhs = t_15.view(self.n, self.g * self.h, self.c * self.w // self.g)
        t_16 = t_16_lhs + t_16_rhs
        t_16 = t_16.view(self.n, self.h, self.w, self.c)
        # Output: p_17
        return t_16.permute(0, 3, 1, 2).contiguous().view(self.n, self.c, self.h, self.w)