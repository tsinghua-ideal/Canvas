import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_17371628126399981364(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_17371628126399981364, self).__init__()
        self.g = 2
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Shift_1/0/H_1/1/W_K2: p_1
        self.p_1_1_0 = random.randint(-2, 2)
        self.p_1_1_1 = random.randint(-2, 2)
        # BMM_0_1: p_2
        pass
        # Convolution_5x1_3x1_DW1: p_3
        self.p_3 = nn.Conv2d(self.c, self.c, (5, 5), dilation=(1, 1), padding=(2, 2), groups=self.c, bias=False)
        # Convolution_3x3_2x2_DW1: p_4
        self.p_4 = nn.Conv2d(self.c, self.c, (3, 3), dilation=(2, 2), padding=(2, 2), groups=self.c, bias=False)
        # Scale_0/1/C_1/1/C: p_5
        self.p_5_w = nn.Parameter(torch.ones((1, self.c, self.c,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_5_w, std=.1)
        # Mix: p_6
        self.p_6_w = nn.Parameter(torch.ones((self.c, 1, )), requires_grad=True)
        nn.init.trunc_normal_(self.p_6_w, std=.1)
        # UnfoldW_K7_D2: p_7
        pass
        # Mix: p_8
        self.p_8_w = nn.Parameter(torch.ones((self.c, self.c * 7, )), requires_grad=True)
        nn.init.trunc_normal_(self.p_8_w, std=.1)
        # BMM_0_1: p_9
        pass
        # BMM_1_0: p_10
        pass
        # BAdd: p_11
        pass
        # Output: p_12
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Shift_1/0/H_1/1/W_K2: p_1
        t_1 = torch.roll(t_0, self.p_1_1_0, 2)
        t_1 = torch.roll(t_1, self.p_1_1_1, 3)
        # BMM_0_1: p_2
        t_0_lhs = t_0.view(self.n, self.c, self.h * self.w)        
        t_1_rhs = t_1.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_2 = torch.bmm(t_0_lhs, t_1_rhs).view(self.n, self.c, self.c) / math.sqrt(self.h * self.w)
        # Convolution_5x1_3x1_DW1: p_3
        t_3 = self.p_3(t_1)
        t_3 = t_3.view(self.n, self.c, self.h, self.w)
        # Convolution_3x3_2x2_DW1: p_4
        t_4 = self.p_4(t_1)
        t_4 = t_4.view(self.n, self.c, self.h, self.w)
        # Scale_0/1/C_1/1/C: p_5
        t_5 = self.p_5_w * t_2
        # Mix: p_6
        t_6 = torch.einsum('abcd,be->aecd', [t_4, self.p_6_w]).view(self.n, self.h, self.w).contiguous()
        # UnfoldW_K7_D2: p_7
        t_7 = F.unfold(t_0, (1, 7), dilation=(1, 2), padding=(0, 6))
        t_7 = t_7.view(self.n, self.c, 7, self.h, self.w)
        # Mix: p_8
        t_8 = torch.einsum('abc,bd->adc', [t_5, self.p_8_w]).view(self.n, self.c * 7, self.c).contiguous()
        # BMM_0_1: p_9
        t_3_lhs = t_3.view(self.n, self.c, self.h * self.w)        
        t_6_rhs = t_6.view(self.n, 1, self.h * self.w).transpose(1, 2)        
        t_9 = torch.bmm(t_3_lhs, t_6_rhs).view(self.n, self.c) / math.sqrt(self.h * self.w)
        # BMM_1_0: p_10
        t_8_lhs = t_8.view(self.n, self.c * 7, self.c).transpose(1, 2)        
        t_7_rhs = t_7.view(self.n, self.c * 7, self.h * self.w)        
        t_10 = torch.bmm(t_8_lhs, t_7_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c * 7)
        # BAdd: p_11
        t_11_lhs = t_9.view(self.n, self.c, 1)
        t_11_rhs = t_10.view(self.n, self.c, self.h * self.w)
        t_11 = t_11_lhs + t_11_rhs
        t_11 = t_11.view(self.n, self.c, self.h, self.w)
        # Output: p_12
        return t_11.view(self.n, self.c, self.h, self.w)