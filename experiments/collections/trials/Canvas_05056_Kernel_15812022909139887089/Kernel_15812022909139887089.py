import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_15812022909139887089(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_15812022909139887089, self).__init__()
        self.g = 16
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldHW_K3_D3: p_1
        pass
        # Scale_0/1/C_1/0/H_1/1/W: p_2
        self.p_2_w = nn.Parameter(torch.ones((1, self.c, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_2_w, std=.02)
        # BMul: p_3
        pass
        # Shift_1/0/H_1/1/W_K1: p_4
        self.p_4_1_0 = random.randint(-1, 1)
        self.p_4_1_1 = random.randint(-1, 1)
        # Softmax_1/0/H: p_5
        pass
        # FC: p_6
        self.p_6 = nn.Conv2d(self.c, self.c // 8, 1, padding=0, groups=1, bias=False)
        # FC: p_7
        self.p_7 = nn.Conv2d(self.c * 9, self.c, 1, padding=0, groups=1, bias=False)
        # BMM_0_1: p_8
        pass
        # Scale_0/1/C_1/0/H_1/1/W: p_9
        self.p_9_w = nn.Parameter(torch.ones((1, self.c // 8, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_9_w, std=.02)
        # BMax: p_10
        pass
        # Mix: p_11
        self.p_11_w = nn.Parameter(torch.ones((self.c // 8, 1, )), requires_grad=True)
        bound = math.sqrt(3.0 / (self.c // 8))
        nn.init.uniform_(self.p_11_w, a=-bound, b=bound)
        # Convolution_3x1_2x1_DW1: p_12
        self.p_12 = nn.Conv2d(self.c, self.c, (3, 1), dilation=(2, 1), padding=(2, 0), groups=self.c, bias=False)
        # BSub: p_13
        pass
        # Shift_1/0/H_K1: p_14
        self.p_14_1_0 = random.randint(-1, 1)
        # FC: p_15
        self.p_15 = nn.Conv2d(self.c, self.c, 1, padding=0, groups=1, bias=False)
        # Mix: p_16
        self.p_16_w = nn.Parameter(torch.ones((self.c, self.h * self.w, )), requires_grad=True)
        bound = math.sqrt(3.0 / (self.c))
        nn.init.uniform_(self.p_16_w, a=-bound, b=bound)
        # BSub: p_17
        pass
        # BMul: p_18
        pass
        # BMM_0_0: p_19
        pass
        # BMM_1_0: p_20
        pass
        # BAdd: p_21
        pass
        # Output: p_22
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldHW_K3_D3: p_1
        t_1 = F.unfold(t_0, (3, 3), dilation=(3, 3), padding=(3, 3))
        t_1 = t_1.view(self.n, self.c, 3, 3, self.h, self.w)
        # Scale_0/1/C_1/0/H_1/1/W: p_2
        t_2 = self.p_2_w * t_0
        # BMul: p_3
        t_3 = t_2 * t_2
        # Shift_1/0/H_1/1/W_K1: p_4
        t_4 = torch.roll(t_3, self.p_4_1_0, 2)
        t_4 = torch.roll(t_4, self.p_4_1_1, 3)
        # Softmax_1/0/H: p_5
        t_5 = F.softmax(t_4, dim=2)
        # FC: p_6
        t_6 = self.p_6(t_3)
        t_6 = t_6.view(self.n, self.c // 8, self.h, self.w)
        # FC: p_7
        t_7 = t_1.view(self.n, self.c * 9, self.h, self.w)
        t_7 = self.p_7(t_7)
        t_7 = t_7.view(self.n, self.c, self.h, self.w)
        # BMM_0_1: p_8
        t_5_lhs = t_5.view(self.n, self.c, self.h * self.w)        
        t_3_rhs = t_3.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_8 = torch.bmm(t_5_lhs, t_3_rhs).view(self.n, self.c, self.c) / math.sqrt(self.h * self.w)
        # Scale_0/1/C_1/0/H_1/1/W: p_9
        t_9 = self.p_9_w * t_6
        # BMax: p_10
        t_10 = torch.maximum(t_5, t_7)
        # Mix: p_11
        t_11 = torch.einsum('abcd,be->aecd', [t_9, self.p_11_w]).view(self.n, self.h, self.w).contiguous()
        # Convolution_3x1_2x1_DW1: p_12
        t_12 = self.p_12(t_10)
        t_12 = t_12.view(self.n, self.c, self.h, self.w)
        # BSub: p_13
        t_13_lhs = t_11.view(self.n, 1, self.h, self.w)
        t_13_rhs = t_12.view(self.n, self.c, self.h, self.w)
        t_13 = t_13_lhs - t_13_rhs
        t_13 = t_13.view(self.n, self.c, self.h, self.w)
        # Shift_1/0/H_K1: p_14
        t_14 = torch.roll(t_0, self.p_14_1_0, 2)
        # FC: p_15
        t_15 = self.p_15(t_14)
        t_15 = t_15.view(self.n, self.c, self.h, self.w)
        # Mix: p_16
        t_16 = torch.einsum('abc,bd->adc', [t_8, self.p_16_w]).view(self.n, self.h * self.w, self.c).contiguous()
        # BSub: p_17
        t_17 = t_3 - t_15
        # BMul: p_18
        t_18 = t_7 * t_17
        # BMM_0_0: p_19
        t_17_lhs = t_17.view(self.n, self.c, self.h * self.w)        
        t_16_rhs = t_16.view(self.n, self.h * self.w, self.c)        
        t_19 = torch.bmm(t_17_lhs, t_16_rhs).view(self.n, self.c, self.c) / math.sqrt(self.h * self.w)
        # BMM_1_0: p_20
        t_19_lhs = t_19.view(self.n, self.c, self.c).transpose(1, 2)        
        t_13_rhs = t_13.view(self.n, self.c, self.h * self.w)        
        t_20 = torch.bmm(t_19_lhs, t_13_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c)
        # BAdd: p_21
        t_21 = t_20 + t_18
        # Output: p_22
        return t_21.view(self.n, self.c, self.h, self.w)