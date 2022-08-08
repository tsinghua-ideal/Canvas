import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_16550548659508808100(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_16550548659508808100, self).__init__()
        self.g = 2
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldW_K3_D3: p_1
        pass
        # Shift_1/0/H_K3: p_2
        self.p_2_1_0 = random.randint(-3, 3)
        # Shift_1/0/H_K1: p_3
        self.p_3_1_0 = random.randint(-1, 1)
        # Fold_0/1/C_Avg: p_4
        pass
        # Shift_1/1/W_K3: p_5
        self.p_5_1_1 = random.randint(-3, 3)
        # Shift_1/0/H_K2: p_6
        self.p_6_1_0 = random.randint(-2, 2)
        # Mix: p_7
        self.p_7_w = nn.Parameter(torch.ones((3, self.w, 1, 1, )), requires_grad=True)
        bound = math.sqrt(3.0 / (self.w * 3))
        nn.init.uniform_(self.p_7_w, a=-bound, b=bound)
        # Scale_0/1/C_1/0/H: p_8
        self.p_8_w = nn.Parameter(torch.ones((1, self.c, self.h, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_8_w, std=.02)
        # BAdd: p_9
        pass
        # BSub: p_10
        pass
        # Convolution_1x3_1x2_DW0: p_11
        self.p_11 = nn.Conv2d(self.c, self.c, (1, 3), dilation=(1, 2), padding=(0, 2), groups=1, bias=False)
        # UnfoldH_K7_D2: p_12
        pass
        # BMax: p_13
        pass
        # BMM_1_0: p_14
        pass
        # Abs: p_15
        pass
        # BSub: p_16
        pass
        # Fold_1/0/H_Avg: p_17
        pass
        # BSub: p_18
        pass
        # Scale_0/1/C: p_19
        self.p_19_w = nn.Parameter(torch.ones((1, self.c, 1, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_19_w, std=.02)
        # BAdd: p_20
        pass
        # BMM_0_0: p_21
        pass
        # Output: p_22
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldW_K3_D3: p_1
        t_1 = F.unfold(t_0, (1, 3), dilation=(1, 3), padding=(0, 3))
        t_1 = t_1.view(self.n, self.c, 3, self.h, self.w)
        # Shift_1/0/H_K3: p_2
        t_2 = torch.roll(t_0, self.p_2_1_0, 2)
        # Shift_1/0/H_K1: p_3
        t_3 = torch.roll(t_1, self.p_3_1_0, 3)
        # Fold_0/1/C_Avg: p_4
        t_4 = t_3.mean(1)
        # Shift_1/1/W_K3: p_5
        t_5 = torch.roll(t_2, self.p_5_1_1, 3)
        # Shift_1/0/H_K2: p_6
        t_6 = torch.roll(t_4, self.p_6_1_0, 2)
        # Mix: p_7
        t_7 = torch.einsum('abcd,bdef->aecf', [t_6, self.p_7_w]).view(self.n, self.h).contiguous()
        # Scale_0/1/C_1/0/H: p_8
        t_8 = self.p_8_w * t_5
        # BAdd: p_9
        t_9 = t_0 + t_8
        # BSub: p_10
        t_10 = t_8 - t_9
        # Convolution_1x3_1x2_DW0: p_11
        t_11 = self.p_11(t_10)
        t_11 = t_11.view(self.n, self.c, self.h, self.w)
        # UnfoldH_K7_D2: p_12
        t_12 = t_7.view(self.n, 1, self.h, 1)
        t_12 = F.unfold(t_12, (7, 1), dilation=(2, 1), padding=(6, 0))
        t_12 = t_12.view(self.n, 7, self.h)
        # BMax: p_13
        t_13 = torch.maximum(t_11, t_0)
        # BMM_1_0: p_14
        t_0_lhs = t_0.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_13_rhs = t_13.view(self.n, self.c, self.h * self.w)        
        t_14 = torch.bmm(t_0_lhs, t_13_rhs).view(self.n, self.h, self.w, self.h, self.w) / math.sqrt(self.c)
        # Abs: p_15
        t_15 = torch.abs(t_12)
        # BSub: p_16
        t_16_lhs = t_7.view(self.n, self.h, 1)
        t_16_rhs = t_14.view(self.n, self.h, self.h * self.w * self.w)
        t_16 = t_16_lhs - t_16_rhs
        t_16 = t_16.view(self.n, self.h, self.w, self.h, self.w)
        # Fold_1/0/H_Avg: p_17
        t_17 = t_15.mean(2)
        # BSub: p_18
        t_18_lhs = t_17.view(self.n, 1, 7)
        t_18_rhs = t_10.view(self.n, self.c * self.h * self.w // 7, 7)
        t_18 = t_18_lhs - t_18_rhs
        t_18 = t_18.view(self.n, self.c, self.h, self.w)
        # Scale_0/1/C: p_19
        t_19 = self.p_19_w * t_18
        # BAdd: p_20
        t_20_lhs = t_7.view(self.n, self.h, 1)
        t_20_rhs = t_16.view(self.n, self.h, self.h * self.w * self.w)
        t_20 = t_20_lhs + t_20_rhs
        t_20 = t_20.view(self.n, self.h, self.w, self.h, self.w)
        # BMM_0_0: p_21
        t_19_lhs = t_19.view(self.n, self.c, self.h * self.w)        
        t_20_rhs = t_20.view(self.n, self.h * self.w, self.h * self.w)        
        t_21 = torch.bmm(t_19_lhs, t_20_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.h * self.w)
        # Output: p_22
        return t_21.view(self.n, self.c, self.h, self.w)