import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_12368652706678037939(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_12368652706678037939, self).__init__()
        self.g = 1
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldH_K3_D2: p_1
        pass
        # Scale_0/1/C_0/2/KH: p_2
        self.p_2_w = nn.Parameter(torch.ones((1, self.c, 3, 1, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_2_w, std=.02)
        # Shift_1/1/W_K3: p_3
        self.p_3_1_1 = random.randint(-3, 3)
        # Convolution_1x3_1x1_DW0: p_4
        self.p_4 = nn.Conv2d(self.c, self.c, (1, 3), dilation=(1, 1), padding=(0, 1), groups=1, bias=False)
        # BMM_0_1: p_5
        pass
        # Shift_1/1/C_K1: p_6
        self.p_6_1_1 = random.randint(-1, 1)
        # TanH: p_7
        pass
        # BMax: p_8
        pass
        # Scale_0/1/C_1/0/H: p_9
        self.p_9_w = nn.Parameter(torch.ones((1, self.c, 1, self.h, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_9_w, std=.02)
        # Scale_0/1/C: p_10
        self.p_10_w = nn.Parameter(torch.ones((1, self.c, 1, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_10_w, std=.02)
        # BMM_0_1: p_11
        pass
        # BMul: p_12
        pass
        # BAdd: p_13
        pass
        # BMM_1_0: p_14
        pass
        # Output: p_15
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldH_K3_D2: p_1
        t_1 = F.unfold(t_0, (3, 1), dilation=(2, 1), padding=(2, 0))
        t_1 = t_1.view(self.n, self.c, 3, self.h, self.w)
        # Scale_0/1/C_0/2/KH: p_2
        t_2 = self.p_2_w * t_1
        # Shift_1/1/W_K3: p_3
        t_3 = torch.roll(t_0, self.p_3_1_1, 3)
        # Convolution_1x3_1x1_DW0: p_4
        t_4 = self.p_4(t_0)
        t_4 = t_4.view(self.n, self.c, self.h, self.w)
        # BMM_0_1: p_5
        t_1_lhs = t_1.view(self.n, self.c * 3, self.h * self.w)        
        t_0_rhs = t_0.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_5 = torch.bmm(t_1_lhs, t_0_rhs).view(self.n, self.c, 3, self.c) / math.sqrt(self.h * self.w)
        # Shift_1/1/C_K1: p_6
        t_6 = torch.roll(t_5, self.p_6_1_1, 3)
        # TanH: p_7
        t_7 = torch.tanh(t_3)
        # BMax: p_8
        t_8 = torch.maximum(t_4, t_0)
        # Scale_0/1/C_1/0/H: p_9
        t_9 = self.p_9_w * t_2
        # Scale_0/1/C: p_10
        t_10 = self.p_10_w * t_6
        # BMM_0_1: p_11
        t_7_lhs = t_7.view(self.n, self.c, self.h * self.w)        
        t_3_rhs = t_3.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_11 = torch.bmm(t_7_lhs, t_3_rhs).view(self.n, self.c, self.c) / math.sqrt(self.h * self.w)
        # BMul: p_12
        t_12_lhs = t_8.view(self.n, self.c, 1, self.h, self.w)
        t_12_rhs = t_9.view(self.n, self.c, 3, self.h, self.w)
        t_12 = t_12_lhs * t_12_rhs
        t_12 = t_12.view(self.n, self.c, 3, self.h, self.w)
        # BAdd: p_13
        t_13_lhs = t_11.view(self.n, self.c, 1, self.c)
        t_13_rhs = t_10.view(self.n, self.c, 3, self.c)
        t_13 = t_13_lhs + t_13_rhs
        t_13 = t_13.view(self.n, self.c, 3, self.c)
        # BMM_1_0: p_14
        t_13_lhs = t_13.view(self.n, self.c * 3, self.c).transpose(1, 2)        
        t_12_rhs = t_12.view(self.n, self.c * 3, self.h * self.w)        
        t_14 = torch.bmm(t_13_lhs, t_12_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c * 3)
        # Output: p_15
        return t_14.view(self.n, self.c, self.h, self.w)