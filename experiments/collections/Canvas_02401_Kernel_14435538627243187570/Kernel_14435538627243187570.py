import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_14435538627243187570(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_14435538627243187570, self).__init__()
        self.g = 1
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # BMM_0_1: p_1
        pass
        # UnfoldW_K5_D3: p_2
        pass
        # Shift_0/3/KW_K2: p_3
        self.p_3_0_3 = random.randint(-2, 2)
        # Scale_0/1/C_1/1/C: p_4
        self.p_4_w = nn.Parameter(torch.ones((1, self.c, self.c,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_4_w, std=.02)
        # FC: p_5
        self.p_5 = nn.Conv2d(self.c * 5, self.c, 1, padding=0, groups=1, bias=False)
        # BMM_1_0: p_6
        pass
        # Output: p_7
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # BMM_0_1: p_1
        t_1_lhs = t_0.view(self.n, self.c, self.h * self.w)        
        t_1_rhs = t_0.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_1 = torch.bmm(t_1_lhs, t_1_rhs) / math.sqrt(self.h * self.w)
        t_1 = t_1.view(self.n, self.c, self.c)
        # UnfoldW_K5_D3: p_2
        t_2 = F.unfold(t_0, (1, 5), dilation=(1, 3), padding=(0, 6))
        t_2 = t_2.view(self.n, self.c, 5, self.h, self.w)
        # Shift_0/3/KW_K2: p_3
        t_3 = torch.roll(t_2, self.p_3_0_3, 2)
        # Scale_0/1/C_1/1/C: p_4
        t_4 = self.p_4_w * t_1
        # FC: p_5
        t_5 = t_3.view(self.n, self.c * 5, self.h, self.w)
        t_5 = self.p_5(t_5)
        t_5 = t_5.view(self.n, self.c, self.h, self.w)
        # BMM_1_0: p_6
        t_6_lhs = t_4.view(self.n, self.c, self.c).transpose(1, 2)        
        t_6_rhs = t_5.view(self.n, self.c, self.h * self.w)        
        t_6 = torch.bmm(t_6_lhs, t_6_rhs) / math.sqrt(self.c)
        t_6 = t_6.view(self.n, self.c, self.h, self.w)
        # Output: p_7
        return t_6.view(self.n, self.c, self.h, self.w)