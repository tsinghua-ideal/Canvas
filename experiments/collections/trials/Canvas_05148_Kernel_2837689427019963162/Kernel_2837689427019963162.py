import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_2837689427019963162(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_2837689427019963162, self).__init__()
        self.g = 32
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldH_K3_D2: p_1
        pass
        # Scale_0/1/C_0/2/KH_1/0/H_1/1/W: p_2
        self.p_2_w = nn.Parameter(torch.ones((1, self.c, 3, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_2_w, std=.02)
        # UnfoldW_K3_D2: p_3
        pass
        # Shift_1/0/H_K1: p_4
        self.p_4_1_0 = random.randint(-1, 1)
        # Scale_0/1/C: p_5
        self.p_5_w = nn.Parameter(torch.ones((1, self.c, 1, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_5_w, std=.02)
        # BAdd: p_6
        pass
        # BMM_0_1: p_7
        pass
        # FC: p_8
        self.p_8 = nn.Conv2d(self.c, self.c // self.g, 1, padding=0, groups=1, bias=False)
        # BMul: p_9
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
        # UnfoldH_K3_D2: p_1
        t_1 = F.unfold(t_0, (3, 1), dilation=(2, 1), padding=(2, 0))
        t_1 = t_1.view(self.n, self.c, 3, self.h, self.w)
        # Scale_0/1/C_0/2/KH_1/0/H_1/1/W: p_2
        t_2 = self.p_2_w * t_1
        # UnfoldW_K3_D2: p_3
        t_3 = F.unfold(t_0, (1, 3), dilation=(1, 2), padding=(0, 2))
        t_3 = t_3.view(self.n, self.c, 3, self.h, self.w)
        # Shift_1/0/H_K1: p_4
        t_4 = torch.roll(t_0, self.p_4_1_0, 2)
        # Scale_0/1/C: p_5
        t_5 = self.p_5_w * t_4
        # BAdd: p_6
        t_6 = t_4 + t_0
        # BMM_0_1: p_7
        t_2_lhs = t_2.view(self.n, self.c * 3, self.h * self.w)        
        t_6_rhs = t_6.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_7 = torch.bmm(t_2_lhs, t_6_rhs).view(self.n, self.c, 3, self.c) / math.sqrt(self.h * self.w)
        # FC: p_8
        t_8 = self.p_8(t_5)
        t_8 = t_8.view(self.n, self.c // self.g, self.h, self.w)
        # BMul: p_9
        t_9_lhs = t_8.view(self.n, 1, self.c // self.g, self.h, self.w)
        t_9_rhs = t_0.view(self.n, self.g, self.c // self.g, self.h, self.w)
        t_9 = t_9_lhs * t_9_rhs
        t_9 = t_9.view(self.n, self.c, self.h, self.w)
        # BMM_1_0: p_10
        t_7_lhs = t_7.view(self.n, self.c * 3, self.c).transpose(1, 2)        
        t_3_rhs = t_3.view(self.n, self.c * 3, self.h * self.w)        
        t_10 = torch.bmm(t_7_lhs, t_3_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c * 3)
        # BAdd: p_11
        t_11 = t_9 + t_10
        # Output: p_12
        return t_11.view(self.n, self.c, self.h, self.w)