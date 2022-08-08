import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_4272043718162930052(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_4272043718162930052, self).__init__()
        self.g = 8
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Scale_0/1/C_1/0/H_1/1/W: p_1
        self.p_1_w = nn.Parameter(torch.ones((1, self.c, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_1_w, std=.1)
        # Shift_0/1/C_K1: p_2
        self.p_2_0_1 = random.randint(-1, 1)
        # UnfoldW_K7_D1: p_3
        pass
        # Shift_1/0/H_1/1/W_K1: p_4
        self.p_4_1_0 = random.randint(-1, 1)
        self.p_4_1_1 = random.randint(-1, 1)
        # BSub: p_5
        pass
        # BAdd: p_6
        pass
        # BMM_0_1: p_7
        pass
        # BMM_0_0: p_8
        pass
        # Output: p_9
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Scale_0/1/C_1/0/H_1/1/W: p_1
        t_1 = self.p_1_w * t_0
        # Shift_0/1/C_K1: p_2
        t_2 = torch.roll(t_1, self.p_2_0_1, 1)
        # UnfoldW_K7_D1: p_3
        t_3 = F.unfold(t_2, (1, 7), dilation=(1, 1), padding=(0, 3))
        t_3 = t_3.view(self.n, self.c, 7, self.h, self.w)
        # Shift_1/0/H_1/1/W_K1: p_4
        t_4 = torch.roll(t_3, self.p_4_1_0, 3)
        t_4 = torch.roll(t_4, self.p_4_1_1, 4)
        # BSub: p_5
        t_5_lhs = t_1.view(self.n, self.c, 1, self.h, self.w)
        t_5_rhs = t_3.view(self.n, self.c, 7, self.h, self.w)
        t_5 = t_5_lhs - t_5_rhs
        t_5 = t_5.view(self.n, self.c, 7, self.h, self.w)
        # BAdd: p_6
        t_6_lhs = t_0.view(self.n, self.c, 1, self.h, self.w)
        t_6_rhs = t_4.view(self.n, self.c, 7, self.h, self.w)
        t_6 = t_6_lhs + t_6_rhs
        t_6 = t_6.view(self.n, self.c, 7, self.h, self.w)
        # BMM_0_1: p_7
        t_0_lhs = t_0.view(self.n, self.c, self.h * self.w)        
        t_5_rhs = t_5.view(self.n, self.c * 7, self.h * self.w).transpose(1, 2)        
        t_7 = torch.bmm(t_0_lhs, t_5_rhs).view(self.n, self.c, self.c, 7) / math.sqrt(self.h * self.w)
        # BMM_0_0: p_8
        t_7_lhs = t_7.view(self.n, self.c, self.c * 7)        
        t_6_rhs = t_6.view(self.n, self.c * 7, self.h * self.w)        
        t_8 = torch.bmm(t_7_lhs, t_6_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c * 7)
        # Output: p_9
        return t_8.view(self.n, self.c, self.h, self.w)