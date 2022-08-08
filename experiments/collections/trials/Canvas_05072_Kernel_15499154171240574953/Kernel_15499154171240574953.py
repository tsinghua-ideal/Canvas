import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_15499154171240574953(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_15499154171240574953, self).__init__()
        self.g = 2
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldHW_K3_D1: p_1
        pass
        # BAdd: p_2
        pass
        # BSub: p_3
        pass
        # Scale_0/1/C_0/2/KH: p_4
        self.p_4_w = nn.Parameter(torch.ones((1, self.c, 3, 1, 1, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_4_w, std=.02)
        # BMM_0_1: p_5
        pass
        # BMM_1_0: p_6
        pass
        # Output: p_7
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldHW_K3_D1: p_1
        t_1 = F.unfold(t_0, (3, 3), dilation=(1, 1), padding=(1, 1))
        t_1 = t_1.view(self.n, self.c, 3, 3, self.h, self.w)
        # BAdd: p_2
        t_2_lhs = t_0.view(self.n, self.c, 1, self.h, self.w)
        t_2_rhs = t_1.view(self.n, self.c, 9, self.h, self.w)
        t_2 = t_2_lhs + t_2_rhs
        t_2 = t_2.view(self.n, self.c, 3, 3, self.h, self.w)
        # BSub: p_3
        t_3 = t_1 - t_2
        # Scale_0/1/C_0/2/KH: p_4
        t_4 = self.p_4_w * t_1
        # BMM_0_1: p_5
        t_5_lhs = t_3.view(self.n, self.c * 9, self.h * self.w)        
        t_5_rhs = t_0.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_5 = torch.bmm(t_5_lhs, t_5_rhs) / math.sqrt(self.h * self.w)
        t_5 = t_5.view(self.n, self.c, 3, 3, self.c)
        # BMM_1_0: p_6
        t_6_lhs = t_4.view(self.n, self.c * 9, self.h * self.w).transpose(1, 2)        
        t_6_rhs = t_5.view(self.n, self.c * 9, self.c)        
        t_6 = torch.bmm(t_6_lhs, t_6_rhs) / math.sqrt(self.c * 9)
        t_6 = t_6.view(self.n, self.h, self.w, self.c)
        # Output: p_7
        return t_6.permute(0, 3, 1, 2).contiguous().view(self.n, self.c, self.h, self.w)