import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_4005987118033986640(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_4005987118033986640, self).__init__()
        self.g = 8
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldW_K3_D2: p_1
        pass
        # UnfoldH_K7_D3: p_2
        pass
        # BMM_0_1: p_3
        pass
        # Group_0_x_0: p_4
        pass
        # Scale_0/1/C_0/2/KH_0/3/KW_1/0/H_1/1/W: p_5
        self.p_5_w = nn.Parameter(torch.ones((1, self.c, 7, 3, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_5_w, std=.1)
        # Scale_0/1/C_0/2/KH_0/3/KW_1/1/C: p_6
        self.p_6_w = nn.Parameter(torch.ones((1, self.c, 7, 3, self.c,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_6_w, std=.1)
        # BAdd: p_7
        pass
        # BMM_1_0: p_8
        pass
        # Output: p_9
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldW_K3_D2: p_1
        t_1 = F.unfold(t_0, (1, 3), dilation=(1, 2), padding=(0, 2))
        t_1 = t_1.view(self.n, self.c, 3, self.h, self.w)
        # UnfoldH_K7_D3: p_2
        t_2 = t_1.view(self.n, self.c * 3, self.h, self.w)
        t_2 = F.unfold(t_2, (7, 1), dilation=(3, 1), padding=(9, 0))
        t_2 = t_2.view(self.n, self.c, 7, 3, self.h, self.w)
        # BMM_0_1: p_3
        t_2_lhs = t_2.view(self.n, self.c * 21, self.h * self.w)        
        t_0_rhs = t_0.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_3 = torch.bmm(t_2_lhs, t_0_rhs).view(self.n, self.c, 7, 3, self.c) / math.sqrt(self.h * self.w)
        # Group_0_x_0: p_4
        t_4 = t_2
        # Scale_0/1/C_0/2/KH_0/3/KW_1/0/H_1/1/W: p_5
        t_5 = self.p_5_w * t_2
        # Scale_0/1/C_0/2/KH_0/3/KW_1/1/C: p_6
        t_6 = self.p_6_w * t_3
        # BAdd: p_7
        t_7_lhs = t_5.view(self.n, 1, self.c, 7, 3, self.h, self.w)
        t_7_rhs = t_4.view(self.n, 1, self.c, 7, 3, self.h, self.w)
        t_7 = t_7_lhs + t_7_rhs
        t_7 = t_7.view(self.n, self.c, 7, 3, self.h, self.w)
        # BMM_1_0: p_8
        t_7_lhs = t_7.view(self.n, self.c * 21, self.h * self.w).transpose(1, 2)        
        t_6_rhs = t_6.view(self.n, self.c * 21, self.c)        
        t_8 = torch.bmm(t_7_lhs, t_6_rhs).view(self.n, self.h, self.w, self.c) / math.sqrt(self.c * 21)
        # Output: p_9
        return t_8.permute(0, 3, 1, 2).contiguous().view(self.n, self.c, self.h, self.w)