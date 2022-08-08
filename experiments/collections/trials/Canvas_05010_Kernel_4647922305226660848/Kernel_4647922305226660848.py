import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_4647922305226660848(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_4647922305226660848, self).__init__()
        self.g = 4
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldH_K3_D2: p_1
        pass
        # Scale_0/2/KH_1/1/W: p_2
        self.p_2_w = nn.Parameter(torch.ones((1, 1, 3, 1, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_2_w, std=.02)
        # FC: p_3
        self.p_3 = nn.Conv2d(self.c * 3, self.c, 1, padding=0, groups=1, bias=False)
        # UnfoldW_K3_D1: p_4
        pass
        # Scale_1/0/H_1/1/W: p_5
        self.p_5_w = nn.Parameter(torch.ones((1, 1, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_5_w, std=.02)
        # Group_0_C/G: p_6
        pass
        # UnfoldW_K3_D2: p_7
        pass
        # Softmax_0/3/KW: p_8
        pass
        # BMM_0_1: p_9
        pass
        # Scale_0/1/C_0/3/KW_1/0/H_1/1/W: p_10
        self.p_10_w = nn.Parameter(torch.ones((1, self.c, 3, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_10_w, std=.02)
        # BSub: p_11
        pass
        # BSub: p_12
        pass
        # BMM_0_0: p_13
        pass
        # Output: p_14
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldH_K3_D2: p_1
        t_1 = F.unfold(t_0, (3, 1), dilation=(2, 1), padding=(2, 0))
        t_1 = t_1.view(self.n, self.c, 3, self.h, self.w)
        # Scale_0/2/KH_1/1/W: p_2
        t_2 = self.p_2_w * t_1
        # FC: p_3
        t_3 = t_1.view(self.n, self.c * 3, self.h, self.w)
        t_3 = self.p_3(t_3)
        t_3 = t_3.view(self.n, self.c, self.h, self.w)
        # UnfoldW_K3_D1: p_4
        t_4 = F.unfold(t_3, (1, 3), dilation=(1, 1), padding=(0, 1))
        t_4 = t_4.view(self.n, self.c, 3, self.h, self.w)
        # Scale_1/0/H_1/1/W: p_5
        t_5 = self.p_5_w * t_3
        # Group_0_C/G: p_6
        t_6 = t_5.view(self.n, self.c // self.g, self.g, self.h, self.w)
        # UnfoldW_K3_D2: p_7
        t_7 = F.unfold(t_0, (1, 3), dilation=(1, 2), padding=(0, 2))
        t_7 = t_7.view(self.n, self.c, 3, self.h, self.w)
        # Softmax_0/3/KW: p_8
        t_8 = F.softmax(t_7, dim=2)
        # BMM_0_1: p_9
        t_0_lhs = t_0.view(self.n, self.c, self.h * self.w)        
        t_4_rhs = t_4.view(self.n, self.c * 3, self.h * self.w).transpose(1, 2)        
        t_9 = torch.bmm(t_0_lhs, t_4_rhs).view(self.n, self.c, self.c, 3) / math.sqrt(self.h * self.w)
        # Scale_0/1/C_0/3/KW_1/0/H_1/1/W: p_10
        t_10 = self.p_10_w * t_8
        # BSub: p_11
        t_11_lhs = t_6.view(self.n, 1, self.c, self.h, self.w)
        t_11_rhs = t_2.view(self.n, 3, self.c, self.h, self.w)
        t_11 = t_11_lhs - t_11_rhs
        t_11 = t_11.view(self.n, self.c, 3, self.h, self.w)
        # BSub: p_12
        t_12 = t_11 - t_10
        # BMM_0_0: p_13
        t_9_lhs = t_9.view(self.n, self.c, self.c * 3)        
        t_12_rhs = t_12.view(self.n, self.c * 3, self.h * self.w)        
        t_13 = torch.bmm(t_9_lhs, t_12_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c * 3)
        # Output: p_14
        return t_13.view(self.n, self.c, self.h, self.w)