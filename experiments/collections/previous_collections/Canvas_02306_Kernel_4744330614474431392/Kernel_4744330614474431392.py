import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_4744330614474431392(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_4744330614474431392, self).__init__()
        self.g = 4
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Group_0_C/G: p_1
        pass
        # UnfoldH_K3_D2: p_2
        pass
        # BMM_0_1: p_3
        pass
        # Scale_0/0/G_0/1/C_0/2/KH_1/1/C: p_4
        self.p_4_w = nn.Parameter(torch.ones((1, self.c // self.g, self.g, 3, self.c,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_4_w, std=.02)
        # Convolution_5x1_1x1_DW0: p_5
        self.p_5 = nn.Conv2d(self.c, self.c * 3, (5, 1), dilation=(1, 1), padding=(2, 0), groups=self.c // self.g, bias=False)
        # BMM_1_0: p_6
        pass
        # Output: p_7
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Group_0_C/G: p_1
        t_1 = t_0.view(self.n, self.c // self.g, self.g, self.h, self.w)
        # UnfoldH_K3_D2: p_2
        t_2 = t_1.view(self.n, self.c, self.h, self.w)
        t_2 = F.unfold(t_2, (3, 1), dilation=(2, 1), padding=(2, 0))
        t_2 = t_2.view(self.n, self.c // self.g, self.g, 3, self.h, self.w)
        # BMM_0_1: p_3
        t_3_lhs = t_2.view(self.n, self.c * 3, self.h * self.w)        
        t_3_rhs = t_0.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_3 = torch.bmm(t_3_lhs, t_3_rhs) / math.sqrt(self.h * self.w)
        t_3 = t_3.view(self.n, self.c // self.g, self.g, 3, self.c)
        # Scale_0/0/G_0/1/C_0/2/KH_1/1/C: p_4
        t_4 = self.p_4_w * t_3
        # Convolution_5x1_1x1_DW0: p_5
        t_5 = t_1.view(self.n, self.c, self.h, self.w)
        t_5 = self.p_5(t_5)
        t_5 = t_5.view(self.n, self.c * 3, self.h, self.w)
        # BMM_1_0: p_6
        t_6_lhs = t_4.view(self.n, self.c * 3, self.c).transpose(1, 2)        
        t_6_rhs = t_5.view(self.n, self.c * 3, self.h * self.w)        
        t_6 = torch.bmm(t_6_lhs, t_6_rhs) / math.sqrt(self.c * 3)
        t_6 = t_6.view(self.n, self.c, self.h, self.w)
        # Output: p_7
        return t_6.view(self.n, self.c, self.h, self.w)