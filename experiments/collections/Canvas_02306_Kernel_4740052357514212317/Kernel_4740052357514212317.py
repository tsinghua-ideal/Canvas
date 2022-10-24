import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_4740052357514212317(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_4740052357514212317, self).__init__()
        self.g = 4
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldW_K5_D2: p_1
        pass
        # BMM_0_1: p_2
        pass
        # ReLU: p_3
        pass
        # UnfoldH_K5_D1: p_4
        pass
        # GeLU: p_5
        pass
        # Scale_0/1/C_1/1/C_1/3/KW: p_6
        self.p_6_w = nn.Parameter(torch.ones((1, self.c, self.c, 5,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_6_w, std=.02)
        # BMM_0_0: p_7
        pass
        # Output: p_8
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldW_K5_D2: p_1
        t_1 = F.unfold(t_0, (1, 5), dilation=(1, 2), padding=(0, 4))
        t_1 = t_1.view(self.n, self.c, 5, self.h, self.w)
        # BMM_0_1: p_2
        t_2_lhs = t_0.view(self.n, self.c, self.h * self.w)        
        t_2_rhs = t_1.view(self.n, self.c * 5, self.h * self.w).transpose(1, 2)        
        t_2 = torch.bmm(t_2_lhs, t_2_rhs) / math.sqrt(self.h * self.w)
        t_2 = t_2.view(self.n, self.c, self.c, 5)
        # ReLU: p_3
        t_3 = torch.relu(t_0)
        # UnfoldH_K5_D1: p_4
        t_4 = F.unfold(t_3, (5, 1), dilation=(1, 1), padding=(2, 0))
        t_4 = t_4.view(self.n, self.c, 5, self.h, self.w)
        # GeLU: p_5
        t_5 = F.gelu(t_4)
        # Scale_0/1/C_1/1/C_1/3/KW: p_6
        t_6 = self.p_6_w * t_2
        # BMM_0_0: p_7
        t_7_lhs = t_6.view(self.n, self.c, self.c * 5)        
        t_7_rhs = t_5.view(self.n, self.c * 5, self.h * self.w)        
        t_7 = torch.bmm(t_7_lhs, t_7_rhs) / math.sqrt(self.c * 5)
        t_7 = t_7.view(self.n, self.c, self.h, self.w)
        # Output: p_8
        return t_7.view(self.n, self.c, self.h, self.w)