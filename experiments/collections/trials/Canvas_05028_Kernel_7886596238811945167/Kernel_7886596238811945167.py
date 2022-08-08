import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_7886596238811945167(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_7886596238811945167, self).__init__()
        self.g = 2
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldW_K3_D2: p_1
        pass
        # Shift_0/1/C_K1: p_2
        self.p_2_0_1 = random.randint(-1, 1)
        # FC: p_3
        self.p_3 = nn.Conv2d(self.c * 3, self.c, 1, padding=0, groups=1, bias=False)
        # Sin: p_4
        pass
        # BMM_0_1: p_5
        pass
        # BMax: p_6
        pass
        # Scale_0/1/C_0/3/KW_1/1/C: p_7
        self.p_7_w = nn.Parameter(torch.ones((1, self.c, 3, self.c,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_7_w, std=.02)
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
        # Shift_0/1/C_K1: p_2
        t_2 = torch.roll(t_1, self.p_2_0_1, 1)
        # FC: p_3
        t_3 = t_1.view(self.n, self.c * 3, self.h, self.w)
        t_3 = self.p_3(t_3)
        t_3 = t_3.view(self.n, self.c, self.h, self.w)
        # Sin: p_4
        t_4 = torch.sin(t_1 * 1.5707963)
        # BMM_0_1: p_5
        t_2_lhs = t_2.view(self.n, self.c * 3, self.h * self.w)        
        t_0_rhs = t_0.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_5 = torch.bmm(t_2_lhs, t_0_rhs).view(self.n, self.c, 3, self.c) / math.sqrt(self.h * self.w)
        # BMax: p_6
        t_6_lhs = t_3.view(self.n, self.c, 1, self.h, self.w)
        t_6_rhs = t_4.view(self.n, self.c, 3, self.h, self.w)
        t_6 = torch.maximum(t_6_lhs, t_6_rhs)
        t_6 = t_6.view(self.n, self.c, 3, self.h, self.w)
        # Scale_0/1/C_0/3/KW_1/1/C: p_7
        t_7 = self.p_7_w * t_5
        # BMM_1_0: p_8
        t_6_lhs = t_6.view(self.n, self.c * 3, self.h * self.w).transpose(1, 2)        
        t_7_rhs = t_7.view(self.n, self.c * 3, self.c)        
        t_8 = torch.bmm(t_6_lhs, t_7_rhs).view(self.n, self.h, self.w, self.c) / math.sqrt(self.c * 3)
        # Output: p_9
        return t_8.permute(0, 3, 1, 2).contiguous().view(self.n, self.c, self.h, self.w)