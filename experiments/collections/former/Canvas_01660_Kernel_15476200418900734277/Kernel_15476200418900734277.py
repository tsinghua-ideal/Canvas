import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_15476200418900734277(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_15476200418900734277, self).__init__()
        self.g = 8
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldH_K7_D3: p_1
        pass
        # Convolution_1x5_1x1_DW0: p_2
        self.p_2 = nn.Conv2d(self.c, self.c, (1, 5), dilation=(1, 1), padding=(0, 2), groups=1, bias=False)
        # Fold_1/1/W_Avg: p_3
        pass
        # Shift_1/0/H_1/1/W_K1: p_4
        self.p_4_1_0 = random.randint(-1, 1)
        self.p_4_1_1 = random.randint(-1, 1)
        # BMul: p_5
        pass
        # BMax: p_6
        pass
        # Output: p_7
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldH_K7_D3: p_1
        t_1 = F.unfold(t_0, (7, 1), dilation=(3, 1), padding=(9, 0))
        t_1 = t_1.view(self.n, self.c, 7, self.h, self.w)
        # Convolution_1x5_1x1_DW0: p_2
        t_2 = self.p_2(t_0)
        t_2 = t_2.view(self.n, self.c, self.h, self.w)
        # Fold_1/1/W_Avg: p_3
        t_3 = t_1.mean(4)
        # Shift_1/0/H_1/1/W_K1: p_4
        t_4 = torch.roll(t_0, self.p_4_1_0, 2)
        t_4 = torch.roll(t_4, self.p_4_1_1, 3)
        # BMul: p_5
        t_5_lhs = t_4.view(self.n, 1, self.c, self.h, self.w)
        t_5_rhs = t_2.view(self.n, 1, self.c, self.h, self.w)
        t_5 = t_5_lhs * t_5_rhs
        t_5 = t_5.view(self.n, self.c, self.h, self.w)
        # BMax: p_6
        t_6_lhs = t_3.view(self.n, 1, self.c * self.h * 7)
        t_6_rhs = t_5.view(self.n, self.w // 7, self.c * self.h * 7)
        t_6 = torch.maximum(t_6_lhs, t_6_rhs)
        t_6 = t_6.view(self.n, self.c, self.h, self.w)
        # Output: p_7
        return t_6.view(self.n, self.c, self.h, self.w)