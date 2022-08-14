import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_11405126940588574002(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_11405126940588574002, self).__init__()
        self.g = 32
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Convolution_5x1_1x1_DW0: p_1
        self.p_1 = nn.Conv2d(self.c, self.c, (5, 1), dilation=(1, 1), padding=(2, 0), groups=1, bias=False)
        # BMul: p_2
        pass
        # Shift_0/1/C_K1: p_3
        self.p_3_0_1 = random.randint(-1, 1)
        # UnfoldW_K5_D1: p_4
        pass
        # Fold_0/3/KW_Avg: p_5
        pass
        # Output: p_6
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Convolution_5x1_1x1_DW0: p_1
        t_1 = self.p_1(t_0)
        t_1 = t_1.view(self.n, self.c, self.h, self.w)
        # BMul: p_2
        t_2_lhs = t_1.view(self.n, 1, self.c, self.h, self.w)
        t_2_rhs = t_0.view(self.n, 1, self.c, self.h, self.w)
        t_2 = t_2_lhs * t_2_rhs
        t_2 = t_2.view(self.n, self.c, self.h, self.w)
        # Shift_0/1/C_K1: p_3
        t_3 = torch.roll(t_2, self.p_3_0_1, 1)
        # UnfoldW_K5_D1: p_4
        t_4 = F.unfold(t_3, (1, 5), dilation=(1, 1), padding=(0, 2))
        t_4 = t_4.view(self.n, self.c, 5, self.h, self.w)
        # Fold_0/3/KW_Avg: p_5
        t_5 = t_4.mean(2)
        # Output: p_6
        return t_5.view(self.n, self.c, self.h, self.w)