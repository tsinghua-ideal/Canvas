import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_9391337695165810855(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_9391337695165810855, self).__init__()
        self.g = 4
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # FC: p_1
        self.p_1 = nn.Conv2d(self.c, self.c, 1, padding=0, groups=1, bias=False)
        # Softmax_0/1/C: p_3
        pass
        # Shift_0/1/C_K2: p_4
        self.p_4_0_1 = random.randint(-2, 2)
        # Neg: p_5
        pass
        # Convolution_3x3_1x1_DW1: p_6
        self.p_6 = nn.Conv2d(self.c, self.c, (3, 3), dilation=(1, 1), padding=(1, 1), groups=self.c, bias=False)
        # Mix: p_7
        self.p_7 = nn.Conv2d(self.c, self.c, (3, 3), dilation=(1, 1), padding=(1, 1), groups=1, bias=False)
        # BMul: p_8
        pass
        # BSub: p_9
        pass
        # BAdd: p_10
        pass
        # Output: p_11
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # FC: p_1
        t_1 = self.p_1(t_0)
        t_1 = t_1.view(self.n, self.c, self.h, self.w)
        # Softmax_0/1/C: p_3
        t_3 = F.softmax(t_1, dim=1)
        # Shift_0/1/C_K2: p_4
        t_4 = torch.roll(t_3, self.p_4_0_1, 1)
        # Neg: p_5
        t_5 = -t_1
        # Convolution_3x3_1x1_DW1: p_6
        t_6 = self.p_6(t_0)
        t_6 = t_6.view(self.n, self.c, self.h, self.w)
        # Mix: p_7
        t_7 = self.p_7(t_0)
        # BMul: p_8
        t_8_lhs = t_7.view(self.n, 1, self.c, self.h, self.w)
        t_8_rhs = t_5.view(self.n, 1, self.c, self.h, self.w)
        t_8 = t_8_lhs * t_8_rhs
        t_8 = t_8.view(self.n, self.c, self.h, self.w)
        # BSub: p_9
        t_9_lhs = t_6.view(self.n, 1, self.c, self.h, self.w)
        t_9_rhs = t_4.view(self.n, 1, self.c, self.h, self.w)
        t_9 = t_9_lhs - t_9_rhs
        t_9 = t_9.view(self.n, self.c, self.h, self.w)
        # BAdd: p_10
        t_10 = t_9 + t_8
        # Output: p_11
        return t_10.view(self.n, self.c, self.h, self.w)