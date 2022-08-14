import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_1614844627337381533(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_1614844627337381533, self).__init__()
        self.g = 1
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Shift_1/1/W_K3: p_1
        self.p_1_1_1 = random.randint(-3, 3)
        # UnfoldH_K3_D3: p_2
        pass
        # Convolution_7x7_3x3_DW1: p_3
        self.p_3 = nn.Conv2d(self.c, self.c, (7, 7), dilation=(3, 3), padding=(9, 9), groups=self.c, bias=False)
        # FC: p_4
        self.p_4 = nn.Conv2d(self.c, self.c, 1, padding=0, groups=1, bias=False)
        # BMax: p_5
        pass
        # FC: p_6
        self.p_6 = nn.Conv2d(self.c * 3, self.c, 1, padding=0, groups=1, bias=False)
        # BMul: p_7
        pass
        # BAdd: p_8
        pass
        # Output: p_9
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Shift_1/1/W_K3: p_1
        t_1 = torch.roll(t_0, self.p_1_1_1, 3)
        # UnfoldH_K3_D3: p_2
        t_2 = F.unfold(t_1, (3, 1), dilation=(3, 1), padding=(3, 0))
        t_2 = t_2.view(self.n, self.c, 3, self.h, self.w)
        # Convolution_7x7_3x3_DW1: p_3
        t_3 = self.p_3(t_1)
        t_3 = t_3.view(self.n, self.c, self.h, self.w)
        # FC: p_4
        t_4 = self.p_4(t_1)
        t_4 = t_4.view(self.n, self.c, self.h, self.w)
        # BMax: p_5
        t_5 = torch.maximum(t_1, t_4)
        # FC: p_6
        t_6 = t_2.view(self.n, self.c * 3, self.h, self.w)
        t_6 = self.p_6(t_6)
        t_6 = t_6.view(self.n, self.c, self.h, self.w)
        # BMul: p_7
        t_7 = t_6 * t_5
        # BAdd: p_8
        t_8 = t_3 + t_7
        # Output: p_9
        return t_8.view(self.n, self.c, self.h, self.w)