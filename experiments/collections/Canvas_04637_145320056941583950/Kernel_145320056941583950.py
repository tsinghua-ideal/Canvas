import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_4262547987266090318(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_4262547987266090318, self).__init__()
        self.g = 32
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldH_K5_D2: p_1
        pass
        # UnfoldW_K7_D3: p_2
        pass
        # FC: p_3
        self.p_3 = nn.Conv2d(self.c * 35, self.c // 8, 1, padding=0, groups=1, bias=False)
        # FC: p_4
        self.p_4 = nn.Conv2d(self.c // 8, self.c // 8, 1, padding=0, groups=1, bias=False)
        # FC: p_5
        self.p_5 = nn.Conv2d(self.c // 8, self.c, 1, padding=0, groups=1, bias=False)
        # Output: p_6
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldH_K5_D2: p_1
        t_1 = F.unfold(t_0, (5, 1), dilation=(2, 1), padding=(4, 0))
        t_1 = t_1.view(self.n, self.c, 5, self.h, self.w)
        # UnfoldW_K7_D3: p_2
        t_2 = t_1.view(self.n, self.c * 5, self.h, self.w)
        t_2 = F.unfold(t_2, (1, 7), dilation=(1, 3), padding=(0, 9))
        t_2 = t_2.view(self.n, self.c, 5, 7, self.h, self.w)
        # FC: p_3
        t_3 = t_2.view(self.n, self.c * 35, self.h, self.w)
        t_3 = self.p_3(t_3)
        t_3 = t_3.view(self.n, self.c // 8, self.h, self.w)
        # FC: p_4
        t_4 = self.p_4(t_3)
        t_4 = t_4.view(self.n, self.c // 8, self.h, self.w)
        # FC: p_5
        t_5 = self.p_5(t_4)
        t_5 = t_5.view(self.n, self.c, self.h, self.w)
        # Output: p_6
        return t_5.view(self.n, self.c, self.h, self.w)