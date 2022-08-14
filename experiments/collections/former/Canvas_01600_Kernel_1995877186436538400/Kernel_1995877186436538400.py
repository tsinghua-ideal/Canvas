import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_1995877186436538400(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_1995877186436538400, self).__init__()
        self.g = 2
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Convolution_5x5_1x1_DW1: p_1
        self.p_1 = nn.Conv2d(self.c, self.c, (5, 5), dilation=(1, 1), padding=(2, 2), groups=self.c, bias=False)
        # Convolution_3x1_1x1_DW1: p_2
        self.p_2 = nn.Conv2d(self.c, self.c, (3, 1), dilation=(1, 1), padding=(1, 0), groups=self.c, bias=False)
        # UnfoldH_K7_D3: p_3
        pass
        # FC: p_4
        self.p_4 = nn.Conv2d(self.c * 7, self.c, 1, padding=0, groups=1, bias=False)
        # BMul: p_5
        pass
        # Output: p_6
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Convolution_5x5_1x1_DW1: p_1
        t_1 = self.p_1(t_0)
        t_1 = t_1.view(self.n, self.c, self.h, self.w)
        # Convolution_3x1_1x1_DW1: p_2
        t_2 = self.p_2(t_0)
        t_2 = t_2.view(self.n, self.c, self.h, self.w)
        # UnfoldH_K7_D3: p_3
        t_3 = F.unfold(t_2, (7, 1), dilation=(3, 1), padding=(9, 0))
        t_3 = t_3.view(self.n, self.c, 7, self.h, self.w)
        # FC: p_4
        t_4 = t_3.view(self.n, self.c * 7, self.h, self.w)
        t_4 = self.p_4(t_4)
        t_4 = t_4.view(self.n, self.c, self.h, self.w)
        # BMul: p_5
        t_5 = t_1 * t_4
        # Output: p_6
        return t_5.view(self.n, self.c, self.h, self.w)