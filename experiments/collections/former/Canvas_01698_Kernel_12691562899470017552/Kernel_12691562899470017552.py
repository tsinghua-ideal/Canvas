import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_12691562899470017552(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_12691562899470017552, self).__init__()
        self.g = 2
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldH_K7_D1: p_1
        pass
        # FC: p_2
        self.p_2 = nn.Conv2d(self.c * 7, self.c, 1, padding=0, groups=1, bias=False)
        # Scale_1/1/W: p_3
        self.p_3_w = nn.Parameter(torch.ones((1, 1, 1, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_3_w, std=.02)
        # BMul: p_4
        pass
        # BMax: p_5
        pass
        # Scale_0/1/C: p_6
        self.p_6_w = nn.Parameter(torch.ones((1, self.c, 1, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_6_w, std=.02)
        # Shift_1/0/H_K1: p_7
        self.p_7_1_0 = random.randint(-1, 1)
        # BMul: p_8
        pass
        # BMax: p_9
        pass
        # Output: p_10
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldH_K7_D1: p_1
        t_1 = F.unfold(t_0, (7, 1), dilation=(1, 1), padding=(3, 0))
        t_1 = t_1.view(self.n, self.c, 7, self.h, self.w)
        # FC: p_2
        t_2 = t_1.view(self.n, self.c * 7, self.h, self.w)
        t_2 = self.p_2(t_2)
        t_2 = t_2.view(self.n, self.c, self.h, self.w)
        # Scale_1/1/W: p_3
        t_3 = self.p_3_w * t_2
        # BMul: p_4
        t_4 = t_0 * t_2
        # BMax: p_5
        t_5 = torch.maximum(t_3, t_2)
        # Scale_0/1/C: p_6
        t_6 = self.p_6_w * t_2
        # Shift_1/0/H_K1: p_7
        t_7 = torch.roll(t_4, self.p_7_1_0, 2)
        # BMul: p_8
        t_8 = t_6 * t_5
        # BMax: p_9
        t_9 = torch.maximum(t_7, t_8)
        # Output: p_10
        return t_9.view(self.n, self.c, self.h, self.w)