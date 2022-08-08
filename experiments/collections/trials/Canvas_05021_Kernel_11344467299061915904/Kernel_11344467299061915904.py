import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_11344467299061915904(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_11344467299061915904, self).__init__()
        self.g = 4
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldH_K7_D2: p_1
        pass
        # Scale_0/1/C_1/1/W: p_2
        self.p_2_w = nn.Parameter(torch.ones((1, self.c, 1, 1, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_2_w, std=.1)
        # Scale_0/1/C_0/2/KH: p_3
        self.p_3_w = nn.Parameter(torch.ones((1, self.c, 7, 1, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_3_w, std=.1)
        # Scale_0/1/C_1/0/H_1/1/W: p_4
        self.p_4_w = nn.Parameter(torch.ones((1, self.c, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_4_w, std=.1)
        # Shift_1/1/W_K2: p_5
        self.p_5_1_1 = random.randint(-2, 2)
        # Convolution_3x1_3x1_DW1: p_6
        self.p_6 = nn.Conv2d(self.c, self.c, (3, 1), dilation=(3, 1), padding=(3, 0), groups=self.c, bias=False)
        # Convolution_1x7_1x1_DW0: p_7
        self.p_7 = nn.Conv2d(self.c, self.c * 7, (1, 7), dilation=(1, 1), padding=(0, 3), groups=1, bias=False)
        # BMM_0_1: p_8
        pass
        # BMM_0_0: p_9
        pass
        # BSub: p_10
        pass
        # Output: p_11
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldH_K7_D2: p_1
        t_1 = F.unfold(t_0, (7, 1), dilation=(2, 1), padding=(6, 0))
        t_1 = t_1.view(self.n, self.c, 7, self.h, self.w)
        # Scale_0/1/C_1/1/W: p_2
        t_2 = self.p_2_w * t_1
        # Scale_0/1/C_0/2/KH: p_3
        t_3 = self.p_3_w * t_2
        # Scale_0/1/C_1/0/H_1/1/W: p_4
        t_4 = self.p_4_w * t_0
        # Shift_1/1/W_K2: p_5
        t_5 = torch.roll(t_4, self.p_5_1_1, 3)
        # Convolution_3x1_3x1_DW1: p_6
        t_6 = self.p_6(t_5)
        t_6 = t_6.view(self.n, self.c, self.h, self.w)
        # Convolution_1x7_1x1_DW0: p_7
        t_7 = self.p_7(t_0)
        t_7 = t_7.view(self.n, self.c * 7, self.h, self.w)
        # BMM_0_1: p_8
        t_0_lhs = t_0.view(self.n, self.c, self.h * self.w)        
        t_3_rhs = t_3.view(self.n, self.c * 7, self.h * self.w).transpose(1, 2)        
        t_8 = torch.bmm(t_0_lhs, t_3_rhs).view(self.n, self.c, self.c, 7) / math.sqrt(self.h * self.w)
        # BMM_0_0: p_9
        t_8_lhs = t_8.view(self.n, self.c, self.c * 7)        
        t_7_rhs = t_7.view(self.n, self.c * 7, self.h * self.w)        
        t_9 = torch.bmm(t_8_lhs, t_7_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c * 7)
        # BSub: p_10
        t_10_lhs = t_9.view(self.n, 1, self.c, self.h, self.w)
        t_10_rhs = t_6.view(self.n, 1, self.c, self.h, self.w)
        t_10 = t_10_lhs - t_10_rhs
        t_10 = t_10.view(self.n, self.c, self.h, self.w)
        # Output: p_11
        return t_10.view(self.n, self.c, self.h, self.w)