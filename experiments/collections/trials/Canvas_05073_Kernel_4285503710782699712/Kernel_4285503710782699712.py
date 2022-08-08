import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_4285503710782699712(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_4285503710782699712, self).__init__()
        self.g = 1
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldW_K7_D2: p_1
        pass
        # Convolution_1_3_1_1_DW0: p_2
        self.p_2 = nn.Conv2d(self.c, self.c, (1, 3), dilation=(1, 1), padding=(0, 1), groups=1, bias=False)
        # BMM_0_1: p_3
        pass
        # UnfoldH_K7_D1: p_4
        pass
        # Group_0_x_1: p_5
        pass
        # Scale_0/1/C_0/3/KW: p_6
        self.p_6_w = nn.Parameter(torch.ones((1, self.c, 7, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_6_w, std=.02)
        # Convolution_1_5_1_1_DW1: p_7
        self.p_7 = nn.Conv2d(self.c, self.c, (1, 5), dilation=(1, 1), padding=(0, 2), groups=self.c, bias=False)
        # Convolution_5_5_2_2_DW1: p_8
        self.p_8 = nn.Conv2d(self.c, self.c, (5, 5), dilation=(2, 2), padding=(4, 4), groups=self.c, bias=False)
        # Scale_0/3/KW_1/1/C: p_9
        self.p_9_w = nn.Parameter(torch.ones((1, 1, 7, self.c,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_9_w, std=.02)
        # BMM_1_0: p_10
        pass
        # BSub: p_11
        pass
        # BSub: p_12
        pass
        # Output: p_13
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldW_K7_D2: p_1
        t_1 = F.unfold(t_0, (1, 7), dilation=(1, 2), padding=(0, 6))
        t_1 = t_1.view(self.n, self.c, 7, self.h, self.w)
        # Convolution_1_3_1_1_DW0: p_2
        t_2 = self.p_2(t_0)
        t_2 = t_2.view(self.n, self.c, self.h, self.w)
        # BMM_0_1: p_3
        t_1_lhs = t_1.view(self.n, self.c * 7, self.h * self.w)        
        t_0_rhs = t_0.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_3 = torch.bmm(t_1_lhs, t_0_rhs).view(self.n, self.c, 7, self.c) / math.sqrt(self.h * self.w)
        # UnfoldH_K7_D1: p_4
        t_4 = F.unfold(t_2, (7, 1), dilation=(1, 1), padding=(3, 0))
        t_4 = t_4.view(self.n, self.c, 7, self.h, self.w)
        # Group_0_x_1: p_5
        t_5 = t_4
        # Scale_0/1/C_0/3/KW: p_6
        t_6 = self.p_6_w * t_3
        # Convolution_1_5_1_1_DW1: p_7
        t_7 = self.p_7(t_2)
        t_7 = t_7.view(self.n, self.c, self.h, self.w)
        # Convolution_5_5_2_2_DW1: p_8
        t_8 = self.p_8(t_0)
        t_8 = t_8.view(self.n, self.c, self.h, self.w)
        # Scale_0/3/KW_1/1/C: p_9
        t_9 = self.p_9_w * t_6
        # BMM_1_0: p_10
        t_9_lhs = t_9.view(self.n, self.c * 7, self.c).transpose(1, 2)        
        t_5_rhs = t_5.view(self.n, self.c * 7, self.h * self.w)        
        t_10 = torch.bmm(t_9_lhs, t_5_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c * 7)
        # BSub: p_11
        t_11 = t_10 - t_7
        # BSub: p_12
        t_12 = t_8 - t_11
        # Output: p_13
        return t_12.view(self.n, self.c, self.h, self.w)