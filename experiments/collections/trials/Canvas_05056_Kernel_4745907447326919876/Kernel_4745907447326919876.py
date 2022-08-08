import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_4745907447326919876(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_4745907447326919876, self).__init__()
        self.g = 1
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Shift_1/0/H_1/1/W_K2: p_1
        self.p_1_1_0 = random.randint(-2, 2)
        self.p_1_1_1 = random.randint(-2, 2)
        # Scale_1/1/W: p_2
        self.p_2_w = nn.Parameter(torch.ones((1, 1, 1, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_2_w, std=.1)
        # Convolution_3x1_1x1_DW0: p_3
        self.p_3 = nn.Conv2d(self.c, self.c * 3, (3, 1), dilation=(1, 1), padding=(1, 0), groups=1, bias=False)
        # UnfoldW_K3_D1: p_4
        pass
        # Convolution_1x7_1x1_DW1: p_5
        self.p_5 = nn.Conv2d(self.c * 3, self.c * 3, (1, 7), dilation=(1, 1), padding=(0, 3), groups=self.c * 3, bias=False)
        # BMM_0_1: p_6
        pass
        # BAdd: p_7
        pass
        # BSub: p_8
        pass
        # TanH: p_9
        pass
        # Convolution_7x1_1x1_DW0: p_10
        self.p_10 = nn.Conv2d(self.c * 3, self.c, (7, 1), dilation=(1, 1), padding=(3, 0), groups=1, bias=False)
        # BMul: p_11
        pass
        # BMM_1_0: p_12
        pass
        # Output: p_13
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Shift_1/0/H_1/1/W_K2: p_1
        t_1 = torch.roll(t_0, self.p_1_1_0, 2)
        t_1 = torch.roll(t_1, self.p_1_1_1, 3)
        # Scale_1/1/W: p_2
        t_2 = self.p_2_w * t_0
        # Convolution_3x1_1x1_DW0: p_3
        t_3 = self.p_3(t_1)
        t_3 = t_3.view(self.n, self.c * 3, self.h, self.w)
        # UnfoldW_K3_D1: p_4
        t_4 = F.unfold(t_0, (1, 3), dilation=(1, 1), padding=(0, 1))
        t_4 = t_4.view(self.n, self.c, 3, self.h, self.w)
        # Convolution_1x7_1x1_DW1: p_5
        t_5 = self.p_5(t_3)
        t_5 = t_5.view(self.n, self.c * 3, self.h, self.w)
        # BMM_0_1: p_6
        t_2_lhs = t_2.view(self.n, self.c, self.h * self.w)        
        t_2_rhs = t_2.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_6 = torch.bmm(t_2_lhs, t_2_rhs).view(self.n, self.c, self.c) / math.sqrt(self.h * self.w)
        # BAdd: p_7
        t_7_lhs = t_4.view(self.n, 1, self.c * 3, self.h, self.w)
        t_7_rhs = t_5.view(self.n, 1, self.c * 3, self.h, self.w)
        t_7 = t_7_lhs + t_7_rhs
        t_7 = t_7.view(self.n, self.c * 3, self.h, self.w)
        # BSub: p_8
        t_8 = t_2 - t_0
        # TanH: p_9
        t_9 = torch.tanh(t_8)
        # Convolution_7x1_1x1_DW0: p_10
        t_10 = self.p_10(t_7)
        t_10 = t_10.view(self.n, self.c, self.h, self.w)
        # BMul: p_11
        t_11_lhs = t_9.view(self.n, 1, self.c, self.h, self.w)
        t_11_rhs = t_10.view(self.n, 1, self.c, self.h, self.w)
        t_11 = t_11_lhs * t_11_rhs
        t_11 = t_11.view(self.n, self.c, self.h, self.w)
        # BMM_1_0: p_12
        t_6_lhs = t_6.view(self.n, self.c, self.c).transpose(1, 2)        
        t_11_rhs = t_11.view(self.n, self.c, self.h * self.w)        
        t_12 = torch.bmm(t_6_lhs, t_11_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c)
        # Output: p_13
        return t_12.view(self.n, self.c, self.h, self.w)