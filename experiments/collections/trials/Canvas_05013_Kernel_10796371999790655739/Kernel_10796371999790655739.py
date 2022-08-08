import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_10796371999790655739(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_10796371999790655739, self).__init__()
        self.g = 16
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldW_K5_D2: p_1
        pass
        # FC: p_2
        self.p_2 = nn.Conv2d(self.c * 5, self.c // 4, 1, padding=0, groups=1, bias=False)
        # Scale_0/1/C_1/0/H_1/1/W: p_3
        self.p_3_w = nn.Parameter(torch.ones((1, self.c, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_3_w, std=.02)
        # Group_0_G: p_4
        pass
        # Shift_1/1/W_K2: p_5
        self.p_5_1_1 = random.randint(-2, 2)
        # Convolution_1_5_1_3_DW0: p_6
        self.p_6 = nn.Conv2d(self.c, self.c, (1, 5), dilation=(1, 3), padding=(0, 6), groups=1, bias=False)
        # Scale_1/0/H_1/1/W: p_7
        self.p_7_w = nn.Parameter(torch.ones((1, 1, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_7_w, std=.02)
        # Convolution_1_5_1_3_DW0: p_8
        self.p_8 = nn.Conv2d(self.c // 4, self.c, (1, 5), dilation=(1, 3), padding=(0, 6), groups=1, bias=False)
        # FC: p_9
        self.p_9 = nn.Conv2d(self.c, self.c, 1, padding=0, groups=1, bias=False)
        # Scale_1/0/H_1/1/W: p_10
        self.p_10_w = nn.Parameter(torch.ones((1, 1, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_10_w, std=.02)
        # BAdd: p_11
        pass
        # BMM_0_1: p_12
        pass
        # BMM_1_0: p_13
        pass
        # Output: p_14
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldW_K5_D2: p_1
        t_1 = F.unfold(t_0, (1, 5), dilation=(1, 2), padding=(0, 4))
        t_1 = t_1.view(self.n, self.c, 5, self.h, self.w)
        # FC: p_2
        t_2 = t_1.view(self.n, self.c * 5, self.h, self.w)
        t_2 = self.p_2(t_2)
        t_2 = t_2.view(self.n, self.c // 4, self.h, self.w)
        # Scale_0/1/C_1/0/H_1/1/W: p_3
        t_3 = self.p_3_w * t_0
        # Group_0_G: p_4
        t_4 = t_3.view(self.n, self.g, self.c // self.g, self.h, self.w)
        # Shift_1/1/W_K2: p_5
        t_5 = torch.roll(t_0, self.p_5_1_1, 3)
        # Convolution_1_5_1_3_DW0: p_6
        t_6 = self.p_6(t_5)
        t_6 = t_6.view(self.n, self.c, self.h, self.w)
        # Scale_1/0/H_1/1/W: p_7
        t_7 = self.p_7_w * t_2
        # Convolution_1_5_1_3_DW0: p_8
        t_8 = self.p_8(t_7)
        t_8 = t_8.view(self.n, self.c, self.h, self.w)
        # FC: p_9
        t_9 = self.p_9(t_8)
        t_9 = t_9.view(self.n, self.c, self.h, self.w)
        # Scale_1/0/H_1/1/W: p_10
        t_10 = self.p_10_w * t_9
        # BAdd: p_11
        t_11 = t_0 + t_10
        # BMM_0_1: p_12
        t_6_lhs = t_6.view(self.n, self.c, self.h * self.w)        
        t_11_rhs = t_11.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_12 = torch.bmm(t_6_lhs, t_11_rhs).view(self.n, self.c, self.c) / math.sqrt(self.h * self.w)
        # BMM_1_0: p_13
        t_4_lhs = t_4.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_12_rhs = t_12.view(self.n, self.c, self.c)        
        t_13 = torch.bmm(t_4_lhs, t_12_rhs).view(self.n, self.h, self.w, self.c) / math.sqrt(self.c)
        # Output: p_14
        return t_13.permute(0, 3, 1, 2).contiguous().view(self.n, self.c, self.h, self.w)