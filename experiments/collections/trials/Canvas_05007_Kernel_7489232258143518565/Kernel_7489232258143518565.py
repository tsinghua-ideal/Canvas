import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_7489232258143518565(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_7489232258143518565, self).__init__()
        self.g = 2
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Convolution_7_1_2_1_DW1: p_1
        self.p_1 = nn.Conv2d(self.c, self.c, (7, 1), dilation=(2, 1), padding=(6, 0), groups=self.c, bias=False)
        # UnfoldHW_K5_D2: p_2
        pass
        # Shift_0/1/C_K1: p_3
        self.p_3_0_1 = random.randint(-1, 1)
        # Group_0_x_0: p_4
        pass
        # BMM_0_1: p_5
        pass
        # TanH: p_6
        pass
        # FC: p_7
        self.p_7 = nn.Conv2d(self.c * 25, self.c, 1, padding=0, groups=1, bias=False)
        # Convolution_5_5_3_3_DW1: p_8
        self.p_8 = nn.Conv2d(self.c, self.c, (5, 5), dilation=(3, 3), padding=(6, 6), groups=self.c, bias=False)
        # Convolution_1_7_1_1_DW1: p_9
        self.p_9 = nn.Conv2d(self.c, self.c, (1, 7), dilation=(1, 1), padding=(0, 3), groups=self.c, bias=False)
        # BMM_1_0: p_10
        pass
        # Output: p_11
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Convolution_7_1_2_1_DW1: p_1
        t_1 = self.p_1(t_0)
        t_1 = t_1.view(self.n, self.c, self.h, self.w)
        # UnfoldHW_K5_D2: p_2
        t_2 = F.unfold(t_1, (5, 5), dilation=(2, 2), padding=(4, 4))
        t_2 = t_2.view(self.n, self.c, 5, 5, self.h, self.w)
        # Shift_0/1/C_K1: p_3
        t_3 = torch.roll(t_0, self.p_3_0_1, 1)
        # Group_0_x_0: p_4
        t_4 = t_3
        # BMM_0_1: p_5
        t_4_lhs = t_4.view(self.n, self.c, self.h * self.w)        
        t_1_rhs = t_1.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_5 = torch.bmm(t_4_lhs, t_1_rhs).view(self.n, self.c, self.c) / math.sqrt(self.h * self.w)
        # TanH: p_6
        t_6 = torch.tanh(t_2)
        # FC: p_7
        t_7 = t_6.view(self.n, self.c * 25, self.h, self.w)
        t_7 = self.p_7(t_7)
        t_7 = t_7.view(self.n, self.c, self.h, self.w)
        # Convolution_5_5_3_3_DW1: p_8
        t_8 = self.p_8(t_7)
        t_8 = t_8.view(self.n, self.c, self.h, self.w)
        # Convolution_1_7_1_1_DW1: p_9
        t_9 = self.p_9(t_8)
        t_9 = t_9.view(self.n, self.c, self.h, self.w)
        # BMM_1_0: p_10
        t_5_lhs = t_5.view(self.n, self.c, self.c).transpose(1, 2)        
        t_9_rhs = t_9.view(self.n, self.c, self.h * self.w)        
        t_10 = torch.bmm(t_5_lhs, t_9_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c)
        # Output: p_11
        return t_10.view(self.n, self.c, self.h, self.w)