import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_11041417490697712865(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_11041417490697712865, self).__init__()
        self.g = 8
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Group_0_x_0: p_1
        pass
        # Group_0_C/G: p_2
        pass
        # Convolution_1_5_1_2_DW0: p_3
        self.p_3 = nn.Conv2d(self.c, self.c, (1, 5), dilation=(1, 2), padding=(0, 4), groups=self.c, bias=False)
        # Convolution_3_1_1_1_DW0: p_4
        self.p_4 = nn.Conv2d(self.c, self.c, (3, 1), dilation=(1, 1), padding=(1, 0), groups=self.c // self.g, bias=False)
        # Shift_0/1/C_K3: p_5
        self.p_5_0_1 = random.randint(-3, 3)
        # UnfoldH_K3_D3: p_6
        pass
        # FC: p_7
        self.p_7 = nn.Conv2d(self.c * 3, self.c, 1, padding=0, groups=1, bias=False)
        # BMM_0_1: p_8
        pass
        # BMM_1_0: p_9
        pass
        # Output: p_10
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Group_0_x_0: p_1
        t_1 = t_0
        # Group_0_C/G: p_2
        t_2 = t_0.view(self.n, self.c // self.g, self.g, self.h, self.w)
        # Convolution_1_5_1_2_DW0: p_3
        t_3 = self.p_3(t_1)
        t_3 = t_3.view(self.n, self.c, self.h, self.w)
        # Convolution_3_1_1_1_DW0: p_4
        t_4 = t_2.view(self.n, self.c, self.h, self.w)
        t_4 = self.p_4(t_4)
        t_4 = t_4.view(self.n, self.c, self.h, self.w)
        # Shift_0/1/C_K3: p_5
        t_5 = torch.roll(t_0, self.p_5_0_1, 1)
        # UnfoldH_K3_D3: p_6
        t_6 = F.unfold(t_5, (3, 1), dilation=(3, 1), padding=(3, 0))
        t_6 = t_6.view(self.n, self.c, 3, self.h, self.w)
        # FC: p_7
        t_7 = t_6.view(self.n, self.c * 3, self.h, self.w)
        t_7 = self.p_7(t_7)
        t_7 = t_7.view(self.n, self.c, self.h, self.w)
        # BMM_0_1: p_8
        t_4_lhs = t_4.view(self.n, self.c, self.h * self.w)        
        t_3_rhs = t_3.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_8 = torch.bmm(t_4_lhs, t_3_rhs).view(self.n, self.c, self.c) / math.sqrt(self.h * self.w)
        # BMM_1_0: p_9
        t_8_lhs = t_8.view(self.n, self.c, self.c).transpose(1, 2)        
        t_7_rhs = t_7.view(self.n, self.c, self.h * self.w)        
        t_9 = torch.bmm(t_8_lhs, t_7_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c)
        # Output: p_10
        return t_9.view(self.n, self.c, self.h, self.w)