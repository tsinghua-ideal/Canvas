import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_105486651608881816(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_105486651608881816, self).__init__()
        self.g = 16
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Scale_0/1/C_1/0/H_1/1/W: p_1
        self.p_1_w = nn.Parameter(torch.ones((1, self.c, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_1_w, std=.02)
        # Group_0_x_0: p_2
        pass
        # UnfoldW_K5_D2: p_3
        pass
        # Fold_0/0/G_Max: p_4
        pass
        # Convolution_3_1_2_1_DW1: p_5
        self.p_5 = nn.Conv2d(self.c, self.c, (3, 1), dilation=(2, 1), padding=(2, 0), groups=self.c, bias=False)
        # BMM_0_1: p_6
        pass
        # Convolution_3_1_2_1_DW0: p_7
        self.p_7 = nn.Conv2d(self.c, self.c * 5, (3, 1), dilation=(2, 1), padding=(2, 0), groups=1, bias=False)
        # Convolution_7_1_3_1_DW0: p_8
        self.p_8 = nn.Conv2d(self.c, self.c, (7, 1), dilation=(3, 1), padding=(9, 0), groups=1, bias=False)
        # BMM_1_0: p_9
        pass
        # BMax: p_10
        pass
        # Output: p_11
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Scale_0/1/C_1/0/H_1/1/W: p_1
        t_1 = self.p_1_w * t_0
        # Group_0_x_0: p_2
        t_2 = t_1
        # UnfoldW_K5_D2: p_3
        t_3 = F.unfold(t_2, (1, 5), dilation=(1, 2), padding=(0, 4))
        t_3 = t_3.view(self.n, self.c, 5, self.h, self.w)
        # Fold_0/0/G_Max: p_4
        t_4 = t_3
        # Convolution_3_1_2_1_DW1: p_5
        t_5 = self.p_5(t_0)
        t_5 = t_5.view(self.n, self.c, self.h, self.w)
        # BMM_0_1: p_6
        t_4_lhs = t_4.view(self.n, self.c * 5, self.h * self.w)        
        t_0_rhs = t_0.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_6 = torch.bmm(t_4_lhs, t_0_rhs).view(self.n, self.c, 5, self.c) / math.sqrt(self.h * self.w)
        # Convolution_3_1_2_1_DW0: p_7
        t_7 = self.p_7(t_5)
        t_7 = t_7.view(self.n, self.c * 5, self.h, self.w)
        # Convolution_7_1_3_1_DW0: p_8
        t_8 = self.p_8(t_0)
        t_8 = t_8.view(self.n, self.c, self.h, self.w)
        # BMM_1_0: p_9
        t_6_lhs = t_6.view(self.n, self.c * 5, self.c).transpose(1, 2)        
        t_7_rhs = t_7.view(self.n, self.c * 5, self.h * self.w)        
        t_9 = torch.bmm(t_6_lhs, t_7_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c * 5)
        # BMax: p_10
        t_10 = torch.maximum(t_9, t_8)
        # Output: p_11
        return t_10.view(self.n, self.c, self.h, self.w)