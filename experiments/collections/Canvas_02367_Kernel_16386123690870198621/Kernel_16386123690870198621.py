import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_16386123690870198621(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_16386123690870198621, self).__init__()
        self.g = 16
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Fold_1/0/H_Max: p_1
        pass
        # UnfoldW_K7_D1: p_2
        pass
        # Convolution_1x7_1x2_DW1: p_3
        self.p_3 = nn.Conv2d(self.c, self.c, (1, 7), dilation=(1, 2), padding=(0, 6), groups=self.c, bias=False)
        # Group_0_C: p_4
        pass
        # UnfoldW_K7_D3: p_5
        pass
        # Mix: p_6
        self.p_6_w = nn.Parameter(torch.ones((self.w, 1, )), requires_grad=True)
        bound = math.sqrt(3.0 / (self.w))
        nn.init.uniform_(self.p_6_w, a=-bound, b=bound)
        # FC: p_7
        self.p_7 = nn.Conv2d(self.c * 7, self.c, 1, padding=0, groups=self.c, bias=False)
        # BMin: p_8
        pass
        # Group_0_G: p_9
        pass
        # BAdd: p_10
        pass
        # Mix: p_11
        self.p_11_w = nn.Parameter(torch.ones((self.h, 1, )), requires_grad=True)
        bound = math.sqrt(3.0 / (self.h))
        nn.init.uniform_(self.p_11_w, a=-bound, b=bound)
        # Convolution_5x1_1x1_DW0: p_12
        self.p_12 = nn.Conv2d(self.c, self.c, (5, 1), dilation=(1, 1), padding=(2, 0), groups=1, bias=False)
        # BMul: p_13
        pass
        # Scale_1/1/W: p_14
        self.p_14_w = nn.Parameter(torch.ones((1, 1, 1, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_14_w, std=.02)
        # BSub: p_15
        pass
        # Output: p_16
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Fold_1/0/H_Max: p_1
        t_1 = t_0.max(2)[0]
        # UnfoldW_K7_D1: p_2
        t_2 = t_1.view(self.n, self.c, 1, self.w)
        t_2 = F.unfold(t_2, (1, 7), dilation=(1, 1), padding=(0, 3))
        t_2 = t_2.view(self.n, self.c, 7, self.w)
        # Convolution_1x7_1x2_DW1: p_3
        t_3 = self.p_3(t_0)
        t_3 = t_3.view(self.n, self.c, self.h, self.w)
        # Group_0_C: p_4
        t_4 = t_2
        # UnfoldW_K7_D3: p_5
        t_5 = F.unfold(t_3, (1, 7), dilation=(1, 3), padding=(0, 9))
        t_5 = t_5.view(self.n, self.c, 7, self.h, self.w)
        # Mix: p_6
        t_6 = torch.einsum('abcd,de->abce', [t_4, self.p_6_w]).view(self.n, self.c, 7).contiguous()
        # FC: p_7
        t_7 = t_6.view(self.n, self.c * 7, 1, 1)
        t_7 = self.p_7(t_7)
        t_7 = t_7.view(self.n, self.c)
        # BMin: p_8
        t_8_lhs = t_7.view(self.n, self.c, 1)
        t_8_rhs = t_0.view(self.n, self.c, self.h * self.w)
        t_8 = torch.minimum(t_8_lhs, t_8_rhs)
        t_8 = t_8.view(self.n, self.c, self.h, self.w)
        # Group_0_G: p_9
        t_9 = t_8.view(self.n, self.g, self.c // self.g, self.h, self.w)
        # BAdd: p_10
        t_10_lhs = t_9.view(self.n, 1, self.c, self.h, self.w)
        t_10_rhs = t_0.view(self.n, 1, self.c, self.h, self.w)
        t_10 = t_10_lhs + t_10_rhs
        t_10 = t_10.view(self.n, self.c, self.h, self.w)
        # Mix: p_11
        t_11 = torch.einsum('abcde,df->abcfe', [t_5, self.p_11_w]).view(self.n, self.c, 7, self.w).contiguous()
        # Convolution_5x1_1x1_DW0: p_12
        t_12 = self.p_12(t_10)
        t_12 = t_12.view(self.n, self.c, self.h, self.w)
        # BMul: p_13
        t_13_lhs = t_12.view(self.n, 1, self.c, self.h, self.w)
        t_13_rhs = t_0.view(self.n, 1, self.c, self.h, self.w)
        t_13 = t_13_lhs * t_13_rhs
        t_13 = t_13.view(self.n, self.c, self.h, self.w)
        # Scale_1/1/W: p_14
        t_14 = self.p_14_w * t_11
        # BSub: p_15
        t_15_lhs = t_14.view(self.n, self.c, 1, 7, self.w)
        t_15_rhs = t_13.view(self.n, self.c, self.h // 7, 7, self.w)
        t_15 = t_15_lhs - t_15_rhs
        t_15 = t_15.view(self.n, self.c, self.h, self.w)
        # Output: p_16
        return t_15.view(self.n, self.c, self.h, self.w)