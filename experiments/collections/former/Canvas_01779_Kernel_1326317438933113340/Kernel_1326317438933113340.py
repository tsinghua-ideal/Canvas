import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_1326317438933113340(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_1326317438933113340, self).__init__()
        self.g = 2
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Neg: p_1
        pass
        # Convolution_1x5_1x3_DW1: p_2
        self.p_2 = nn.Conv2d(self.c, self.c, (1, 5), dilation=(1, 3), padding=(0, 6), groups=self.c, bias=False)
        # UnfoldW_K7_D2: p_3
        pass
        # Mix: p_4
        self.p_4_w = nn.Parameter(torch.ones((self.c, self.c, )), requires_grad=True)
        bound = math.sqrt(3.0 / (self.c))
        nn.init.uniform_(self.p_4_w, a=-bound, b=bound)
        # Convolution_7x7_1x1_DW1: p_5
        self.p_5 = nn.Conv2d(self.c, self.c, (7, 7), dilation=(1, 1), padding=(3, 3), groups=self.c, bias=False)
        # Convolution_5x1_3x1_DW1: p_6
        self.p_6 = nn.Conv2d(self.c, self.c, (5, 1), dilation=(3, 1), padding=(6, 0), groups=self.c, bias=False)
        # Convolution_1x3_1x1_DW0: p_7
        self.p_7 = nn.Conv2d(self.c, self.c, (1, 3), dilation=(1, 1), padding=(0, 1), groups=1, bias=False)
        # Fold_1/0/H_Avg: p_8
        pass
        # Mix: p_9
        self.p_9_w = nn.Parameter(torch.ones((self.w, self.w // 7, )), requires_grad=True)
        bound = math.sqrt(3.0 / (self.w))
        nn.init.uniform_(self.p_9_w, a=-bound, b=bound)
        # BMul: p_10
        pass
        # BSub: p_11
        pass
        # Output: p_12
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Neg: p_1
        t_1 = -t_0
        # Convolution_1x5_1x3_DW1: p_2
        t_2 = self.p_2(t_0)
        t_2 = t_2.view(self.n, self.c, self.h, self.w)
        # UnfoldW_K7_D2: p_3
        t_3 = F.unfold(t_2, (1, 7), dilation=(1, 2), padding=(0, 6))
        t_3 = t_3.view(self.n, self.c, 7, self.h, self.w)
        # Mix: p_4
        t_4 = torch.einsum('abcd,be->aecd', [t_0, self.p_4_w]).view(self.n, self.c, self.h, self.w).contiguous()
        # Convolution_7x7_1x1_DW1: p_5
        t_5 = self.p_5(t_4)
        t_5 = t_5.view(self.n, self.c, self.h, self.w)
        # Convolution_5x1_3x1_DW1: p_6
        t_6 = self.p_6(t_5)
        t_6 = t_6.view(self.n, self.c, self.h, self.w)
        # Convolution_1x3_1x1_DW0: p_7
        t_7 = self.p_7(t_6)
        t_7 = t_7.view(self.n, self.c, self.h, self.w)
        # Fold_1/0/H_Avg: p_8
        t_8 = t_3.mean(3)
        # Mix: p_9
        t_9 = torch.einsum('abcd,de->abce', [t_8, self.p_9_w]).view(self.n, self.c, 7, self.w // 7).contiguous()
        # BMul: p_10
        t_10_lhs = t_7.view(self.n, 1, self.c, self.h, self.w)
        t_10_rhs = t_1.view(self.n, 1, self.c, self.h, self.w)
        t_10 = t_10_lhs * t_10_rhs
        t_10 = t_10.view(self.n, self.c, self.h, self.w)
        # BSub: p_11
        t_11_lhs = t_9.view(self.n, self.c, 1, self.w)
        t_11_rhs = t_10.view(self.n, self.c, self.h, self.w)
        t_11 = t_11_lhs - t_11_rhs
        t_11 = t_11.view(self.n, self.c, self.h, self.w)
        # Output: p_12
        return t_11.view(self.n, self.c, self.h, self.w)