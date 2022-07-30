import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_13877190235998553350(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_13877190235998553350, self).__init__()
        self.g = 16
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Mix: p_1
        self.p_1_w = nn.Parameter(torch.ones((self.h, self.c // 8, )), requires_grad=True)
        bound = math.sqrt(3.0 / self.h)
        nn.init.uniform_(self.p_1_w, a=-bound, b=bound)
        # UnfoldH_K5_D3: p_2
        pass
        # Convolution_5x5_1x1_DW1: p_3
        self.p_3 = nn.Conv2d(self.c, self.c, (9, 9), dilation=(1, 1), padding=(4, 4), groups=self.c, bias=False)
        # Convolution_7x1_2x1_DW0: p_4
        self.p_4 = nn.Conv2d(self.c, self.c, (5, 1), dilation=(1, 1), padding=(2, 0), groups=self.c // 4, bias=False)
        # Mix: p_5
        self.p_5_w = nn.Parameter(torch.ones((self.c // 8, self.w, 1, self.c * 3, )), requires_grad=True)
        bound = math.sqrt(3.0 / (self.c * self.w // 8))
        nn.init.uniform_(self.p_5_w, a=-bound, b=bound)
        # Convolution_1x3_1x2_DW1: p_6
        self.p_6 = nn.Conv2d(self.c, self.c, (1, 3), dilation=(1, 1), padding=(0, 1), groups=self.c, bias=False)
        # Group_0_C/G: p_7
        pass
        # BMM_0_0: p_8
        pass
        # Output: p_9
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Mix: p_1
        t_1 = torch.einsum('abcd,ce->abed', [t_0, self.p_1_w]).view(self.n, self.c, self.c // 8, self.w).contiguous()
        # UnfoldH_K5_D3: p_2
        t_2 = F.unfold(t_0, (3, 1), dilation=(1, 1), padding=(1, 0))
        t_2 = t_2.view(self.n, self.c, 3, self.h, self.w)
        # Convolution_5x5_1x1_DW1: p_3
        t_3 = self.p_3(t_1)
        t_3 = t_3.view(self.n, self.c, self.c // 8, self.w)
        # Convolution_7x1_2x1_DW0: p_4
        t_4 = self.p_4(t_3)
        t_4 = t_4.view(self.n, self.c, self.c // 8, self.w)
        # Mix: p_5
        t_5 = torch.einsum('abcd,cdef->abef', [t_4, self.p_5_w]).view(self.n, self.c, self.c * 3).contiguous()
        # Convolution_1x3_1x2_DW1: p_6
        t_6 = t_5.view(self.n, self.c, 1, self.c * 3)
        t_6 = self.p_6(t_6)
        t_6 = t_6.view(self.n, self.c, self.c * 3)
        # BMM_0_0: p_8
        t_8_lhs = t_6.view(self.n, self.c, self.c * 3)
        t_8_rhs = t_2.view(self.n, self.c * 3, self.h * self.w)
        t_8 = torch.bmm(t_8_lhs, t_8_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c * 3)
        # Output: p_9
        return t_8.view(self.n, self.c, self.h, self.w)