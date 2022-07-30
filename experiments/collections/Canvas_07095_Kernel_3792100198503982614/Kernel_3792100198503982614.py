import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_3792100198503982614(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_3792100198503982614, self).__init__()
        self.g = 2
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Scale_0/1/C_1/0/H_1/1/W: p_1
        self.p_1_w = nn.Parameter(torch.ones((1, self.c, self.h, self.w,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_1_w, std=.02)
        # Mix: p_2
        self.p_2_w = nn.Parameter(torch.ones((self.h, self.w, self.c // self.g, 1, )), requires_grad=True)
        bound = math.sqrt(3.0 / (self.h * self.w))
        nn.init.uniform_(self.p_2_w, a=-bound, b=bound)
        # BMM_0_1: p_3
        pass
        # Convolution_3x3_3x3_DW0: p_4
        self.p_4 = nn.Conv2d(self.c, self.c, (3, 3), dilation=(3, 3), padding=(3, 3), groups=self.c // 4, bias=False)
        # FC: p_5
        self.p_5 = nn.Conv2d(self.c, self.c, 1, padding=0, groups=1, bias=False)
        # UnfoldHW_K3_D3: p_6
        pass
        # BSub: p_7
        pass
        # FC: p_8
        self.p_8 = nn.Conv2d(self.c * 9, self.c, 1, padding=0, groups=self.c // 4, bias=False)
        # BMax: p_9
        pass
        # BMM_0_0: p_10
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
        # Mix: p_2
        t_2 = torch.einsum('abcd,cdef->abef', [t_1, self.p_2_w]).view(self.n, self.c, self.c // self.g).contiguous()
        # BMM_0_1: p_3
        t_1_lhs = t_1.view(self.n, self.c, self.h * self.w)        
        t_0_rhs = t_0.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_3 = torch.bmm(t_1_lhs, t_0_rhs).view(self.n, self.c, self.c) / math.sqrt(self.h * self.w)
        # Convolution_3x3_3x3_DW0: p_4
        t_4 = t_2.view(self.n, self.c, self.c // self.g, 1)
        t_4 = self.p_4(t_4)
        t_4 = t_4.view(self.n, self.c, self.c // self.g)
        # FC: p_5
        t_5 = self.p_5(t_0)
        t_5 = t_5.view(self.n, self.c, self.h, self.w)
        # UnfoldHW_K3_D3: p_6
        t_6 = F.unfold(t_0, (3, 3), dilation=(3, 3), padding=(3, 3))
        t_6 = t_6.view(self.n, self.c, 3, 3, self.h, self.w)
        # BSub: p_7
        t_7_lhs = t_4.view(self.n, self.c, 1, self.c // self.g)
        t_7_rhs = t_3.view(self.n, self.c, self.g, self.c // self.g)
        t_7 = t_7_lhs - t_7_rhs
        t_7 = t_7.view(self.n, self.c, self.c)
        # FC: p_8
        t_8 = t_6.view(self.n, self.c * 9, self.h, self.w)
        t_8 = self.p_8(t_8)
        t_8 = t_8.view(self.n, self.c, self.h, self.w)
        # BMax: p_9
        t_9_lhs = t_5.view(self.n, 1, self.c, self.h, self.w)
        t_9_rhs = t_8.view(self.n, 1, self.c, self.h, self.w)
        t_9 = torch.maximum(t_9_lhs, t_9_rhs)
        t_9 = t_9.view(self.n, self.c, self.h, self.w)
        # BMM_0_0: p_10
        t_7_lhs = t_7.view(self.n, self.c, self.c)        
        t_9_rhs = t_9.view(self.n, self.c, self.h * self.w)        
        t_10 = torch.bmm(t_7_lhs, t_9_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c)
        # Output: p_11
        return t_10.view(self.n, self.c, self.h, self.w)