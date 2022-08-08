import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_17611448876263927411(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_17611448876263927411, self).__init__()
        self.g = 2
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Mix: p_1
        self.p_1_w = nn.Parameter(torch.ones((self.h, self.w, self.c, 1, )), requires_grad=True)
        nn.init.trunc_normal_(self.p_1_w, std=.1)
        # FC: p_3
        self.p_3 = nn.Conv2d(self.c, self.c, 3, padding=1, groups=2, bias=False)
        # Scale_0/1/C_1/0/H: p_4
        self.p_4_w = nn.Parameter(torch.ones((1, self.c, self.c,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_4_w, std=.1)
        # Mix: p_5
        self.p_5_w = nn.Parameter(torch.ones((self.c, 1, )), requires_grad=True)
        nn.init.trunc_normal_(self.p_5_w, std=.1)
        # BMM_0_0: p_6
        pass
        # BMM_1_0: p_7
        pass
        # BAdd: p_8
        pass
        # Output: p_9
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Mix: p_1
        t_1 = torch.einsum('abcd,cdef->abef', [t_0, self.p_1_w]).view(self.n, self.c, self.c).contiguous()
        # FC: p_3
        t_3 = self.p_3(t_0)
        # Scale_0/1/C_1/0/H: p_4
        t_4 = self.p_4_w * t_1
        # Mix: p_5
        t_5 = torch.einsum('abc,cd->abd', [t_4, self.p_5_w]).view(self.n, self.c).contiguous()
        # BMM_0_0: p_6
        t_4_lhs = t_4.view(self.n, self.c, self.c)        
        t_3_rhs = t_3.view(self.n, self.c, self.h * self.w)        
        t_6 = torch.bmm(t_4_lhs, t_3_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.c)
        # BMM_1_0: p_7
        t_5_lhs = t_5.view(self.n, self.c, 1).transpose(1, 2)        
        t_0_rhs = t_0.view(self.n, self.c, self.h * self.w)        
        t_7 = torch.bmm(t_5_lhs, t_0_rhs).view(self.n, self.h, self.w) / math.sqrt(self.c)
        # BAdd: p_8
        t_8_lhs = t_7.view(self.n, 1, self.h, self.w)
        t_8_rhs = t_6.view(self.n, self.c, self.h, self.w)
        t_8 = t_8_lhs + t_8_rhs
        t_8 = t_8.view(self.n, self.c, self.h, self.w)
        # Output: p_9
        return t_8.view(self.n, self.c, self.h, self.w)