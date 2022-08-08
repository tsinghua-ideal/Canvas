import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_13601609615069579155(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_13601609615069579155, self).__init__()
        self.g = 1
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # BMM_0_1: p_1
        pass
        # Scale_1/0/H: p_2
        self.p_2_w = nn.Parameter(torch.ones((1, 1, self.h, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_2_w, std=.02)
        # Convolution_3x3_2x2_DW1: p_3
        self.p_3 = nn.Conv2d(self.c, self.c, (3, 3), dilation=(2, 2), padding=(2, 2), groups=self.c, bias=False)
        # Scale_1/1/C: p_4
        self.p_4_w = nn.Parameter(torch.ones((1, 1, self.c,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_4_w, std=.02)
        # Fold_0/1/C_Max: p_5
        pass
        # Convolution_3x3_1x1_DW1: p_7
        self.p_7 = nn.Conv2d(1, self.c, (3, 3), dilation=(1, 1), padding=(1, 1), groups=1, bias=False)
        # Mix: p_9
        self.p_9_w = nn.Parameter(torch.ones((self.c, 1, )), requires_grad=True)
        bound = math.sqrt(3.0 / (self.c))
        nn.init.uniform_(self.p_9_w, a=-bound, b=bound)
        # FC: p_10
        self.p_10 = nn.Conv2d(self.c, self.c, 3, padding=1, groups=1, bias=False)
        # BMM_1_1: p_11
        pass
        # BMax: p_12
        pass
        # Output: p_13
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # BMM_0_1: p_1
        t_1_lhs = t_0.view(self.n, self.c, self.h * self.w)        
        t_1_rhs = t_0.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_1 = torch.bmm(t_1_lhs, t_1_rhs) / math.sqrt(self.h * self.w)
        t_1 = t_1.view(self.n, self.c, self.c)
        # Scale_1/0/H: p_2
        t_2 = self.p_2_w * t_0
        # Convolution_3x3_2x2_DW1: p_3
        t_3 = self.p_3(t_2)
        t_3 = t_3.view(self.n, self.c, self.h, self.w)
        # Scale_1/1/C: p_4
        t_4 = self.p_4_w * t_1
        # Fold_0/1/C_Max: p_5
        t_5 = t_3.max(1)[0]
        # Convolution_3x3_1x1_DW1: p_7
        t_7 = t_5.view(self.n, 1, self.h, self.w)
        t_7 = self.p_7(t_7)
        t_7 = t_7.view(self.n, self.c, self.h, self.w)
        # Mix: p_9
        t_9 = torch.einsum('abcd,be->aecd', [t_7, self.p_9_w]).view(self.n, self.h, self.w).contiguous()
        # FC: p_10
        t_10 = self.p_10(t_0)
        t_10 = t_10.view(self.n, self.c, self.h, self.w)
        # BMM_1_1: p_11
        t_11_lhs = t_10.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_11_rhs = t_4.view(self.n, self.c, self.c).transpose(1, 2)        
        t_11 = torch.bmm(t_11_lhs, t_11_rhs) / math.sqrt(self.c)
        t_11 = t_11.view(self.n, self.h, self.w, self.c)
        # BMax: p_12
        t_12_lhs = t_9.view(self.n, self.h, self.w, 1)
        t_12_rhs = t_11.view(self.n, self.h, self.w, self.c)
        t_12 = torch.maximum(t_12_lhs, t_12_rhs)
        t_12 = t_12.view(self.n, self.h, self.w, self.c)
        # Output: p_13
        return t_12.permute(0, 3, 1, 2).contiguous().view(self.n, self.c, self.h, self.w)