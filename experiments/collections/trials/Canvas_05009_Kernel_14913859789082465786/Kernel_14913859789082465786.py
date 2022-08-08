import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_14913859789082465786(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_14913859789082465786, self).__init__()
        self.g = 2
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldW_K7_D1: p_1
        pass
        # Scale_0/1/C_1/0/H: p_2
        self.p_2_w = nn.Parameter(torch.ones((1, self.c, self.h, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_2_w, std=.02)
        # Convolution_1x5_1x1_DW0: p_3
        self.p_3 = nn.Conv2d(self.c, self.c, (1, 5), dilation=(1, 1), padding=(0, 2), groups=1, bias=False)
        # Scale_0/1/C_0/3/KW: p_4
        self.p_4_w = nn.Parameter(torch.ones((1, self.c, 7, 1, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_4_w, std=.02)
        # ReLU: p_5
        pass
        # Mix: p_6
        self.p_6_w = nn.Parameter(torch.ones((7, 1, )), requires_grad=True)
        bound = 7
        nn.init.uniform_(self.p_6_w, a=-bound, b=bound)
        # BMM_0_1: p_7
        pass
        # BMM_1_0: p_8
        pass
        # Output: p_9
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldW_K7_D1: p_1
        t_1 = F.unfold(t_0, (1, 7), dilation=(1, 1), padding=(0, 3))
        t_1 = t_1.view(self.n, self.c, 7, self.h, self.w)
        # Scale_0/1/C_1/0/H: p_2
        t_2 = self.p_2_w * t_0
        # Convolution_1x5_1x1_DW0: p_3
        t_3 = self.p_3(t_2)
        t_3 = t_3.view(self.n, self.c, self.h, self.w)
        # Scale_0/1/C_0/3/KW: p_4
        t_4 = self.p_4_w * t_1
        # ReLU: p_5
        t_5 = torch.relu(t_0)
        # Mix: p_6
        t_6 = torch.einsum('abcde,cf->abfde', [t_4, self.p_6_w]).view(self.n, self.c, self.h, self.w).contiguous()
        # BMM_0_1: p_7
        t_3_lhs = t_3.view(self.n, self.c, self.h * self.w)        
        t_6_rhs = t_6.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_7 = torch.bmm(t_3_lhs, t_6_rhs).view(self.n, self.c, self.c) / math.sqrt(self.h * self.w)
        # BMM_1_0: p_8
        t_5_lhs = t_5.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_7_rhs = t_7.view(self.n, self.c, self.c)        
        t_8 = torch.bmm(t_5_lhs, t_7_rhs).view(self.n, self.h, self.w, self.c) / math.sqrt(self.c)
        # Output: p_9
        return t_8.permute(0, 3, 1, 2).contiguous().view(self.n, self.c, self.h, self.w)