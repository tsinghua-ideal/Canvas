import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_7187357794635299218(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_7187357794635299218, self).__init__()
        self.g = 8
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Mix: p_2
        self.p_2 = nn.Conv2d(self.c, self.c, (3, 3), dilation=(1, 1), padding=(1, 1), groups=self.c, bias=False)
        # BMM_1_0: p_3
        pass
        # Fold_1/0/H_1/1/W_Avg: p_4
        pass
        # Convolution_1x7_1x1_DW0: p_5
        self.p_5 = nn.Conv2d(self.c, self.c, (3, 3), dilation=(1, 1), padding=(1, 1), groups=1, bias=False)
        # BMM_0_0: p_6
        pass
        # BSub: p_7
        pass
        # BMax: p_8
        pass
        # Output: p_9
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Mix: p_2
        t_2 = self.p_2(t_0)
        # BMM_1_0: p_3
        t_2_lhs = t_2.view(self.n, self.c, self.h * self.w).transpose(1, 2)        
        t_0_rhs = t_0.view(self.n, self.c, self.h * self.w)        
        t_3 = torch.bmm(t_2_lhs, t_0_rhs).view(self.n, self.h, self.w, self.h, self.w) / math.sqrt(self.c)
        # Fold_1/0/H_1/1/W_Avg: p_4
        t_4 = t_2.mean(2).mean(2)
        # Convolution_1x7_1x1_DW0: p_5
        t_5 = self.p_5(t_2)
        t_5 = t_5.view(self.n, self.c, self.h, self.w)
        # BMM_0_0: p_6
        t_0_lhs = t_0.view(self.n, self.c, self.h * self.w)        
        t_3_rhs = t_3.view(self.n, self.h * self.w, self.h * self.w)        
        t_6 = torch.bmm(t_0_lhs, t_3_rhs).view(self.n, self.c, self.h, self.w) / math.sqrt(self.h * self.w)
        # BSub: p_7
        t_7_lhs = t_6.view(self.n, 1, self.c, self.h, self.w)
        t_7_rhs = t_5.view(self.n, 1, self.c, self.h, self.w)
        t_7 = t_7_lhs - t_7_rhs
        t_7 = t_7.view(self.n, self.c, self.h, self.w)
        # BMax: p_8
        t_8_lhs = t_4.view(self.n, 1, self.c)
        t_8_rhs = t_7.view(self.n, self.h * self.w, self.c)
        t_8 = torch.maximum(t_8_lhs, t_8_rhs)
        t_8 = t_8.view(self.n, self.c, self.h, self.w)
        # Output: p_9
        return t_8.view(self.n, self.c, self.h, self.w)