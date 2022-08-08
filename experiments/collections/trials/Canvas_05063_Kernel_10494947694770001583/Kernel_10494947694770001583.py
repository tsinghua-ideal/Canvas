import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_10494947694770001583(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_10494947694770001583, self).__init__()
        self.g = 16
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Fold_1/1/W_Max: p_1
        pass
        # UnfoldHW_K5_D3: p_2
        pass
        # ReLU: p_3
        pass
        # TanH: p_4
        pass
        # Shift_1/0/H_K3: p_5
        self.p_5_1_0 = random.randint(-3, 3)
        # Mix: p_6
        self.p_6_w = nn.Parameter(torch.ones((self.c, 1, )), requires_grad=True)
        nn.init.trunc_normal_(self.p_6_w, std=.1)
        # FC: p_7
        self.p_7 = nn.Conv2d(self.c * 25, self.c, 1, padding=0, groups=1, bias=False)
        # BMul: p_8
        pass
        # Softmax_1/1/W: p_9
        pass
        # BMul: p_10
        pass
        # BAdd: p_11
        pass
        # BSub: p_12
        pass
        # Output: p_13
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Fold_1/1/W_Max: p_1
        t_1 = t_0.max(3)[0]
        # UnfoldHW_K5_D3: p_2
        t_2 = F.unfold(t_0, (5, 5), dilation=(3, 3), padding=(6, 6))
        t_2 = t_2.view(self.n, self.c, 5, 5, self.h, self.w)
        # ReLU: p_3
        t_3 = torch.relu(t_0)
        # TanH: p_4
        t_4 = torch.tanh(t_3)
        # Shift_1/0/H_K3: p_5
        t_5 = torch.roll(t_0, self.p_5_1_0, 2)
        # Mix: p_6
        t_6 = torch.einsum('abcd,be->aecd', [t_4, self.p_6_w]).view(self.n, self.h, self.w).contiguous()
        # FC: p_7
        t_7 = t_2.view(self.n, self.c * 25, self.h, self.w)
        t_7 = self.p_7(t_7)
        t_7 = t_7.view(self.n, self.c, self.h, self.w)
        # BMul: p_8
        t_8 = t_0 * t_7
        # Softmax_1/1/W: p_9
        t_9 = F.softmax(t_6, dim=2)
        # BMul: p_10
        t_10_lhs = t_9.view(self.n, 1, self.h, self.w)
        t_10_rhs = t_5.view(self.n, self.c, self.h, self.w)
        t_10 = t_10_lhs * t_10_rhs
        t_10 = t_10.view(self.n, self.c, self.h, self.w)
        # BAdd: p_11
        t_11 = t_10 + t_8
        # BSub: p_12
        t_12_lhs = t_1.view(self.n, self.c, self.h, 1)
        t_12_rhs = t_11.view(self.n, self.c, self.h, self.w)
        t_12 = t_12_lhs - t_12_rhs
        t_12 = t_12.view(self.n, self.c, self.h, self.w)
        # Output: p_13
        return t_12.view(self.n, self.c, self.h, self.w)