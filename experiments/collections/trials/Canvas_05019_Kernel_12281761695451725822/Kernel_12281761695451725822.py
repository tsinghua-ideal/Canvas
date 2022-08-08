import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_12281761695451725822(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_12281761695451725822, self).__init__()
        self.g = 8
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldW_K3_D2: p_1
        pass
        # Group_0_C: p_2
        pass
        # Convolution_1x5_1x1_DW0: p_3
        self.p_3 = nn.Conv2d(self.c, self.c, (1, 5), dilation=(1, 1), padding=(0, 2), groups=1, bias=False)
        # UnfoldH_K3_D1: p_4
        pass
        # Shift_1/1/W_K2: p_5
        self.p_5_1_1 = random.randint(-2, 2)
        # BMax: p_6
        pass
        # FC: p_7
        self.p_7 = nn.Conv2d(self.c * 9, self.c, 1, padding=0, groups=self.c, bias=False)
        # BMul: p_8
        pass
        # Output: p_9
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldW_K3_D2: p_1
        t_1 = F.unfold(t_0, (1, 3), dilation=(1, 2), padding=(0, 2))
        t_1 = t_1.view(self.n, self.c, 3, self.h, self.w)
        # Group_0_C: p_2
        t_2 = t_1
        # Convolution_1x5_1x1_DW0: p_3
        t_3 = self.p_3(t_0)
        t_3 = t_3.view(self.n, self.c, self.h, self.w)
        # UnfoldH_K3_D1: p_4
        t_4 = t_2.view(self.n, self.c * 3, self.h, self.w)
        t_4 = F.unfold(t_4, (3, 1), dilation=(1, 1), padding=(1, 0))
        t_4 = t_4.view(self.n, self.c, 3, 3, self.h, self.w)
        # Shift_1/1/W_K2: p_5
        t_5 = torch.roll(t_4, self.p_5_1_1, 5)
        # BMax: p_6
        t_6_lhs = t_1.view(self.n, self.c, 1, 3, self.h, self.w)
        t_6_rhs = t_5.view(self.n, self.c, 3, 3, self.h, self.w)
        t_6 = torch.maximum(t_6_lhs, t_6_rhs)
        t_6 = t_6.view(self.n, self.c, 3, 3, self.h, self.w)
        # FC: p_7
        t_7 = t_6.view(self.n, self.c * 9, self.h, self.w)
        t_7 = self.p_7(t_7)
        t_7 = t_7.view(self.n, self.c, self.h, self.w)
        # BMul: p_8
        t_8 = t_7 * t_3
        # Output: p_9
        return t_8.view(self.n, self.c, self.h, self.w)