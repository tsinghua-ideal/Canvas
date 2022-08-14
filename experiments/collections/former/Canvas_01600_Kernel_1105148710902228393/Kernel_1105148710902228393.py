import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_1105148710902228393(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_1105148710902228393, self).__init__()
        self.g = 4
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # UnfoldH_K3_D2: p_1
        pass
        # Shift_1/0/H_1/1/W_K2: p_2
        self.p_2_1_0 = random.randint(-2, 2)
        self.p_2_1_1 = random.randint(-2, 2)
        # Convolution_1x7_1x3_DW0: p_3
        self.p_3 = nn.Conv2d(self.c, self.c, (1, 7), dilation=(1, 3), padding=(0, 9), groups=1, bias=False)
        # Mix: p_4
        self.p_4_w = nn.Parameter(torch.ones((3, 1, )), requires_grad=True)
        bound = math.sqrt(3.0 / (3))
        nn.init.uniform_(self.p_4_w, a=-bound, b=bound)
        # BAdd: p_5
        pass
        # BMul: p_6
        pass
        # Output: p_7
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # UnfoldH_K3_D2: p_1
        t_1 = F.unfold(t_0, (3, 1), dilation=(2, 1), padding=(2, 0))
        t_1 = t_1.view(self.n, self.c, 3, self.h, self.w)
        # Shift_1/0/H_1/1/W_K2: p_2
        t_2 = torch.roll(t_1, self.p_2_1_0, 3)
        t_2 = torch.roll(t_2, self.p_2_1_1, 4)
        # Convolution_1x7_1x3_DW0: p_3
        t_3 = self.p_3(t_0)
        t_3 = t_3.view(self.n, self.c, self.h, self.w)
        # Mix: p_4
        t_4 = torch.einsum('abcde,cf->abfde', [t_2, self.p_4_w]).view(self.n, self.c, self.h, self.w).contiguous()
        # BAdd: p_5
        t_5 = t_0 + t_3
        # BMul: p_6
        t_6 = t_5 * t_4
        # Output: p_7
        return t_6.view(self.n, self.c, self.h, self.w)