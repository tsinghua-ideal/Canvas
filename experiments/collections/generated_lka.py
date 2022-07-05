import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_17579745839098906056(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_17579745839098906056, self).__init__()
        self.g = 1
        self.n, self.c, self.h, self.w = None, c, h, w

        # Kernels
        # Input: p_0
        pass
        # FC: p_1
        self.p_1 = nn.Conv2d(self.c, self.c, 1, padding=0, groups=1, bias=False)
        # GeLU: p_2
        pass
        # Convolution_5_5_1_1: p_3
        self.p_3 = nn.Conv2d(self.c, self.c, (5, 5), dilation=(1, 1), padding=(2, 2), groups=self.c, bias=False)
        # Convolution_7_7_3_3: p_4
        self.p_4 = nn.Conv2d(self.c, self.c, (7, 7), dilation=(3, 3), padding=(9, 9), groups=self.c, bias=False)
        # FC: p_5
        self.p_5 = nn.Conv2d(self.c, self.c, 1, padding=0, groups=1, bias=False)
        # BMul: p_6
        pass
        # FC: p_7
        self.p_7 = nn.Conv2d(self.c, self.c, 1, padding=0, groups=1, bias=False)
        # BAdd: p_8
        pass
        # Output: p_9
        pass

    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # FC: p_1
        t_1 = self.p_1(t_0)
        t_1 = t_1.view(self.n, self.c, self.h, self.w)
        # GeLU: p_2
        t_2 = F.gelu(t_1)
        # Convolution_5_5_1_1: p_3
        t_3 = self.p_3(t_2)
        t_3 = t_3.view(self.n, self.c, self.h, self.w)
        # Convolution_7_7_3_3: p_4
        t_4 = self.p_4(t_3)
        t_4 = t_4.view(self.n, self.c, self.h, self.w)
        # FC: p_5
        t_5 = self.p_5(t_4)
        t_5 = t_5.view(self.n, self.c, self.h, self.w)
        # BMul: p_6
        t_6 = t_5 * t_2
        # FC: p_7
        t_7 = self.p_7(t_6)
        t_7 = t_7.view(self.n, self.c, self.h, self.w)
        # BAdd: p_8
        t_8 = t_0 + t_7
        # Output: p_9
        return t_8.view(self.n, self.c, self.h, self.w)
