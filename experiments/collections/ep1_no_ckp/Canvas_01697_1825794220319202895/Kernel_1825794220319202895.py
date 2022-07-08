import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_13538758526885243862(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_13538758526885243862, self).__init__()
        self.g = 32
        self.n, self.c, self.h, self.w = None, c, h, w
        
        # Kernels
        # Input: p_0
        pass
        # Convolution_1_3_1_3: p_1
        self.p_1 = nn.Conv2d(self.c, self.c // 8, (1, 3), dilation=(1, 3), padding=(0, 3), groups=1, bias=False)
        # Scale_0/1/C: p_2
        self.p_2_w = nn.Parameter(torch.ones((1, self.c // 8, 1, 1,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_2_w, std=.02)
        # Convolution_3_1_2_1: p_3
        self.p_3 = nn.Conv2d(self.c // 8, self.c, (3, 1), dilation=(2, 1), padding=(2, 0), groups=1, bias=False)
        # Convolution_1_3_1_3: p_4
        self.p_4 = nn.Conv2d(self.c, 1, (1, 3), dilation=(1, 3), padding=(0, 3), groups=1, bias=False)
        # Convolution_3_1_3_1: p_5
        self.p_5 = nn.Conv2d(self.c // 8, self.c // 8, (3, 1), dilation=(3, 1), padding=(3, 0), groups=self.c // 8, bias=False)
        # UnfoldHW_K7_D3: p_6
        pass
        # Fold_0/2/KH_Max: p_7
        pass
        # Convolution_1_7_1_2: p_8
        self.p_8 = nn.Conv2d(self.c // 8, self.c // 8, (1, 7), dilation=(1, 2), padding=(0, 6), groups=1, bias=False)
        # ReLU: p_9
        pass
        # UnfoldH_K3_D1: p_10
        pass
        # Fold_0/2/KH_0/3/KW_Avg: p_11
        pass
        # Convolution_1_7_1_1: p_12
        self.p_12 = nn.Conv2d(self.c // 8, self.c, (1, 7), dilation=(1, 1), padding=(0, 3), groups=1, bias=False)
        # UnfoldH_K3_D2: p_13
        pass
        # Convolution_3_3_1_1: p_14
        self.p_14 = nn.Conv2d(1, 1, (3, 3), dilation=(1, 1), padding=(1, 1), groups=1, bias=False)
        # FC: p_15
        self.p_15 = nn.Conv2d(self.c * 3, self.c, 1, padding=0, groups=1, bias=False)
        # Convolution_1_3_1_3: p_16
        self.p_16 = nn.Conv2d(self.c, self.c, (1, 3), dilation=(1, 3), padding=(0, 3), groups=self.c, bias=False)
        # BAdd: p_17
        pass
        # Output: p_18
        pass
    
    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        assert (self.n, self.c, self.h, self.w) == tuple(t_0.size())
        # Convolution_1_3_1_3: p_1
        t_1 = self.p_1(t_0)
        t_1 = t_1.view(self.n, self.c // 8, self.h, self.w)
        # Scale_0/1/C: p_2
        t_2 = self.p_2_w * t_1
        # Convolution_3_1_2_1: p_3
        t_3 = self.p_3(t_1)
        t_3 = t_3.view(self.n, self.c, self.h, self.w)
        # Convolution_1_3_1_3: p_4
        t_4 = self.p_4(t_3)
        t_4 = t_4.view(self.n, self.h, self.w)
        # Convolution_3_1_3_1: p_5
        t_5 = self.p_5(t_2)
        t_5 = t_5.view(self.n, self.c // 8, self.h, self.w)
        # UnfoldHW_K7_D3: p_6
        t_6 = t_4.view(self.n, 1, self.h, self.w)
        t_6 = F.unfold(t_6, (7, 7), dilation=(3, 3), padding=(9, 9))
        t_6 = t_6.view(self.n, 7, 7, self.h, self.w)
        # Fold_0/2/KH_Max: p_7
        t_7 = t_6.max(1)[0]
        # Convolution_1_7_1_2: p_8
        t_8 = self.p_8(t_5)
        t_8 = t_8.view(self.n, self.c // 8, self.h, self.w)
        # ReLU: p_9
        t_9 = torch.relu(t_7)
        # UnfoldH_K3_D1: p_10
        t_10 = F.unfold(t_9, (3, 1), dilation=(1, 1), padding=(1, 0))
        t_10 = t_10.view(self.n, 3, 7, self.h, self.w)
        # Fold_0/2/KH_0/3/KW_Avg: p_11
        t_11 = t_10.mean(1).mean(1)
        # Convolution_1_7_1_1: p_12
        t_12 = self.p_12(t_8)
        t_12 = t_12.view(self.n, self.c, self.h, self.w)
        # UnfoldH_K3_D2: p_13
        t_13 = F.unfold(t_12, (3, 1), dilation=(2, 1), padding=(2, 0))
        t_13 = t_13.view(self.n, self.c, 3, self.h, self.w)
        # Convolution_3_3_1_1: p_14
        t_14 = t_11.view(self.n, 1, self.h, self.w)
        t_14 = self.p_14(t_14)
        t_14 = t_14.view(self.n, self.h, self.w)
        # FC: p_15
        t_15 = t_13.view(self.n, self.c * 3, self.h, self.w)
        t_15 = self.p_15(t_15)
        t_15 = t_15.view(self.n, self.c, self.h, self.w)
        # Convolution_1_3_1_3: p_16
        t_16 = self.p_16(t_15)
        t_16 = t_16.view(self.n, self.c, self.h, self.w)
        # BAdd: p_17
        t_17_lhs = t_14.view(self.n, 1, self.h, self.w)
        t_17_rhs = t_16.view(self.n, self.c, self.h, self.w)
        t_17 = t_17_lhs + t_17_rhs
        t_17 = t_17.view(self.n, self.c, self.h, self.w)
        # Output: p_18
        return t_17.view(self.n, self.c, self.h, self.w)