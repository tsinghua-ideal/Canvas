import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_14952580559822654753(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_14952580559822654753, self).__init__()
        self.g = 4
        self.n, self.c, self.h, self.w = 0, c, h, w

        # Kernels
        # Input: p_0
        pass
        # Sigmoid: p_1
        pass
        # FC: p_2
        self.p_2 = nn.Conv2d(self.c, self.g * self.c, 1, padding=0, groups=1, bias=False)
        # ReLU: p_3
        pass
        # GroupByFactor_0: p_4
        pass
        # ReLU: p_5
        pass
        # FC: p_6
        self.p_6 = nn.Conv2d(self.c, self.c // 2, 1, padding=0, groups=1, bias=False)
        # Shift_0/1/C_K1: p_7
        self.p_7_0_1 = random.randint(-1, 1)
        # Fold_0/1/C_Max: p_8
        pass
        # Shift_1/1/W_K3: p_9
        self.p_9_1_1 = random.randint(-3, 3)
        # UnfoldW_K7_D3: p_10
        pass
        # GroupAll_0: p_11
        pass
        # Fold_0/0/G_Avg: p_12
        pass
        # Sin: p_13
        pass
        # Fold_1/1/W_Max: p_14
        pass
        # Convolution_x_3_5_3_2_3: p_15
        self.p_15 = nn.Conv2d(self.c, self.c, (5, 3), dilation=(2, 3), padding=(4, 3), groups=self.c, bias=False)
        # Sigmoid: p_16
        pass
        # Shift_1/0/H_K2: p_17
        self.p_17_1_0 = random.randint(-2, 2)
        # GeLU: p_18
        pass
        # FC: p_19
        self.p_19 = nn.Conv2d(self.c, self.c, 1, padding=0, groups=1, bias=False)
        # BAdd: p_20
        pass
        # Output: p_21
        pass

    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = x
        self.n = t_0.size(0)
        # Sigmoid: p_1
        t_1 = torch.sigmoid(t_0)
        # FC: p_2
        t_2 = self.p_2(t_0)
        t_2 = t_2.view(self.n, self.g * self.c, self.h, self.w)
        # GroupByFactor_0: p_4
        t_4 = t_2.view(self.n, self.g, self.c, self.h, self.w)
        # ReLU: p_5
        t_5 = torch.relu(t_4)
        # FC: p_6
        t_6 = self.p_6(t_1)
        t_6 = t_6.view(self.n, self.c // 2, self.h, self.w)
        # Shift_0/1/C_K1: p_7
        t_7 = torch.roll(t_6, self.p_7_0_1, 1)
        # Fold_0/1/C_Max: p_8
        t_8 = t_7.max(1)[0]
        # Shift_1/1/W_K3: p_9
        t_9 = torch.roll(t_8, self.p_9_1_1, 2)
        # UnfoldW_K7_D3: p_10
        t_10 = t_9.view(self.n, 1, self.h, self.w)
        t_10 = F.unfold(t_10, (1, 7), dilation=(1, 3), padding=(0, 9))
        t_10 = t_10.view(self.n, 7, self.h, self.w)
        # GroupAll_0: p_11
        t_11 = t_10
        # Fold_0/0/G_Avg: p_12
        t_12 = t_5.mean(1)
        # Sin: p_13
        t_13 = torch.sin(t_11 * 1.5707963)
        # Fold_1/1/W_Max: p_14
        t_14 = t_13.max(3)[0]
        # Convolution_x_3_5_3_2_3: p_15
        t_15 = self.p_15(t_12)
        t_15 = t_15.view(self.n, self.c, self.h, self.w)
        # Sigmoid: p_16
        t_16 = torch.sigmoid(t_14)
        # Shift_1/0/H_K2: p_17
        t_17 = torch.roll(t_15, self.p_17_1_0, 2)
        # GeLU: p_18
        t_18 = F.gelu(t_17)
        # FC: p_19
        t_19 = self.p_19(t_18)
        t_19 = t_19.view(self.n, self.c, self.h, self.w)
        # BAdd: p_20
        t_20_lhs = t_16.view(self.n, 1, self.h * 7)
        t_20_rhs = t_19.view(self.n, self.c * self.w // 7, self.h * 7)
        t_20 = t_20_lhs + t_20_rhs
        t_20 = t_20.view(self.n, self.c, self.h, self.w)
        # Output: p_21
        return t_20.view(self.n, self.c, self.h, self.w)
