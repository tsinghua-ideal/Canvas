import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Kernel_7886596238811945167(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(Kernel_7886596238811945167, self).__init__()
        self.n, self.c, self.h, self.w = None, c, h, w

        # Kernels
        self.proj = nn.Sequential(
            nn.Conv2d(c, c, 5, padding=2, groups=c),
            nn.Conv2d(c, c, 7, padding=9, groups=c, dilation=3),
            nn.Conv2d(c, c, 1)
        )
        # FC: p_3
        self.p_3 = nn.Conv2d(self.c, self.c, 1)
        # Scale_0/1/C_0/3/KW_1/1/C: p_7
        self.p_7_w = nn.Parameter(torch.ones((1, self.c, self.c,)), requires_grad=True)
        nn.init.trunc_normal_(self.p_7_w, std=.02)

    def forward(self, x: torch.Tensor):
        # Input: p_0
        t_0 = self.proj(x)
        self.n = t_0.size(0)
        # FC: p_3
        t_3 = self.p_3(t_0)
        # Sin: p_4
        t_4 = torch.sin(t_0 * (math.pi / 2))
        # BMM_0_1: p_5
        t_5_lhs = t_0.view(self.n, self.c, self.h * self.w)
        t_5_rhs = t_0.view(self.n, self.c, self.h * self.w).transpose(1, 2)
        t_5 = torch.bmm(t_5_lhs, t_5_rhs).view(self.n, self.c, self.c) / math.sqrt(self.h * self.w)
        # BMax: p_6
        t_6 = torch.maximum(t_3, t_4)
        # Scale_0/1/C_0/3/KW_1/1/C: p_7
        t_7 = self.p_7_w * t_5
        # BMM_1_0: p_8
        t_6_lhs = t_6.view(self.n, self.c, self.h * self.w).transpose(1, 2)
        t_7_rhs = t_7.view(self.n, self.c, self.c)
        t_8 = torch.bmm(t_6_lhs, t_7_rhs).view(self.n, self.h, self.w, self.c)
        # Output: p_9
        return t_8.permute(0, 3, 1, 2).contiguous().view(self.n, self.c, self.h, self.w)
