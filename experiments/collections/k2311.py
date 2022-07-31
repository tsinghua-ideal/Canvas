import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class K2311(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(K2311, self).__init__()
        self.n, self.c, self.h, self.w = None, c, h, w
        self.g, self.kh, self.kw = 4, 3, 3
        self.ph, self.pw = (self.kh - 1) // 2, (self.kw - 1) // 2
        assert self.c % self.g == 0

        self.proj = nn.Conv2d(self.c, self.c, kernel_size=(self.kh, self.kw), padding=(self.ph, self.pw),
                              groups=self.g, bias=False)
        self.scale = nn.Parameter(torch.ones((1, self.g, self.c // self.g * self.kh * self.kw * self.c // self.g,)),
                                  requires_grad=True)
        nn.init.trunc_normal_(self.scale, std=.02)

    def forward(self, x: torch.Tensor):
        self.n = x.size(0)
        unfolded = F.unfold(x, kernel_size=(self.kh, self.kw), padding=(self.ph, self.pw)) \
            .view(self.n * self.g, self.c // self.g * self.kh * self.kw, self.h * self.w)
        attn_rhs = x.view(self.n * self.g, self.c // self.g, self.h * self.w).transpose(1, 2)
        attn = torch.bmm(unfolded, attn_rhs) / math.sqrt(self.h * self.w)
        attn = (attn.view(self.n, self.g, -1) * self.scale) \
            .view(self.n * self.g, self.c // self.g * self.kh * self.kw, self.c // self.g)
        sin_unfold = F.unfold(torch.sin(x), kernel_size=(self.kh, self.kw), padding=(self.ph, self.pw)) \
            .view(self.n * self.g, self.c // self.g, self.kh * self.kw, self.h * self.w)
        proj_x = self.proj(x).view(self.n * self.g, self.c // self.g, 1, self.h * self.w)
        proj_x = torch.maximum(proj_x, sin_unfold) \
            .view(self.n * self.g, self.c // self.g * self.kh * self.kw, self.h * self.w)
        # attn: (ng, ckk/g, c/g), proj_x: (ng, ckk/g, hw)
        return torch.bmm(attn.transpose(1, 2), proj_x).view(self.n, self.c, self.h, self.w) \
            / math.sqrt(self.c // self.g * self.kh * self.kw)
