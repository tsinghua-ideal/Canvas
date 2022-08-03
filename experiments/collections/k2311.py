import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial


class K2311(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        # Configurations
        super(K2311, self).__init__()
        self.n, self.c, self.h, self.w = None, c, h, w
        self.g, self.kh, self.kw, self.dh, self.dw = 16, 5, 5, 3, 3
        self.gkh, self.gkw = max(1, self.dh * 2 + 5), max(1, self.dw * 2 + 5)
        self.gph, self.gpw = (self.gkh - 1) // 2, (self.gkw - 1) // 2
        self.ph, self.pw = self.dh * ((self.kh - 1) // 2), self.dw * ((self.kw - 1) // 2)
        assert self.c % self.g == 0

        self.gather = nn.Conv2d(self.c, self.c,
                                kernel_size=(self.gkh, self.gkw), padding=(self.gph, self.gpw),
                                groups=self.c)
        self.proj = nn.Conv2d(self.c, self.c,
                              kernel_size=(self.kh, self.kw), dilation=(self.dh, self.dw), padding=(self.ph, self.pw),
                              groups=self.g)
        self.scale = nn.Parameter(torch.ones((1, self.g, self.c // self.g * self.kh * self.kw * self.c // self.g,)),
                                  requires_grad=True)
        self.attn_proj = nn.Conv1d(self.c // self.g * self.kh * self.kw, self.c // self.g, kernel_size=1)
        nn.init.trunc_normal_(self.scale, std=.02)
        self.unfold = partial(F.unfold, kernel_size=(self.kh, self.kw),
                              dilation=(self.dh, self.dw), padding=(self.ph, self.pw))

    def forward(self, x: torch.Tensor):
        self.n = x.size(0)
        x = self.gather(x)
        unfolded = self.unfold(x) \
            .view(self.n * self.g, self.c // self.g * self.kh * self.kw, self.h * self.w)
        attn_rhs = x.view(self.n * self.g, self.c // self.g, self.h * self.w).transpose(1, 2)
        attn = torch.bmm(unfolded, attn_rhs) / math.sqrt(self.h * self.w)
        attn = (attn.view(self.n, self.g, -1) * self.scale) \
            .view(self.n * self.g, self.c // self.g * self.kh * self.kw, self.c // self.g)
        attn = self.attn_proj(attn)
        sin = torch.sin(x * (math.pi / 2))
        proj_x = torch.maximum(self.proj(x), sin).view(self.n * self.g, self.c // self.g, self.h * self.w)
        # attn: (ng, c/g, c/g), proj_x: (ng, c/g, hw)
        return torch.bmm(attn.transpose(1, 2), proj_x).view(self.n, self.c, self.h, self.w) \
            / math.sqrt(self.c // self.g)
