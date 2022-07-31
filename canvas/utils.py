import math
from collections.abc import Iterable
from typing import Tuple
from torch import nn


def int_range_check(r: Tuple[int, int],
                    min_value: int = 0,
                    max_value: int = 0x7fffffff):
    if len(r) != 2:
        return False
    if type(r[0]) != int or type(r[1]) != int:
        return False
    if not (min_value <= r[0] <= max_value and min_value <= r[1] <= max_value):
        return False
    return r[0] <= r[1]


def float_range_check(r: Tuple[float, float],
                      min_value: float = 0,
                      max_value: float = float('inf')):
    if len(r) != 2:
        return False
    if type(r[0]) != float and type(r[0]) != int:
        return False
    if type(r[1]) != float and type(r[1]) != int:
        return False
    if not (min_value <= r[0] <= max_value and min_value <= r[1] <= max_value):
        return False
    return r[0] <= r[1]


def is_type_range(r, type_cls):
    if not isinstance(r, Iterable):
        return False
    for item in r:
        if type(item) != type_cls:
            return False
    return True


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
        nn.init.trunc_normal_(m.weight.data, std=.1)
        if m.bias is not None:
            m.bias.data.zero_()
