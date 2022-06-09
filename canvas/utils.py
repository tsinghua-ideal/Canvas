from collections.abc import Iterable
from typing import Tuple


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
