"""ch09 —— padding：把动态边界 tile 磨平成静态尺寸（m17/t5）。

只实现 [Linalg §3.2] 三种缓解手段里的第 2 种（padding）。第 1 种
（peeling/版本化）与第 3 种（显式 masking）不实现——masking 论文自己写明
"work in progress and outside the scope of this paper"（paper.md:L347），
本包同样不展开；peeling/版本化是控制流层面的改写，不是这里要讲的代数问题。
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

# PAPER: [Linalg §3.2] "padding must be the neutral for the consuming operation."
# (arXiv:2202.03293 §3.2；paper.md:L346 的中文对应句) —— 这是一条正确性条件，
# 不是「随便补 0」。这里只登记本章会用到的几种归约的幺元。
NEUTRAL_ELEMENTS = {
    "sum": 0.0,
    "prod": 1.0,
    "max": float("-inf"),
    "min": float("inf"),
}


# PAPER: [Linalg §3.2] padding 的正确性条件（paper.md:L346）。
def neutral_element(combine: str) -> float:
    """按消费该 tile 的归约算子查表取幺元——加法补 0、乘法补 1、max 补 -inf、
    min 补 +inf，都是这条正确性条件的实例。
    """
    try:
        return NEUTRAL_ELEMENTS[combine]
    except KeyError as exc:
        raise ValueError(f"没有为 combine={combine!r} 登记幺元") from exc


# PAPER: [Linalg §3.2] padding 把动态 tile 补到更大的静态尺寸（paper.md:L341-L346）。
def pad_to_static(
    tile_arr: np.ndarray, target_shape: Tuple[int, ...], neutral: float
) -> np.ndarray:
    """把（可能不满的）`tile_arr` 在每个轴的尾端补到 `target_shape`，补的值是
    `neutral`。只在尾端补，对应「边界 tile 是某一维上的最后一块、天生比满 tile
    短」这个场景（不是通用的居中/前端 padding）。
    """
    pad_width = [(0, t - s) for s, t in zip(tile_arr.shape, target_shape)]
    if any(w < 0 for _, w in pad_width):
        raise ValueError("target_shape 必须在每个轴上都不小于 tile_arr.shape")
    return np.pad(tile_arr, pad_width, mode="constant", constant_values=neutral)
