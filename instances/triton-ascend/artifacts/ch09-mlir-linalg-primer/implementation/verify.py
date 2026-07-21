"""ch09 —— "legal by design" 的可执行校验：不同变换路径必须数值一致。

# PAPER: [Linalg §3.6] "These transformations are legal by design, in the
sense that their legality and applicability derive from the operation's
properties and structure." (paper.md:L385) —— 全章方法论的收口一句：合法性
不是靠低层 IR 分析恢复的，而是从算子自身的性质与结构推出来的。本文件把这句
话变成一个可以在测试里直接调用的断言。
"""
from __future__ import annotations

import numpy as np


# PAPER: [Linalg §3.6] "legal by design" (paper.md:L385)。
def assert_same_result(*results: np.ndarray, rtol: float = 1e-6, atol: float = 1e-9) -> None:
    """断言给定的若干条结果（原始 / tiled / tiled+padded / tiled+vectorized /
    bufferize_naive / bufferize_dps ……）在数值上彼此一致——用来把
    "legal by design" 落成一个可运行的交叉检验，而不是一句口号。
    """
    if len(results) < 2:
        return
    ref = results[0]
    for other in results[1:]:
        if ref.shape != other.shape:
            raise AssertionError(f"结果形状不一致: {ref.shape} vs {other.shape}")
        if not np.allclose(ref, other, rtol=rtol, atol=atol):
            diff = np.max(np.abs(ref.astype(float) - other.astype(float)))
            raise AssertionError(f"结果数值不一致：最大绝对误差 = {diff}")
