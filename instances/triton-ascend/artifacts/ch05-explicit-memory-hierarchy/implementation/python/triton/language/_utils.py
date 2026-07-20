# SOURCE: python/triton/language/_utils.py（全文件，仅取 validate_block_shape 一个
# 依赖：buffer_type/block_type 构造时用它校验 shape 元素为 int 且总元素数不超上限。
# 与本章内存层级/copy/fixpipe 主线无关，只是 tl.block_type 的支撑函数，逐字保留。）
from typing import List

TRITON_MAX_TENSOR_NUMEL = 1048576


# SOURCE: python/triton/language/_utils.py:L17-29
def validate_block_shape(shape: List[int]):
    numel = 1
    for i, d in enumerate(shape):
        if not isinstance(d, int):
            raise TypeError(f"Shape element {i} must have type `constexpr[int]`, got `constexpr[{type(d)}]")
        numel *= d

    if numel > TRITON_MAX_TENSOR_NUMEL:
        raise ValueError(f"numel ({numel}) exceeds triton maximum tensor numel ({TRITON_MAX_TENSOR_NUMEL})")
    return numel
