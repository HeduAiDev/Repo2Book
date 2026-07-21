# 精简版 triton.language 包——真实 __init__.py 重导出上百个算子/类型名，本章
# mem_ops.py/vec_ops.py 用 `import triton.language as tl` 只取用到的这些。
#
# SOURCE: python/triton/language/__init__.py（重导出清单节选）

from .core import (  # noqa: F401
    constexpr,
    dtype,
    pointer_type,
    block_type,
    tensor,
    builtin,
    is_builtin,
    int1,
    int8,
    int16,
    int32,
    int64,
    uint8,
    uint16,
    uint32,
    float16,
    bfloat16,
    float32,
    float8e4nv,
    float8e5,
    void,
    get_int_dtype,
    reshape,
    static_assert,
)
