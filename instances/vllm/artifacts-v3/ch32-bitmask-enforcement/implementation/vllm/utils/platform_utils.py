# SOURCE: vllm/utils/platform_utils.py
# HOST SEAM：本章消费面一件——is_pin_memory_available（torch_utils 的
# PIN_MEMORY 派生：pinned H2D 链 m11 的物质前提）。真实实现经
# current_platform 多态（CUDA 平台还要查 WSL 内核版本）；HOST SEAM 保留
# 委托结构 + platforms 承载。
from __future__ import annotations


# SOURCE: vllm/utils/platform_utils.py:L44-L47 is_pin_memory_available —— 逐字
def is_pin_memory_available() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_pin_memory_available()
