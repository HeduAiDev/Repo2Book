# SOURCE: vllm/utils/platform_utils.py
# HOST SEAM：本章消费面两件——is_pin_memory_available（torch_utils 的
# PIN_MEMORY 派生）与 num_compute_units（Triton wrapper 的
# NUM_PROGRAMS=min(SM 数, batch)，topk_topp_triton.py:L909）。逐字。
from __future__ import annotations


# SOURCE: vllm/utils/platform_utils.py:L44-L47 is_pin_memory_available —— 逐字
def is_pin_memory_available() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_pin_memory_available()


# SOURCE: vllm/utils/platform_utils.py:L61-L65 num_compute_units —— 逐字
def num_compute_units(device_id: int = 0) -> int:
    """Get the number of compute units of the current device."""
    from vllm.platforms import current_platform

    return current_platform.num_compute_units(device_id)
