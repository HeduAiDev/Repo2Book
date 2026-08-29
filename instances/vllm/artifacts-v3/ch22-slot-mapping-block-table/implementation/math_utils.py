# SOURCE: vllm/utils/math_utils.py
# ch22 消费面：cdiv（get_block_table_width 的对齐乘子 / kv_cache_interface 的
# max_num_blocks_per_req 均用它）。本文件其余成员归他章域，不进。
from __future__ import annotations


# SOURCE: vllm/utils/math_utils.py:L~8 cdiv
def cdiv(a: int, b: int) -> int:
    """Ceiling division."""
    return -(a // -b)
