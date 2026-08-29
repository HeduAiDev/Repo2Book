# SOURCE: vllm/v1/attention/backends/utils.py
# ch22 切面：两个 pad 常量的出生地（m3「PAD 语义」的两块界碑）——
# PAD_SLOT_ID = -1（token 侧：slot_mapping 的哨兵）与 NULL_BLOCK_ID = 0
# （行侧：块表尾行的空块号）。原文件其余成员（布局覆写/mm range 张量等）
# → ch21 后端域。
from __future__ import annotations

from typing import Literal

# SOURCE: vllm/v1/attention/backends/utils.py:L42-L43 布局类型名（注解面）
KVCacheLayoutType = Literal["NHD", "HND"]
# SOURCE: vllm/v1/attention/backends/utils.py:L43
_KV_CACHE_LAYOUT_OVERRIDE: KVCacheLayoutType | None = None

# SOURCE: vllm/v1/attention/backends/utils.py:L45 PAD_SLOT_ID（token 侧 pad 值）
PAD_SLOT_ID = -1
# SOURCE: vllm/v1/attention/backends/utils.py:L46 NULL_BLOCK_ID（行侧 pad 值）
NULL_BLOCK_ID = 0
