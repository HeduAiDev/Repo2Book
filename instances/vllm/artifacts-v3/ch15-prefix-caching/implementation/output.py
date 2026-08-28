# SOURCE: vllm/v1/core/sched/output.py
# SchedulerOutput 的**CoW 过线切面**（m14 第 9 站的负载形态）：kv_cache_
# block_copies——调度器只给 (src,dst) 块号对，真拷贝在 worker 的
# copy_kv_cache_blocks_inplace；new_block_ids_to_zero——新块清零账（拷贝
# 先于 forward、晚于清零）。
# SUBTRACTED: NewRequestData/CachedRequestData 全量块表/差量账（ch07/ch13
#   切面已建）、spec/encoder/structured/connector/V2 观测等其余字段
#   （dossier.delete 第 3/5 条 + 各邻章边界）；make_empty 工厂。
from dataclasses import dataclass

from .kv_cache_utils import KVCacheBlockCopy


# SOURCE: vllm/v1/core/sched/output.py:L193 SchedulerOutput（CoW 过线切面）
@dataclass
class SchedulerOutput:
    # Block IDs freshly allocated from the pool during this scheduling step.
    # The worker zeros the corresponding GPU memory before the blocks are used,
    # preventing stale NaN/data from corrupting attention or SSM computation.
    # SOURCE: vllm/v1/core/sched/output.py:L253-L256
    new_block_ids_to_zero: list[int] | None = None

    # CoW copies to apply after zeroing new blocks and before forward.
    # SOURCE: vllm/v1/core/sched/output.py:L258-L259
    kv_cache_block_copies: list[KVCacheBlockCopy] | None = None
