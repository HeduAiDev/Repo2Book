# SOURCE: vllm/config/scheduler.py
# 本章消费面：watermark（护轨分配的 free−reserved−watermark 第三项）、
# disable_hybrid_kv_cache_manager（factory 的 HMA 门）、max_num_batched_
# tokens（token 预算面账位）、scheduler_reserve_full_isl。
# SUBTRACTED: 队列策略/chunked prefill/DelayFactor/结构化输出等调度面
#   （ch10/11 各章切面）。
from dataclasses import dataclass


# SOURCE: vllm/config/scheduler.py:L~30 SchedulerConfig（切面直供——pydantic
#   装配链归 ch03）
@dataclass
class SchedulerConfig:
    """Scheduler-related configuration（本章消费面）。"""

    # SOURCE: vllm/config/scheduler.py:L49 max_num_batched_tokens
    max_num_batched_tokens: int = 8192
    # SOURCE: vllm/config/scheduler.py:L63 max_num_seqs（账位保留）
    max_num_seqs: int = 1024
    # SOURCE: vllm/config/scheduler.py:L122 disable_hybrid_kv_cache_manager
    #   （默认 None → False：HMA 开——factory 的门读它）
    disable_hybrid_kv_cache_manager: bool | None = None
    # SOURCE: vllm/config/scheduler.py:L130 scheduler_reserve_full_isl
    #   （默认 True——full-ISL 准入门开关，ch14 已立、本章透传）
    scheduler_reserve_full_isl: bool = True
    # SOURCE: vllm/config/scheduler.py:L136 watermark（默认 0.0 关——
    #   「Fraction of total KV cache blocks to keep free」）
    watermark: float = 0.0
