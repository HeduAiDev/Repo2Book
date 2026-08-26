# SOURCE: vllm/config/scheduler.py
# 预算与切块的配置真相源（m4）：DEFAULT_MAX_NUM_BATCHED_TOKENS=2048（『主要为
# 测试便利』）、max_num_scheduled_tokens 缺省回落、max_num_seqs=128、
# long_prefill_token_threshold=0（不钳）、enable_chunked_prefill=True（v1 默认开）、
# scheduler_reserve_full_isl=True（整序列准入门）。真实部署的预算不来自这里的
# 默认值而在 EngineArgs.create_engine_config 按硬件仲裁（见 arg_utils.py）。
# SUBTRACTED: pydantic 装配与 __post_init__ 校验/派生（default_factory、
#   encoder 预算派生、chunked-prefill 与 encoder-decoder 的互斥关闭 L228-L236、
#   async_scheduling/stream_interval 等场景开关）——换纯 dataclass 等价承载
#   本章旋钮；其余字段属 dossier.delete 批准的子系统或邻章。
from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

# SOURCE: vllm/config/scheduler.py:L22 SchedulerPolicy 字面量类型
SchedulerPolicy = Literal["fcfs", "priority"]
# SUBTRACTED: RunnerType 字面量（L21）——runner 类型装配属 ch03。


# SOURCE: vllm/config/scheduler.py:L25-L26 SchedulerConfig
@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    # SUBTRACTED: max_model_len / is_encoder_decoder InitVar（L29-L40）——
    #   本章构造不走 pydantic InitVar 装配；encoder-decoder 关 chunked 的
    #   __post_init__ 逻辑随 encoder 子系统删除。

    # SOURCE: vllm/config/scheduler.py:L42-L44 三个 ClassVar 默认
    DEFAULT_MAX_NUM_BATCHED_TOKENS: ClassVar[int] = 2048
    DEFAULT_MAX_NUM_BATCHED_TOKENS_FOR_BATCHED_DP: ClassVar[int] = 256
    DEFAULT_MAX_NUM_SEQS: ClassVar[int] = 128

    # SOURCE: vllm/config/scheduler.py:L49-L54 max_num_batched_tokens
    max_num_batched_tokens: int = 2048
    """Maximum number of tokens that can be processed in a single iteration.

    The default value here is mainly for convenience when testing.
    In real usage, this should be set in `EngineArgs.create_engine_config`.
    """

    # SOURCE: vllm/config/scheduler.py:L56-L61 max_num_scheduled_tokens
    max_num_scheduled_tokens: int | None = None
    """Maximum number of tokens that the scheduler may issue in a single iteration.

    This is usually equal to max_num_batched_tokens, but can be smaller in cases
    when the model might append tokens into the batch (such as speculative decoding).
    Defaults to max_num_batched_tokens."""

    # SOURCE: vllm/config/scheduler.py:L63-L68 max_num_seqs
    max_num_seqs: int = 128
    """Maximum number of sequences to be processed in a single iteration.

    The default value here is mainly for convenience when testing.
    In real usage, this should be set in `EngineArgs.create_engine_config`.
    """

    # SOURCE: vllm/config/scheduler.py:L70-L72 long_prefill_token_threshold
    long_prefill_token_threshold: int = 0
    """For chunked prefill, a request is considered long if the prompt is
    longer than this number of tokens. 0 disables the cap (default)."""

    # SOURCE: vllm/config/scheduler.py:L74-L80 enable_chunked_prefill
    enable_chunked_prefill: bool = True
    """If True, prefill requests can be chunked based
    on the remaining `max_num_batched_tokens`.

    The default value here is mainly for convenience when testing.
    In real usage, this should be set in `EngineArgs.create_engine_config`.
    """

    # SOURCE: vllm/config/scheduler.py:L99-L105 policy
    policy: SchedulerPolicy = "fcfs"
    """The scheduling policy to use:

    - "fcfs" means first come first served, i.e., requests are handled in order
      of arrival.
    - "priority" means requests are handled based on given priority (lower
      value means earlier handling) and time of arrival deciding any ties)."""

    # SOURCE: vllm/config/scheduler.py:L130-L134 scheduler_reserve_full_isl
    scheduler_reserve_full_isl: bool = True
    """If True, the scheduler checks whether the full input sequence length
    fits in the KV cache before admitting a new request, rather than only
    checking the first chunk. Prevents over-admission and KV cache thrashing
    with chunked prefill."""

    # SOURCE: vllm/config/scheduler.py:L136-L141 watermark
    watermark: float = 0.0
    """Fraction of total KV cache blocks to keep free (the watermark) when
    admitting waiting or preempted requests into the running queue. This headroom
    helps avoid frequent KV cache eviction and the resulting repeated preemption
    of requests when GPU memory is scarce. Must be in the range [0.0, 1.0); 0.0
    (the default) disables the watermark."""
    # 注：watermark 的显存账本深挖归 ch14——本章精简版的 KVCacheManager 契约面
    # 不消费它（默认 0.0 关闭，与真实默认一致）。

    # SUBTRACTED: runner_type / is_multimodal_model / max_num_encoder_input_
    #   tokens / encoder_cache_size（L46-L97，encoder/mm）、disable_chunked_mm_
    #   input（L107-L113，mm 切块）、scheduler_cls（L115-L120，类装配）、
    #   disable_hybrid_kv_cache_manager（L122-L128，混合注意力分组）、
    #   prefill_schedule_interval（L143-L146，DP prefill balancing——dossier.delete
    #   第 7 条）、async_scheduling（L148-L151，ch12）、stream_interval（L153-L157，
    #   流式输出）——各属 dossier.delete 批准的子系统或邻章。
