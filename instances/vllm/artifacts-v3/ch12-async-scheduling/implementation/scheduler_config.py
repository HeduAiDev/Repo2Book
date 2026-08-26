# SOURCE: vllm/config/scheduler.py
# 本章两个配置真相源：async_scheduling 配置位（L148-L151——None=默认，交给
# VllmConfig 仲裁）与 get_scheduler_cls 换型点（L170-L178——async → AsyncScheduler）。
# 预算类字段沿用 ch10 立过的基线；pydantic 装配换 dataclass 等价承载。
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .interface import SchedulerInterface


# SOURCE: vllm/config/scheduler.py:L22 SchedulerPolicy 字面量类型
SchedulerPolicy = Literal["fcfs", "priority"]


# SOURCE: vllm/config/scheduler.py:L25 SchedulerConfig
@dataclass
class SchedulerConfig:
    """Scheduler configuration."""

    # SUBTRACTED: max_model_len / is_encoder_decoder InitVar 与 encoder 预算
    #   派生（L29-L46——装配层）、runner_type/is_multimodal_model/
    #   max_num_encoder_input_tokens（L46-L97——mm/encoder）、
    #   scheduler_cls/disable_hybrid_kv_cache_manager（L115-L128）、
    #   stream_interval（L153-L157——流式输出）、prefill_schedule_interval
    #   （L143-L146——DP）。

    # SOURCE: vllm/config/scheduler.py:L49-L54 max_num_batched_tokens
    max_num_batched_tokens: int = 2048
    """Maximum number of tokens that can be processed in a single iteration."""

    # SOURCE: vllm/config/scheduler.py:L56-L61 max_num_scheduled_tokens
    max_num_scheduled_tokens: int | None = None
    """Maximum number of tokens that the scheduler may issue in a single iteration.

    This is usually equal to max_num_batched_tokens, but can be smaller in cases
    when the model might append tokens into the batch (such as speculative decoding).
    Defaults to max_num_batched_tokens."""

    # SOURCE: vllm/config/scheduler.py:L63-L68 max_num_seqs
    max_num_seqs: int = 128
    """Maximum number of sequences to be processed in a single iteration."""

    # SOURCE: vllm/config/scheduler.py:L70-L72 long_prefill_token_threshold
    long_prefill_token_threshold: int = 0
    """For chunked prefill, a request is considered long if the prompt is
    longer than this number of tokens. 0 disables the cap (default)."""

    # SOURCE: vllm/config/scheduler.py:L74-L80 enable_chunked_prefill
    enable_chunked_prefill: bool = True
    """If True, prefill requests can be chunked based
    on the remaining `max_num_batched_tokens`."""

    # SOURCE: vllm/config/scheduler.py:L99-L105 policy
    policy: SchedulerPolicy = "fcfs"
    """The scheduling policy to use: "fcfs" or "priority"."""

    # SUBTRACTED: scheduler_cls（L115-L120——自定义调度器装配）——精简版恒走
    #   内置两型，不可达分支不保留。

    # SOURCE: vllm/config/scheduler.py:L148-L151 async_scheduling 配置位（m1）
    async_scheduling: bool | None = None
    """If set to False, disable async scheduling. Async scheduling helps to
    avoid gaps in GPU utilization, leading to better latency and throughput.
    """

    # SUBTRACTED: scheduler_reserve_full_isl/watermark（L130-L141——ch11 立过，
    #   本章准入深水不展开；kv seam 不消费）。

    # SOURCE: vllm/config/scheduler.py:L170-L178 get_scheduler_cls（m3 换型点；
    #   真实代码在 scheduler_cls is None 守卫内——该守卫随自定义调度器删除）
    def get_scheduler_cls(self) -> type["SchedulerInterface"]:
        # SOURCE: vllm/config/scheduler.py:L172-L175 async → AsyncScheduler
        if self.async_scheduling:
            from .async_scheduler import AsyncScheduler

            return AsyncScheduler
        # SOURCE: vllm/config/scheduler.py:L176-L178 否则 Scheduler
        from .scheduler import Scheduler

        return Scheduler
