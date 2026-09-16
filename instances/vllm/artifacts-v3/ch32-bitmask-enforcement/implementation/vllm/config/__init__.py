# SOURCE: vllm/config/__init__.py
# HOST SEAM：真实 config 包 re-export 各子配置模块（model.py/scheduler.py/
# parallel.py/…，ch03 全文已立）。本章消费字段面以最小 dataclass 承载；
# VllmConfig 本体（两个 property 逐字）在 vllm/config/vllm.py。
from __future__ import annotations

from dataclasses import dataclass, field


# SOURCE: vllm/config/scheduler.py SchedulerConfig —— 消费字段承载
#   （本章消费：max_num_seqs——掩码行数预算的批维）
@dataclass
class SchedulerConfig:
    max_num_seqs: int = 1024
    # SUBTRACTED: max_num_batched_tokens/chunked_prefill_enabled/priority 等
    #   调度约束字段（ch10/ch11 消费面）


# SOURCE: vllm/config/model.py ModelConfig —— 消费字段承载
#   （本章消费：is_diffusion（bonus 行跳过）/skip_tokenizer_init（装配分支）/
#   architectures·is_moe·runner_type·is_hybrid·is_attention_free（V2 判据））
@dataclass
class ModelConfig:
    architectures: tuple[str, ...] = ()
    is_moe: bool = False
    runner_type: str = "generate"
    is_diffusion: bool = False
    skip_tokenizer_init: bool = False
    is_hybrid: bool = False
    is_attention_free: bool = False
    # SUBTRACTED: hf_config/tokenizer/max_model_len/dtype 等模型装配面（ch03）


# SOURCE: vllm/config/parallel.py ParallelConfig —— 消费字段承载
@dataclass
class ParallelConfig:
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    distributed_executor_backend: str = "mp"
    # SUBTRACTED: data_parallel_size/decode_context_parallel_size 等（ch05）


# SOURCE: vllm/config/speculative.py SpeculativeConfig —— 消费字段承载
@dataclass
class SpeculativeConfig:
    method: str | None = None
    num_speculative_tokens: int | None = None
    # SUBTRACTED: draft model 装配面（ch32/33）


# SOURCE: vllm/config/diffusion.py DiffusionConfig —— 消费字段承载
@dataclass
class DiffusionConfig:
    canvas_length: int | None = None


# SOURCE: vllm/config/structured_outputs.py StructuredOutputsConfig —— 消费字段承载
@dataclass
class StructuredOutputsConfig:
    reasoning_parser: str | None = None
    reasoning_parser_plugin: str | None = None
    enable_in_reasoning: bool = False


def _as_config(cls, value):
    return cls(**value) if isinstance(value, dict) else value


from vllm.config.vllm import VllmConfig  # noqa: E402  (真实同款 re-export 位)

__all__ = [
    "VllmConfig",
    "SchedulerConfig",
    "ModelConfig",
    "ParallelConfig",
    "SpeculativeConfig",
    "DiffusionConfig",
    "StructuredOutputsConfig",
]
