# SOURCE: vllm/config/__init__.py
# HOST SEAM：config 面的最小承载（ch23/ch25 同款骨架 + 本章消费字段）。
# VllmConfig 真实本体是海量 namespace 聚合；本切面只承载本章消费的六个
# namespace 的字段子集（model/cache/scheduler/parallel/compilation/attention）
# + get_current_vllm_config 家族（模型装配期 DeepseekV32IndexerCache 注册
# static_forward_context、SparseAttnIndexer 读 parallel_config 都从这里取）。
# SUBTRACTED：各 namespace 的非本章字段（MoE/量化/多模态/…）——按消费面裁剪，
# 字段默认值与真实定义处一致。
from __future__ import annotations

import enum
from dataclasses import dataclass, field

import torch

from vllm.config.attention import AttentionConfig  # noqa: F401  (再导出消费位)

# SOURCE: vllm/config/compilation.py:L53-L61 CUDAGraphMode —— 逐字枚举
class CUDAGraphMode(enum.Enum):
    """Constants for the cudagraph mode in CompilationConfig.
    Meanwhile, the subset enum `NONE`, `PIECEWISE` and `FULL` are also
    treated as concrete runtime mode for cudagraph runtime dispatching.
    """

    NONE = 0
    PIECEWISE = 1
    FULL = 2
    FULL_AND_PIECEWISE = 3


# SOURCE: vllm/config/cache.py CacheDType —— Literal 面（ch25 同款）
CacheDType = str  # HOST SEAM：Literal 字符串面（"auto"/"bfloat16"/"fp8_ds_mla"/…)


# SOURCE: vllm/config/model.py ModelConfig —— HOST SEAM 消费字段子集
@dataclass
class ModelConfig:
    hf_config: object
    max_model_len: int
    # SOURCE: vllm/config/model.py dtype —— torch dtype 面（默认跟随 torch）
    dtype: torch.dtype = field(default_factory=torch.get_default_dtype)

    # SUBTRACTED: rope/量化/多模态/架构旗标族——非本章消费面


# SOURCE: vllm/config/cache.py CacheConfig —— HOST SEAM 消费字段子集
@dataclass
class CacheConfig:
    block_size: int = 16
    cache_dtype: CacheDType = "auto"
    # SOURCE: vllm/config/cache.py calculate_kv_scales —— 默认 False
    calculate_kv_scales: bool = False
    # SOURCE: vllm/config/cache.py enable_prefix_caching —— 默认 False 位
    enable_prefix_caching: bool = False
    # SOURCE: vllm/config/cache.py kv_cache_dtype_skip_layers —— 默认空
    kv_cache_dtype_skip_layers: list = field(default_factory=list)
    # SOURCE: vllm/config/cache.py user_specified_block_size —— selector 消费位
    user_specified_block_size: bool = False


# SOURCE: vllm/config/scheduler.py SchedulerConfig —— HOST SEAM 消费字段子集
@dataclass
class SchedulerConfig:
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 1024
    # SUBTRACTED: chunked prefill/抢占/前缀缓存调度族——ch10/ch11/ch15 域


# SOURCE: vllm/config/parallel.py ParallelConfig —— HOST SEAM 消费字段子集
@dataclass
class ParallelConfig:
    pipeline_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    cp_kv_cache_interleave_size: int = 1
    dcp_comm_backend: str = ""
    # SUBTRACTED: TP/PP/DP/EPLB 全族——单进程退化（=1 默认即真实单卡形态）


# SOURCE: vllm/config/compilation.py CompilationConfig —— HOST SEAM 消费字段
#   子集（static_forward_context 是本章命脉：IndexCache/MLA 层注册与
#   ForwardContext 查找的账本；enabled/disabled_custom_ops 是 CustomOp 派发面）
@dataclass
class CompilationConfig:
    static_forward_context: dict = field(default_factory=dict)
    enabled_custom_ops: set = field(default_factory=set)
    disabled_custom_ops: set = field(default_factory=set)
    # SOURCE: vllm/config/compilation.py cudagraph_mode —— 默认 NONE
    cudagraph_mode: CUDAGraphMode = CUDAGraphMode.NONE
    # SUBTRACTED: torch.compile/分片/缓存目录族——ch19 域


# SOURCE: vllm/config/__init__.py VllmConfig —— HOST SEAM 聚合位
#   （真实本体的 namespace 聚合子集；本章消费面）
@dataclass
class VllmConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    attention_config: AttentionConfig = field(default_factory=AttentionConfig)
    quant_config: object | None = None
    speculative_config: object | None = None
    kv_transfer_config: object | None = None


# SOURCE: vllm/config/__init__.py get_current_vllm_config 家族 —— HOST SEAM
_current_vllm_config: VllmConfig | None = None


# SOURCE: vllm/config/__init__.py set_current_vllm_config —— 上下文设置位
def set_current_vllm_config(vllm_config: VllmConfig) -> None:
    # SOURCE: vllm/config/__init__.py set_current_vllm_config —— HOST SEAM 位
    global _current_vllm_config
    _current_vllm_config = vllm_config


# SOURCE: vllm/config/__init__.py get_current_vllm_config —— HOST SEAM
def get_current_vllm_config() -> VllmConfig:
    # SOURCE: vllm/config/__init__.py get_current_vllm_config —— HOST SEAM
    assert _current_vllm_config is not None, (
        "Current VllmConfig is not set. Please use `set_current_vllm_config` "
        "to set the current config."
    )
    return _current_vllm_config


# SOURCE: vllm/config/__init__.py get_current_vllm_config_or_none —— HOST SEAM
def get_current_vllm_config_or_none() -> VllmConfig | None:
    # SOURCE: vllm/config/__init__.py —— HOST SEAM
    return _current_vllm_config


# SOURCE: vllm/config/__init__.py get_cached_compilation_config —— HOST SEAM
#   （CustomOp.dispatch_forward 的消费位；真实有 lru_cache，host 单配置等价）
def get_cached_compilation_config() -> CompilationConfig:
    # SOURCE: vllm/config/__init__.py get_cached_compilation_config —— HOST SEAM
    return get_current_vllm_config().compilation_config
