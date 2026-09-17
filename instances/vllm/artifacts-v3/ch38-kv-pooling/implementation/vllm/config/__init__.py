# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""配置层载体：VllmConfig 与本章消费到的子配置。

# SOURCE: vllm/config/__init__.py:L1-L120 + vllm/config/vllm.py:L331-L700
# SUBTRACTED: 配置 __post_init__ 校验链/多模态/编译/LoRA 等字段族——本章的
#   池化面只读：cache.block_size/cache_dtype/enable_prefix_caching/prefix_match_unit、
#   parallel 各并行轴与 rank/world_size、model.use_mla/model/dtype、
#   kv_events_config、use_v2_model_runner。
"""

from dataclasses import dataclass, field
from typing import Any

from vllm.config.kv_transfer import KVTransferConfig


# SOURCE: vllm/config/vllm.py KVEventsConfig 条目（kv_events 的开关面）
@dataclass
class KVEventsConfig:
    """KV 事件发布配置（本章只读两个开关）。"""

    enable_kv_cache_events: bool = False
    enable_event_buffer: bool = False
    reception_method: str = "zmq"
    self_describing_kv_events: bool = False


# SOURCE: vllm/config/model.py:L120-L400（模型侧只读面）
@dataclass
class ModelConfig:
    """模型身份与 KV 几何。"""

    model: str = "dummy"
    dtype: str = "float16"
    use_mla: bool = False

    # SOURCE: vllm/config/model.py:L900-L950
    def get_total_num_kv_heads(self) -> int:
        # SOURCE: vllm/config/model.py:L900-L930
        return 8

    # SOURCE: vllm/config/model.py:L930-L950
    def get_head_size(self) -> int:
        # SOURCE: vllm/config/model.py:L930-L950
        return 128


# SOURCE: vllm/config/cache.py:L20-L200（cache 侧只读面）
@dataclass
class CacheConfig:
    block_size: int = 16
    cache_dtype: str = "auto"
    enable_prefix_caching: bool = True
    prefix_match_unit: int | None = None


# SOURCE: vllm/config/parallel.py:L120-L500（并行侧只读面）
@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    prefill_context_parallel_size: int = 1
    decode_context_parallel_size: int = 1
    data_parallel_index: int = 0
    rank: int = 0
    # SOURCE: vllm/config/parallel.py:L900-L960（真实为 tp*pp 的派生+校验位）
    world_size: int = 1
    distributed_executor_backend: str = "mp"


# SOURCE: vllm/config/scheduler.py:L100-L200（调度侧只读面）
@dataclass
class SchedulerConfig:
    # 默认 False = 混合 KV 分配器（HMA）开着。
    disable_hybrid_kv_cache_manager: bool = False


# SOURCE: vllm/config/vllm.py:L331-L700（VllmConfig 字段族）
@dataclass
class VllmConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    kv_transfer_config: KVTransferConfig | None = None
    kv_events_config: KVEventsConfig | None = None
    speculative_config: Any | None = None
    # SOURCE: vllm/config/vllm.py use_v2_model_runner 条目
    use_v2_model_runner: bool = False

    # SOURCE: vllm/config/vllm.py:L1180-L1210
    def __post_init__(self) -> None:
        # SOURCE: vllm/config/vllm.py:L1180-L1210
        if self.kv_transfer_config is not None:
            assert self.kv_transfer_config.engine_id is not None


__all__ = [
    "VllmConfig",
    "ModelConfig",
    "CacheConfig",
    "ParallelConfig",
    "SchedulerConfig",
    "KVEventsConfig",
    "KVTransferConfig",
]
