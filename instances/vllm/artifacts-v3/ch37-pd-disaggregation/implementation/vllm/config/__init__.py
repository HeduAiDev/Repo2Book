# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""配置层载体：VllmConfig 与本章消费到的四个子配置。

# SOURCE: vllm/config/__init__.py:L1-L120 + vllm/config/vllm.py:L331-L700
# SUBTRACTED: 配置的 __post_init__ 校验/多模态/编译/LoRA/投机解码等字段族——
#   本章的 NIXL connector 只读这几个面：cache.block_size / cache.cache_dtype /
#   parallel.tensor_parallel_size / parallel.data_parallel_index /
#   scheduler.disable_hybrid_kv_cache_manager / model.get_total_num_kv_heads()...
"""

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from vllm.config.kv_transfer import KVTransferConfig

# SOURCE: vllm/config/__init__.py:L40-L120（子配置重导出）


# SOURCE: vllm/config/model.py:L120-L400（模型侧只读面）
@dataclass
class ModelConfig:
    """模型身份与 KV 几何（本章只要这几项参与兼容 hash 与描述符推算）。"""

    model: str = "dummy"
    dtype: str = "float16"
    num_hidden_layers: int = 1
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    head_size: int = 128
    use_mla: bool = False

    # SOURCE: vllm/config/model.py:L900-L930
    def get_total_num_kv_heads(self) -> int:
        return self.num_key_value_heads

    # SOURCE: vllm/config/model.py:L930-L950
    def get_head_size(self) -> int:
        return self.head_size

    # SOURCE: vllm/config/model.py:L950-L980
    def get_total_num_hidden_layers(self) -> int:
        return self.num_hidden_layers


# SOURCE: vllm/config/cache.py:L20-L200（cache 侧只读面）
@dataclass
class CacheConfig:
    block_size: int = 16
    cache_dtype: str = "auto"
    enable_prefix_caching: bool = True


# SOURCE: vllm/config/parallel.py:L120-L500（并行侧只读面）
@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_index: int = 0

    # SOURCE: vllm/config/parallel.py:L900-L960
    @property
    def world_size(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size


# SOURCE: vllm/config/scheduler.py:L100-L200（调度侧只读面）
@dataclass
class SchedulerConfig:
    # 默认 False = 混合 KV 分配器（HMA）开着——`_is_hma_required` 的输入。
    disable_hybrid_kv_cache_manager: bool = False
    max_num_seqs: int = 256


# SOURCE: vllm/config/vllm.py:L331-L700（VllmConfig 字段族）
@dataclass
class VllmConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    kv_transfer_config: KVTransferConfig | None = None
    # SUBTRACTED: speculative_config / lora_config / compilation_config / ... ——
    #   本章的 connector 面不读它们（EngineCore 的 use_spec_decode 判定除外，
    #   见 vllm/v1/engine/core.py 的删除标记）。
    speculative_config: Any | None = None

    # SOURCE: vllm/config/vllm.py:L1180-L1210（compute_hash / __str__ 之外的接口位）
    def __post_init__(self) -> None:
        if self.kv_transfer_config is not None:
            assert self.kv_transfer_config.engine_id is not None


# SOURCE: vllm/config/__init__.py:L600-L660（vllm_config 上下文）
_CURRENT_VLLM_CONFIG: Any | None = None


# SOURCE: vllm/config/__init__.py:L610-L625
def get_current_vllm_config() -> VllmConfig:
    global _CURRENT_VLLM_CONFIG
    if _CURRENT_VLLM_CONFIG is None:
        raise RuntimeError("No current vLLM config is set.")
    return _CURRENT_VLLM_CONFIG


# SOURCE: vllm/config/__init__.py:L628-L640
@contextlib.contextmanager
def set_current_vllm_config(vllm_config: VllmConfig) -> Iterator[None]:
    global _CURRENT_VLLM_CONFIG
    old = _CURRENT_VLLM_CONFIG
    _CURRENT_VLLM_CONFIG = vllm_config
    try:
        yield
    finally:
        _CURRENT_VLLM_CONFIG = old


# SOURCE: vllm/config/__init__.py:L645-L680（get_layers_from_vllm_config）
def get_layers_from_vllm_config(
    vllm_config: VllmConfig, layer_type: type, layer_names: list[str] | None = None
) -> dict[str, Any]:
    """本章不跑模型装配，静态层表恒为空 → 调用方走 fallback 分支
    （源码里注释就写着这是 tests 路径：kv_connector/utils.py:L348-L362）。"""
    return {}


__all__ = [
    "VllmConfig",
    "ModelConfig",
    "CacheConfig",
    "ParallelConfig",
    "SchedulerConfig",
    "KVTransferConfig",
    "get_current_vllm_config",
    "set_current_vllm_config",
    "get_layers_from_vllm_config",
]
