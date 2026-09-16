# SOURCE: vllm/config/__init__.py —— re-export 子集（真实文件是全配置装配线的
# 出口，ch03 域）。本章消费面：ParallelConfig / VllmConfig / holder 三件套。

from vllm.config.parallel import ParallelConfig
from vllm.config.vllm import (
    VllmConfig,
    get_current_vllm_config,
    get_current_vllm_config_or_none,
    set_current_vllm_config,
)

__all__ = [
    "ParallelConfig",
    "VllmConfig",
    "get_current_vllm_config",
    "get_current_vllm_config_or_none",
    "set_current_vllm_config",
]
# SUBTRACTED: 其余 re-export（ModelConfig/CacheConfig/get_layers...）——ch03 域。
