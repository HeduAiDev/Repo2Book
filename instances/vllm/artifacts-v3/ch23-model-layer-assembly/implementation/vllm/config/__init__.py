# SOURCE: vllm/config/__init__.py（真实为 config/ 包各类的 re-export 门面）
# HOST SEAM：配置面的最小承载。真实 VllmConfig 是 ch03 域的大配置对象
# （vllm/config/vllm.py:L331 起、数千行装配链）——本章只消费其字段面
# （model_config/cache_config/quant_config/load_config/device_config/
# compilation_config/parallel_config），以同名字段载体镜像；
# set_current_vllm_config / get_current_vllm_config / get_cached_compilation_
# config 是真实函数的减法子集（语义逐字）。
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Optional

import torch


# SOURCE: vllm/config/cache.py:L44 CacheConfig —— HOST SEAM 字段面
#   （本章消费：cache_dtype/calculate_kv_scales/sliding_window——Attention
#   __init__ L264-L276 读这三个）
@dataclass
class CacheConfig:
    # SOURCE: vllm/config/cache.py cache_dtype —— HOST SEAM 取默认 "auto"
    cache_dtype: str = "auto"
    # SOURCE: vllm/config/cache.py calculate_kv_scales —— HOST SEAM 默认 False
    calculate_kv_scales: bool = False
    # SOURCE: vllm/config/cache.py sliding_window —— HOST SEAM 默认 None
    sliding_window: Optional[int] = None


# SOURCE: vllm/config/model.py:L122 ModelConfig —— HOST SEAM 字段面
#   （本章消费：hf_config/dtype/registry/model/quantization/model_impl）
@dataclass
class ModelConfig:
    # SOURCE: vllm/config/model.py hf_config 字段 —— HOST SEAM
    hf_config: Any = None
    # SOURCE: vllm/config/model.py dtype 字段 —— HOST SEAM
    dtype: torch.dtype = torch.float32
    # SOURCE: vllm/config/model.py model 字段 —— HOST SEAM
    model: str = ""
    # SOURCE: vllm/config/model.py model_impl 字段 —— HOST SEAM 默认 "auto"
    model_impl: str = "auto"
    # SOURCE: vllm/config/model.py trust_remote_code 字段 —— HOST SEAM 默认
    #   False（get_model_architecture 的 hash 键消费面）
    trust_remote_code: bool = False
    # SOURCE: vllm/config/model.py quantization 字段 —— HOST SEAM 默认 None
    quantization: Optional[str] = None
    # SOURCE: vllm/config/model.py convert_type 字段 —— HOST SEAM 默认 "none"
    convert_type: str = "none"
    # SOURCE: vllm/config/model.py runner_type 字段 —— HOST SEAM 默认位
    runner_type: str = "generate"
    # SOURCE: vllm/config/model.py head_dtype 字段（v0.27 新面：--hf-overrides
    #   可置 fp32 供 RL 训推一致）—— HOST SEAM 默认 None
    head_dtype: Optional[torch.dtype] = None

    # SOURCE: vllm/config/model.py:L930-L933 registry 属性 —— HOST SEAM 镜像
    #   （真实经 _maybe_register_model_class_overrides 回 ModelRegistry）
    @property
    def registry(self):
        # SOURCE: vllm/config/model.py:L930-L933 registry 属性 —— HOST SEAM 镜像
        from vllm.model_executor.models.registry import ModelRegistry

        return ModelRegistry


# SOURCE: vllm/config/load.py:L27 LoadConfig —— HOST SEAM 字段面
@dataclass
class LoadConfig:
    # SOURCE: vllm/config/load.py load_format 字段 —— HOST SEAM 默认 "auto"
    load_format: str = "auto"
    # SOURCE: vllm/config/load.py device 字段 —— HOST SEAM 默认 None
    device: Optional[str] = None
    # SOURCE: vllm/config/load.py model_loader_extra_config —— HOST SEAM
    model_loader_extra_config: dict = field(default_factory=dict)


# SOURCE: vllm/config/device.py:L17 DeviceConfig —— HOST SEAM 字段面
@dataclass
class DeviceConfig:
    # SOURCE: vllm/config/device.py device 字段 —— HOST SEAM 默认 "cpu"
    device: str = "cpu"


# SOURCE: vllm/config/compilation.py:L398 CompilationConfig —— HOST SEAM 字段面
#   （本章消费：static_forward_context（m11 的账本，compilation.py:L753）、
#   custom_ops/enabled_custom_ops/disabled_custom_ops（CustomOp.enabled 的
#   读面）、custom_op_log_check（compilation.py:L1309，no-op 位））
@dataclass
class CompilationConfig:
    # SOURCE: vllm/config/compilation.py:L753 static_forward_context —— 逐字字段
    static_forward_context: dict[str, Any] = field(default_factory=dict, init=False)
    # SOURCE: vllm/config/compilation.py:L495 custom_ops —— HOST SEAM 字段位
    #   （默认 ["none"]：Inductor 后端的真实默认基模式——custom_ops 未配置时
    #   count_none==1，CustomOp.default_on() 为 False，dispatch 走 native）
    custom_ops: list = field(default_factory=lambda: ["none"])
    # SOURCE: vllm/config/compilation.py:L742 enabled_custom_ops —— 字段位
    enabled_custom_ops: list = field(default_factory=list)
    # SOURCE: vllm/config/compilation.py:L744 disabled_custom_ops —— 字段位
    disabled_custom_ops: list = field(default_factory=list)

    # SOURCE: vllm/config/compilation.py:L1309 custom_op_log_check —— HOST SEAM
    #   no-op 位（日志与 custom_ops 校验面；host 无自定义算子）
    def custom_op_log_check(self):
        # SOURCE: vllm/config/compilation.py:L1309 custom_op_log_check —— HOST SEAM
        return None


# SOURCE: vllm/config/parallel.py ParallelConfig —— HOST SEAM 字段面
@dataclass
class ParallelConfig:
    # SOURCE: vllm/config/parallel.py tensor_parallel_size —— HOST SEAM 默认 1
    tensor_parallel_size: int = 1
    # SOURCE: vllm/config/parallel.py pipeline_parallel_size —— HOST SEAM 默认 1
    pipeline_parallel_size: int = 1
    # SOURCE: vllm/config/parallel.py data_parallel_size —— HOST SEAM 默认 1
    data_parallel_size: int = 1
    # SOURCE: vllm/config/parallel.py enable_eplb —— HOST SEAM 默认 False
    enable_eplb: bool = False


# SOURCE: vllm/config/vllm.py:L331 VllmConfig —— HOST SEAM 字段面载体
#   （真实是 ch03 域的大配置对象；本章六消费面 + quant_config）
@dataclass
class VllmConfig:
    # SOURCE: vllm/config/vllm.py model_config 字段 —— HOST SEAM
    model_config: ModelConfig = field(default_factory=ModelConfig)
    # SOURCE: vllm/config/vllm.py cache_config 字段 —— HOST SEAM
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    # SOURCE: vllm/config/vllm.py quant_config 字段 —— HOST SEAM 默认 None
    quant_config: Any = None
    # SOURCE: vllm/config/vllm.py load_config 字段 —— HOST SEAM
    load_config: LoadConfig = field(default_factory=LoadConfig)
    # SOURCE: vllm/config/vllm.py device_config 字段 —— HOST SEAM
    device_config: DeviceConfig = field(default_factory=DeviceConfig)
    # SOURCE: vllm/config/vllm.py compilation_config 字段 —— HOST SEAM
    compilation_config: CompilationConfig = field(default_factory=CompilationConfig)
    # SOURCE: vllm/config/vllm.py parallel_config 字段 —— HOST SEAM
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)


# SOURCE: vllm/config/vllm.py:L2374-L2408 set_current_vllm_config —— 减法子集
#   （语义逐字：全局暂存/恢复 + check_compile 尾检查；compilation_counter
#   计数与 VLLM_COMPILE 告警段删除——ch19 编译域）
_current_vllm_config: VllmConfig | None = None
_current_prefix: str | None = None


# SOURCE: vllm/config/vllm.py:L2374 set_current_vllm_config
@contextmanager
def set_current_vllm_config(
    vllm_config: VllmConfig, check_compile=False, prefix: str | None = None
):
    """
    Temporarily set the current vLLM config.
    Used during model initialization.
    We save the current vLLM config in a global variable,
    so that all modules can access it, e.g. custom ops
    can access the vLLM config to determine how to dispatch.
    """
    # SOURCE: vllm/config/vllm.py:L2374 set_current_vllm_config
    global _current_vllm_config, _current_prefix
    old_vllm_config = _current_vllm_config
    old_prefix = _current_prefix
    try:
        # SUBTRACTED: compilation_counter 计数与 get_cached_compilation_config
        #   缓存清理（vllm/config/vllm.py:L2387-L2394）——ch19 编译域
        _current_vllm_config = vllm_config
        _current_prefix = prefix
        yield
    except Exception:
        raise
    else:
        if check_compile:
            vllm_config.compilation_config.custom_op_log_check()
        # SUBTRACTED: VLLM_COMPILE 模式的 num_models_seen 告警段
        #   （vllm/config/vllm.py:L2401-L2415）——ch19 编译域
    finally:
        _current_vllm_config = old_vllm_config
        _current_prefix = old_prefix


# SOURCE: vllm/config/vllm.py:L2434-L2447 get_current_vllm_config
def get_current_vllm_config() -> VllmConfig:
    if _current_vllm_config is None:
        raise AssertionError(
            "Current vLLM config is not set. This typically means "
            "get_current_vllm_config() was called outside of a "
            "set_current_vllm_config() context, or a CustomOp was instantiated "
            "at module import time or model forward time when config is not set. "
        )
    return _current_vllm_config


# SOURCE: vllm/config/vllm.py:L2429-L2431 get_cached_compilation_config
#   子集（真实带 lru_cache(1)——host 单线程直调等价）
def get_cached_compilation_config():
    """Cache config to avoid repeated calls to get_current_vllm_config()"""
    # SOURCE: vllm/config/vllm.py:L2429-L2431 get_cached_compilation_config
    return get_current_vllm_config().compilation_config
