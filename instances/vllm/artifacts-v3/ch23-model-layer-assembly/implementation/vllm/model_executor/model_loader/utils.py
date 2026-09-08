# SOURCE: vllm/model_executor/model_loader/utils.py
# ch23 主角文件之十一（m8/m10）：initialize_model（新式签名校验 (vllm_config,
# prefix) + set_current_vllm_config 上下文内构造）+ process_weights_after_
# loading 两轮（quant_method 重打包 → Attention 层 scale）+ get_model_
# architecture（arch→类查表，hash 缓存）。
# SUBTRACTED：老式 kwargs 猜参兼容段（L62-L97）、configure_quant_config
#   （量化分发 ch27）、record_metadata_for_reloading（reload 域）、ParamMapping、
#   device_loading_context 的 UVA 面。
from __future__ import annotations

import inspect
from contextlib import contextmanager

import torch
from torch import nn

from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.quantization.base_config import (
    QuantizeMethodBase,
)

logger = init_logger(__name__)

# SUBTRACTED: import vllm.envs / MMEncoderAttention / HpcModule / instrument /
#   reload 族 / release_device_memory_under_pressure / pin_memory / UVA 工具
#   （utils.py:L15-L35）——观测/多模态/量化/权重 offload 域


# SUBTRACTED: @instrument(span_name="Initialize model") 观测装饰
#   （utils.py:L40）

# SOURCE: vllm/model_executor/model_loader/utils.py:L41-L64 initialize_model
#   减法子集（新式签名校验三行逐字——inspect.signature 验 (vllm_config,
#   prefix)；set_current_vllm_config 上下文内构造；老式 kwargs 猜参兼容段
#   L66-L97 与 record_metadata_for_reloading 删除——deprecated 面）
def initialize_model(
    vllm_config: VllmConfig,
    *,
    prefix: str = "",
    model_class: type[nn.Module] | None = None,
    model_config: ModelConfig | None = None,
) -> nn.Module:
    """Initialize a model with the given configurations."""
    # SOURCE: vllm/model_executor/model_loader/utils.py:L41-L64 initialize_model
    if model_config is None:
        model_config = vllm_config.model_config
    if model_class is None:
        model_class, _ = get_model_architecture(model_config)

    # SUBTRACTED: configure_quant_config(vllm_config.quant_config, model_class)
    #   （utils.py:L54-L55）——量化分发 ch27

    signatures = inspect.signature(model_class.__init__)
    all_params = [param.name for param in signatures.parameters.values()]
    if "vllm_config" in all_params and "prefix" in all_params:
        # new-style model class
        with set_current_vllm_config(vllm_config, check_compile=True, prefix=prefix):
            model = model_class(vllm_config=vllm_config, prefix=prefix)
            # SUBTRACTED: record_metadata_for_reloading（utils.py:L63）
            #   ——torchao reload 域
            return model

    # SUBTRACTED: 老式模型类的 kwargs 猜参兼容段（utils.py:L66-L97）——
    #   DeprecationWarning 面；本章六条目全是新式
    raise ValueError(
        "vLLM model class should accept `vllm_config` and `prefix` as "
        "input arguments. (old-style model class: deprecated — SUBTRACTED)"
    )


# SOURCE: vllm/model_executor/model_loader/utils.py:L100-L152
#   process_weights_after_loading —— 减法子集（两轮主干逐字：第一轮
#   quant_method.process_weights_after_loading + update_param_tp_status 调和；
#   第二轮 AttentionLayerBase 层的 process_weights_after_loading(act_dtype)；
#   release_device_memory_under_pressure/HpcModule/torchao 尾段删除）
def process_weights_after_loading(
    model: nn.Module, model_config: ModelConfig, target_device: torch.device
) -> None:
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            # When quant methods need to process weights after loading
            # (for repacking, quantizing, etc), they expect parameters
            # to be on the global target device. This scope is for the
            # case where cpu offloading is used, where we will move the
            # parameters onto device for processing and back off after.
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)
            # process_weights_after_loading may swap in freshly-created
            # Parameters (e.g. FP8 requantization), which are stamped with the
            # global rank in BasevLLMParameter.__init__. Re-reconcile their TP
            # state to the layer so a later weight reload / RL weight-refit
            # narrows replicated (disable_tp) weights at the correct offset.
            if hasattr(module, "update_param_tp_status"):
                module.update_param_tp_status()
            # SUBTRACTED: release_device_memory_under_pressure
            #   （utils.py:L120-L122）——UMA 设备域

    # Initialize post-load attention weights for any attention layer and MM
    # encoder. NOTE: Happens after other modules so we can easily decompress
    # weights.
    # SOURCE: vllm/model_executor/model_loader/utils.py:L124-L134 第二轮
    #   子集：MMEncoderAttention 从 isinstance 判删除——多模态域
    for _, module in model.named_modules():
        if isinstance(module, AttentionLayerBase) and hasattr(
            module, "process_weights_after_loading"
        ):
            # TODO(lucas): see if there is a way to unify the signatures
            # of process_weights_after_loading
            with device_loading_context(module, target_device):
                module.process_weights_after_loading(model_config.dtype)

    # SUBTRACTED: HpcModule 轮与模型级 hook、torchao reload attrs
    #   （utils.py:L136-L152）——HPC/torchao 域


# SOURCE: vllm/model_executor/model_loader/utils.py:L155-L197 device_loading_
#   context —— 减法子集（CPU 目标早退 + 参数搬运/恢复主干逐字；pin_memory/
#   UVA 回迁面删除）
@contextmanager
def device_loading_context(module: torch.nn.Module, target_device: torch.device):
    # SOURCE: vllm/model_executor/model_loader/utils.py:L155-L197 device_loading_
    if target_device.type == "cpu":
        # If target is CPU, no need to move anything
        yield module
        return

    original_device_states: dict[str, torch.device] = {}
    # SUBTRACTED: uva_offloaded_parameters 追踪（utils.py:L162-L163、L188-L197）
    #   ——UVA offload 域

    # Store original device states and move parameters to GPU if they're on CPU
    for name, p in module.named_parameters():
        if p.device.type == "cpu":
            original_device_states[name] = p.device
            p.data = p.data.to(target_device)
        # Parameters already on target device are not touched

    try:
        yield module

    finally:
        # Restore parameters to their original devices, ignoring new parameters
        for name, p in module.named_parameters():
            if name in original_device_states:
                original_device: torch.device = original_device_states[name]
                p.data = p.data.to(original_device)


# SOURCE: vllm/model_executor/model_loader/utils.py:L200-L201 _MODEL_ARCH_BY_HASH
#   缓存声明（逐字）
_MODEL_ARCH_BY_HASH = dict[int, tuple[type[nn.Module], str]]()
"""Caches the outputs of `_get_model_architecture`."""


# SOURCE: vllm/model_executor/model_loader/utils.py:L204-L236 _get_model_
#   architecture —— 减法子集（registry.resolve_model_cls 主干逐字；
#   transformers 后端回退告警与 convert_type embed/classify 适配转换删除）
def _get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    # SUBTRACTED: from vllm.model_executor.models.adapters import ... 位
    #   （utils.py:L205）——适配器面只作叙事（m15）

    # SOURCE: vllm/model_executor/model_loader/utils.py:L204-L236 _get_model_
    architectures = getattr(model_config.hf_config, "architectures", None) or []

    model_cls, arch = model_config.registry.resolve_model_cls(
        architectures,
        model_config=model_config,
    )

    # SUBTRACTED: transformers 后端回退告警（utils.py:L214-L222）——另一条
    #   产品线（delete[8]）
    # SUBTRACTED: convert_type embed/classify 适配转换（utils.py:L224-L234）
    #   ——delete[3]，m15 叙事面

    return model_cls, arch


# SOURCE: vllm/model_executor/model_loader/utils.py:L239-L255 get_model_
#   architecture（逐字——hash 键缓存：model/convert_type/runner_type/
#   trust_remote_code/model_impl/architectures 六元组）
def get_model_architecture(model_config: ModelConfig) -> tuple[type[nn.Module], str]:
    # SOURCE: vllm/model_executor/model_loader/utils.py:L239-L255 get_model_
    key = hash(
        (
            model_config.model,
            model_config.convert_type,
            model_config.runner_type,
            model_config.trust_remote_code,
            model_config.model_impl,
            tuple(getattr(model_config.hf_config, "architectures", None) or []),
        )
    )
    if key in _MODEL_ARCH_BY_HASH:
        return _MODEL_ARCH_BY_HASH[key]

    model_cls_and_arch = _get_model_architecture(model_config)
    _MODEL_ARCH_BY_HASH[key] = model_cls_and_arch
    return model_cls_and_arch


# SOURCE: vllm/model_executor/model_loader/utils.py:L258-L259 get_model_cls
#   （逐字）
def get_model_cls(model_config: ModelConfig) -> type[nn.Module]:
    # SOURCE: vllm/model_executor/model_loader/utils.py:L258-L259 get_model_cls
    return get_model_architecture(model_config)[0]


# SOURCE: vllm/model_executor/model_loader/utils.py:L262-L263
#   get_architecture_class_name（逐字）
def get_architecture_class_name(model_config: ModelConfig) -> str:
    # SOURCE: vllm/model_executor/model_loader/utils.py:L262-L263
    return get_model_architecture(model_config)[1]


# SUBTRACTED: ParamMapping/configure_quant_config（utils.py:L266-L317）——
#   量化分发域（ch27）
