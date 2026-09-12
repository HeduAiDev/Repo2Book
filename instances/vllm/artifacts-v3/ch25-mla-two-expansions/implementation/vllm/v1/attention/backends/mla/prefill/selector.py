# SOURCE: vllm/v1/attention/backends/mla/prefill/selector.py
# ch25 切面（站 3 的第二选择轴）：prefill 家族选后端入口——MLAPrefill
# SelectorConfig + get_mla_prefill_backend（显式 mla_prefill_backend 支 +
# 算力不可得回退支逐字）+ _auto_select_mla_prefill_backend 的优先级循环
# 主干。优先级表按算力代分档逐字（Blackwell 分档含 DSV3 dims 特判）。
from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch

from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backends.mla.prefill.base import MLADimensions
from vllm.v1.attention.backends.mla.prefill.registry import MLAPrefillBackendEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend

logger = init_logger(__name__)


# SOURCE: vllm/v1/attention/backends/mla/prefill/selector.py:L23-L40
#   MLAPrefillSelectorConfig —— 逐字
class MLAPrefillSelectorConfig(NamedTuple):
    # SOURCE: vllm/v1/attention/backends/mla/prefill/selector.py:L23-L40（锚点双置：声明上方注释同文）
    """Hashable configuration for MLA prefill backend selection.

    This is analogous to AttentionSelectorConfig and contains model-specific
    configuration needed to select an MLA prefill backend, extracted from
    VllmConfig into a hashable form for caching.
    """

    dtype: torch.dtype
    mla_dimensions: MLADimensions = MLADimensions(
        qk_nope_head_dim=0,
        qk_rope_head_dim=0,
        v_head_dim=0,
    )

    def __repr__(self):
        # SOURCE: vllm/v1/attention/backends/mla/prefill/selector.py:L23-L40（锚点双置：声明上方注释同文）
        return (
            f"MLAPrefillSelectorConfig(dtype={self.dtype}, "
            f"mla_dimensions={self.mla_dimensions})"
        )


# SOURCE: vllm/v1/attention/backends/mla/prefill/selector.py:L43-L82
#   _get_mla_prefill_backend_priorities —— 逐字
def _get_mla_prefill_backend_priorities(
    device_capability: DeviceCapability,
    mla_dimensions: MLADimensions,
) -> list[MLAPrefillBackendEnum]:
    # SOURCE: vllm/v1/attention/backends/mla/prefill/selector.py:L43-L82（锚点双置：声明上方注释同文）
    """Get MLA prefill backend priorities based on device capability.

    Args:
        device_capability: The device's compute capability.
        mla_dimensions: The model's MLA head dimensions.

    Returns:
        List of backends in priority order (highest priority first).
    """
    from vllm.platforms import current_platform

    if current_platform.is_rocm():
        return [
            MLAPrefillBackendEnum.ROCM_AITER_FA,
            MLAPrefillBackendEnum.FLASH_ATTN,
        ]

    if device_capability.major == 10:  # Blackwell
        if mla_dimensions == MLADimensions(
            qk_nope_head_dim=192,
            qk_rope_head_dim=64,
            v_head_dim=256,
        ):
            return [
                MLAPrefillBackendEnum.TRTLLM_RAGGED,
                MLAPrefillBackendEnum.FLASH_ATTN,
                MLAPrefillBackendEnum.FLASHINFER,
                MLAPrefillBackendEnum.TOKENSPEED_MLA,
            ]
        return [
            MLAPrefillBackendEnum.FLASH_ATTN,
            MLAPrefillBackendEnum.TRTLLM_RAGGED,
            MLAPrefillBackendEnum.FLASHINFER,
            MLAPrefillBackendEnum.TOKENSPEED_MLA,
        ]
    else:  # Hopper (SM90) and older
        return [
            MLAPrefillBackendEnum.FLASH_ATTN,
        ]


# SOURCE: vllm/v1/attention/backends/mla/prefill/selector.py:L85-L160
#   get_mla_prefill_backend —— 逐字
def get_mla_prefill_backend(
    vllm_config: "VllmConfig",
) -> "type[MLAPrefillBackend]":
    # SOURCE: vllm/v1/attention/backends/mla/prefill/selector.py:L85-L160（锚点双置：声明上方注释同文）
    """Select the MLA prefill backend based on configuration and device.

    This function first checks for explicit user preferences via
    mla_prefill_backend in AttentionConfig, then falls back to automatic
    priority-based selection.

    Args:
        vllm_config: The vLLM configuration.

    Returns:
        The selected prefill backend class.
    """
    from vllm.platforms import current_platform

    device_capability = current_platform.get_device_capability()
    if device_capability is None:
        logger.info_once(
            "Device capability not available, using FlashAttention MLA prefill backend."
        )
        return MLAPrefillBackendEnum.FLASH_ATTN.get_class()

    attention_config = vllm_config.attention_config

    model_config = vllm_config.model_config
    if model_config is None:
        selector_config = MLAPrefillSelectorConfig(dtype=torch.get_default_dtype())
    else:
        hf_text_config = model_config.hf_text_config
        selector_config = MLAPrefillSelectorConfig(
            dtype=model_config.dtype,
            mla_dimensions=MLADimensions(
                qk_nope_head_dim=getattr(hf_text_config, "qk_nope_head_dim", 0),
                qk_rope_head_dim=getattr(hf_text_config, "qk_rope_head_dim", 0),
                v_head_dim=getattr(hf_text_config, "v_head_dim", 0),
            ),
        )

    if attention_config.mla_prefill_backend is not None:
        selected_backend = attention_config.mla_prefill_backend
        backend_cls: type[MLAPrefillBackend] | None = None
        try:
            backend_cls = selected_backend.get_class()
            invalid_reasons = backend_cls.validate_configuration(
                device_capability, selector_config
            )
        except ImportError:
            invalid_reasons = ["ImportError"]
        if invalid_reasons:
            raise ValueError(
                f"Selected MLA prefill backend {selected_backend.name} "
                f"is not valid for this configuration. "
                f"Reason: {invalid_reasons}"
            )
        assert backend_cls is not None
        logger.info_once("Using %s MLA prefill backend.", selected_backend.name)
        return backend_cls

    return _auto_select_mla_prefill_backend(
        device_capability,
        selector_config,
    )


def _auto_select_mla_prefill_backend(
    device_capability: DeviceCapability,
    selector_config: MLAPrefillSelectorConfig,
) -> "type[MLAPrefillBackend]":
    """Auto-select the best available MLA prefill backend.

    Args:
        device_capability: The device's compute capability.
        selector_config: Hashable configuration for backend selection.

    Returns:
        The selected prefill backend class.
    """
    # SOURCE: vllm/v1/attention/backends/mla/prefill/selector.py:L163-L211 —— 逐字
    priorities = _get_mla_prefill_backend_priorities(
        device_capability,
        selector_config.mla_dimensions,
    )
    all_invalid_reasons: dict[str, list[str]] = {}

    for backend_enum in priorities:
        backend_cls: type[MLAPrefillBackend] | None = None
        try:
            backend_cls = backend_enum.get_class()
            invalid_reasons = backend_cls.validate_configuration(
                device_capability, selector_config
            )
        except ImportError:
            invalid_reasons = ["ImportError"]
        if not invalid_reasons:
            assert backend_cls is not None
            logger.info_once("Using %s MLA prefill backend.", backend_enum.name)
            return backend_cls
        all_invalid_reasons[backend_enum.name] = invalid_reasons

    reasons_str = (
        "{"
        + ", ".join(
            f"{name}: [{', '.join(reasons)}]"
            for name, reasons in all_invalid_reasons.items()
        )
        + "}"
    )
    raise ValueError(f"No available MLA prefill backend, reasons: {reasons_str}")
