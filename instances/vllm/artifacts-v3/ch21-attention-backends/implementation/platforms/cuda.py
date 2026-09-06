# Subtract-only companion for v3 ch21 — vllm/platforms/cuda.py
# (pin v0.27.1 / 6e448d0ea). 本章切面（站 3-5）：CUDA 平台的选后端三件——
# _get_backend_priorities 优先级表（use_mla × 算力代 × fp8 KV × num_heads）、
# get_valid_backends 逐个 validate 回退（ImportError 容忍）、get_attn_backend_cls
# 显式/自动两路裁决 + 胜者日志。CudaPlatform 类只保留选后端线消费面
# （device_name / get_device_capability / 两 classmethod），NVML/NCCL/内存
# 等平台域以章界注记收窄。
from __future__ import annotations

from functools import cache
from typing import NamedTuple, TYPE_CHECKING

from .._host_seams import DeviceCapability, init_logger
from ..utils.torch_utils import is_quantized_kv_cache
from ..v1.attention.backends.registry import AttentionBackendEnum

if TYPE_CHECKING:
    from ..v1.attention.backend import AttentionBackend
    from ..v1.attention.selector import AttentionSelectorConfig

logger = init_logger(__name__)


# SOURCE: vllm/platforms/cuda.py:L82-L163 _get_backend_priorities ——（逐字）
#   优先级表本体：列表序即优先级（priority=enumerate 下标）
@cache
def _get_backend_priorities(  # SOURCE: vllm/platforms/cuda.py
    use_mla: bool,
    device_capability: DeviceCapability,
    num_heads: int | None = None,
    kv_cache_dtype: str | None = None,
    use_non_causal: bool = False,
) -> list[AttentionBackendEnum]:
    """Get backend priorities with lazy import to avoid circular dependency."""

    if use_mla:
        if device_capability.major == 10:
            # Sparse MLA backend priorities
            # See https://github.com/vllm-project/vllm/issues/35807 for
            # benchmark results
            if kv_cache_dtype is not None and is_quantized_kv_cache(kv_cache_dtype):
                # Prefer FlashInfer for fp8 kv cache
                sparse_backends = [
                    AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
                    AttentionBackendEnum.FLASHMLA_SPARSE,
                ]
            else:
                # BF16 KV Cache
                # Prefer FlashInfer at low head counts (FlashMLA uses padding)
                if num_heads is not None and num_heads <= 16:
                    sparse_backends = [
                        AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
                        AttentionBackendEnum.FLASHMLA_SPARSE,
                    ]
                else:
                    sparse_backends = [
                        AttentionBackendEnum.FLASHMLA_SPARSE,
                        AttentionBackendEnum.FLASHINFER_MLA_SPARSE,
                    ]

            return [
                AttentionBackendEnum.FLASHINFER_MLA,
                # R1 dims + FP8 KV only; rejected by supports_combination
                # otherwise. Behind FLASHINFER_MLA: wins past bs≈8, regresses
                # at bs≤2.
                AttentionBackendEnum.TOKENSPEED_MLA,
                AttentionBackendEnum.CUTLASS_MLA,
                AttentionBackendEnum.FLASH_ATTN_MLA,
                AttentionBackendEnum.FLASHMLA,
                AttentionBackendEnum.TRITON_MLA,
                *sparse_backends,
            ]
        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120,
            ]
        else:
            return [
                AttentionBackendEnum.FLASH_ATTN_MLA,
                AttentionBackendEnum.FLASHMLA,
                AttentionBackendEnum.FLASHINFER_MLA,
                AttentionBackendEnum.TRITON_MLA,
                AttentionBackendEnum.FLASH_ATTN_MLA_SPARSE,
                AttentionBackendEnum.FLASHMLA_SPARSE,
            ]
    else:
        # SM100f defaults to FlashInfer for TRTLLM causal attention, but its non-causal
        # cutlass path (used for dflash attention) is known to have problems.
        # So prefer FlashAttention when non-causal on SM100f.
        if device_capability.major == 10 and not use_non_causal:
            return [
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
                AttentionBackendEnum.TURBOQUANT,
            ]
        else:
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
                AttentionBackendEnum.TURBOQUANT,
            ]


# SOURCE: vllm/platforms/cuda.py:L166-L168 _backend_cls_path ——（逐字）
def _backend_cls_path(backend_cls: type[AttentionBackend]) -> str:
    module, qualname = backend_cls.full_cls_name()
    return f"{module}.{qualname}"


# SOURCE: vllm/platforms/cuda.py:L171-L172 _get_attn_backend_class ——（逐字）
def _get_attn_backend_class(backend: AttentionBackendEnum) -> type[AttentionBackend]:
    return backend.get_class()


# SOURCE: vllm/platforms/cuda.py:L175-L178 _BackendCandidate ——（逐字）
#   幸存候选三元组：类 + 枚举 + 优先级
class _BackendCandidate(NamedTuple):  # SOURCE: vllm/platforms/cuda.py
    backend_class: type[AttentionBackend]
    backend: AttentionBackendEnum
    priority: int


# SUBTRACTED: with_nvml_context / _get_wsl_kernel_version / pynvml 装配与
#   torch.backends 面板（L181-L207 平台域）——NVML 探测不进 host。


# SOURCE: vllm/platforms/cuda.py:L208 CudaPlatformBase —— 本章切面：只保留
#   选后端线消费面（device_name / get_device_capability / get_valid_backends /
#   get_attn_backend_cls）；import_kernels/supported_dtypes/NCCL/内存等平台域
#   （L222-L255、L494-L1016）以章界注记收窄。
class CudaPlatform:
    device_name: str = "cuda"  # SOURCE: vllm/platforms/cuda.py:L210（逐字）
    device_type: str = "cuda"  # SOURCE: vllm/platforms/cuda.py:L211（逐字）

    # HOST SEAM 装配位：真实 get_device_capability 经 NVML
    #   （vllm/platforms/cuda.py:L733-L742 nvmlDeviceGetCudaComputeCapability，
    #   不受 CUDA_VISIBLE_DEVICES 影响、不初始化 CUDA）；host 无 pynvml——
    #   以类属性注入算力代（测试注入 SM90/SM100/SM12 驱动优先级表分档）。
    _seam_device_capability: DeviceCapability | None = DeviceCapability(9, 0)

    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> DeviceCapability | None:
        # SOURCE: vllm/platforms/cuda.py:L733-L742（HOST SEAM 装配位）
        return cls._seam_device_capability


    @classmethod
    def get_valid_backends(  # SOURCE: vllm/platforms/cuda.py:L358-L394
        cls,
        device_capability: DeviceCapability,
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> tuple[
        list[_BackendCandidate],
        dict[AttentionBackendEnum, tuple[int, list[str]]],
    ]:
        valid_backends_priorities = []
        invalid_reasons: dict[AttentionBackendEnum, tuple[int, list[str]]] = {}

        backend_priorities = _get_backend_priorities(
            attn_selector_config.use_mla,
            device_capability,
            num_heads,
            attn_selector_config.kv_cache_dtype,
            attn_selector_config.use_non_causal,
        )
        for priority, backend in enumerate(backend_priorities):
            try:
                backend_class = _get_attn_backend_class(backend)
                invalid_reasons_i = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                invalid_reasons_i = ["ImportError"]
            if invalid_reasons_i:
                invalid_reasons[backend] = (priority, invalid_reasons_i)
            else:
                valid_backends_priorities.append(
                    _BackendCandidate(backend_class, backend, priority)
                )

        return valid_backends_priorities, invalid_reasons


    @classmethod
    def get_attn_backend_cls(  # SOURCE: vllm/platforms/cuda.py:L396-L492
        cls,
        selected_backend: AttentionBackendEnum | None,
        attn_selector_config: "AttentionSelectorConfig",
        num_heads: int | None = None,
    ) -> str:
        device_capability = cls.get_device_capability()
        assert device_capability is not None

        # First try checking just the selected backend, if there is one.
        if selected_backend is not None:
            try:
                backend_class = _get_attn_backend_class(selected_backend)
                invalid_reasons = backend_class.validate_configuration(
                    device_capability=device_capability,
                    **attn_selector_config._asdict(),
                )
            except ImportError:
                invalid_reasons = ["ImportError"]
            if invalid_reasons:
                raise ValueError(
                    f"Selected backend {selected_backend} is not valid for "
                    f"this configuration. Reason: {invalid_reasons}"
                )
            else:
                logger.info("Using %s backend.", selected_backend)
                return _backend_cls_path(backend_class)

        # No selected backend or the selected backend is invalid,
        # so we try finding a valid backend.
        valid_backends_priorities, all_invalid_reasons = cls.get_valid_backends(
            device_capability=device_capability,
            attn_selector_config=attn_selector_config,
            num_heads=num_heads,
        )
        reasons_str = (
            "{"
            + ", ".join(
                f"{backend.name}: [{', '.join(reasons)}]"
                for backend, (_, reasons) in all_invalid_reasons.items()
            )
            + "}"
        )
        config_str = attn_selector_config.__repr__()
        logger.debug_once(
            f"Some attention backends are not valid for {cls.device_name} with "
            f"{config_str}. Reasons: {reasons_str}."
        )
        if len(valid_backends_priorities) == 0:
            raise ValueError(
                f"No valid attention backend found for {cls.device_name} "
                f"with {config_str}. Reasons: {reasons_str}."
            )

        # We have found some valid backends. Select the one with the
        # highest priority.
        selected_candidate = min(
            valid_backends_priorities,
            key=lambda candidate: candidate.priority,
        )
        selected_backend_class = selected_candidate.backend_class
        selected_backend = selected_candidate.backend
        selected_priority = selected_candidate.priority

        # If the user specified --block-size (but not --attention-backend),
        # check whether that constraint precluded any higher-priority backends.
        if attn_selector_config.block_size is not None:
            excluded = [
                backend
                for backend, (priority, reasons) in all_invalid_reasons.items()
                if priority < selected_priority
                and reasons == ["block_size not supported"]
            ]
            if excluded:
                names = ", ".join(b.name for b in excluded)
                logger.warning(
                    "--block-size %d precluded higher-priority backend(s) "
                    "%s. Using %s instead, which may result in reduced "
                    "performance. Consider removing --block-size to "
                    "auto-select the optimal block size.",
                    attn_selector_config.block_size,
                    names,
                    selected_backend.name,
                )

        logger.info_once(
            "Using %s attention backend out of potential backends: %s.",
            selected_backend.name,
            "["
            + ", ".join(
                f"'{candidate.backend.name}'" for candidate in valid_backends_priorities
            )
            + "]",
        )

        return _backend_cls_path(selected_backend_class)

    # SUBTRACTED: get_supported_vit_attn_backends / get_vit_attn_backend
    #   （L494-L528——ViT 接口；TORCH_SDPA 枚举成员为它保留在 registry）；
    #   其余平台域（L529-L1016：memory/NCCL/graph capture/事件面）。


# SUBTRACTED: NvmlCudaPlatform / NonNvmlCudaPlatform 尾部装配（L717-L1016）——
#   NVML 工具族与最终平台类装配（detect_platform 域）；本切面由
#   _host_seams.current_platform 惰性转出（get_attn_backend_cls 等）。
