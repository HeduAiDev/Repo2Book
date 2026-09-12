# SOURCE: vllm/platforms/__init__.py
# HOST SEAM：current_platform 的单进程退化承载（ch23 同款骨架 + 本章消费面）。
# 真实 platforms/__init__.py 按可见加速器实例化 Platform 子类；host 测试无任何
# 加速器——按 Platform 基类（platforms/interface.py）的默认行为回答：
#   is_cuda/is_rocm/… → False（因此 _is_flashmla_available() 恒 False——
#     flashmla.py 的 else 支即真实「无 CUDA 扩展」形态）
#   opaque_attention_op → False（interface.py:L1116-L1121 基类默认——
#     use_direct_call=True，MLAAttention.forward 走直调路径）
#   fp8_dtype → torch.float8_e4m3fn（CUDA/ROCm 同款，interface.py 基类位）
#   get_device_capability → None（无设备 → prefill selector 走 FLASH_ATTN
#     回退支/显式 CUSTOM 支）
#   get_attn_backend_cls → 只实现显式后端支（自动优先级表归 ch21；host 平台
#     无可用 MLA 后端——与真实 CPU 平台同型的失败面）
from __future__ import annotations

import torch


# SOURCE: vllm/platforms/interface.py Platform —— HOST SEAM：基类默认行为位
class _HostPlatform:
    # SOURCE: vllm/platforms/interface.py Platform.is_cuda —— HOST SEAM
    @classmethod
    def is_cuda(cls) -> bool:
        # SOURCE: vllm/platforms/interface.py Platform.is_cuda —— HOST SEAM
        return False

    # SOURCE: vllm/platforms/interface.py Platform.is_rocm —— HOST SEAM
    @classmethod
    def is_rocm(cls) -> bool:
        # SOURCE: vllm/platforms/interface.py Platform.is_rocm —— HOST SEAM
        return False

    # SOURCE: vllm/platforms/interface.py Platform.is_cpu —— HOST SEAM
    @classmethod
    def is_cpu(cls) -> bool:
        # SOURCE: vllm/platforms/interface.py Platform.is_cpu —— HOST SEAM
        return True

    # SOURCE: vllm/platforms/interface.py:L1116-L1121 opaque_attention_op
    #   —— 基类默认 False
    @classmethod
    def opaque_attention_op(cls) -> bool:
        """
        Returns True if we register attention as one giant opaque custom op
        on the current platform
        """
        # SOURCE: vllm/platforms/interface.py:L1116-L1121 opaque_attention_op
        return False

    # SOURCE: vllm/platforms/interface.py Platform.fp8_dtype —— HOST SEAM
    #   （CUDA/ROCm 均 float8_e4m3fn）
    @classmethod
    def fp8_dtype(cls) -> torch.dtype:
        # SOURCE: vllm/platforms/interface.py Platform.fp8_dtype —— HOST SEAM
        return torch.float8_e4m3fn

    # SOURCE: vllm/platforms/interface.py Platform.get_device_capability
    #   —— HOST SEAM：无设备 → None（get_mla_prefill_backend 的
    #   "Device capability not available" 分支同型）
    @classmethod
    def get_device_capability(cls):
        # SOURCE: vllm/platforms/interface.py get_device_capability —— HOST SEAM
        return None

    # SOURCE: vllm/platforms/interface.py Platform.is_device_capability_family
    #   —— HOST SEAM：无设备恒 False
    @classmethod
    def is_device_capability_family(cls, major: int) -> bool:
        # SOURCE: vllm/platforms/interface.py —— HOST SEAM
        return False

    # SOURCE: vllm/platforms/interface.py Platform.device_name —— HOST SEAM
    device_name = "cpu"

    # SOURCE: vllm/platforms/interface.py Platform.dispatch_key —— HOST SEAM
    #   （CPU 平台位；本章无 torch.library 注册消费它）
    dispatch_key = "CPU"

    # SOURCE: vllm/v1/attention/backend.py AttentionBackend.indexes_kv_by_
    #   block_stride 的平台面经由 selector；本章经 get_attn_backend_cls 消费。
    # SOURCE: vllm/platforms/cuda.py CudaPlatform.get_attn_backend_cls（显式
    #   后端支）—— HOST SEAM 子集：显式 backend 名走注册表覆盖表解析
    #   （_ATTN_OVERRIDES→register_backend 的真实第三方注册流）；自动优先级
    #   表（use_mla×算力代分档，cuda.py:L82-L163）归 ch21。
    @classmethod
    def get_attn_backend_cls(cls, backend, attn_selector_config=None,
                             num_heads=None):
        # SOURCE: vllm/platforms/cuda.py:L396-L423 显式指定只校验不回退——
        #   HOST SEAM 子集（validate_configuration 全探针面归 ch21）
        if backend is None:
            raise NotImplementedError(
                "Automatic attention backend selection is ch21's domain; "
                "no MLA backend is available on a CPU host (same failure "
                "mode as the real selector on a platform without backends)."
            )
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        selected = AttentionBackendEnum[backend]
        cls = selected.get_class()
        return f"{cls.__module__}.{cls.__qualname__}"


# SOURCE: vllm/platforms/__init__.py current_platform —— HOST SEAM 实例位
current_platform = _HostPlatform()
