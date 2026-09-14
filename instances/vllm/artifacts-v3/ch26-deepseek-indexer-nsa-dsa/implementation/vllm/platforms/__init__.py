# SOURCE: vllm/platforms/__init__.py
# HOST SEAM：current_platform 的单进程退化承载（ch23/ch25 同款骨架 + 本章消费面）。
# 真实 platforms/__init__.py 按可见加速器实例化 Platform 子类；host 测试无任何
# 加速器——按 Platform 基类（platforms/interface.py）的默认行为回答：
#   is_cuda/is_rocm/is_xpu… → False（sparse_attn_indexer 的 use_cooperative/
#     use_persistent 判定恒 False → 走 top_k_per_row_decode 兜底核——真实
#     非 CUDA 平台同型分派；Indexer.forward 的 ROCm/fused 分支同为死支）
#   opaque_attention_op → False（use_direct_call=True）
#   fp8_dtype → torch.float8_e4m3fn
#   get_device_capability → None / is_device_capability_family → False
#     （builder 的 use_flattening 与 FlashMLASparseImpl 的 prefill_padding=64
#     均按真实非 SM100 形态取值）
#   device_type → "cpu"（topk_indices_buffer 等分配在 CPU）
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

    # SOURCE: vllm/platforms/interface.py Platform.is_xpu —— HOST SEAM
    @classmethod
    def is_xpu(cls) -> bool:
        # SOURCE: vllm/platforms/interface.py Platform.is_xpu —— HOST SEAM
        return False

    # SOURCE: vllm/platforms/interface.py Platform.is_cpu —— HOST SEAM
    @classmethod
    def is_cpu(cls) -> bool:
        # SOURCE: vllm/platforms/interface.py Platform.is_cpu —— HOST SEAM
        return True

    # SOURCE: vllm/platforms/interface.py Platform.is_out_of_tree —— HOST SEAM
    @classmethod
    def is_out_of_tree(cls) -> bool:
        return False

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
    #   —— HOST SEAM：无设备 → None
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

    # SOURCE: vllm/platforms/interface.py Platform.has_device_capability
    #   —— HOST SEAM：无设备恒 False
    @classmethod
    def has_device_capability(cls, major: int) -> bool:
        # SOURCE: vllm/platforms/interface.py —— HOST SEAM
        return False

    # SOURCE: vllm/platforms/interface.py Platform.device_name —— HOST SEAM
    device_name = "cpu"

    # SOURCE: vllm/platforms/interface.py Platform.device_type —— HOST SEAM
    device_type = "cpu"

    # SOURCE: vllm/platforms/interface.py Platform.dispatch_key —— HOST SEAM
    #   （CPU 平台位——direct_register_custom_op 的算子按 CPU 键注册）
    dispatch_key = "CPU"

    # SOURCE: vllm/platforms/cuda.py CudaPlatform.get_attn_backend_cls（显式
    #   后端支）—— HOST SEAM 子集：显式 backend 名走注册表解析（真实第三方
    #   显式指定路径）；backend=None 的自动优先级面（use_mla×算力代分档）
    #   归 ch21——host 平台无可用 MLA 后端，与真实 CPU 平台同型的失败面。
    @classmethod
    def get_attn_backend_cls(cls, backend, attn_selector_config=None,
                             num_heads=None):
        # SOURCE: vllm/platforms/cuda.py:L396-L423 显式指定只校验不回退
        #   —— HOST SEAM 子集
        if backend is None:
            raise NotImplementedError(
                "Automatic attention backend selection is ch21's domain; "
                "no MLA backend is available on a CPU host (same failure "
                "mode as the real selector on a platform without backends)."
            )
        from vllm.v1.attention.backends.registry import AttentionBackendEnum

        selected = AttentionBackendEnum[backend]
        cls_ = selected.get_class()
        return f"{cls_.__module__}.{cls_.__qualname__}"


# SOURCE: vllm/platforms/__init__.py current_platform —— HOST SEAM 实例位
current_platform = _HostPlatform()
