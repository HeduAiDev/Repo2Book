"""ch24 精简版 — FlashInfer 注意力后端（只做减法，选后端表的代表项）。

对应 vllm/v1/attention/backends/flashinfer.py 的 FlashInferBackend。与 triton_attn.py 同理，
FLASHINFER 是 `_get_backend_priorities` 优先级列表里保留的**代表项**（在 Blackwell sm100 上它
被提到优先级第一）——做成自包含忠实缩影，好让选后端逻辑对三个候选都真跑
validate_configuration。

只保留决定合法性的能力探针 + 身份 staticmethod；Impl/Builder/CUDA 前向、nvfp4/fp8 量化 KV
布局全部 # SUBTRACTED。命名/签名/判据与 pin v0.21.0 一致。
"""

from __future__ import annotations

from typing import ClassVar

import torch

from backend import AttentionBackend


# SOURCE: vllm/v1/attention/backends/flashinfer.py:L327 (FlashInferBackend)
class FlashInferBackend(AttentionBackend):
    # SOURCE: vllm/v1/attention/backends/flashinfer.py:L328 (supported_dtypes)
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    # SOURCE: vllm/v1/attention/backends/flashinfer.py:L329 (supported_kv_cache_dtypes)
    # SUBTRACTED: 真实还含 fp8/fp8_e4m3/fp8_e5m2/nvfp4 量化 dtype；本章非量化主路径，保留标准三项。
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:  # SOURCE: vllm/v1/attention/backends/flashinfer.py:L340 (get_supported_kernel_block_sizes)
        # Blackwell 上 FlashInfer 只支持页大小 16/32/64。
        return [16, 32, 64]

    @staticmethod
    def get_name() -> str:  # SOURCE: vllm/v1/attention/backends/flashinfer.py:L346 (get_name)
        return "FLASHINFER"

    @staticmethod
    # SOURCE: vllm/v1/attention/backends/flashinfer.py:L357 (get_kv_cache_shape)
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # SUBTRACTED: nvfp4 时最后一维打包 fp4 数据 + fp8 块缩放（flashinfer.py:L365-L368）。
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:  # SOURCE: vllm/v1/attention/backends/flashinfer.py:L404 (get_supported_head_sizes)
        return [64, 128, 256]

    @classmethod
    # SOURCE: vllm/v1/attention/backends/flashinfer.py:L409 (supports_compute_capability)
    def supports_compute_capability(cls, capability) -> bool:
        from platform_cuda import DeviceCapability

        return capability >= DeviceCapability(7, 5) and capability <= DeviceCapability(
            12, 1
        )

    @classmethod
    # SOURCE: vllm/v1/attention/backends/flashinfer.py:L415 (supports_sink)
    def supports_sink(cls) -> bool:
        # SUBTRACTED: 真实探测 TRTLLM attention 是否可用（SM100 才支持 sink，flashinfer.py:L417-L421）；
        # host 无 flashinfer 运行时，保守返回基类默认 False（本章标准配置 has_sink=False 不触发此判据）。
        return False

    # SUBTRACTED: get_impl_cls/get_builder_cls 指向 FlashInferImpl/FlashInferMetadataBuilder
    # （flashinfer.py:L349/L353）及 CUDA 前向——选后端逻辑不实例化、不前向，代表项只需可被判定合法性。
    @staticmethod
    def get_impl_cls():  # SOURCE: vllm/v1/attention/backends/flashinfer.py:L349 (get_impl_cls)
        raise NotImplementedError("SUBTRACTED: FlashInfer 前向不在本章范围")

    @staticmethod
    def get_builder_cls():  # SOURCE: vllm/v1/attention/backends/flashinfer.py:L353 (get_builder_cls)
        raise NotImplementedError("SUBTRACTED: FlashInfer metadata builder 不在本章范围")
