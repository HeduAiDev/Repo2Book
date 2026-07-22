"""ch24 精简版 — Triton 注意力后端（只做减法，选后端表的代表项）。

对应 vllm/v1/attention/backends/triton_attn.py 的 TritonAttentionBackend。本章 focus 是
FLASH_ATTN；TRITON_ATTN 是 `_get_backend_priorities` 优先级列表里保留的**代表项**——把它做成
与 flash_attn.py 同构的自包含忠实缩影，好让平台层选后端逻辑对三个候选**都真跑**
validate_configuration（而非依赖宿主装没装真 vllm）。

只保留决定「该后端在给定配置下是否合法」的能力探针（validate_configuration 经 backend.py
基类聚合调用它们）+ 身份 staticmethod。真实的 Impl/Builder/CUDA 前向、per-token-head 量化
KV 布局等全部 # SUBTRACTED —— 选后端逻辑不碰它们。命名/签名/判据与 pin v0.21.0 一致。
"""

from __future__ import annotations

from typing import ClassVar

import torch

from backend import AttentionBackend, AttentionType


# SOURCE: vllm/v1/attention/backends/triton_attn.py:L265 (TritonAttentionBackend)
class TritonAttentionBackend(AttentionBackend):
    # SOURCE: vllm/v1/attention/backends/triton_attn.py:L266 (supported_dtypes)
    # 注意：Triton 比 FA 多支持 float32（FA 仅 fp16/bf16）——这是两后端合法域的真实差异。
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    # SOURCE: vllm/v1/attention/backends/triton_attn.py:L271 (supported_kv_cache_dtypes)
    # SUBTRACTED: 真实还含 fp8/fp8_e4m3/fp8_e5m2/int8_per_token_head/fp8_per_token_head 等量化
    # dtype；本章非量化主路径，保留 auto/float16/bfloat16 即可判定标准配置合法性。
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "float16", "bfloat16"]

    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:  # SOURCE: vllm/v1/attention/backends/triton_attn.py:L283 (get_supported_kernel_block_sizes)
        # SUBTRACTED: 真实返回 [MultipleOf(16)]；本章用具体整数表达「块大小须为 16 的倍数」。
        return [16]

    @classmethod
    # SOURCE: vllm/v1/attention/backends/triton_attn.py:L287 (supports_block_size)
    def supports_block_size(cls, block_size: int | None) -> bool:
        if block_size is None:
            return True
        return block_size % 16 == 0

    @staticmethod
    def get_name() -> str:  # SOURCE: vllm/v1/attention/backends/triton_attn.py:L295 (get_name)
        return "TRITON_ATTN"

    @classmethod
    def supports_batch_invariance(cls) -> bool:  # SOURCE: vllm/v1/attention/backends/triton_attn.py:L299 (supports_batch_invariance)
        return True

    @staticmethod
    # SOURCE: vllm/v1/attention/backends/triton_attn.py:L306 (get_kv_cache_shape)
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        # SUBTRACTED: per-token-head 量化时 head_size 会 pad 出 scale 位（triton_attn.py:L316-L327）。
        # 注意 Triton 的逻辑 shape 是 (num_blocks, 2, ...)，与 FA 的 (2, num_blocks, ...) 不同。
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:  # SOURCE: vllm/v1/attention/backends/triton_attn.py:L360 (supports_head_size)
        return head_size >= 32

    @classmethod
    def supports_mm_prefix(cls) -> bool:  # SOURCE: vllm/v1/attention/backends/triton_attn.py:L364 (supports_mm_prefix)
        return True

    @classmethod
    def supports_sink(cls) -> bool:  # SOURCE: vllm/v1/attention/backends/triton_attn.py:L368 (supports_sink)
        return True

    @classmethod
    # SOURCE: vllm/v1/attention/backends/triton_attn.py:L372 (supports_attn_type)
    def supports_attn_type(cls, attn_type: str) -> bool:
        """TritonAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_alibi_sqrt(cls) -> bool:  # SOURCE: vllm/v1/attention/backends/triton_attn.py:L382 (supports_alibi_sqrt)
        return True

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:  # SOURCE: vllm/v1/attention/backends/triton_attn.py:L386 (supports_compute_capability)
        return True

    # SUBTRACTED: get_impl_cls/get_builder_cls 指向 TritonAttentionImpl/TritonAttentionMetadataBuilder
    # （triton_attn.py:L302/L355）及其 CUDA/Triton 前向、metadata 翻译——选后端逻辑不实例化后端、
    # 不触发前向，本代表项只需能被 validate_configuration 聚合判定合法性。
    @staticmethod
    def get_impl_cls():  # SOURCE: vllm/v1/attention/backends/triton_attn.py:L302 (get_impl_cls)
        raise NotImplementedError("SUBTRACTED: Triton 前向不在本章范围")

    @staticmethod
    def get_builder_cls():  # SOURCE: vllm/v1/attention/backends/triton_attn.py:L355 (get_builder_cls)
        raise NotImplementedError("SUBTRACTED: Triton metadata builder 不在本章范围")
