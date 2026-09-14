# SOURCE: vllm/models/deepseek_v4/sparse_mla.py
# ch26 切面（站 14 消费面）：DeepseekV4FlashMLABackend（L31-L105 减法子集）
# + DeepseekV4FlashMLAMetadata（L108-… 消费字段——C128A 的 metadata 期直算
# 全选表三字段）。SUBTRACTED：DeepseekV4FlashMLAMetadataBuilder（L120-L400+
# ——V4 metadata 构建/C128A 直算/tile scheduler 域；测试直接构造 metadata，
# 正文按真实源码 excerpt 讲 _build_c128a_metadata）。
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    MultipleOf,
)


# SOURCE: vllm/models/deepseek_v4/sparse_mla.py:L31-L105 DeepseekV4FlashMLA-
#   Backend —— 减法子集（名字/块型/head/身份证 + 584B 布局）
class DeepseekV4FlashMLABackend(AttentionBackend):
    """Deepseek-V4 sparse-MLA backend.

    Subclasses ``AttentionBackend`` directly (not the V3.2
    ``FlashMLASparseBackend``): Deepseek-V4 runs its own attention layer
    (``DeepseekV4Attention``), so it does not reuse the V3.2 builder or impl, and
    only needs to declare its own metadata builder, KV-cache layout, and the
    sparse-MLA capability flags.
    """

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # SOURCE: vllm/models/deepseek_v4/sparse_mla.py:L54-L56
        return [256]

    @staticmethod
    def get_name() -> str:
        # SOURCE: vllm/models/deepseek_v4/sparse_mla.py:L58-L60
        return "FLASHMLA_SPARSE_DSV4"

    @staticmethod
    def get_builder_cls() -> type:
        # SOURCE: vllm/models/deepseek_v4/sparse_mla.py —— 章界位（builder
        #   域删；见文件头）
        raise NotImplementedError(
            "DSV4 FlashMLA metadata builder is not carried in the ch26 "
            "subtract-only facet (tests construct metadata directly)."
        )

    @staticmethod
    def get_impl_cls() -> type[Any]:
        # Deepseek-V4 runs its attention through ``DeepseekV4Attention.forward``,
        # not the generic ``Attention``/``MLAAttention`` layer, so the backend's
        # impl class is never instantiated.
        # SOURCE: vllm/models/deepseek_v4/sparse_mla.py:L68-L75
        raise NotImplementedError(
            "DeepseekV4FlashMLABackend has no separate impl class; Deepseek-V4 "
            "attention runs through DeepseekV4Attention."
        )

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # DeepSeek V4 layout: 448 NoPE + 64 RoPE = 512.
        # SOURCE: vllm/models/deepseek_v4/sparse_mla.py:L84-L86
        return [512]

    @classmethod
    def is_mla(cls) -> bool:
        # SOURCE: vllm/models/deepseek_v4/sparse_mla.py:L88-L90
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        # SOURCE: vllm/models/deepseek_v4/sparse_mla.py:L92-L94
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # SOURCE: vllm/models/deepseek_v4/sparse_mla.py:L103-L113 —— 逐字
        if cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 main MLA: 584B per token (448 NoPE + 128 RoPE + 8 fp8 scale).
            # head_size passed in is the semantic head_dim (512).
            return (num_blocks, block_size, 584)
        else:
            return (num_blocks, block_size, head_size)


# SOURCE: vllm/models/deepseek_v4/sparse_mla.py:L108-… DeepseekV4FlashMLA-
#   Metadata —— 减法子集（本章消费字段）
@dataclass
# SOURCE: vllm/models/deepseek_v4/sparse_mla.py:L108-…（锚点双置）
class DeepseekV4FlashMLAMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int
    topk_tokens: int

    # C128A：metadata 期直算的全选表（压缩后 candidates≤topk → 全选即最优）
    c128a_global_decode_topk_indices: torch.Tensor | None = None
    c128a_decode_topk_lens: torch.Tensor | None = None
    c128a_prefill_topk_indices: torch.Tensor | None = None

    # SUBTRACTED: DeepseekV4FlashMLAMetadataBuilder（sparse_mla.py:L120-…：
    #   compress slot 换算调用 + _build_c128a_metadata 直算 + tile scheduler
    #   构建）——V4 metadata 构建域；正文以真实源码 excerpt 呈现
