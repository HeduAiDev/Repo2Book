# SOURCE: vllm/v1/kv_cache_interface.py
# ch26 切面（ch25 同源切面·本章消费字段）：spec 协议层——
#   KVQuantMode/get_kv_quant_mode + KVCacheSpec/AttentionSpec/FullAttentionSpec +
#   _apply_alignment_padding + MLAAttentionSpec（L388-L468 逐字——cache_dtype_str/
#   alignment/compress_ratio/model_version 字段是 IndexCache/压缩层与主 KV 分组
#   定池的原料；storage_block_size=block_size//compress_ratio）+
#   SlidingWindowSpec/SlidingWindowMLASpec（SWA 支路与压缩机状态缓存自报）。
# SUBTRACTED：spec 注册表族/分组账本（→ ch14 站 6 / ch25 站 6 域）。
from __future__ import annotations

from dataclasses import dataclass, fields
from enum import IntEnum
from typing import Self

import torch

from vllm.utils.math_utils import cdiv, round_up
from vllm.utils.torch_utils import get_dtype_size  # noqa: F401  (real_page_size_bytes 面)


# SOURCE: vllm/v1/kv_cache_interface.py:L33 KVQuantMode —— 减法子集（枚举位）
class KVQuantMode(IntEnum):
    """KV cache quantization mode.

    Used by attention backends and kernels to dispatch quantization logic
    without string matching on ``kv_cache_dtype``.
    """

    NONE = 0
    INT8_PER_TOKEN_HEAD = 1


# SOURCE: vllm/v1/kv_cache_interface.py:L68 get_kv_quant_mode —— 减法子集
#   （本章消费：非量化 → NONE）
def get_kv_quant_mode(kv_cache_dtype: str) -> KVQuantMode:
    """Map a ``kv_cache_dtype`` string to a :class:`KVQuantMode`."""
    # SOURCE: vllm/v1/kv_cache_interface.py:L68-…
    if kv_cache_dtype == "int8_per_token_head":
        return KVQuantMode.INT8_PER_TOKEN_HEAD
    return KVQuantMode.NONE


# SOURCE: vllm/v1/kv_cache_interface.py:L107-L108 KVCacheSpec —— 标记基类
@dataclass(frozen=True)
# SOURCE: vllm/v1/kv_cache_interface.py:L107-L108（锚点双置）
class KVCacheSpec:
    block_size: int

    # SUBTRACTED: spec kind/分组记账族——ch14 域


# SOURCE: vllm/v1/kv_cache_interface.py:L183-L231 AttentionSpec —— 减法子集
#   （消费字段；byte 账面按元素尺寸承载）
@dataclass(frozen=True, kw_only=True)
class AttentionSpec(KVCacheSpec):
    num_kv_heads: int
    head_size: int
    dtype: torch.dtype
    kv_quant_mode: KVQuantMode = KVQuantMode.NONE
    page_size_padded: int | None = None
    indexes_kv_by_block_stride: bool = False

    @property
    def real_page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py AttentionSpec.real_page_size_
        #   bytes —— 减法子集（非量化面）
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )

    def max_num_blocks_per_req(self, vllm_config, max_len: int) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py AttentionSpec.max_num_blocks_
        #   per_req —— 逐字
        parallel_config = vllm_config.parallel_config
        kv_shard_count = parallel_config.decode_context_parallel_size
        return cdiv(max_len, self.block_size * kv_shard_count)


# SOURCE: vllm/v1/kv_cache_interface.py:L234-L259 FullAttentionSpec —— 减法
#   子集（sliding_window 透传字段）
@dataclass(frozen=True, kw_only=True)
# SOURCE: vllm/v1/kv_cache_interface.py:L234-L259（锚点双置）
class FullAttentionSpec(AttentionSpec):
    """When hybrid allocator is disabled and the model contains both full
    attention layers and sliding window attention layers, sliding window
    attention are regarded as full attention in KV cache manager.
    """

    head_size_v: int = None  # type: ignore[assignment]

    sliding_window: int | None = None
    attention_chunk_size: int | None = None
    non_causal: bool = False


# SOURCE: vllm/v1/kv_cache_interface.py:L353-L359 _apply_alignment_padding
#   —— 逐字
# SOURCE: vllm/v1/kv_cache_interface.py:L353-L359（锚点双置）
def _apply_alignment_padding(spec: "MLAAttentionSpec | SlidingWindowMLASpec"):
    if spec.alignment is None:
        return
    actual_page_size = spec.real_page_size_bytes
    padded_page_size = round_up(actual_page_size, spec.alignment)
    if padded_page_size != actual_page_size:
        object.__setattr__(spec, "page_size_padded", padded_page_size)


# SOURCE: vllm/v1/kv_cache_interface.py:L388-L468 MLAAttentionSpec —— 逐字
#   （本章消费：compress_ratio/storage_block_size/584B/656B 特账/merge 四字段
#   断言）
@dataclass(frozen=True, kw_only=True)
class MLAAttentionSpec(FullAttentionSpec):
    # TODO(Lucas/Chen): less hacky way to do this
    cache_dtype_str: str | None = None
    # DeepseekV4 only fields. Non-DeepseekV4 MLA models leave these at defaults.
    alignment: int | None = None  # Default to None for no padding.
    compress_ratio: int = 1  # Default to 1 for no compression.
    model_version: str | None = None
    # Marks draft groups that flatten a non-causal query block into decode rows.
    non_causal_multi_token_decode: bool = False

    # SOURCE: vllm/v1/kv_cache_interface.py:L563-L565（锚点双置）
    def __post_init__(self):
        # SOURCE: vllm/v1/kv_cache_interface.py:L397-L399（锚点双置）
        super().__post_init__() if hasattr(super(), "__post_init__") else None
        _apply_alignment_padding(self)

    @property
    def storage_block_size(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L401-L405 storage_block_size
        return self.block_size // self.compress_ratio

    @property
    def real_page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L407-L426 —— 逐字（584B/656B
        #   特账）
        if self.cache_dtype_str == "fp8_ds_mla":
            if self.model_version == "deepseek_v4":
                # DeepseekV4: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B per token.
                # head_size stays semantic (512); bytes are determined here.
                return self.storage_block_size * 584
            # V3.2 main MLA: 656-byte custom layout (kv_lora_rank=512 +
            # qk_rope_head_dim=64, head_size=576). See flashmla_sparse.py.
            return self.block_size * 656
        if self.kv_quant_mode == KVQuantMode.INT4_PER_TOKEN_HEAD:
            head_dim = self.head_size // 2
        else:
            head_dim = self.head_size
        return (
            self.storage_block_size
            * self.num_kv_heads
            * head_dim
            * get_dtype_size(self.dtype)
        )

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        # SOURCE: vllm/v1/kv_cache_interface.py:L429-L468 merge —— 逐字
        #   （compress_ratio/model_version/cache_dtype_str/block_stride 四字段
        #   断言）
        assert all(isinstance(spec, MLAAttentionSpec) for spec in specs), (
            "All attention layers in the same KV cache group must be MLAAttentionSpec."
        )
        cache_dtype_str_set = set(spec.cache_dtype_str for spec in specs)
        compress_ratio_set = set(spec.compress_ratio for spec in specs)
        model_version_set = set(spec.model_version for spec in specs)
        block_stride_set = set(spec.indexes_kv_by_block_stride for spec in specs)
        assert (
            len(cache_dtype_str_set) == 1
            and len(compress_ratio_set) == 1
            and len(model_version_set) == 1
            and len(block_stride_set) == 1
        ), (
            "All attention layers in the same KV cache group must use the same "
            "quantization method, compress ratio, model version, and KV block "
            "stride indexing."
        )
        merged_spec = cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            dtype=specs[0].dtype,
            kv_quant_mode=specs[0].kv_quant_mode,
            page_size_padded=specs[0].page_size_padded,
            indexes_kv_by_block_stride=block_stride_set.pop(),
            cache_dtype_str=cache_dtype_str_set.pop(),
            compress_ratio=compress_ratio_set.pop(),
            model_version=model_version_set.pop(),
            non_causal_multi_token_decode=any(
                spec.non_causal_multi_token_decode for spec in specs
            ),
        )
        for spec in specs:
            for f in fields(AttentionSpec):
                assert getattr(spec, f.name) == getattr(merged_spec, f.name), (
                    "All attention layers in the same KV cache group must have "
                    "the same attention spec."
                )
        return merged_spec


# SOURCE: vllm/v1/kv_cache_interface.py:L558-L… SlidingWindowSpec —— 减法子集
@dataclass(frozen=True, kw_only=True)
# SOURCE: vllm/v1/kv_cache_interface.py:L558-L…（锚点双置）
class SlidingWindowSpec(AttentionSpec):
    sliding_window: int
    head_size_v: int = None  # type: ignore[assignment]

        # SOURCE: vllm/v1/kv_cache_interface.py:L563-L565（锚点双置）
    def __post_init__(self):
        if self.head_size_v is None:
            object.__setattr__(self, "head_size_v", self.head_size)


# SOURCE: vllm/v1/kv_cache_interface.py:L630-L… SlidingWindowMLASpec —— 逐字
#   字段面（SWA 支路缓存与压缩机状态缓存自报的 spec）
@dataclass(frozen=True, kw_only=True)
class SlidingWindowMLASpec(SlidingWindowSpec):
    """Sliding window attention with MLA cache format."""

    cache_dtype_str: str | None = None
    # DeepseekV4-only: see MLAAttentionSpec.model_version.
    alignment: int | None = None  # Default to None for no padding.
    compress_ratio: int = 1
    model_version: str | None = None

    # SOURCE: vllm/v1/kv_cache_interface.py:L563-L565（锚点双置）
    def __post_init__(self):
        # SOURCE: vllm/v1/kv_cache_interface.py:L640-L641（锚点双置）
        _apply_alignment_padding(self)

    @property
    def storage_block_size(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L643-L645 storage_block_size
        return self.block_size // self.compress_ratio

    @property
    def real_page_size_bytes(self) -> int:
        # SOURCE: vllm/v1/kv_cache_interface.py:L647-L… —— 584B 特账（逐字）
        if self.model_version == "deepseek_v4" and self.cache_dtype_str == "fp8_ds_mla":
            # DeepseekV4 FlashMLA: 448B NoPE + 128B RoPE + 8B fp8 scale = 584B
            # per token. FlashInfer's contiguous bf16/fp8 cache falls through to
            # the element-size formula below.
            return self.storage_block_size * 584
        return (
            2
            * self.block_size
            * self.num_kv_heads
            * self.head_size
            * get_dtype_size(self.dtype)
        )
