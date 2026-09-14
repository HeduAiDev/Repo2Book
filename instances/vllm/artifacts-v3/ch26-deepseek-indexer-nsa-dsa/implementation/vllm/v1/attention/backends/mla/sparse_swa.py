# SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py
# ch26 切面（m10/m13）：DeepseekV4SWACache（L56-L104 逐字——SWA 支路缓存以
# 独立 prefix 注册 + 自报 SlidingWindowMLASpec + 与 C4A 压缩块共享物理张量的
# block_size=64 账注释原文）+ 三类层分型常量（L38-L52 逐字）+
# DeepseekSparseSWAMetadata（L163-… 消费字段 + get_prefill_chunk_plan 逐字）
# + DeepseekSparseSWABackend 名字位。
# SUBTRACTED：DeepseekSparseSWAMetadataBuilder/ComputePrefillMetadataKernel/
# tile scheduler 构建（L286-L856）——SWA metadata 构建域（→ ch25/ch21；
# 测试直接构造 metadata）。is_valid_token/decode_swa_indices 等字段面保留。
from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm.config import CacheConfig, VllmConfig, get_current_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    MultipleOf,
)
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    SlidingWindowMLASpec,
    get_kv_quant_mode,
)

# DeepseekV4 decode layer types, keyed by compress_ratio. Each type has a distinct
# (topk, extra_topk, extra_page_block_size) config, so they cannot share a
# FlashMLA tile-scheduler plan. Within a type, all ~60 DeepseekV4 layers share one
# plan per step because b / s_q / h_q / page_block_sizes / topks are identical.
# SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L38-L41 —— 逐字
_LAYER_TYPE_SWAONLY = "swaonly"
_LAYER_TYPE_C4A = "c4a"
_LAYER_TYPE_C128A = "c128a"


# SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L43-L52 _layer_type_for
#   —— 逐字
# SOURCE: sparse_swa.py:L43-L52（锚点双置）
def _layer_type_for(compress_ratio: int) -> str:
    if compress_ratio <= 1:
        return _LAYER_TYPE_SWAONLY
    if compress_ratio == 4:
        return _LAYER_TYPE_C4A
    if compress_ratio == 128:
        return _LAYER_TYPE_C128A
    raise ValueError(
        f"Unsupported DeepseekV4 compress_ratio={compress_ratio}; "
        "expected 1, 4, or 128."
    )


# SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L56-L104
#   DeepseekV4SWACache —— 逐字
class DeepseekV4SWACache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        window_size: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
    ):
        # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L57-L89
        #   （锚点双置）
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.window_size = window_size
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        # Block size is constrained by tensor sharing between SWA and C4A KV blocks.
        # Since both block types share the same physical tensor, they must use the
        # same page size. The C4A KV block shape [256//4, head_dim] = [64, head_dim]
        # determines the SWA block size of 64 tokens per block.
        # TODO(yifan): make SWA block size automatically determined and configurable.
        self.block_size = 64
        # uint8: fp8_ds_mla UE8M0 paged layout. bfloat16 / float8_e4m3fn:
        # contiguous full-cache layout.
        assert self.dtype in (torch.uint8, torch.bfloat16, torch.float8_e4m3fn)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L91-L102 —— 逐字
        # fp8_ds_mla's UE8M0 paged layout needs 576B alignment; contiguous
        # bf16/fp8 cache uses the natural element-size page.
        uses_fp8_ds_mla_layout = self.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            sliding_window=self.window_size,
            cache_dtype_str=self.cache_config.cache_dtype,
            # 576B for FlashMLA packing; 512B for FlashInfer sparse (#44577).
            alignment=576 if uses_fp8_ds_mla_layout else 512,
            model_version="deepseek_v4",
            kv_quant_mode=get_kv_quant_mode(self.cache_config.cache_dtype),
        )

    # SOURCE: sparse_swa.py:L103-L104（锚点双置）
    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L106-L107
        return DeepseekSparseSWABackend


# SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L110-… DeepseekSparseSWA-
#   Backend —— 减法子集（名字/块型/head 面）
class DeepseekSparseSWABackend(AttentionBackend):
    @staticmethod
    def get_impl_cls() -> type:
        # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py —— 章界位
        raise NotImplementedError(
            "DSV4 sparse SWA impl face lives on the platform attention "
            "subclass (nvidia/flashmla.py::forward_mqa, ch26)."
        )

    @staticmethod
    def get_name() -> str:
        # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L112-L114
        return "DEEPSEEK_SPARSE_SWA"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L116-L118
        return [MultipleOf(64)]

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:
        # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L120-L122
        return 256

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L124-L126
        return [512]

    @staticmethod
    def get_builder_cls() -> type:
        # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py —— 章界位
        raise NotImplementedError(
            "SWA metadata builder is ch25/ch21 domain (ch26 tests construct "
            "DeepseekSparseSWAMetadata directly)."
        )

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py —— fp8_ds_mla
        #   584B 位（逐字语义）
        if cache_dtype_str == "fp8_ds_mla":
            return (num_blocks, block_size, 584)
        return (num_blocks, block_size, head_size)


# SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L163-… DeepseekSparseSWA-
#   Metadata —— 减法子集（本章消费字段 + tile_sched 三槽 + chunk plan 逐字）
@dataclass
class DeepseekSparseSWAMetadata(AttentionMetadata):
    block_size: int
    num_decodes: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_prefill_tokens: int = 0
    # [num_decode_tokens]（1=有效 token、0=pad）
    is_valid_token: torch.Tensor | None = None  # [num_tokens]
    token_to_req_indices: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    # SWA 滑窗支路：decode 侧的滑窗 slot 与长度（NSA 三支路中『滑窗』的字面载体）
    decode_swa_indices: torch.Tensor | None = None  # [num_decode_tokens, window]
    decode_swa_lens: torch.Tensor | None = None  # [num_decode_tokens]
    prefill_seq_lens: torch.Tensor | None = None
    prefill_seq_lens_cpu: torch.Tensor | None = None
    prefill_gather_lens: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
    query_start_loc_cpu: torch.Tensor | None = None
    # chunk plan 的预算字段（builder 侧填）
    prefill_query_lens_cpu: torch.Tensor | None = None
    prefill_window_size: int = 0
    prefill_max_model_len: int = 0
    prefill_max_num_batched_tokens: int = 0
    # One FlashMLASchedMeta per layer type, shared across all same-type
    # layers within this decode step（三类层各有 planner、不共享）.
    tile_sched_swaonly: object | None = None
    tile_sched_c4a: object | None = None
    tile_sched_c128a: object | None = None

    # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L213-… get_prefill_
    #   chunk_plan —— 逐字
    def get_prefill_chunk_plan(
        self, compress_ratio: int, prefill_chunk_size: int
    ) -> list[tuple[int, int, int, int]]:
        # SOURCE: vllm/v1/attention/backends/mla/sparse_swa.py:L213-…（锚点双置）
        if self.num_prefills == 0:
            return []

        assert self.prefill_seq_lens_cpu is not None
        assert self.prefill_query_lens_cpu is not None

        # query_len <= max_num_batched_tokens and
        # gather_len = query_len + min(prefix_len, window_size - 1), so the
        # worst-case gathered width is bounded by
        # max_num_batched_tokens + window_size - 1. The compressed prefix pool
        # is bounded by ceil(max_model_len / compress_ratio).
        max_workspace_area = prefill_chunk_size * (
            (
                0
                if compress_ratio <= 1
                else cdiv(self.prefill_max_model_len, compress_ratio)
            )
            + self.prefill_window_size
            + self.prefill_max_num_batched_tokens
        )
        prefix_lens_cpu = self.prefill_seq_lens_cpu - self.prefill_query_lens_cpu
        gather_lens_cpu = self.prefill_query_lens_cpu + torch.clamp(
            prefix_lens_cpu, min=0, max=self.prefill_window_size - 1
        )
        compressed_lens_cpu = (
            torch.zeros_like(self.prefill_seq_lens_cpu)
            if compress_ratio <= 1
            else torch.div(
                self.prefill_seq_lens_cpu,
                compress_ratio,
                rounding_mode="floor",
            )
        )

        chunk_plan: list[tuple[int, int, int, int]] = []
        chunk_start = 0
        while chunk_start < self.num_prefills:
            chunk_max_compressed = int(compressed_lens_cpu[chunk_start].item())
            chunk_max_gather = int(gather_lens_cpu[chunk_start].item())
            chunk_end = chunk_start + 1

            while chunk_end < self.num_prefills:
                candidate_max_compressed = max(
                    chunk_max_compressed,
                    int(compressed_lens_cpu[chunk_end].item()),
                )
                candidate_max_gather = max(
                    chunk_max_gather,
                    int(gather_lens_cpu[chunk_end].item()),
                )
                candidate_width = candidate_max_compressed + candidate_max_gather
                candidate_area = (chunk_end - chunk_start + 1) * candidate_width
                if candidate_area > max_workspace_area:
                    break
                chunk_max_compressed = candidate_max_compressed
                chunk_max_gather = candidate_max_gather
                chunk_end += 1

            chunk_plan.append(
                (
                    chunk_start,
                    chunk_end,
                    chunk_max_compressed,
                    chunk_max_compressed + chunk_max_gather,
                )
            )
            chunk_start = chunk_end

        return chunk_plan

    # SUBTRACTED: DeepseekSparseSWAMetadataBuilder / ComputePrefillMetadata-
    #   Kernel / build_tile_scheduler（sparse_swa.py:L286-L856）——SWA metadata
    #   构建域（→ ch25/ch21）
