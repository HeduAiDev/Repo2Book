# SOURCE: vllm/models/deepseek_v4/compressor.py —— ch26 主文件（站 13/m11）
# 全文减法：CompressorBackend/CompressorMetadata/CompressorMetadataBuilder
#（L61-L152 减法）+ CompressorStateCache（L155-L208 逐字——kv_state 前半+
# score_state 后半、压缩窗跨页边界的『等满 4 个再压』状态、与 KV 块共享物理
# 张量的 block_size 账注释原文）+ DeepseekCompressor（L211-L478 减法——
# fused_wkv_wgate 一枪 GEMM 出 KV+门控 score（L284-L292）、save_partial_states
# 前置、compress→norm→RoPE→量化写缓存融合尾步分派）。
# SUBTRACTED：_prefer_two_stage_compressor 的 ROCm two-stage 路径（L42-L45
# 保留函数、L259-L272 的 scratch 分支）与 head=512 cutedsl 分支（L422-L439
# ——主压缩 KV 池域 → ch25/ch28）；本章承载 indexer 路径（head_dim==128 的
# triton 分派臂逐字）。
from dataclasses import dataclass
from typing import Any, ClassVar, cast

import torch
from torch import nn

from vllm.config import CUDAGraphMode, VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import MergedColumnParallelLinear
from vllm.models.deepseek_v4.common.ops.fused_compress_quant_cache import (
    compress_norm_rope_store_triton,
)
from vllm.models.deepseek_v4.common.ops.fused_indexer_q import MXFP4_BLOCK_SIZE
from vllm.models.deepseek_v4.common.ops.save_partial_states import (
    save_partial_states,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import split_decodes_and_prefills
from vllm.v1.kv_cache_interface import (
    KVCacheSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


# SOURCE: vllm/models/deepseek_v4/compressor.py:L42-L45 _prefer_two_stage_
#   compressor —— 逐字（ROCm 判定；host 恒 False）
# SOURCE: vllm/models/deepseek_v4/compressor.py:L42-L45 —— 逐字（锚点双置）
def _prefer_two_stage_compressor() -> bool:
    # Platforms that favor the triton variant of two-stage compressor split.
    # Currently only tested on ROCm
    return current_platform.is_rocm()


# SOURCE: vllm/models/deepseek_v4/compressor.py:L48-L58 _get_c128_boundary
#   —— 逐字
# SOURCE: vllm/models/deepseek_v4/compressor.py:L48-L58 —— 逐字（锚点双置）
def _get_c128_boundary(metadata: CommonAttentionMetadata) -> bool | None:
    starts = metadata._num_computed_tokens_cpu
    if starts is None:
        return None

    starts_list = starts.tolist()
    query_start_loc = metadata.query_start_loc_cpu.tolist()
    return any(
        start % 128 + query_start_loc[i + 1] - query_start_loc[i] >= 128
        for i, start in enumerate(starts_list)
    )


# SOURCE: vllm/models/deepseek_v4/compressor.py:L61-L98 CompressorBackend
#   —— 减法子集（名字/块型/shape/stride_order）
class CompressorBackend(AttentionBackend):
    @staticmethod
    def get_impl_cls() -> type:
        # SOURCE: vllm/models/deepseek_v4/compressor.py —— HOST SEAM 章界位
        raise NotImplementedError(
            "Compressor impl face is carried by DeepseekCompressor.forward "
            "(ch26); generic impl plumbing is ch25/ch28 domain."
        )

    @staticmethod
    def get_name() -> str:
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L65-L67
        return "CompressorBackend"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L69-L71
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L73-L75
        return [512, 1024]

    @staticmethod
    def get_builder_cls() -> type["CompressorMetadataBuilder"]:
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L77-L79
        return CompressorMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L81-L90 —— 逐字
        assert num_kv_heads == 1
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L92-L98 —— 逐字
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


# SOURCE: vllm/models/deepseek_v4/compressor.py:L101-L109 CompressorMetadata
#   —— 逐字
@dataclass
# SOURCE: vllm/models/deepseek_v4/compressor.py:L101-L109 —— 逐字（锚点双置）
class CompressorMetadata:
    block_table: torch.Tensor
    slot_mapping: torch.Tensor
    block_size: int

    token_to_req_indices: torch.Tensor | None = None  # [num_tokens]
    num_decode_tokens: int | None = None
    c128_boundary: bool | None = None


# SOURCE: vllm/models/deepseek_v4/compressor.py:L112-L152
#   CompressorMetadataBuilder —— 逐字
class CompressorMetadataBuilder(AttentionMetadataBuilder):
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.ALWAYS

    def __init__(self, *args, **kwargs):
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L115-L125（锚点双置）
        super().__init__(*args, **kwargs)
        assert isinstance(self.kv_cache_spec, SlidingWindowMLASpec | MLAAttentionSpec)
        mla_spec = cast(SlidingWindowMLASpec | MLAAttentionSpec, self.kv_cache_spec)
        self.block_size = mla_spec.block_size

        self.token_to_req_indices = torch.zeros(
            self.vllm_config.scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device=self.device,
        )

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CompressorMetadata:
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L127-L152 —— 逐字
        token_to_req_indices = common_attn_metadata.token_to_req_indices(
            self.token_to_req_indices
        )
        num_decode_tokens = None
        if _prefer_two_stage_compressor():
            _, _, num_decode_tokens, _ = split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=1
            )
        return CompressorMetadata(
            block_table=common_attn_metadata.block_table_tensor.clamp_(min=0),
            slot_mapping=common_attn_metadata.slot_mapping,
            block_size=self.block_size,
            token_to_req_indices=token_to_req_indices,
            num_decode_tokens=num_decode_tokens,
            c128_boundary=(
                _get_c128_boundary(common_attn_metadata)
                if self.block_size == 8
                else None
            ),
        )


# SOURCE: vllm/models/deepseek_v4/compressor.py:L155-L208 CompressorStateCache
#   —— 逐字
class CompressorStateCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        prefix: str,
    ):
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L156-L189（锚点双置）
        super().__init__()
        self.state_dim = state_dim
        self.dtype = dtype
        self.prefix = prefix
        self.kv_cache = torch.tensor([])
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

        assert self.dtype == torch.float32
        assert compress_ratio in [4, 128]
        coff = 1 + (compress_ratio == 4)
        self.sliding_window = coff * compress_ratio
        # Block size is constrained by tensor sharing between compressor states
        # and KV blocks. Since compressor states share the same physical tensor
        # as KV blocks, they must use the same page size.
        # The KV block shape [256//4, head_dim] = [64, 584] determines:
        # - C4 compressor block shape [4, 2*512*2*4] -> block_size = 4
        # - C128 compressor block shape [8, 512*2*4] -> block_size = 8
        # TODO(yifan): make block size automatically determined and configurable.
        if compress_ratio == 4:
            self.block_size = 4
        elif compress_ratio == 128:
            self.block_size = 8
        else:
            raise ValueError(f"Invalid compress ratio: {compress_ratio}")

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L191-L203 —— 逐字
        # fp8_ds_mla is the UE8M0 paged layout and needs 576B alignment. Plain
        # full-cache rows share state pages with contiguous KV pages, so padding
        # would break page matching.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return SlidingWindowMLASpec(  # only has one vector instead of K + V
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
        )

    # SOURCE: vllm/models/deepseek_v4/compressor.py:L329-L478 —— 减法子集（锚点双置）
    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L207-L208
        return CompressorBackend


# SOURCE: vllm/models/deepseek_v4/compressor.py:L211-L478 DeepseekCompressor
#   —— 减法子集（indexer 路径逐字；head=512 cutedsl / two-stage 分支 →
#   ch25/ch28 章界）
class DeepseekCompressor(nn.Module):
    """DeepSeek V4 KV/score compressor.

    Owns the linear / norm / state-cache / ape state and the shared forward
    prologue (kv/score split, save_partial_states launch). The
    compress → norm → RoPE → store step is dispatched to a triton kernel
    (``compress_norm_rope_store_triton``) by default, except for the NVIDIA
    head_dim=128 indexer path which uses the cutedsl kernel
    (``compress_norm_rope_store_cutedsl``) for better performance.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        compress_ratio: int,
        hidden_size: int,
        head_dim: int,
        rotate: bool = False,
        prefix: str = "",
        k_cache_prefix="",
        use_fp4_cache: bool = False,
        eager_scratch_pool=None,
    ):
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L222-L327 __init__
        #   —— 减法子集（two-stage scratch 分支删——ROCm only）
        super().__init__()
        self.compress_ratio = compress_ratio
        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.rotate = rotate
        self.prefix = prefix
        self.k_cache_prefix = k_cache_prefix
        self.use_fp4_cache = use_fp4_cache
        self.eager_scratch_pool = eager_scratch_pool

        config = vllm_config.model_config.hf_config
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = self.head_dim - self.rope_head_dim
        self.rms_norm_eps = config.rms_norm_eps
        self.device = current_platform.device_type
        self.max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        self.overlap = compress_ratio == 4
        self.coff = 1 + self.overlap

        # SUBTRACTED: two-stage 压缩机的 fp32 scratch（L259-L272）——ROCm
        #   gfx950 专属（_prefer_two_stage_compressor host 恒 False）
        self.max_num_batched_tokens = (
            vllm_config.scheduler_config.max_num_batched_tokens
        )

        state_dtype = torch.float32
        self.ape = nn.Parameter(
            torch.empty(
                (compress_ratio, self.coff * self.head_dim),
                dtype=state_dtype,
                device=self.device,
            ),
            requires_grad=False,
        )

        # SOURCE: vllm/models/deepseek_v4/compressor.py:L284-L292 fused_wkv_
        #   wgate —— 逐字（一枪 GEMM 出 [coff·head_dim] 的 KV 与同宽门控
        #   score 两半）
        self.fused_wkv_wgate = MergedColumnParallelLinear(
            self.hidden_size,
            [self.coff * self.head_dim, self.coff * self.head_dim],
            bias=False,
            return_bias=False,
            quant_config=None,
            disable_tp=True,
            prefix=f"{prefix}.fused_wkv_wgate",
        )
        self.norm = RMSNorm(self.head_dim, self.rms_norm_eps)

        # SOURCE: vllm/models/deepseek_v4/compressor.py:L295-L300 —— 逐字
        self.state_cache = CompressorStateCache(
            state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
            dtype=state_dtype,
            compress_ratio=compress_ratio,
            prefix=f"{prefix}.state_cache",
        )

        # Save reference to static_forward_context for forward-time KV cache lookup.
        # get_current_vllm_config() is only available during __init__, not forward.
        self._static_forward_context = (
            vllm_config.compilation_config.static_forward_context
        )

        # SOURCE: vllm/models/deepseek_v4/compressor.py:L308-L327 —— 减法
        #   子集（head_dim==512 的量化布局位 → ch25/ch28；indexer 128 的
        #   fp8/fp4 两臂保留）
        if self.head_dim == 512:
            assert not use_fp4_cache, (
                "MXFP4 cache is only supported for indexer (head=128)"
            )
            self._quant_block = 64
            self._token_stride = self.nope_head_dim + self.rope_head_dim * 2
            self._scale_dim = self.nope_head_dim // 64 + 1  # 7 real + 1 pad
        elif self.head_dim == 128:
            if use_fp4_cache:
                self._quant_block = MXFP4_BLOCK_SIZE
                self._token_stride = self.head_dim // 2
                self._scale_dim = self.head_dim // MXFP4_BLOCK_SIZE
            else:
                self._quant_block = 128
                self._token_stride = self.head_dim
                self._scale_dim = 4  # single float32 scale
        else:
            raise ValueError(
                f"Unsupported head_dim for fused quant+cache: {self.head_dim}"
            )

    def forward(
        self,
        # [num_tokens, 2 * self.coff * self.head_dim]
        kv_score: torch.Tensor,
        # [num_tokens]
        positions: torch.Tensor,
        rotary_emb,
    ) -> None:
        # SOURCE: vllm/models/deepseek_v4/compressor.py:L329-L478 forward
        #   —— 减法子集（cutedsl/two-stage 分派臂 → ch25/ch28；indexer
        #   triton 臂逐字）
        # Each of shape [num_tokens, coff * self.head_dim]
        # input bf16, output are fp32
        kv, score = kv_score.split(
            [self.coff * self.head_dim, self.coff * self.head_dim], dim=-1
        )

        # Get the metadata and handle dummy profiling run.
        forward_context = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if not isinstance(attn_metadata, dict):
            return

        state_metadata = cast(
            CompressorMetadata, attn_metadata[self.state_cache.prefix]
        )
        token_to_req_indices = state_metadata.token_to_req_indices
        slot_mapping = state_metadata.slot_mapping
        num_actual = slot_mapping.shape[0]
        block_table = state_metadata.block_table
        block_size = state_metadata.block_size

        # [num_blocks, block_size, kv_dim+score_dim], where kv_dim == score_dim
        state_cache = self.state_cache.kv_cache
        # kv_state stored in first half, score_state stored in second half
        state_width = state_cache.shape[-1] // 2
        pdl_kwargs = (
            {}
            if current_platform.is_rocm() or current_platform.is_xpu()
            else {"launch_pdl": False}
        )

        # Store the KV and score (with fused APE addition) in the state.
        save_partial_states(
            kv=kv,
            score=score,
            ape=self.ape,
            positions=positions,
            state_cache=state_cache,
            slot_mapping=slot_mapping,
            block_size=block_size,
            state_width=state_width,
            compress_ratio=self.compress_ratio,
            pdl_kwargs=pdl_kwargs,
        )

        # full graph cannot branch on per-step CPU metadata after capture
        if (
            current_platform.is_cuda()
            and self.head_dim == 512
            and self.compress_ratio == 128
            and forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL
            and state_metadata.c128_boundary is False
        ):
            return

        # Fused: compress → RMSNorm → RoPE → FP8 quant → KV cache write.
        # RoPE requirements (kernel applies forward GPT-J style rotation):
        # - is_neox_style=False (interleaved pairs, NOT split-half)
        # - cos_sin_cache layout: [max_pos, rope_head_dim] with first half cos,
        #   second half sin (per-pair, length rope_head_dim // 2 each)
        # - applied to LAST rope_head_dim elements of head_dim
        # - position used: (positions // compress_ratio) * compress_ratio
        cos_sin_cache = rotary_emb.cos_sin_cache
        k_cache_metadata = cast(Any, attn_metadata[self.k_cache_prefix])
        k_cache_layer = self._static_forward_context[self.k_cache_prefix]
        kv_cache = k_cache_layer.kv_cache

        # SUBTRACTED: head=512 CUDA 的 cutedsl 分派（L422-L439）与 two-stage
        #   分派（L440-L448）——主压缩 KV 池域（→ ch25/ch28）；本章承载
        #   indexer（head_dim==128）/非 CUDA 的 triton 臂（L449-L452 原文）
        compress_norm_rope_store_fn = compress_norm_rope_store_triton
        extra_kwargs: dict[str, Any] = {}

        compress_norm_rope_store_fn(
            state_cache=state_cache,
            num_actual=num_actual,
            token_to_req_indices=token_to_req_indices,
            positions=positions,
            slot_mapping=slot_mapping,
            block_table=block_table,
            block_size=block_size,
            state_width=state_width,
            cos_sin_cache=cos_sin_cache,
            kv_cache=kv_cache,
            k_cache_metadata=k_cache_metadata,
            pdl_kwargs=pdl_kwargs,
            head_dim=self.head_dim,
            rope_head_dim=self.rope_head_dim,
            compress_ratio=self.compress_ratio,
            overlap=self.overlap,
            use_fp4_cache=self.use_fp4_cache,
            rms_norm_weight=self.norm.weight,
            rms_norm_eps=self.rms_norm_eps,
            quant_block=self._quant_block,
            token_stride=self._token_stride,
            scale_dim=self._scale_dim,
            **extra_kwargs,
        )
