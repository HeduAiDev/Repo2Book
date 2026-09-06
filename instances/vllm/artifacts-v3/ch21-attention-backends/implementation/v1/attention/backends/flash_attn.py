# Subtract-only companion for v3 ch21 — vllm/v1/attention/backends/flash_attn.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions (delete[5]/delete[6], each marked
# `# SUBTRACTED:`), plus 章范围外域段以 SUBTRACTED+归属注记收窄。
#
# 本章主角文件之一：代表性具体后端四件套——FlashAttentionBackend（能力探针
# + KV 布局声明）、FlashAttentionMetadata（后端专属 dataclass——对比 Common
# 看『翻译』）、FlashAttentionMetadataBuilder（build 翻译 + FA3/FA2 CG 档 +
# update_block_table 换表复用）、FlashAttentionImpl（写腿 do_kv_cache_update +
# 读腿 forward）。
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...._host_seams import (
    current_platform,
    envs,
    init_logger,
)
from ....utils.torch_utils import (
    canonicalize_singleton_dim_strides,
    is_quantized_kv_cache,
)
from .utils import get_kv_cache_layout
from ..backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from .fa_utils import (
    flash_attn_supports_kv_cache_dtype,
    flash_attn_supports_quant_query_input,
    flash_attn_supports_sinks,
    get_flash_attn_version,
    is_fa_version_supported,
    is_flash_attn_varlen_func_available,
)

# SOURCE: vllm/v1/attention/backends/flash_attn.py:L37-L43 平台 import 面
#   （host 由 fa_utils 的 HOST SEAM 镜像承载，见彼处注记）
if is_flash_attn_varlen_func_available():
    from .fa_utils import (
        flash_attn_varlen_func,
        get_scheduler_metadata,
        reshape_and_cache_flash,
    )
from ...._host_seams import DeviceCapability
from ....utils.math_utils import round_up
from ...kv_cache_interface import AttentionSpec

if TYPE_CHECKING:
    from ...._host_seams import VllmConfigSeam as VllmConfig

logger = init_logger(__name__)

# SUBTRACTED: vllm.vllm_flash_attn 的 merge_attn_states / cp_lse_ag_out_rs /
#   dcp_a2a_lse_reduce / current_workspace_manager / cp_utils import（L32-L67）
#   ——cascade 数学（ch20 已立）与 DCP/工作空间域（delete[5] 同域）。


# SOURCE: vllm/v1/attention/backends/flash_attn.py:L72-L239 FlashAttentionBackend
#   ——（逐字）四件套之一：能力探针 + KV 布局声明
class FlashAttentionBackend(AttentionBackend):
    supported_dtypes: list[torch.dtype] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: list[str] = [
        "auto",
        "float16",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L82-L84
        return [MultipleOf(16)]

    # SOURCE: vllm/v1/attention/backends/flash_attn.py:L86 —— KV 写不含在
    #   forward() 里（独立写腿算子先写）
    forward_includes_kv_cache_update: bool = False

    @classmethod
    def get_preferred_block_size(cls, default_block_size: int) -> int:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L88-L92
        if current_platform.is_xpu():
            return max(default_block_size, 64)
        return super().get_preferred_block_size(default_block_size)

    @staticmethod
    def get_name() -> str:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L94-L96
        return "FLASH_ATTN"

    @classmethod
    def supports_sliding_window(cls) -> bool:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L98-L100
        return True

    @classmethod
    def supports_batch_invariance(cls) -> bool:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L102-L104
        return True

    @classmethod
    def supports_non_causal(cls) -> bool:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L106-L108
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L110-L118
        """FlashAttention supports all attention types."""
        return attn_type in (
            AttentionType.DECODER,
            AttentionType.ENCODER,
            AttentionType.ENCODER_ONLY,
            AttentionType.ENCODER_DECODER,
        )

    @classmethod
    def supports_per_head_quant_scales(cls) -> bool:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L120-L123
        fa_version = get_flash_attn_version()
        return fa_version is not None and fa_version >= 3

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L125-L127
        return FlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["FlashAttentionMetadataBuilder"]:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L129-L131
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L133-L144
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        # K and V are packed into the content dim: logical (B, H, N, 2*D).
        return (num_blocks, num_kv_heads, block_size, 2 * head_size)

    @staticmethod
    def get_kv_cache_stride_order(  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L146-L168
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        # `stride_order` indicates the permutation that gets us from
        # `get_kv_cache_shape` (logical (B, H, N, 2*D)) to the actual memory
        # layout we want.
        cache_layout = get_kv_cache_layout()
        if cache_layout == "NHD" and include_num_layers_dimension:
            # (num_blocks, num_layers, block_size, num_kv_heads, 2*head_size)
            return (1, 0, 3, 2, 4)
        elif cache_layout == "NHD":
            # (num_blocks, block_size, num_kv_heads, 2*head_size)
            stride_order = (0, 2, 1, 3)
        elif cache_layout == "HND" and include_num_layers_dimension:
            # (num_blocks, num_kv_heads, num_layers, block_size, 2*head_size)
            return (1, 2, 0, 3, 4)
        elif cache_layout == "HND":
            # (num_blocks, num_kv_heads, block_size, 2*head_size)
            stride_order = (0, 1, 2, 3)
        else:
            raise ValueError(f"Unknown cache layout format {cache_layout}.")
        return stride_order

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L170-L178
        if head_size % 8 != 0:
            return False
        if head_size <= 256:
            return True
        if is_fa_version_supported(4):
            return head_size <= 512
        return False

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: str | None) -> bool:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L180-L188
        if kv_cache_dtype is None:
            return True
        if kv_cache_dtype not in cls.supported_kv_cache_dtypes:
            return False
        if is_quantized_kv_cache(kv_cache_dtype):
            return flash_attn_supports_kv_cache_dtype(kv_cache_dtype)
        return True

    @classmethod
    def supports_mm_prefix(cls) -> bool:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L190-L192
        return is_fa_version_supported(4)

    @classmethod
    def supports_sink(cls) -> bool:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L194-L198
        if not is_flash_attn_varlen_func_available():
            return False
        return flash_attn_supports_sinks()

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L200-L202
        return capability >= DeviceCapability(8, 0)

    @classmethod
    def supports_combination(  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L204-L239
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        if has_sink and device_capability < DeviceCapability(9, 0):
            return "sink not supported on compute capability < 9.0"
        if (
            kv_cache_dtype is not None
            and is_quantized_kv_cache(kv_cache_dtype)
            and not flash_attn_supports_kv_cache_dtype(
                kv_cache_dtype,
                head_size=head_size,
                head_size_v=head_size,
                has_sinks=has_sink,
            )
        ):
            return "FP8 KV cache requires FA3 on SM90 or FA4 on SM100"
        if (
            use_mm_prefix
            and get_flash_attn_version(head_size=head_size, has_sinks=has_sink) != 4
        ):
            return (
                "mm_prefix (PrefixLM bidirectional attention) requires "
                "FlashAttention v4, which does not resolve for this "
                "head_size"
            )
        return None


# SOURCE: vllm/v1/attention/backends/flash_attn.py:L242-L299 FlashAttentionMetadata
#   ——（逐字）四件套之二：后端专属 dataclass（Common 字段改名搬入 +
#   FA 特有字段——DCP/cascade/mm/rswa 族字段面全保留，其生产支已删、
#   恒为默认值）
@dataclass
class FlashAttentionMetadata:  # SOURCE: vllm/v1/attention/backends/flash_attn.py
    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention.
    use_cascade: bool
    common_prefix_len: int
    cu_prefix_query_lens: torch.Tensor | None
    prefix_kv_lens: torch.Tensor | None
    suffix_kv_lens: torch.Tensor | None

    # For GQA DCP
    max_dcp_context_kv_len: int | None = None
    dcp_context_kv_lens: torch.Tensor | None = None

    # Split counts for FA2 DCP context attention. num_prefill_* tracks
    # context-bearing extend rows; pure prefills do not attend to DCP context.
    num_decode_reqs: int = 0
    num_prefill_reqs: int = 0
    num_decode_tokens: int = 0
    num_prefill_tokens: int = 0

    # Optional aot scheduling
    scheduler_metadata: torch.Tensor | None = None
    prefix_scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0

    causal: bool | torch.Tensor = True

    sliding_window: tuple[int, int] | None = None

    # PrefixLM bidirectional ranges for multimodal tokens.
    # Shape: (num_seqs, max_ranges, 2) int32, [start, end] per range.
    mm_prefix_range_tensor: torch.Tensor | None = None

    # Reference Sliding Window Attention (R-SWA) fields.
    # rswa_prefix_lens:  per-request prompt lengths [num_reqs], int32, CUDA.
    # rswa_window:       sliding window size (scalar int, for logic checks).
    # rswa_window_tensor: [1] int32 CUDA tensor — pre-allocated in build() so
    #   no CPU→CUDA copy is needed inside forward() during CUDA graph capture.
    # Only populated when the model uses R-SWA (Unlimited-OCR).
    rswa_prefix_lens: torch.Tensor | None = None
    rswa_window: int | None = None
    rswa_window_tensor: torch.Tensor | None = None


# SUBTRACTED: _get_sliding_window_configs（L302-L316）——唯一消费方是 build()
#   的 aot_sliding_window 多窗口回退（delete[6] L484-L498 连带删）。


# SOURCE: vllm/v1/attention/backends/flash_attn.py:L319-L330 _maybe_symmetrize_
#   window ——（逐字）非 causal 时把 (w, 0) 对称化成 (w, w)
def _maybe_symmetrize_window(  # SOURCE: vllm/v1/attention/backends/flash_attn.py
    window: tuple[int, int] | None,
    causal: bool | torch.Tensor,
) -> tuple[int, int] | None:
    """Make a causal sliding window ``(w, 0)`` symmetric ``(w, w)`` when attention
    is non-causal, so bidirectional queries attend in both directions. Leaves
    full-attention ``(-1, -1)`` and already-symmetric windows untouched.
    """
    non_causal = isinstance(causal, torch.Tensor) or causal is False
    if window is not None and window[0] >= 0 and window[1] == 0 and non_causal:
        return (window[0], window[0])
    return window


# SOURCE: vllm/v1/attention/backends/flash_attn.py:L333-L357 FlashAttention-
#   MetadataBuilder ——（逐字）四件套之三：FA3=ALWAYS / FA2=UNIFORM_BATCH 的
#   CG 档分叉 + 换表复用资格
class FlashAttentionMetadataBuilder(AttentionMetadataBuilder[FlashAttentionMetadata]):
    # FA3:
    # Supports full cudagraphs for all cases.
    #
    # FA2:
    # For FA2, a graph is captured with max_query_len=1, (which is what we
    # capture by default for num_tokens <= max_num_seqs when there is no
    # spec-decode) then these graphs will not work for mixed prefill-decode
    # (unlike FA3). This is due to special max_query_len=1 packed-GQA handling
    # in FA2.
    # In summary if we are running with spec decodes the graphs would
    # work for mixed prefill-decode and uniform-decode. But for non-spec decodes
    # the graphs would not work for mixed prefill-decode; sorta the inverse
    # of UNIFORM_SINGLE_TOKEN_DECODE.
    # There's probably a better way to describe this using `AttentionCGSupport`
    # but for now just set it to `UNIFORM_BATCH` to get use to drop down
    # to FULL_AND_PIECEWISE.
    # TODO(luka, lucas): audit FA2 as part of:
    #  https://github.com/vllm-project/vllm/issues/22945
    _cudagraph_support = (
        AttentionCGSupport.ALWAYS
        if get_flash_attn_version() == 3
        else AttentionCGSupport.UNIFORM_BATCH
    )
    supports_update_block_table: bool = True

    @classmethod
    def get_cudagraph_support(  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L359-L365
        cls,
        vllm_config: "VllmConfig",
        kv_cache_spec: "AttentionSpec",
    ) -> AttentionCGSupport:
        return cls._cudagraph_support

    def __init__(  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L367-L456（delete[6] 删 DCP/rswa 段）
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: "VllmConfig",
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.attention_config = vllm_config.attention_config

        self.num_heads_q = self.model_config.get_num_attention_heads(
            self.parallel_config
        )
        self.num_heads_kv = self.model_config.get_num_kv_heads(self.parallel_config)
        self.kv_cache_dtype = kv_cache_spec.dtype
        self.headdim = self.model_config.get_head_size()
        self.block_size = kv_cache_spec.block_size

        self.max_num_splits = 0  # No upper bound on the number of splits.
        self.aot_schedule = get_flash_attn_version() == 3

        try:
            from ...._host_seams import get_dcp_group

            self.dcp_world_size = get_dcp_group().world_size
            self.dcp_rank = get_dcp_group().rank_in_group
        except AssertionError:
            # DCP might not be initialized in testing
            self.dcp_world_size = 1
            self.dcp_rank = 0

        # SUBTRACTED: cp_kv_cache_interleave_size（L402-L404）——delete[6]：
        #   唯一消费方是 build() 的 DCP 分支（delete[5] L556-L616 删）。

        self.use_full_cuda_graph = (
            self.compilation_config.cudagraph_mode.has_full_cudagraphs()
        )
        self.max_cudagraph_size = self.compilation_config.max_cudagraph_capture_size

        if self.use_full_cuda_graph and self.aot_schedule:
            # FA3 scheduler_metadata size: 1 + round_up(batch_size, 4) * 4
            # The +1 is for the tile_count_semaphore (synchronization).
            # The 4 slots per batch element (num_prepare_batch_vectors) are:
            #   prepare_varlen + dynamic_split + sort_batches + head_swizzle
            # See: https://github.com/vllm-project/flash-attention/blob/5824e6e/hopper/flash_api.cpp#L664-L671  # noqa: E501
            max_batch_size = max(
                vllm_config.scheduler_config.max_num_seqs,
                self.max_cudagraph_size or 0,
            )
            self.scheduler_metadata = torch.zeros(
                1 + round_up(max_batch_size, 4) * 4,
                dtype=torch.int32,
                device=self.device,
            )
            # When using cuda graph, we need to set the upper bound of the
            # number of splits so that large enough intermediate buffers are
            # pre-allocated during capture.
            self.max_num_splits = (
                self.attention_config.flash_attn_max_num_splits_for_cuda_graph
            )

        # SUBTRACTED: _dcp_context_kv_lens 预分配（L433-L440）——delete[6]：
        #   消费方是 build() 的 DCP 分支（delete[5] 删）。

        # Sliding window size to be used with the AOT scheduler will be
        # populated on first build() call.
        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L441-L443（逐字——
        #   schedule 闭包 L537 读它，删则 AttributeError）
        self.aot_sliding_window: tuple[int, int] | None = None

        # SUBTRACTED: R-SWA 持久缓冲族（L444-L456——rswa_window/persistent_
        #   rswa_*）——delete[6]：消费方是 build() 尾段（delete[5] L713-L724 删）。

    def build(  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L458-L726
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> FlashAttentionMetadata:
        """
        fast_build disables AOT scheduling, used when there will be few
        iterations i.e. spec-decode
        """
        num_reqs = common_attn_metadata.num_reqs
        num_actual_tokens = common_attn_metadata.num_actual_tokens
        max_query_len = common_attn_metadata.max_query_len
        max_seq_len = common_attn_metadata.max_seq_len
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        block_table_tensor = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping
        causal = common_attn_metadata.causal

        # Disable AOT schedule for spec-decode proposer (not worth the overhead)
        # and for batch invariance (schedule varies with max_seqlen_q/k).
        aot_schedule = (
            self.aot_schedule and not fast_build and not envs.VLLM_BATCH_INVARIANT
        )

        # SUBTRACTED: aot_sliding_window 多窗口回退（L484-L498）——delete[6]：
        #   首次 build 时扫描全层滑窗配置、多配置则关 AOT；__init__ 的
        #   aot_sliding_window=None 保留（schedule 闭包读它）。

        max_num_splits = 0  # 0 means use FA3's heuristics, not CG compatible
        if (
            self.use_full_cuda_graph
            and self.max_cudagraph_size is not None
            and num_actual_tokens <= self.max_cudagraph_size
        ):
            # NOTE(woosuk): Setting num_splits > 1 may increase the memory
            # usage, because the intermediate buffers of size [num_splits,
            # num_heads, num_tokens, head_size] are allocated. Therefore,
            # we only set num_splits when using cuda graphs.
            max_num_splits = self.max_num_splits

        if envs.VLLM_BATCH_INVARIANT:
            max_num_splits = 1

        def schedule(  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L515-L541（逐字）
            batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
        ):
            cache_dtype = self.cache_config.cache_dtype
            if is_quantized_kv_cache(cache_dtype):
                qkv_dtype = current_platform.fp8_dtype()
            else:
                qkv_dtype = self.kv_cache_dtype
            if aot_schedule:
                return get_scheduler_metadata(
                    batch_size=batch_size,
                    max_seqlen_q=max_query_len,
                    max_seqlen_k=max_seq_len,
                    num_heads_q=self.num_heads_q * self.dcp_world_size,
                    num_heads_kv=self.num_heads_kv,
                    headdim=self.headdim,
                    cache_seqlens=seqlens,
                    qkv_dtype=qkv_dtype,
                    cu_seqlens_q=cu_query_lens,
                    page_size=self.block_size,
                    causal=causal,
                    window_size=_maybe_symmetrize_window(
                        self.aot_sliding_window, causal
                    ),
                    num_splits=max_num_splits,
                )
            return None

        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L543-L554（逐字——
        #   DCP/cascade 分支的产物初值；分支删后恒为这些默认）
        use_cascade = common_prefix_len > 0
        max_dcp_context_kv_len = 0
        dcp_context_kv_lens = None
        num_decode_reqs = 0
        num_prefill_reqs = 0
        num_decode_tokens = 0
        num_prefill_tokens = 0

        cu_prefix_query_lens = None
        prefix_kv_lens = None
        suffix_kv_lens = None
        prefix_scheduler_metadata = None

        # SUBTRACTED: DCP 分支（L556-L616）——delete[5]：dcp_world_size>1 的
        #   本地序列切分与上下文调度（分布式 Part）。
        # SUBTRACTED: cascade 分支（L617-L641）——delete[5]：公共前缀的两段
        #   调度（cascade 的数学与调用现场 ch20 已立，含 merge_attn_states）。
        # SOURCE: vllm/v1/attention/backends/flash_attn.py:L642-L650（else 支
        #   去缩进无条件——标准 decoder 主路径）
        scheduler_metadata = schedule(
            batch_size=num_reqs,
            cu_query_lens=query_start_loc,
            max_query_len=max_query_len,
            seqlens=seq_lens,
            max_seq_len=max_seq_len,
            causal=causal,
        )
        # For FA3 + full cudagraph
        if self.use_full_cuda_graph and scheduler_metadata is not None:
            n = scheduler_metadata.shape[0]
            self.scheduler_metadata[:n] = scheduler_metadata
            # NOTE(woosuk): We should zero out the rest of the scheduler
            # metadata to guarantee the correctness. Otherwise, some thread
            # blocks may use the invalid scheduler metadata and overwrite the
            # output buffer.
            self.scheduler_metadata[n:] = 0
            scheduler_metadata = self.scheduler_metadata[:n]

        if isinstance(causal, torch.Tensor) and causal.dtype != torch.int32:
            causal = causal.to(torch.int32)

        # Symmetrize the spec's sliding_window for non-causal attention
        group_sliding_window = getattr(self.kv_cache_spec, "sliding_window", None)
        base_window = (
            (-1, -1) if group_sliding_window is None else (group_sliding_window - 1, 0)
        )
        effective_sliding_window = _maybe_symmetrize_window(base_window, causal)

        attn_metadata = FlashAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=max_query_len,
            query_start_loc=query_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=seq_lens,
            block_table=block_table_tensor,
            slot_mapping=slot_mapping,
            max_dcp_context_kv_len=max_dcp_context_kv_len,
            dcp_context_kv_lens=dcp_context_kv_lens,
            num_decode_reqs=num_decode_reqs,
            num_prefill_reqs=num_prefill_reqs,
            num_decode_tokens=num_decode_tokens,
            num_prefill_tokens=num_prefill_tokens,
            use_cascade=use_cascade,
            common_prefix_len=common_prefix_len,
            scheduler_metadata=scheduler_metadata,
            cu_prefix_query_lens=cu_prefix_query_lens,
            prefix_kv_lens=prefix_kv_lens,
            suffix_kv_lens=suffix_kv_lens,
            prefix_scheduler_metadata=prefix_scheduler_metadata,
            max_num_splits=max_num_splits,
            causal=causal,
            sliding_window=effective_sliding_window,
        )

        # SUBTRACTED: build() 尾段（L698-L724）——delete[5]：mm_prefix_range_
        #   tensor 计算与 R-SWA 拷入持久缓冲（消费被删的 mm_req_doc_ranges/
        #   rswa_prefix_lens/rswa_window/persistent_rswa_*）。

        return attn_metadata

    def update_block_table(  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L728-L737（逐字）
        self,
        metadata: FlashAttentionMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> FlashAttentionMetadata:
        new_metadata = copy.copy(metadata)
        new_metadata.block_table = blk_table
        new_metadata.slot_mapping = slot_mapping
        return new_metadata

    # SUBTRACTED: use_cascade_attention 方法（L739-L740）——delete[5]：转发到
    #   已删的模块级 use_cascade_attention（ch20 cascade 域）。


# SOURCE: vllm/v1/attention/backends/flash_attn.py:L743 FlashAttentionImpl ——
#   四件套之四：写腿 do_kv_cache_update + 读腿 forward
class FlashAttentionImpl(AttentionImpl):
    can_return_lse_for_decode: bool = True

    def __init__(  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L746-L820（delete[5] 删 DCP 尾段）
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        elif attn_type == AttentionType.ENCODER_ONLY:
            self.sliding_window = (sliding_window - 1, sliding_window - 1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        self.attn_type = attn_type
        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=alibi_slopes is not None,
            requires_local_attention=sliding_window is not None,
            head_size=head_size,
            has_sinks=sinks is not None,
        )
        logger.info_once(
            "Using FlashAttention version %s",
            self.vllm_flash_attn_version,
        )
        # Cache the batch invariant result for use in forward passes
        self.batch_invariant_enabled = envs.VLLM_BATCH_INVARIANT

        if is_quantized_kv_cache(
            self.kv_cache_dtype
        ) and not flash_attn_supports_kv_cache_dtype(
            self.kv_cache_dtype,
            requires_alibi=alibi_slopes is not None,
            head_size=head_size,
            head_size_v=head_size,
            has_sinks=sinks is not None,
        ):
            raise NotImplementedError(
                f"FlashAttention does not support {self.kv_cache_dtype}"
                " kv-cache on this device."
            )

        self.sinks = sinks
        if self.sinks is not None:
            assert flash_attn_supports_sinks(), (
                "Sinks are only supported in FlashAttention 3"
            )
            assert self.sinks.shape[0] == num_heads, (
                "Sinks must have the same number of heads as the number of "
                "heads in the layer"
            )

        self.supports_quant_query_input = flash_attn_supports_quant_query_input()

        # SUBTRACTED: dcp_a2a combine 装配与 _dcp_dtype/_dcp_max_num_tokens
        #   （L822-L836）——delete[5]：Impl.__init__ 的 DCP/rswa/mm 字段
        #   （_forward_with_dcp 与 DCP combine 的消费面，彼支已删）。

    def forward(  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L838-L1096（delete[5] 删特例支）
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [num_blocks, num_kv_heads, block_size, 2 * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert self.vllm_flash_attn_version is not None, (
            "FlashAttention version not detected."
        )

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for FlashAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        attn_type = self.attn_type

        # IMPORTANT!
        # NOTE(woosuk): With piece-wise CUDA graphs, this method is executed in
        # eager-mode PyTorch. Thus, we need to be careful about any CPU overhead
        # in this method. For example, `view` and `slice` (or `[:n]`) operations
        # are surprisingly slow even in the case they do not invoke any GPU ops.
        # Minimize the PyTorch ops in this method as much as possible.
        # Whenever making a change in this method, please benchmark the
        # performance to make sure it does not introduce any overhead.

        num_actual_tokens = attn_metadata.num_actual_tokens

        # SUBTRACTED: encoder 早退段（L891-L902——if attn_type 判定与 return
        #   _forward_encoder_attention，delete[5]：_forward_encoder_attention
        #   模块级函数已删；标准 decoder 不进此支）。

        # (B, H, N, 2*D) -> ((B, N, H, D), (B, N, H, D))
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        # Fix degenerate strides on size-1 dims (e.g. num_kv_heads=1 with TP).
        # FA3/4 on H100+ uses TMA, which requires ≥16-byte stride alignment.
        # See vllm.utils.torch_utils.canonicalize_singleton_dim_strides.
        fixed_k = canonicalize_singleton_dim_strides(key_cache)
        fixed_v = canonicalize_singleton_dim_strides(value_cache)
        if fixed_k is not key_cache or fixed_v is not value_cache:
            logger.debug(
                "Canonicalized degenerate KV cache strides (FlashAttention): "
                "shape=%s, key strides before=%s after=%s, "
                "value strides before=%s after=%s",
                key_cache.shape,
                key_cache.stride(),
                fixed_k.stride(),
                value_cache.stride(),
                fixed_v.stride(),
            )
        key_cache, value_cache = fixed_k, fixed_v

        # SUBTRACTED: descale view（L924-L927——量化 KV 的 fp8 视图，delete[5]：
        #   descale kwargs 与量化面已删，标准 decoder 不进此支）。

        if not attn_metadata.use_cascade:
            cu_seqlens_q = attn_metadata.query_start_loc
            seqused_k = attn_metadata.seq_lens
            max_seqlen_q = attn_metadata.max_query_len
            max_seqlen_k = attn_metadata.max_seq_len
            block_table = attn_metadata.block_table
            scheduler_metadata = attn_metadata.scheduler_metadata

            # SUBTRACTED: descale 展开（L937-L945——descale_shape/q_descale/
            #   k_descale/v_descale，delete[5]）。

            # SUBTRACTED: DCP 前向支（L947-L960——_forward_with_dcp 调用，
            #   delete[5]；单卡 dcp_world_size=1 恒走 else——去缩进无条件）。
            window = (
                attn_metadata.sliding_window
                if attn_metadata.sliding_window is not None
                else self.sliding_window
            )
            sliding_window_size: list[int] | None = (
                list(window) if window is not None else None
            )

            causal = attn_metadata.causal
            is_dynamic_causal = isinstance(causal, torch.Tensor)

            # SUBTRACTED: mm_prefix/R-SWA/dynamic_causal 段（L974-L1039——
            #   delete[5]：CuTE-DSL mask_mod 构造与 FA4 per-seq causal 的
            #   FA 版本守卫；张量 causal 直接下传 kernel（FA4 dynamic_causal
            #   的消费面）。

            # SOURCE: vllm/v1/attention/backends/flash_attn.py:L1041-L1067 ——
            #   读腿本体：flash_attn_varlen_func 一次调用吃下整批——varlen
            #   打平（cu_seqlens_q 前缀和切序列）+ paged KV（k/v 是整块
            #   cache、block_table 指路）。ch20 的 tiling kernel 在此喂料。
            flash_attn_varlen_func(
                q=query[:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[:num_actual_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=max_seqlen_q,
                seqused_k=seqused_k,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=self.scale,
                causal=causal,
                alibi_slopes=self.alibi_slopes,
                window_size=sliding_window_size,
                block_table=block_table,
                softcap=self.logits_soft_cap,
                scheduler_metadata=scheduler_metadata,
                fa_version=self.vllm_flash_attn_version,
                # SUBTRACTED: q_descale/k_descale/v_descale/dynamic_causal/
                #   s_aux/mask_mod/aux_tensors kwargs 行（L1058-L1065——
                #   delete[5]：descale 视图、sinks（s_aux）、mm/R-SWA mask 与
                #   FA 版本守卫已删，其产物无生产者）。
                num_splits=attn_metadata.max_num_splits,
            )
            return output

        # SUBTRACTED: cascade 调用（L1069-L1095——cascade_attention 模块级
        #   函数 delete[5] 已删；ch20 已立其数学与调用现场）。
        return output

    def do_kv_cache_update(  # SOURCE: vllm/v1/attention/backends/flash_attn.py:L1098-L1132（逐字）
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            # For encoder attention,
            # we use direct Q, K, V tensors without caching
            return

        # Scatter write into the KV cache using slot_mapping indices.
        # No TMA kernel is invoked here, so stride canonicalization is not needed.
        # (B, H, N, 2*D) -> ((B, N, H, D), (B, N, H, D))
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)

        # Reshape the input keys and values and store them in the cache.
        # Skip this if sharing KV cache with an earlier attention layer.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    # SUBTRACTED: _forward_with_dcp / _forward_encoder_attention（L1134-L1378）
    #   与 _make_mm_prefix_mask_mod / _make_rswa_mask_mod / use_cascade_
    #   attention / cascade_attention（L1381-L1691）——delete[5]：DCP 前向、
    #   encoder 前向、CuTE-DSL mask_mod 工厂、cascade 启发式与两段调用
    #   （cascade 数学 ch20 已立；DCP 分布式 Part；mm/R-SWA 扩展态）。
