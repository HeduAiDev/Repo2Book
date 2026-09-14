# SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py —— ch26 主文件
#（站 12/m14：V3.2 消费——buffer 行取出 → 逻辑位换算物理 slot → 只对选中
# latent 条目真算 O(Lk)）
# 全文减法：656B/584B 布局 docstring（L63-L85 逐字）+ FlashMLASparseBackend
#（L88-L142 逐字）+ FlashMLASparseMetadata（L145-L219 逐字）+
# get_prefill_workspace_size（L222-L229 逐字——魔数 5 注释原文）+
# FlashMLASparseImpl（L514-L875 减法——__init__/_forward_bf16_kv/
# _forward_fp8_kv_separate_prefill_decode/_forward_fp8_kv_mixed_batch/
# _fp8_flash_mla_kernel/_bf16_flash_mla_kernel/forward_mqa 三条 KV 路径）。
# SUBTRACTED（章界 → ch21/ch25）：FlashMLASparseMetadataBuilder（L232-L511
#   ——reorder 阈值表/fp8 sched 元数据/pinned workspace 族；测试直接构造
#   metadata）。
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar  # noqa: F401

import torch

from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config  # noqa: F401
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonPrefillMetadata,
)
from vllm.model_executor.layers.attention.sparse_mla_attention import (
    SparseMLACommonImpl,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionLayer,
    AttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)
from vllm.v1.attention.backends.utils import (
    reshape_attn_output_for_spec_decode,
    reshape_query_for_spec_decode,
    split_prefill_chunks,
)
from vllm.v1.attention.ops.flashmla import (
    FlashMLASchedMeta,
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
    get_mla_metadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec  # noqa: F401
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.worker.workspace import current_workspace_manager

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer

logger = init_logger(__name__)

# For FP8 sparse attention we have two implementations:
# 1. Mixed batch mode: use the FP8 decode kernel for both prefill and decode this is
#    done by treating all tokens as single batch.
# 2. Separate prefill and decode mode: use the BF16 prefill kernel for prefill
#    (upconverting the FP8 cache to BF16 then calling the prefill kernel) and using
#    the FP8 decode kernel for decode.
# Currently we use #1 when the number of heads per rank is low (i.e. TP) since the BF16
# prefill kernel requires padding the number of heads to 128 while the decode does not
# so when the per-rank head count is below MIN_HEADS_FOR_BF16_PREFILL we use the mixed
# batch mode (#1).
# SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L51-L61 —— 逐字
MIN_HEADS_FOR_BF16_PREFILL = 32

"""
NOTE: FlashMLA Sparse uses an fp8 cache with the following format

For DeepSeek V3.2, in the "FP8 with scale" format, each token's KV cache is 656
Bytes, structured as:
-   **First 512 bytes:** The "quantized NoPE" part, containing 512
    `float8_e4m3` values.
-   **Next 16 bytes:** Scale factors, containing 4 `float32` values.
    The first `float32` is the scale for the first 128 `float8_e4m3` values,
    the second for the next 128, and so on.
-   **Last 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This
    part is not quantized for accuracy.

For DeepSeek V4, in the "FP8 with scale" format, each token's KV cache is 584
Bytes, structured as:
-   **First 448 bytes:** The "quantized NoPE" part, containing 448
    `float8_e4m3` values.
-   **Next 128 bytes:** The "RoPE" part, containing 64 `bfloat16` values. This
    part is not quantized for accuracy.
-   **Last 8 bytes:** Scale factors, containing 7 `ue8m0` values + 1B pad.
    The first `ue8m0` is the scale for the first 64 `float8_e4m3` values,
    the second for the next 64, and so on.
"""
# SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L63-L85 —— 逐字


# SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L88-L142
#   FlashMLASparseBackend —— 逐字
class FlashMLASparseBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8_ds_mla",
        "fp8",  # alias for fp8_ds_mla
    ]

    @staticmethod
    def get_impl_cls() -> type["FlashMLASparseImpl"]:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L97-L99
        return FlashMLASparseImpl

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L101-L103
        return [64]

    @staticmethod
    def get_name() -> str:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L105-L107
        return "FLASHMLA_SPARSE"

    @staticmethod
    def get_builder_cls() -> type:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py —— 章界位
        #   （builder 域删；见文件头）
        raise NotImplementedError(
            "FlashMLASparse metadata builder is ch21/ch25 domain; ch26 tests "
            "construct FlashMLASparseMetadata directly."
        )

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # DeepSeek V3.2 layout: 512 NoPE + 64 RoPE = 576.
        # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L113-L116
        return [576]

    @classmethod
    def is_mla(cls) -> bool:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L118-L120
        return True

    @classmethod
    def is_sparse(cls) -> bool:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L122-L124
        return True

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L130-L142
        #   —— 逐字
        if cache_dtype_str == "fp8_ds_mla":
            # V3.2 main MLA: 656-byte custom storage format. See module docstring.
            return (num_blocks, block_size, 656)
        else:
            return (num_blocks, block_size, head_size)


# SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L145-L219
#   FlashMLASparseMetadata —— 逐字
@dataclass
# SOURCE: flashmla_sparse.py:L145-L219（锚点双置）
class FlashMLASparseMetadata(AttentionMetadata):
    num_reqs: int
    max_query_len: int
    max_seq_len: int

    num_actual_tokens: int  # Number of tokens excluding padding.
    query_start_loc: torch.Tensor
    slot_mapping: torch.Tensor

    block_table: torch.Tensor
    req_id_per_token: torch.Tensor
    block_size: int = 64
    topk_tokens: int = 2048

    num_decodes: int = 0
    num_prefills: int = 0
    num_decode_tokens: int = 0
    seq_lens: torch.Tensor | None = None
    prefill_max_seq_len: int = 0
    prefill: MLACommonPrefillMetadata | None = None
    cp_kv_cache_interleave_size: int = 1

    @dataclass
    # SOURCE: flashmla_sparse.py:L168-L172（锚点双置）
    class FP8KernelMetadata:
        scheduler_metadata: FlashMLASchedMeta
        dummy_block_table: torch.Tensor
        cache_lens: torch.Tensor

    @dataclass
    # SOURCE: flashmla_sparse.py:L174-L216（锚点双置）
    class FP8SeparatePrefillDecode:
        @dataclass
        # SOURCE: flashmla_sparse.py:L176-L180（锚点双置）
        class Decode:
            seq_lens: torch.Tensor
            kernel_metadata: "FlashMLASparseMetadata.FP8KernelMetadata"
            decode_query_len: int  # needed for reshape in spec decode

        @dataclass
        # SOURCE: flashmla_sparse.py:L182-L208（锚点双置）
        class Prefill:
            # Request ID for each token: -1 for decode tokens, request index
            # (0, 1, 2, ...) for prefill tokens.
            # Shape: [num_actual_tokens]
            request_ids: torch.Tensor

            # Workspace start offsets for all prefill requests
            # Shape: [num_prefill_reqs], adjusted in-place per chunk to be
            # 0-indexed within each chunk. Used to map prefill tokens to workspace
            # offsets in convert_logical_index_to_physical_index
            workspace_starts: torch.Tensor

            @dataclass
            # SOURCE: flashmla_sparse.py:L195-L206（锚点双置）
            class Chunk:
                """Metadata for a chunk of prefill requests.

                Prefill requests may be chunked to fit within the fixed workspace size.
                """

                tokens_slice: slice
                block_table: torch.Tensor
                req_start_idx: int
                workspace_starts: torch.Tensor
                chunk_tot_seqlen: int

            chunks: list[Chunk]

        num_prefills: int = 0
        num_decodes: int = 0
        num_prefill_tokens: int = 0
        num_decode_tokens: int = 0

        decode: Decode | None = None
        prefill: Prefill | None = None

    fp8_extra_metadata: FP8SeparatePrefillDecode | FP8KernelMetadata | None = None
    fp8_use_mixed_batch: bool = False


# SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L222-L229
#   get_prefill_workspace_size —— 逐字（魔数 5 注释原文）
# SOURCE: flashmla_sparse.py:L222-L229（锚点双置）
def get_prefill_workspace_size(max_model_len: int):
    # NOTE(Lucas): 5 is a magic number for controlling the prefill buffer size.
    # May be tuned later.
    # Memory usage: 5 * max_model_len * 576 * 2 bytes
    #   Example: DeepSeek-V3.2 with max_model_len=163840 ->
    #            5 * 163840 * 576 * 2 = ~900 MB
    # This fits nicely below the typical MoE workspace size of >2GB so this is "free"
    return max_model_len * 5


# SUBTRACTED: FlashMLASparseMetadataBuilder（L232-L511）——reorder 阈值表/
#   fp8 sched 元数据构建/pinned workspace 族——ch21/ch25 域


# SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L514-L875
#   FlashMLASparseImpl —— 减法子集（三条 KV 路径齐备）
class FlashMLASparseImpl(SparseMLACommonImpl[FlashMLASparseMetadata]):
    @staticmethod
    def _compute_fp8_decode_padded_heads(num_heads: int) -> int:
        # FP8 decode kernel only supports h_q = 64 or 128
        # Compute padded head count for decode
        # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L516-L519
        return 64 if num_heads <= 64 else 128

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA Specific Arguments
        topk_indices_buffer: torch.Tensor | None = None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L521-L585
        #   —— 逐字
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            indexer=indexer,
            topk_indices_buffer=topk_indices_buffer,
            **mla_args,
        )
        self.softmax_scale = scale
        # Prefill BF16 kernel requires 64 on Hopper, 128 on Blackwell
        self.prefill_padding = (
            128 if current_platform.is_device_capability_family(100) else 64
        )
        self.fp8_decode_padded_heads = self._compute_fp8_decode_padded_heads(num_heads)

        vllm_config = get_current_vllm_config()
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        q_concat_shape = (max_tokens, num_heads, head_size)
        if is_quantized_kv_cache(kv_cache_dtype):
            assert kv_cache_dtype == "fp8_ds_mla", (
                "FlashMLA Sparse Attention backend fp8 only supports "
                "fp8_ds_mla kv-cache dtype"
            )

        if kv_cache_dtype == "fp8_ds_mla":
            # Reserve workspace during initialization
            assert vllm_config is not None and vllm_config.model_config is not None
            prefill_workspace_size = get_prefill_workspace_size(
                vllm_config.model_config.max_model_len
            )
            self.prefill_workspace_shape = (prefill_workspace_size, head_size)
            self.q_concat_buffer, self.prefill_bf16_workspace = (
                current_workspace_manager().get_simultaneous(
                    (q_concat_shape, torch.bfloat16),
                    (self.prefill_workspace_shape, torch.bfloat16),
                )
            )
        else:
            (self.q_concat_buffer,) = current_workspace_manager().get_simultaneous(
                (q_concat_shape, torch.bfloat16),
            )

    # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L587-L611
    #   _forward_bf16_kv —— 逐字
    def _forward_bf16_kv(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> torch.Tensor:
        # Convert per-request indices to global slots (decode) or workspace
        # offsets (prefill). req_id_per_token covers the whole batch; slice it
        # to the MQA tokens (q may exclude prefill tokens routed to dense MHA).
        # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L587-L611
        #   （锚点双置）
        topk_indices, topk_length = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[: topk_indices.shape[0]],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )

        return self._bf16_flash_mla_kernel(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            topk_length,
        )

    # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L613-L722
    #   _forward_fp8_kv_separate_prefill_decode —— 逐字
    def _forward_fp8_kv_separate_prefill_decode(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> torch.Tensor:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L613-L722
        #   （锚点双置）
        fp8_metadata = attn_metadata.fp8_extra_metadata
        assert isinstance(fp8_metadata, FlashMLASparseMetadata.FP8SeparatePrefillDecode)
        num_decodes = fp8_metadata.num_decodes
        num_mqa_tokens = q.shape[0]
        num_decode_tokens = fp8_metadata.num_decode_tokens
        num_prefill_tokens = num_mqa_tokens - num_decode_tokens
        assert num_prefill_tokens in (0, fp8_metadata.num_prefill_tokens), (
            "FP8 sparse MLA expects either the decode subset or the full batch"
        )

        prefill_request_ids = None
        prefill_workspace_starts = None
        has_prefill_workspace = False
        if num_prefill_tokens > 0:
            assert fp8_metadata.prefill is not None
            prefill_request_ids = fp8_metadata.prefill.request_ids
            prefill_workspace_starts = fp8_metadata.prefill.workspace_starts
            has_prefill_workspace = True

        # Convert per-request indices to global slots (decode) or workspace
        # offsets (prefill).
        # For FP8 cache: prefill uses workspace mapping (upconverted to BF16)
        # For BF16 cache: always use global cache slots (no workspace)
        # prefill_workspace_starts has been adjusted in-place per chunk so
        # prefill indices automatically come out chunk-local
        topk_indices, topk_length = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[: topk_indices.shape[0]],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            HAS_PREFILL_WORKSPACE=has_prefill_workspace,
            prefill_workspace_request_ids=prefill_request_ids,
            prefill_workspace_starts=prefill_workspace_starts,
            return_valid_counts=True,
        )

        fp8_metadata = attn_metadata.fp8_extra_metadata
        assert isinstance(fp8_metadata, FlashMLASparseMetadata.FP8SeparatePrefillDecode)

        # SOURCE: flashmla_sparse.py:L660-L680（锚点双置）
        def _fp8_decode(
            q: torch.Tensor,
            topk_indices: torch.Tensor,
        ) -> torch.Tensor:
            # Reshape q: (num_decode_tokens, num_heads, head_dim)
            #         -> (num_decodes, seq_len, num_heads, head_dim)
            q = reshape_query_for_spec_decode(q, num_decodes)
            seq_len = q.shape[1]
            # Reshape topk_indices: (num_decode_tokens, topk)
            #                    -> (num_decodes, seq_len, topk)
            topk_indices = topk_indices.view(num_decodes, seq_len, -1)
            assert fp8_metadata.decode is not None
            attn_out, _ = self._fp8_flash_mla_kernel(
                q=q,
                kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
                topk_indices=topk_indices,
                kernel_metadata=fp8_metadata.decode.kernel_metadata,
            )
            # Reshape output: (num_decodes, seq_len, num_heads, head_dim_v)
            #              -> (num_decode_tokens, num_heads, head_dim_v)
            return reshape_attn_output_for_spec_decode(attn_out)

        # Pure decode: direct call without allocation
        if num_decode_tokens > 0 and num_prefill_tokens == 0:
            assert fp8_metadata.decode is not None
            attn_out = _fp8_decode(q, topk_indices)
        else:
            # Mixed or pure prefill: allocate output tensor
            attn_out = q.new_empty(
                (num_mqa_tokens, self.num_heads, self.kv_lora_rank),
                dtype=q.dtype,
                device=q.device,
            )

            if num_decode_tokens > 0:
                attn_out[:num_decode_tokens] = _fp8_decode(
                    q[:num_decode_tokens],
                    topk_indices[:num_decode_tokens],
                )

            assert fp8_metadata.prefill is not None
            for chunk in fp8_metadata.prefill.chunks:
                chunk_workspace = self.prefill_bf16_workspace[: chunk.chunk_tot_seqlen]
                ops.cp_gather_and_upconvert_fp8_kv_cache(
                    kv_c_and_k_pe_cache,
                    chunk_workspace,
                    chunk.block_table,
                    chunk.workspace_starts,
                    len(chunk.block_table),
                )

                chunk_q = q[chunk.tokens_slice]
                chunk_topk_indices_workspace = topk_indices[chunk.tokens_slice]
                chunk_topk_length = topk_length[chunk.tokens_slice]

                attn_out[chunk.tokens_slice] = self._bf16_flash_mla_kernel(
                    chunk_q,
                    chunk_workspace,
                    chunk_topk_indices_workspace,
                    chunk_topk_length,
                )

        return attn_out

    # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L724-L761
    #   _forward_fp8_kv_mixed_batch —— 逐字
    def _forward_fp8_kv_mixed_batch(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
    ) -> torch.Tensor:
        """Mixed batch FP8 forward path that treats all tokens as one batch.

        This is equivalent to main branch's approach and avoids the BF16
        prefill kernel which has head padding overhead when num_heads is small.
        Used when use_mixed_batch is True.
        """
        # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L724-L761
        #   （锚点双置）
        # Convert per-request indices to global slots (decode) or workspace
        # offsets (prefill).
        topk_indices = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[: topk_indices.shape[0]],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
        )

        assert attn_metadata.fp8_extra_metadata is not None
        assert isinstance(
            attn_metadata.fp8_extra_metadata, FlashMLASparseMetadata.FP8KernelMetadata
        )
        fp8_metadata = attn_metadata.fp8_extra_metadata

        _attn_out, _ = self._fp8_flash_mla_kernel(
            q=q.unsqueeze(0),  # unsqueeze to add batch_dim: (T, H, D) -> (1, T, H, D)
            kv_c_and_k_pe_cache=kv_c_and_k_pe_cache,
            topk_indices=topk_indices.unsqueeze(0),  # (T, topk) -> (1, T, topk)
            kernel_metadata=fp8_metadata,
        )

        # Output is (1, T, H, D_v), squeeze back to (T, H, D_v)
        return _attn_out.squeeze(0)

    # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L763-L800
    #   _fp8_flash_mla_kernel —— 逐字（kernel 经 HOST SEAM 镜像）
    def _fp8_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        kernel_metadata: FlashMLASparseMetadata.FP8KernelMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L763-L800
        #   （锚点双置）
        # q shape: (batch, seq_len, num_heads, head_dim)
        actual_num_heads = q.size(2)
        padded_num_heads = self.fp8_decode_padded_heads

        # Pad query if needed (kernel only supports h_q = 64 or 128)
        if actual_num_heads < padded_num_heads:
            logger.warning_once(
                f"Padding num_heads from {actual_num_heads} to "
                f"{padded_num_heads} for FP8 sparse decode kernel"
            )
            q_padded = q.new_zeros((q.size(0), q.size(1), padded_num_heads, q.size(3)))
            q_padded[:, :, :actual_num_heads, :] = q
            q = q_padded

        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(-2),
            block_table=kernel_metadata.dummy_block_table,
            head_dim_v=512,
            cache_seqlens=kernel_metadata.cache_lens,
            tile_scheduler_metadata=kernel_metadata.scheduler_metadata,
            is_fp8_kvcache=True,
            indices=topk_indices,
            softmax_scale=self.softmax_scale,
        )

        # Slice output back to actual head count if we padded
        if actual_num_heads < padded_num_heads:
            out = out[:, :, :actual_num_heads, :]

        return out, lse

    # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L802-L836
    #   _bf16_flash_mla_kernel —— 逐字
    def _bf16_flash_mla_kernel(
        self,
        q: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        topk_indices: torch.Tensor,
        topk_length: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L802-L836
        #   （锚点双置）
        num_tokens = q.shape[0]
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.view(
            -1, 1, kv_c_and_k_pe_cache.shape[-1]
        )

        # NOTE(Chen): kernel requires num_local_head to be a multiple of
        # 64 on hopper and 128 on blackwell
        if self.num_heads % self.prefill_padding != 0:
            assert self.prefill_padding % self.num_heads == 0
            logger.warning_once(
                f"Padding num_heads from {self.num_heads} to "
                f"{self.prefill_padding} for BF16 sparse prefill kernel"
            )
            q_padded = q.new_empty((q.shape[0], self.prefill_padding, q.shape[2]))
            q_padded[:, : self.num_heads, :] = q
            q = q_padded

        topk_indices = topk_indices.view(num_tokens, 1, -1)
        output = flash_mla_sparse_fwd(
            q,
            kv_c_and_k_pe_cache,
            topk_indices,
            self.softmax_scale,
            topk_length=topk_length,
        )[0]

        output = output[:, : self.num_heads, :]
        return output

    # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L838-L875
    #   forward_mqa —— 逐字（站 12：取 buffer 前 num_actual_toks 行）
    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # SOURCE: vllm/v1/attention/backends/mla/flashmla_sparse.py:L838-L875
        #   （锚点双置）
        # NOTE(lucas): for the sparse FlashMLA kernels the kernels want to use
        # MQA 576/512 approach for both prefill and decode

        # Concatenate q if it's a tuple (ql_nope, q_pe)
        if isinstance(q, tuple):
            ql_nope, q_pe = q
            q = self.q_concat_buffer[: ql_nope.shape[0]]
            ops.concat_mla_q(ql_nope, q_pe, q)

        num_actual_toks = q.shape[0]

        # Get topk indices
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"

        if not use_fp8_cache:
            attn_out = self._forward_bf16_kv(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )
        elif attn_metadata.fp8_use_mixed_batch:
            attn_out = self._forward_fp8_kv_mixed_batch(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )
        else:
            attn_out = self._forward_fp8_kv_separate_prefill_decode(
                q, kv_c_and_k_pe_cache, topk_indices, attn_metadata
            )

        return attn_out, None
