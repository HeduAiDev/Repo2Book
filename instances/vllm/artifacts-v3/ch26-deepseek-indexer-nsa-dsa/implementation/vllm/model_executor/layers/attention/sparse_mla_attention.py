# SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py
# ch26 主文件（m14/m08 消费侧公共基类）：
#   GLOBAL_TOPK_MASK_MAX_BYTES（L43）+ _is_masked_mha_available（L46-L73 减法
#   ——host 无 SM100/FA4 恒 False，真实非 Blackwell 形态）+
#   SparseMLACommonImpl（L441-L829 减法——__init__ 的 buffer 接管【indexer
#   自带 or 显式传参——skip 层没 indexer 也能读】逐字、_slice_topk_per_req/
#   _remap_topk_to_ranges/_project_kv 逐字、_try_build_global_mask/_run_
#   masked_mha 的 topk 位掩码路径、forward_mha 的 sparse 前置分流逐字）。
# SUBTRACTED（章界 → ch25）：SparseMLACommonMetadataBuilder（L75-L384——
#   req_id_per_token/chunked workspace 族）与 _compute_context_mha 的
#   chunked-context 机器（L637-L720——gather/LSE merge 归 ch25 m08）；
#   _build_topk_mask 的 Triton 核以 HOST SEAM 位掩码数学承载。
from shutil import which  # noqa: F401  (ninja 探针位)
from typing import ClassVar, Generic, TypeVar, cast  # noqa: F401

import torch

from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBaseImpl,
    MLACommonPrefillMetadata,
)
from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.flashinfer import has_flashinfer  # noqa: F401
from vllm.utils.torch_utils import is_quantized_kv_cache
from vllm.v1.attention.backend import AttentionMetadata, AttentionMetadataBuilder
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states

logger = init_logger(__name__)

T = TypeVar("T", bound=AttentionMetadata)

# SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py:L43
GLOBAL_TOPK_MASK_MAX_BYTES = 64 * 1024 * 1024  # 64 MiB


# SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py:L46-L73
#   _is_masked_mha_available —— 减法子集（FA4 探针面 → ch21；host 恒 False）
def _is_masked_mha_available(
    num_heads_total: int,
    kv_lora_rank: int,
    qk_nope_head_dim: int,
    qk_rope_head_dim: int,
    v_head_dim: int,
    kv_cache_dtype: str,
) -> bool:
    """Check if masked MHA can ever fire for this model configuration."""
    # SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py:L57-L73
    if not current_platform.is_device_capability_family(100):
        return False
    if (
        num_heads_total != 128
        or kv_lora_rank != 512
        or qk_nope_head_dim != 128
        or qk_rope_head_dim != 64
        or v_head_dim != 128
    ):
        return False
    return not is_quantized_kv_cache(kv_cache_dtype)


# SUBTRACTED: SparseMLACommonMetadataBuilder（L75-L384）——ch25 域（req_id_
#   per_token/chunked workspace/determine_chunked_prefill_workspace_size 族）


# SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py:L385-L438
#   _build_topk_mask —— HOST SEAM 位掩码数学（word = idx>>5、bit = 1<<(idx&31)
#   的原子 or 散射；行零初始化——真实 _scatter_topk_kernel L311-L350/
#   _scatter_topk_single_req L352-L384 的语义）
def _build_topk_mask(
    topk_per_req: list[torch.Tensor],
    q_lens: list[int],
    padded_q_len: int,
    max_seq_len: int,
    out: torch.Tensor,
) -> torch.Tensor:
    # SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py:
    #   L385-L438 —— HOST SEAM（逐行置零 + 位散射）
    batch_size = len(q_lens)
    num_words = (max_seq_len + 31) // 32 + 1
    out[:batch_size, :, :num_words] = 0
    for b, (topk, q_len) in enumerate(zip(topk_per_req, q_lens)):
        for t in range(q_len):
            row = out[b, t]
            for idx in topk[t].tolist():
                if idx >= 0:
                    row[idx >> 5] |= 1 << (idx & 31)
    return out[:batch_size, :padded_q_len, :num_words]


# SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py:L441-L829
#   SparseMLACommonImpl —— 减法子集
class SparseMLACommonImpl(MLACommonBaseImpl[T], Generic[T]):
    """Sparse MLA base with dense and masked-MHA prefill paths."""

    is_sparse = True

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
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj,
        indexer: object | None = None,
        topk_indices_buffer: torch.Tensor | None = None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        # SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py:
        #   L446-L505 —— 逐字
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            kv_cache_dtype,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_b_proj,
        )

        # The indexer carries the shared buffer for normal layers and tests;
        # the explicitly-passed buffer covers backbone skip layers, whose
        # indexer is not constructed (see deepseek_v2.py).
        # SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py:
        #   L483-L490 —— 逐字（装配期接管 buffer 的字面证据）
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer  # type: ignore[attr-defined]
            if indexer is not None
            else topk_indices_buffer
        )
        # SUBTRACTED: _use_flashinfer_concat_mla_k 探测（L491-L497）——host
        #   has_flashinfer 恒 False，真实无 flashinfer 形态同型跳过
        self.masked_mha_available = _is_masked_mha_available(
            num_heads_total=num_heads * get_tensor_model_parallel_world_size(),
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            kv_cache_dtype=kv_cache_dtype,
        )

    # SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py:
    #   L507-L517 _slice_topk_per_req —— 逐字
    @staticmethod
    # SOURCE: sparse_mla_attention.py:L507-L517（锚点双置）
    def _slice_topk_per_req(
        topk_all: torch.Tensor,
        q_lens: list[int],
    ) -> list[torch.Tensor]:
        topk_per_req = []
        offset = 0
        for q_len in q_lens:
            topk_per_req.append(topk_all[offset : offset + q_len])
            offset += q_len
        return topk_per_req

    # SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py:
    #   L519-L529 _remap_topk_to_ranges —— 逐字
    @staticmethod
    # SOURCE: sparse_mla_attention.py:L519-L529（锚点双置）
    def _remap_topk_to_ranges(
        topk_per_req: list[torch.Tensor],
        range_starts,
        range_lens: list[int],
    ) -> list[torch.Tensor]:
        remapped = []
        for topk, start, length in zip(topk_per_req, range_starts, range_lens):
            valid = (topk >= start) & (topk < start + length)
            remapped.append(torch.where(valid, topk - start, -1))
        return remapped

    # SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py:
    #   L531-L540 _project_kv —— 逐字
    # SOURCE: sparse_mla_attention.py:L531-L540（锚点双置）
    def _project_kv(
        self, kv_c_normed: torch.Tensor, k_pe: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kv_nope = self.kv_b_proj(kv_c_normed)[0].view(
            -1,
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
        )
        k_nope, v = kv_nope.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        return self._concat_k_nope_k_pe(k_nope, k_pe), v

    # SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py:
    #   L542-L575 _try_build_global_mask —— 逐字（预算判定 + _build_topk_mask）
    def _try_build_global_mask(
        self,
        topk_per_req: list[torch.Tensor],
        q_lens: list[int],
        max_query_len: int,
        max_seq_len: int,
        topk_mask_workspace: torch.Tensor,
    ) -> torch.Tensor | None:
        """Build a full-sequence top-k mask if it fits within the budget.

        When the mask fits, it is reused across the suffix and all context
        chunks, avoiding per-chunk mask rebuilds.  Returns None when the
        mask is too large, signalling the caller to fall back to per-chunk
        index remapping.
        """
        # SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py:
        #   L550-L575（锚点双置）
        batch_size = len(q_lens)
        tile_m = 128 if max_query_len <= 128 else 256
        padded_q_len = triton.cdiv(max_query_len, tile_m) * tile_m
        num_words_padded = (max_seq_len + 31) // 32 + 1
        needed = batch_size * padded_q_len * num_words_padded
        if needed * torch.int32.itemsize > GLOBAL_TOPK_MASK_MAX_BYTES:
            return None

        mask = topk_mask_workspace[:needed].view(
            batch_size, padded_q_len, num_words_padded
        )
        _build_topk_mask(
            topk_per_req,
            q_lens,
            padded_q_len,
            max_seq_len,
            mask,
        )
        return mask

    # SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py:
    #   L577-L635 _run_masked_mha —— 减法子集（FA varlen 入口经 HOST SEAM
    #   镜像承载位掩码数学；mask_mod/aux_tensors 契约面逐字）
    def _run_masked_mha(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        topk_per_req: list[torch.Tensor],
        q_lens: list[int],
        causal: bool,
        return_softmax_lse: bool = False,
        dense_mask: torch.Tensor | None = None,
        key_starts: torch.Tensor | None = None,
        topk_mask_workspace: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py:
        #   L577-L635 —— 减法子集（sparse_mla_mask 的 mask_mod 族以契约为
        #   参数透传；host 的 vllm_flash_attn 镜像按 aux 位掩码实现）
        from vllm.vllm_flash_attn import flash_attn_varlen_func

        tile_m = 128 if max_seqlen_q <= 128 else 256
        padded_q_len = triton.cdiv(max_seqlen_q, tile_m) * tile_m
        if dense_mask is None:
            batch_size = len(q_lens)
            num_words = (max_seqlen_k + 31) // 32
            assert topk_mask_workspace is not None
            workspace_3d = topk_mask_workspace[
                : batch_size * padded_q_len * num_words
            ].view(batch_size, padded_q_len, num_words)
            dense_mask = _build_topk_mask(
                topk_per_req,
                q_lens,
                padded_q_len,
                max_seqlen_k,
                workspace_3d,
            )
        if key_starts is not None:
            dense_mask[:, 0, -1].copy_(key_starts)
        kwargs = {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": max_seqlen_q,
            "max_seqlen_k": max_seqlen_k,
            "softmax_scale": self.scale,
            "return_softmax_lse": return_softmax_lse,
            "fa_version": 4,
            "aux_tensors": [dense_mask],
            "aux_tensor_leading_dims": [2],
            "causal": causal,
        }

        return flash_attn_varlen_func(**kwargs)

    # SUBTRACTED: _compute_context_mha（L637-L720）——章界 → ch25（chunked
    #   context 的 gather/​LSE merge 机器 = ch25 m08 域；正文按真实源码
    #   excerpt 作 supporting 展开）

    # SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py:
    #   L722-L829 forward_mha —— 减法子集（sparse 前置分流与 chunked-
    #   context=None 的 masked 路径逐字；chunked 分支 → ch25 章界）
    def forward_mha(  # type: ignore[override]
        self,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: T,
        k_scale: torch.Tensor,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
    ) -> None:
        # SOURCE: vllm/model_executor/layers/attention/sparse_mla_attention.py:
        #   L722-L781 —— 逐字减章界
        prefill_max_seq_len = attn_metadata.prefill_max_seq_len  # type: ignore[attr-defined]
        topk_tokens = attn_metadata.topk_tokens  # type: ignore[attr-defined]
        force_dense = getattr(self, "_sparse_mla_force_dense_mha", False)
        force_masked = getattr(self, "_sparse_mla_force_masked_mha", False)
        if force_dense or (prefill_max_seq_len <= topk_tokens and not force_masked):
            return super().forward_mha(
                q,
                kv_c_normed,
                k_pe,
                kv_c_and_k_pe_cache,
                cast("object", attn_metadata),
                k_scale,
                output,
                output_scale,
            )

        assert output_scale is None
        assert self.masked_mha_available
        prefill_metadata = attn_metadata.prefill  # type: ignore[attr-defined]
        assert prefill_metadata is not None
        assert prefill_metadata.query_lens_cpu is not None
        assert self.topk_indices_buffer is not None

        q_lens = prefill_metadata.query_lens_cpu.tolist()
        num_decode_tokens = attn_metadata.num_decode_tokens  # type: ignore[attr-defined]
        topk_all = self.topk_indices_buffer[
            num_decode_tokens : num_decode_tokens + q.shape[0]
        ]
        topk_per_req = self._slice_topk_per_req(topk_all, q_lens)

        k, v = self._project_kv(kv_c_normed, k_pe)
        chunked_context = prefill_metadata.chunked_context
        if chunked_context is None:
            attn_out = self._run_masked_mha(
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=prefill_metadata.query_start_loc,
                cu_seqlens_k=prefill_metadata.query_start_loc,
                max_seqlen_q=prefill_metadata.max_query_len,
                max_seqlen_k=prefill_metadata.max_query_len,
                topk_per_req=topk_per_req,
                q_lens=q_lens,
                causal=True,
                topk_mask_workspace=prefill_metadata.topk_mask_workspace,
            )
            assert isinstance(attn_out, torch.Tensor)
            output.copy_(attn_out[..., : self.v_head_dim].flatten(start_dim=-2))
            return

        # SUBTRACTED: chunked-context 分支（L783-L829——_try_build_global_mask
        #   的全局掩码重用 + suffix/context 两段 merge_attn_states 合并）——
        #   章界 → ch25 m08（chunked context workspace/LSE merge 机器）；
        #   正文按真实源码 excerpt 作 supporting 展开
        raise NotImplementedError(
            "Sparse MLA chunked-context prefill (remap + merge_attn_states) "
            "rides ch25's chunked-context machinery; not carried here."
        )
