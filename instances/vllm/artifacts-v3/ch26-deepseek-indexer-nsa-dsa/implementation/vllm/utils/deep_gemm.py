# SOURCE: vllm/utils/deep_gemm.py
# HOST SEAM（B1 CUDA kernel 镜像·精确数学）：DSA 打分核族的 host 参考镜像。
# 真实实现是 DeepGEMM CUDA kernel（sm90/sm100 fp8/fp4 MQA logits）——host 无
# DeepGEMM；本镜像承载同一函数签名与打分数学：
#   I_{t,s} = Σ_j w_{t,j}·ReLU(q_{t,j}·k_s)     （DSA arXiv:2512.02556 Eq.(1)）
#   FP8 路径 q_scale 已折进 weights（fused_indexer_q_rope_quant 的折叠契约），
#   k 侧逐 token 反量化（128B fp8 值 + 4B fp32 scale）。clean_logits=False 时
#   因果窗外保留 0（真实核为未写垃圾值——topk 核按 cu_seqlen/seq_lens 界定
#   有效窗，窗外值不参与选择；镜像以 0 承载同一『不参与』语义）。
from __future__ import annotations

import torch


# SOURCE: vllm/utils/deep_gemm.py has_deep_gemm —— HOST SEAM：恒 False
#   （host 无 DeepGEMM 包；SparseAttnIndexer 的 CUDA 断言因此只在 CUDA 平台触）
def has_deep_gemm() -> bool:
    # SOURCE: vllm/utils/deep_gemm.py has_deep_gemm —— HOST SEAM
    return False


# SOURCE: vllm/utils/deep_gemm.py —— HOST SEAM（锚点双置）
def _dequant_rows(k_quant: torch.Tensor, k_scale: torch.Tensor) -> torch.Tensor:
    """FP8 布局反量化：k_quant [N,128] fp8 + k_scale [N] fp32 → [N,128] fp32。"""
    return k_quant.view(torch.float8_e4m3fn).float() * k_scale.float().unsqueeze(-1)


# SOURCE: vllm/utils/deep_gemm.py:L499-L542 fp8_fp4_mqa_logits —— HOST SEAM
#   参考数学（prefill 打分核，O(L²) 本尊）
def fp8_fp4_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv: tuple[torch.Tensor, torch.Tensor],
    weights: torch.Tensor,
    cu_seqlen_ks: torch.Tensor,
    cu_seqlen_ke: torch.Tensor,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute MQA logits for a single sequence without KV paging.

    Unified FP8/FP4 dispatch — the underlying DeepGEMM kernel takes
    ``q = (values, scales_or_None)`` where ``scales`` is None for FP8 Q
    (per-token scale is folded into ``weights``) and a packed block-scale
    tensor for MXFP4 Q.

    Args:
        q: Tuple ``(q_values, q_scale)``. FP8 path: q_values is [M, H, D]
            float8_e4m3fn and q_scale is None (per-token scale is folded
            into ``weights``). FP4 path: q_values is packed uint8 and
            q_scale is the companion block-scale tensor.
        kv: Tuple `(k_packed, k_scales)` — FP8 layout is [N, D]
            float8_e4m3fn plus fp32 scales [N]; FP4 layout is packed uint8.
        weights: weights of shape [M, H], dtype `torch.float32`.
        cu_seqlen_ks: Start indices (inclusive) for valid K per query
            position, shape [M], dtype int32.
        cu_seqlen_ke: End indices (exclusive) for valid K per query
            position, shape [M], dtype int32.
        clean_logits: Whether to clean the unfilled logits into `-inf`.

    Returns:
        Logits tensor of shape [M, N], dtype `torch.float32`.
    """
    # SOURCE: vllm/utils/deep_gemm.py:L499-L542 —— HOST SEAM 参考数学
    q_values, _q_scale = q
    k_quant, k_scale = kv
    q_deq = q_values.float()                       # [M, H, D]（scale 已折 weights）
    k_deq = _dequant_rows(k_quant, k_scale.view(-1))  # [N, D]
    dots = torch.einsum("mhd,nd->mhn", q_deq, k_deq)   # [M, H, N]
    logits = torch.einsum("mhn,mh->mn", torch.relu(dots), weights.float())
    return logits


# SOURCE: vllm/utils/deep_gemm.py:L544-L562 get_paged_mqa_logits_metadata
#   —— HOST SEAM：调度元数据容器（真实为 DeepGEMM scheduler 张量；host 返回
#   上下文长张量作名义容器——不参与打分数值）
def get_paged_mqa_logits_metadata(
    context_lens: torch.Tensor, block_size: int, num_sms: int
) -> torch.Tensor:
    """Build scheduling metadata for paged MQA logits.

    Args:
        context_lens: Tensor of shape [B], dtype int32; effective context length
            per batch element.
        block_size: KV-cache block size in tokens (e.g., 64).
        num_sms: Number of SMs available. 132 for Hopper

    Returns:
        Backend-specific tensor consumed by `fp8_fp4_paged_mqa_logits` to
        schedule work across SMs.
    """
    # SOURCE: vllm/utils/deep_gemm.py:L544-L562 —— HOST SEAM 名义容器
    return context_lens.to(torch.int32).contiguous()


# SOURCE: vllm/utils/deep_gemm.py:L565-… fp8_fp4_paged_mqa_logits —— HOST SEAM
#   参考数学（decode 打分核：免 gather 直接对分页 IndexCache 打分）
def fp8_fp4_paged_mqa_logits(
    q: tuple[torch.Tensor, torch.Tensor | None],
    kv_cache: torch.Tensor,
    weights: torch.Tensor,
    context_lens: torch.Tensor,
    block_tables: torch.Tensor,
    schedule_metadata: torch.Tensor,
    max_model_len: int,
    clean_logits: bool,
) -> torch.Tensor:
    """Compute MQA logits using a paged KV-cache.

    Unified FP8/FP4 dispatch — the underlying DeepGEMM kernel takes
    ``q = (values, scales_or_None)``; pass ``(q_tensor, None)`` for the FP8
    path and ``(q_values, q_scale)`` for MXFP4.

    Args:
        q: Tuple ``(q_values, q_scale)``. FP8 path: q_values is
            [B, next_n, H, D] float8_e4m3fn and q_scale is None. FP4 path:
            q_values is packed uint8 and q_scale is the companion
            block-scale tensor.
        kv_cache: Paged KV-cache. FP8 layout is [num_blocks, block_size, 1,
            D+4], dtype `torch.uint8`, with the last 4 bytes per (block, pos)
            storing the float dequant scale.
        weights: Tensor of shape [B * next_n, H], dtype `torch.float32`.
        context_lens: Tensor of shape [B], dtype int32; effective context
            length for each batch element.
        block_tables: Tensor of shape [B, max_blocks], dtype int32; maps logical
            block indices to physical blocks in the paged cache.
        schedule_metadata: Returned by `get_paged_mqa_logits_metadata`;
            used to distribute work across SMs.
        max_model_len: Maximum sequence length; the logits width N.
        clean_logits: Whether to clean the unfilled logits into `-inf`.

    Returns:
        Logits tensor of shape [B * next_n, max_model_len], dtype float32.
    """
    # SOURCE: vllm/utils/deep_gemm.py:L565-… —— HOST SEAM 参考数学
    #   （分页读：block_table[req, pos//block_size]*block_size + pos%block_size）
    q_values, _q_scale = q
    B, next_n, H, D = q_values.shape
    block_size = kv_cache.shape[1]
    flat = kv_cache.view(-1, kv_cache.shape[-1])   # [num_blocks*block, width]
    width = kv_cache.shape[-1]
    lens = context_lens
    if lens.dim() == 2:  # (B, next_n) native spec → 逐 token 上下文长
        lens = lens.reshape(-1)
    logits = torch.zeros(B * next_n, max_model_len, dtype=torch.float32)
    for r in range(B * next_n):
        b = r // next_n
        n = int(lens[r].item())
        ids = []
        for pos in range(n):
            blk = int(block_tables[b, pos // block_size].item())
            ids.append(blk * block_size + pos % block_size)
        if not ids:
            continue
        rows = flat[ids]                             # [n, width]
        k_deq = (rows[:, : D].view(torch.float8_e4m3fn).float()
                 * rows[:, D:].view(torch.float32).float())
        q_deq = q_values.reshape(B * next_n, H, D)[r].float()
        dots = torch.einsum("hd,nd->hn", q_deq, k_deq)
        logits[r, :n] = torch.einsum("hn,h->n", torch.relu(dots),
                                     weights.float()[r])
    return logits
