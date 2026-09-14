# SOURCE: vllm/models/deepseek_v4/common/ops/cache_utils.py
# ch26 切面（m07/m13）：V4 消费侧三个索引算子的 HOST SEAM 镜像——
#   compute_global_topk_indices_and_lens（L435-L474：局部压缩 index → 全局
#     物理 slot（block_table 换算）+ 有效计数 + pad token 计数清零；真实核
#     _compute_global_topk_indices_and_lens_kernel L477-L525）
#   combine_topk_swa_indices（L536-L580：topk 段 + SWA 滑窗段拼进 gathered
#     buffer 的联合索引；_SPARSE_PREFILL_TOPK_ALIGNMENT=128 对齐；真实核
#     CombineTopkSwaIndicesKernel L631-…）
#   dequantize_and_gather_k_cache（L390-L432：fp8_ds_mla 584B 分页 dequant
#     gather——448B fp8(7×64 ue8m0 块) + 128B bf16 rope + 8B scale；真实核
#     _dequantize_and_gather_k_kernel）
from __future__ import annotations

import torch

from vllm.utils.import_utils import has_cutedsl  # noqa: F401  (cutedsl 探针位)

# FlashMLA sparse prefill asserts `params.topk % B_TOPK == 0` (see
# flashmla/csrc/sm100/prefill/sparse/fwd/head{64,128}/phase1.cuh). B_TOPK is
# 64 for the h_q=64 kernel and 128 for h_q=128; pad to 128 to satisfy both.
# The extra slots stay as -1 sentinels and `combined_lens` caps the valid
# range via `topk_length`, so padding is a no-op at kernel level.
# SOURCE: vllm/models/deepseek_v4/common/ops/cache_utils.py:L528-L533
#   _SPARSE_PREFILL_TOPK_ALIGNMENT —— 逐字
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


# SOURCE: vllm/models/deepseek_v4/common/ops/cache_utils.py:L390-L432
#   dequantize_and_gather_k_cache —— HOST SEAM 参考数学（fp8_ds_mla 584B：
#   值 [pos, :448] fp8 · 块 scale 尾区 [bs*576, +bs*8) 的 7×ue8m0+1pad →
#   2^(ue8m0-127) · rope [pos, 448:576) bf16 直通）
def dequantize_and_gather_k_cache(
    # [num_reqs, max_num_tokens, head_size]
    out: torch.Tensor,
    # [num_blocks, block_size, head_bytes]
    k_cache: torch.Tensor,
    # [num_reqs]
    seq_lens: torch.Tensor,
    # [num_reqs]
    gather_lens: torch.Tensor | None,
    # [num_reqs, max_blocks_per_seq]
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
    use_fnuz: bool = False,
) -> None:
    """Dequantize and gather a paged DSv4 K cache."""
    # SOURCE: vllm/models/deepseek_v4/common/ops/cache_utils.py:L390-L432
    #   —— HOST SEAM
    flat = k_cache.reshape(-1)
    width = k_cache.shape[-1]  # 584B/token
    scale_area = k_cache.shape[1] * width  # 块尾 scale 区起点
    for b in range(seq_lens.shape[0]):
        seq_len = int(seq_lens[b].item())
        n = (int(gather_lens[b].item()) if gather_lens is not None
             else seq_len)
        for pos in range(n):
            blk = int(block_table[b, pos // block_size].item())
            src = blk * block_size * width + (pos % block_size) * width
            row = flat[src : src + width]
            nope = row[:448].view(torch.float8_e4m3fn).float().view(7, 64)
            scales_b = flat[
                blk * block_size * width + scale_area
                + (pos % block_size) * 8 : blk * block_size * width + scale_area
                + (pos % block_size) * 8 + 7
            ].float()
            scales = torch.pow(2.0, scales_b - 127.0)
            deq = (nope * scales.unsqueeze(1)).reshape(-1)
            rope = row[448:576].view(torch.bfloat16).float()
            out[b, offset + pos] = torch.cat([deq, rope])


# SOURCE: vllm/models/deepseek_v4/common/ops/cache_utils.py:L435-L474
#   compute_global_topk_indices_and_lens —— HOST SEAM 参考数学
def compute_global_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
    output_buffers: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local topk indices to global KV cache slots and count valid entries.

    Fuses three operations into a single kernel:
    1. Block-table lookup (local index → global slot id)
    2. Valid-entry counting (topk_lens per token)
    3. Masking padding tokens to length 0
    """
    # SOURCE: vllm/models/deepseek_v4/common/ops/cache_utils.py:L435-L474
    #   —— HOST SEAM（kernel L477-L525 逐式）
    num_tokens = topk_indices.shape[0]
    if output_buffers is None:
        global_topk_indices = torch.empty_like(topk_indices)
        topk_lens = torch.empty(
            num_tokens, dtype=torch.int32, device=topk_indices.device
        )
    else:
        global_topk_indices, topk_lens = output_buffers
        assert global_topk_indices.shape == topk_indices.shape
        assert topk_lens.shape == (num_tokens,)
    topk = topk_indices.shape[-1]
    for token_idx in range(num_tokens):
        is_valid = int(is_valid_token[token_idx].item()) != 0
        req_idx = int(token_to_req_indices[token_idx].item())
        count = 0
        for i in range(topk):
            local_idx = int(topk_indices[token_idx, i].item())
            if local_idx < 0:
                global_topk_indices[token_idx, i] = -1
                continue
            block_idx = local_idx // block_size
            block_number = int(block_table[req_idx, block_idx].item())
            global_topk_indices[token_idx, i] = (
                block_number * block_size + local_idx % block_size)
            count += 1
        # Zero out length for padding tokens.
        topk_lens[token_idx] = count if is_valid else 0
    return global_topk_indices, topk_lens


# SOURCE: vllm/models/deepseek_v4/common/ops/cache_utils.py:L536-L580
#   combine_topk_swa_indices —— HOST SEAM 参考数学
def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
    out: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    # SOURCE: vllm/models/deepseek_v4/common/ops/cache_utils.py:L536-L580
    #   —— HOST SEAM（kernel L650-… 逐式：topk_len = min((pos+1)//
    #   compress_ratio, TOP_K)；swa_len = min(pos+1, WINDOW_SIZE)；SWA 段
    #   偏移 = M*b + N + pos - swa_len + 1 - gather_start）
    num_tokens = topk_indices.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    if out is None:
        combined_indices = torch.full(
            (num_tokens, combined_topk),
            fill_value=-1,
            dtype=torch.int32,
            device=topk_indices.device,
        )
        combined_lens = torch.empty(
            num_tokens, dtype=torch.int32, device=topk_indices.device
        )
    else:
        combined_indices, combined_lens = out

    base = int(query_start_loc[0].item())
    num_reqs = seq_lens.shape[0]
    for batch_idx in range(num_reqs):
        query_start = int(query_start_loc[batch_idx].item()) - base
        query_end = int(query_start_loc[batch_idx + 1].item()) - base
        query_len = query_end - query_start
        seq_len = int(seq_lens[batch_idx].item())
        gather_len = int(gather_lens[batch_idx].item())
        start_pos = seq_len - query_len
        # The SWA portion of the gathered buffer starts from position
        # (seq_len - gather_len), not position 0.
        gather_start = seq_len - gather_len

        for token_idx in range(query_start, query_end):
            # topk_len is fully determined by the query token's absolute
            # position: both the C4A indexer and the C128A metadata builder
            # emit min((pos + 1) // compress_ratio, topk_tokens) valid entries.
            # Caller passes TOP_K=0 for SWA-only layers to zero this out.
            token_idx_in_query = token_idx - query_start
            pos = start_pos + token_idx_in_query
            topk_len = min((pos + 1) // compress_ratio, topk)
            swa_len = min(pos + 1, window_size)

            for i in range(topk_len):
                combined_indices[token_idx, i] = (
                    int(topk_indices[token_idx, i].item()) + M * batch_idx)
            # Index into gathered buffer: N + (position - gather_start)
            # For positions [pos - swa_len + 1, pos], the buffer indices are:
            # [N + pos - swa_len + 1 - gather_start, N + pos - gather_start]
            for off in range(swa_len):
                combined_indices[token_idx, topk_len + off] = (
                    M * batch_idx + N + off + pos - swa_len + 1 - gather_start)

            combined_len = topk_len + swa_len
            combined_lens[token_idx] = combined_len
    return combined_indices, combined_lens
