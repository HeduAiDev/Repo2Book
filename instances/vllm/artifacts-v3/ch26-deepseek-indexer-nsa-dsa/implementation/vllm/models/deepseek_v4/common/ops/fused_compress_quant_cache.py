# SOURCE: vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py
# ch26 切面（m11）：compress_norm_rope_store_triton 的 HOST SEAM 镜像——
# 压缩机的融合尾步（compress → RMSNorm → RoPE → FP8 量化 → KV 缓存写），
# indexer 变体（head_dim=128、全 FP8、1 块/token）逐式：
#   边界 token（(pos+1)%compress_ratio==0）才压缩；窗口 =
#   [pos-(1+overlap)*ratio+1, pos]、长 (1+overlap)*ratio；
#   overlap 时第二半窗读 state 第二头（head_offset = (tokens>=ratio)*HEAD）；
#   score=softmax(窗内 score_state)、compressed = Σ kv·score；
#   RMSNorm(weight) → GPT-J interleave RoPE 打**末** rope_dim 维（位置 =
#   (pos//ratio)*ratio 的压缩位）→ 单块 FP8 ue8m0 量化 + fp32 scale 写缓存。
# SUBTRACTED：head=512 sparse_attn 变体与 MXFP4 变体核、two-stage 分裂核
#   （_compress_gather_split 等）——主压缩 KV 池路径（ch25/ch28 域）与
#   delete[2]（FP4）；本章只承载 indexer（head=128）路径。
from __future__ import annotations

import math
from typing import Any

import torch


# SOURCE: vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py
#   :L32-L107 compress_norm_rope_store_triton —— HOST SEAM 参考数学
def compress_norm_rope_store_triton(
    state_cache: torch.Tensor,
    num_actual: int,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    state_width: int,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    k_cache_metadata: Any,
    pdl_kwargs: dict,
    head_dim: int,
    rope_head_dim: int,
    compress_ratio: int,
    overlap: bool,
    use_fp4_cache: bool,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    quant_block: int,
    token_stride: int,
    scale_dim: int,
) -> None:
    """Shared triton launcher for the fused compress+norm+RoPE+insert path."""
    # SOURCE: vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py
    #   :L32-L107 —— HOST SEAM（indexer 变体核 L319-… 逐式；head=512/
    #   mxfp4 变体按章界删）
    assert head_dim == 128 and not use_fp4_cache, (
        "host mirror carries the indexer (head=128, FP8) variant")
    fp8_max = 448.0
    state_flat = state_cache.reshape(-1, state_cache.shape[-1])
    kv_flat = kv_cache.reshape(-1, kv_cache.shape[-1])
    kv_slot_mapping = k_cache_metadata.slot_mapping
    win = (1 + int(overlap)) * compress_ratio
    for token_idx in range(num_actual):
        slot_id = int(slot_mapping[token_idx].item())
        if slot_id < 0:
            continue
        position = int(positions[token_idx].item())
        if (position + 1) % compress_ratio != 0:
            continue
        req_idx = int(token_to_req_indices[token_idx].item())

        start = position - win + 1
        # Gather state cache entries（分页：block_table 寻址；row_base 的
        # head_offset 同管 kv 半区与 score 半区——overlap 第二半窗读第二头）
        kv_w, sc_w = [], []
        for t_i in range(win):
            pos = start + t_i
            if pos < 0:
                continue
            blk = int(block_table[req_idx, pos // block_size].item())
            src = blk * block_size + pos % block_size
            head_off = head_dim if t_i >= compress_ratio else 0
            kv_w.append(state_flat[src, head_off : head_off + head_dim].float())
            sc_w.append(
                state_flat[src, state_width + head_off : state_width + head_off
                           + head_dim].float())
        kv_w = torch.stack(kv_w)      # kv_state（窗内逐 token）
        sc_w = torch.stack(sc_w)      # score_state（窗内逐 token）
        # Softmax + weighted sum
        score = torch.softmax(sc_w, dim=0)
        compressed_kv = (kv_w * score).sum(0)
        # RMSNorm (fp32 throughout)
        variance = (compressed_kv * compressed_kv).sum() / head_dim
        rrms = torch.rsqrt(variance + rms_norm_eps)
        normed = compressed_kv * rrms * rms_norm_weight.float()

        kv_slot_idx = int(kv_slot_mapping[token_idx].item())
        if kv_slot_idx < 0:
            continue
        # Register-based GPT-J forward RoPE（打末 rope_dim 维；压缩位）
        nope = head_dim - rope_head_dim
        half = rope_head_dim // 2
        compressed_pos = (position // compress_ratio) * compress_ratio
        cos = cos_sin_cache[compressed_pos, :half].float()
        sin = cos_sin_cache[compressed_pos, half:].float()
        even = normed[nope::2]
        odd = normed[nope + 1 :: 2]
        new_even = even * cos - odd * sin
        new_odd = odd * cos + even * sin
        result = torch.empty_like(normed)
        result[:nope] = normed[:nope]
        result[nope::2] = new_even
        result[nope + 1 :: 2] = new_odd

        # FP8 UE8M0 quant: single block, flat reduction
        # （cast fp32 → bf16 → fp32 before quant to match reference）
        quant_input = result.to(torch.bfloat16).float()
        block_absmax = torch.clamp(quant_input.abs().max(), min=1e-4)
        raw_scale = float(block_absmax) / fp8_max
        exponent = math.ceil(math.log2(raw_scale))
        scale = 2.0 ** exponent
        x_fp8 = torch.clamp(quant_input / scale, -fp8_max, fp8_max).to(
            torch.float8_e4m3fn)
        kv_flat[kv_slot_idx, :head_dim] = x_fp8.view(torch.uint8)
        kv_flat[kv_slot_idx, head_dim : head_dim + scale_dim] = (
            torch.tensor([scale], dtype=torch.float32).view(torch.uint8))
