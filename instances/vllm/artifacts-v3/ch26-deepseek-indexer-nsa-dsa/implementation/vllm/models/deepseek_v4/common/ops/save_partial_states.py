# SOURCE: vllm/models/deepseek_v4/common/ops/save_partial_states.py
# HOST SEAM（B1 Triton kernel 镜像·精确数学）：save_partial_states——
# kv 前半 / score+ape（score += ape[pos % compress_ratio]）后半写进分页
# state_cache；slot<0（pad）跳过。真实核 L31-L91；host 镜像逐式。
from __future__ import annotations

import torch


# SOURCE: vllm/models/deepseek_v4/common/ops/save_partial_states.py:L8-L30
#   save_partial_states —— HOST SEAM 参考数学
def save_partial_states(
    kv: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    positions: torch.Tensor,
    state_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_size: int,
    state_width: int,
    compress_ratio: int,
    pdl_kwargs: dict | None = None,
) -> None:
    """Write packed [kv, score+ape] partial states into the compressor cache.

    One program per token; pads (slot_id == -1) are skipped.
    """
    # SOURCE: vllm/models/deepseek_v4/common/ops/save_partial_states.py:L8-L30
    #   —— HOST SEAM（kernel L31-L91 逐式：kv 前半/score+ape 后半）
    head_size = kv.shape[-1]
    flat = state_cache.reshape(-1, state_cache.shape[-1])
    for token_idx in range(slot_mapping.shape[0]):
        slot_id = int(slot_mapping[token_idx].item())
        # Skip padded / invalid tokens (slot_id == -1 is the PAD sentinel used
        # by vLLM).
        if slot_id < 0:
            continue
        base = flat[slot_id]
        base[:head_size] = kv[token_idx, :head_size].float()
        position = int(positions[token_idx].item())
        ape_row = position % compress_ratio
        base[state_width : state_width + head_size] = (
            score[token_idx, :head_size].float() + ape[ape_row, :head_size].float())
