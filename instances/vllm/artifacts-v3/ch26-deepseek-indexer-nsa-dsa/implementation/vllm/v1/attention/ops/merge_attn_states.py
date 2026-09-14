# SOURCE: vllm/v1/attention/ops/merge_attn_states.py
# HOST SEAM（ch25 同款骨架）：merge_attn_states 的 Triton 分派面（真实
# CUDA/Triton 双核分派；host 走 Triton 支镜像 triton_merge_attn_states）。
from __future__ import annotations

import torch


# SOURCE: vllm/v1/attention/ops/merge_attn_states.py merge_attn_states ——
#   HOST SEAM 分派位
def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,
):
    # SOURCE: vllm/v1/attention/ops/merge_attn_states.py —— HOST SEAM（真实
    #   按 CUDA/Triton 分派；host 走 Triton 支）
    from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states

    return merge_attn_states(
        output,
        prefix_output,
        prefix_lse,
        suffix_output,
        suffix_lse,
        output_lse,
        prefill_tokens_with_context,
        output_scale,
    )
