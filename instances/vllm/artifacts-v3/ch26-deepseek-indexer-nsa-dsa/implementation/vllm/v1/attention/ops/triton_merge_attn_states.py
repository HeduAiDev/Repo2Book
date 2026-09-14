# SOURCE: vllm/v1/attention/ops/triton_merge_attn_states.py
# HOST SEAM（ch25 同款·精确数学）：Triton merge kernel 的 host 参考镜像——
# LSE 重标定 softmax 分块合并精确恒等式：
#   M = max(prefix_lse, suffix_lse)
#   out = (prefix·exp(prefix_lse−M) + suffix·exp(suffix_lse−M)) /
#         (exp(prefix_lse−M) + exp(suffix_lse−M))
from __future__ import annotations

import torch


# SOURCE: vllm/v1/attention/ops/triton_merge_attn_states.py merge_attn_states
#   —— HOST SEAM 参考数学
def merge_attn_states(
    output: torch.Tensor,
    prefix_output: torch.Tensor,
    prefix_lse: torch.Tensor,
    suffix_output: torch.Tensor,
    suffix_lse: torch.Tensor,
    output_lse: torch.Tensor | None = None,
    prefill_tokens_with_context: int | None = None,
    output_scale: torch.Tensor | None = None,
) -> None:
    # SOURCE: vllm/v1/attention/ops/triton_merge_attn_states.py —— HOST SEAM
    T = prefix_output.shape[0]
    gate = T if prefill_tokens_with_context is None else prefill_tokens_with_context
    if gate == 0:
        output.copy_(suffix_output)
        if output_lse is not None:
            output_lse.copy_(suffix_lse)
        return
    m = torch.maximum(prefix_lse[:, :gate], suffix_lse[:, :gate])  # [N, gate]
    wp = torch.exp(prefix_lse[:, :gate] - m)
    ws = torch.exp(suffix_lse[:, :gate] - m)
    wp_t = wp.t().unsqueeze(-1)
    ws_t = ws.t().unsqueeze(-1)
    merged = (prefix_output[:gate] * wp_t + suffix_output[:gate] * ws_t) / (
        wp_t + ws_t
    )
    output[:gate] = merged
    if gate < T:
        output[gate:] = suffix_output[gate:]
    if output_lse is not None:
        lse = torch.logaddexp(prefix_lse[:, :gate], suffix_lse[:, :gate])
        output_lse[:, :gate] = lse
        if gate < T:
            output_lse[:, gate:] = suffix_lse[:, gate:]
