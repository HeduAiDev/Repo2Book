# SOURCE: vllm/v1/attention/ops/triton_merge_attn_states.py
# HOST SEAM：Triton merge kernel 的 host 参考镜像。真实实现是 @triton.jit
# kernel（GPU 编译执行）；host 无 triton/GPU——本镜像承载同一函数签名与
# LSE 重标定数学（softmax 分块合并精确恒等式）：
#   M = max(prefix_lse, suffix_lse)
#   out = (prefix·exp(prefix_lse−M) + suffix·exp(suffix_lse−M)) /
#         (exp(prefix_lse−M) + exp(suffix_lse−M))
# prefill_tokens_with_context 门与 CUDA kernel 同语义：行号 ≥ 该值的 token
# 无上下文，suffix 直通（其 prefix 分母为 0/−inf）。mask_empty_context
# 同理（空上下文请求行置 lse=−inf、out=0——merge 端自然退化 suffix 直通）。
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
    wp = torch.exp(prefix_lse[:, :gate] - m)  # [N, gate]
    ws = torch.exp(suffix_lse[:, :gate] - m)
    # [N, gate] → [gate, N, 1] 广播到 [gate, N, V]
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


# SOURCE: vllm/v1/attention/ops/triton_merge_attn_states.py mask_empty_context
#   —— HOST SEAM 参考数学（空上下文请求的 query 行置 lse=−inf / out=0）
def mask_empty_context(
    attn_softmax_lse: torch.Tensor,
    attn_output: torch.Tensor,
    query_start_loc: torch.Tensor,
    cu_seq_lens: torch.Tensor,
) -> None:
    # SOURCE: vllm/v1/attention/ops/triton_merge_attn_states.py —— HOST SEAM
    cu = cu_seq_lens.tolist()
    qsl = query_start_loc.tolist()
    for r in range(len(qsl) - 1):
        if cu[r + 1] == cu[r]:  # 本请求在当前 chunk 无上下文行
            attn_softmax_lse[:, qsl[r]:qsl[r + 1]] = -float("inf")
            attn_output[qsl[r]:qsl[r + 1]] = 0
