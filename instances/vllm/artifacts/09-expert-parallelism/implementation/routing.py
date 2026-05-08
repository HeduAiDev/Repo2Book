"""MoE Top-K routing — pure-NumPy reproductions of vLLM's two router paths.

Two routers are exposed:

- ``fused_topk``  — the Mixtral path. softmax(logits) → torch.topk → optional
  renormalize so the K weights sum to 1.
- ``grouped_topk`` — the DeepSeek-V2/V3 path. Score experts inside groups,
  pick ``topk_group`` groups first, then top-K within.

The math here is the *plain PyTorch* fallback that the production kernel
delegates to when CUDA fast-paths are unavailable. The Triton kernel
(``ops.topk_softmax``) and the Marlin/CUTLASS variants are referenced but
not reimplemented — the math is identical, only the layout differs.
"""

from __future__ import annotations

from typing import Tuple

import torch

# REFERENCE: instances/vllm/source/vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:L69-L113
# REFERENCE: instances/vllm/source/vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:L81-L162


def fused_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool = True,
    scoring_func: str = "softmax",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mixtral-style top-K routing.

    Mirrors ``fused_topk`` in
    ``vllm/model_executor/layers/fused_moe/router/fused_topk_router.py``.
    The production version dispatches to a Triton kernel
    (``ops.topk_softmax`` for softmax, ``ops.topk_sigmoid`` for sigmoid);
    the math is identical to what we do here in plain PyTorch.

    Args:
        hidden_states: ``[M, hidden]`` — only used for shape assert and device.
        gating_output: ``[M, E]`` raw router logits.
        topk: K — number of experts each token activates.
        renormalize: if True, the K weights are divided by their sum so they
            sum to exactly 1 (this is what Mixtral / Switch use). When False,
            weights are the raw softmax values and sum to ≤ 1.
        scoring_func: ``"softmax"`` (Mixtral, default) or ``"sigmoid"``
            (some DeepSeek variants).

    Returns:
        ``(topk_weights, topk_ids, token_expert_indices)``:
            topk_weights: ``[M, K]`` float32. The per-(token, slot) weight to
                multiply the expert output by.
            topk_ids: ``[M, K]`` int32. The global expert IDs each token routes
                to.
            token_expert_indices: ``[M, K]`` int32. The per-(token, slot)
                position used by some kernels for scatter/gather. We compute
                it as ``[[0, 1, ..., K-1]] * M`` to match the production
                contract — vLLM's Triton kernel produces the same.
    """
    # REFERENCE: vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:L77 assert
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    M, E = gating_output.shape

    # REFERENCE: vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:L94-L100
    # softmax-first then top-K. They do NOT commute (Trap G).
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output.to(torch.float32), dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.to(torch.float32).sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    # REFERENCE: torch.topk returns (values, indices); same call as the Triton fast-path.
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1)

    # REFERENCE: vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:L20-L30
    # The fused kernel takes a pre-allocated ``token_expert_indices`` buffer
    # and fills it with [0, 1, ..., K-1] per token. We reproduce that.
    token_expert_indices = (
        torch.arange(topk, dtype=torch.int32, device=gating_output.device)
        .unsqueeze(0)
        .expand(M, topk)
        .contiguous()
    )

    if renormalize:
        # REFERENCE: production kernel applies the same divide inside the Triton op.
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return (
        topk_weights.to(torch.float32),
        topk_ids.to(torch.int32),
        token_expert_indices,
    )


def grouped_topk(
    hidden_states: torch.Tensor,
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool = True,
    num_expert_group: int = 0,
    topk_group: int = 0,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    e_score_correction_bias: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DeepSeek-V2 / V3 grouped top-K routing.

    Mirrors ``grouped_topk`` in
    ``vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py``.
    The two-stage routing is:

    1. Score every expert (via softmax or sigmoid).
    2. **Group score** = max-of-group (or top-2-sum for the noaux_tc bias path).
    3. Pick the ``topk_group`` highest-scoring groups.
    4. Mask out experts not in those groups (``-inf``).
    5. Top-K over the masked scores.

    The ``e_score_correction_bias`` (DeepSeek-V3 ``noaux_tc``) is added to
    scores ONLY for selection — the *unbiased* original scores are returned
    as the routing weights.
    """
    # REFERENCE: vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:L111
    assert hidden_states.size(0) == gating_output.size(0), "Number of tokens mismatch"

    # REFERENCE: vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:L113-L118
    if scoring_func == "softmax":
        scores = torch.softmax(gating_output.to(torch.float32), dim=-1)
    elif scoring_func == "sigmoid":
        scores = gating_output.to(torch.float32).sigmoid()
    else:
        raise ValueError(f"Unsupported scoring function: {scoring_func}")

    num_token = scores.size(0)

    # REFERENCE: vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:L121-L132
    if e_score_correction_bias is not None:
        original_scores = scores
        scores = scores + e_score_correction_bias.unsqueeze(0).to(scores.dtype)
        # noaux_tc uses sum-of-top-2 within each group as the group score.
        group_scores = (
            scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        )
    else:
        # Default: max within group is the group score.
        group_scores = (
            scores.view(num_token, num_expert_group, -1).max(dim=-1).values
        )

    # REFERENCE: vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:L136-L146
    group_idx = torch.topk(group_scores, k=topk_group, dim=-1, sorted=False)[1]
    group_mask = torch.zeros_like(group_scores)
    group_mask.scatter_(1, group_idx, 1)
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
        .reshape(num_token, -1)
    )
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))

    # REFERENCE: vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:L148-L155
    if e_score_correction_bias is not None:
        topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)[1]
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = torch.topk(tmp_scores, k=topk, dim=-1, sorted=False)

    # REFERENCE: vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:L157-L161
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


def expert_load_counts(topk_ids: torch.Tensor, global_num_experts: int) -> torch.Tensor:
    """Histogram of how many (token, slot) pairs map to each expert id.

    Used by EPLB and by the §3 placement demo. Returns a length-E int64
    tensor. Equivalent to ``torch.bincount`` after flattening; we use
    ``index_add_`` directly so behavior is well-defined when no token
    selects a particular expert (the cell stays at 0).
    """
    # REFERENCE: vllm/distributed/eplb/eplb_state.py:L210 EplbState tracks per-expert load
    counts = torch.zeros(global_num_experts, dtype=torch.int64, device=topk_ids.device)
    flat = topk_ids.reshape(-1).to(torch.int64)
    counts.index_add_(0, flat, torch.ones_like(flat))
    return counts
