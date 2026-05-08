"""Expert placement — building the global→local expert map.

Reproduces ``determine_expert_map`` from
``vllm/model_executor/layers/fused_moe/layer.py``. Two strategies:

- ``"linear"``: rank ``r`` owns the contiguous block
  ``[r*base + min(r, rem), r*base + min(r, rem) + local)``.
- ``"round_robin"``: rank ``r`` owns experts ``r, r+P, r+2P, ...``.

The returned ``expert_map`` has shape ``(global_num_experts,)``: each entry
is the local index on this rank, or ``-1`` if the expert is not on this
rank. The ``-1`` sentinel is the same convention vLLM uses; the FusedMoE
forward pass uses it to skip tokens routed to off-rank experts.
"""

from __future__ import annotations

from typing import Tuple

import torch

# REFERENCE: instances/vllm/source/vllm/model_executor/layers/fused_moe/layer.py:L70-L157

VALID_STRATEGIES = ("linear", "round_robin")


def determine_expert_map(
    ep_size: int,
    ep_rank: int,
    global_num_experts: int,
    expert_placement_strategy: str = "linear",
) -> Tuple[int, torch.Tensor | None]:
    """Compute (local_num_experts, expert_map) for a single rank.

    Args:
        ep_size: total number of EP ranks.
        ep_rank: this rank's id in ``[0, ep_size)``.
        global_num_experts: E in the model.
        expert_placement_strategy: ``"linear"`` (default) or ``"round_robin"``.

    Returns:
        Tuple ``(local_num_experts, expert_map_or_None)``. When ``ep_size==1``
        we return ``(global_num_experts, None)`` — there is no sharding to
        do, and FusedMoE's forward pass takes a fast path on ``None``.
    """
    # REFERENCE: vllm/model_executor/layers/fused_moe/layer.py:L107-L109
    assert ep_size > 0
    assert 0 <= ep_rank < ep_size, f"ep_rank={ep_rank} out of range for ep_size={ep_size}"
    if ep_size == 1:
        return (global_num_experts, None)

    if global_num_experts < ep_size:
        # vLLM never hits this in practice (E >> P) but our toy demos do.
        # The behavior must mirror what the source would do if you ran it:
        # remainder ranks get one expert, the rest get zero.
        pass

    # REFERENCE: vllm/model_executor/layers/fused_moe/layer.py:L112-L114
    base_experts = global_num_experts // ep_size
    remainder = global_num_experts % ep_size
    local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts

    # REFERENCE: vllm/model_executor/layers/fused_moe/layer.py:L117 the -1 sentinel
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)

    if expert_placement_strategy == "linear":
        # REFERENCE: vllm/model_executor/layers/fused_moe/layer.py:L119-L123
        # Block placement: rank r owns experts [start, start+local).
        # The min(ep_rank, remainder) accounts for the +1 ranks at the front.
        start_idx = ep_rank * base_experts + min(ep_rank, remainder)
        expert_map[start_idx : start_idx + local_num_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    elif expert_placement_strategy == "round_robin":
        # REFERENCE: vllm/model_executor/layers/fused_moe/layer.py:L124-L131
        local_log_experts = torch.arange(
            ep_rank, global_num_experts, ep_size, dtype=torch.int64
        )
        expert_map[local_log_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    else:
        raise ValueError(
            f"Unsupported expert placement strategy '{expert_placement_strategy}', "
            f"expected one of {VALID_STRATEGIES}"
        )

    return local_num_experts, expert_map


def get_compressed_expert_map(expert_map: torch.Tensor) -> str:
    """Pretty-print the local→global mapping (logging helper).

    Mirrors ``get_compressed_expert_map`` at
    ``vllm/model_executor/layers/fused_moe/layer.py:L196-L214``.
    """
    # REFERENCE: vllm/model_executor/layers/fused_moe/layer.py:L209
    global_indices = torch.where(expert_map != -1)[0]
    local_indices = expert_map[global_indices]
    return ", ".join(
        f"{local_index.item()}->{global_index.item()}"
        for local_index, global_index in zip(local_indices, global_indices)
    )


def all_rank_maps(
    ep_size: int,
    global_num_experts: int,
    expert_placement_strategy: str = "linear",
) -> list[tuple[int, torch.Tensor | None]]:
    """Build ``determine_expert_map`` for all ranks at once (single-process demo).

    Useful for the placement demo so we can show all ranks' coverage in one
    figure without standing up real ``torch.distributed``.
    """
    return [
        determine_expert_map(
            ep_size, ep_rank, global_num_experts, expert_placement_strategy
        )
        for ep_rank in range(ep_size)
    ]


def per_rank_token_load(
    topk_ids: torch.Tensor, expert_maps: list[torch.Tensor | None]
) -> torch.Tensor:
    """How many (token, slot) routed pairs land on each rank.

    For each rank ``r`` with map ``M_r``, we count entries of ``topk_ids``
    that satisfy ``M_r[id] != -1``. Returns shape ``[P]``, int64.
    """
    P = len(expert_maps)
    flat = topk_ids.reshape(-1).to(torch.int64)
    out = torch.zeros(P, dtype=torch.int64)
    for r, m in enumerate(expert_maps):
        if m is None:
            # ep_size==1: every token lands on the only rank.
            out[r] = flat.numel()
            continue
        on_rank = (m[flat] != -1).sum().item()
        out[r] = int(on_rank)
    return out
