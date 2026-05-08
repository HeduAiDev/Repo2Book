"""Side-by-side: Mixtral (E=8, K=2) vs DeepSeek-V2 grouped (E=64, K=8).

These are the two reference sites the brief calls out:

- ``vllm/model_executor/models/mixtral.py:L77 class MixtralMoE`` — the
  simple case. ``ReplicatedLinear`` gate, ``FusedMoE(num_experts=8,
  top_k=2, use_grouped_topk=False)``, no shared expert.
- ``vllm/model_executor/models/deepseek_v2.py:L244 class DeepseekV2MoE`` —
  the production case. ``GateLinear`` gate with optional
  ``e_score_correction_bias``, ``FusedMoE(use_grouped_topk=True,
  num_expert_group=n_group, topk_group=topk_group, n_shared_experts=…)``.

The point of the side-by-side is to see how the SAME ``FusedMoE`` constructor
takes you from one to the other through different argument values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .fused_moe_block import FusedMoEBlock

# REFERENCE: instances/vllm/source/vllm/model_executor/models/mixtral.py:L77-L154
# REFERENCE: instances/vllm/source/vllm/model_executor/models/deepseek_v2.py:L244-L386


@dataclass
class MoEConfig:
    """A minimal config bundle covering the args FusedMoEBlock cares about."""

    num_experts: int
    top_k: int
    hidden_size: int
    intermediate_size: int
    use_grouped_topk: bool = False
    num_expert_group: int = 0
    topk_group: int = 0
    scoring_func: str = "softmax"
    has_shared_expert: bool = False
    name: str = "moe"


# Mixtral ships with E=8, K=2 per attention block.
# REFERENCE: vllm/model_executor/models/mixtral.py:L132-L145 num_experts=8, top_k=2
MIXTRAL_8x7B = MoEConfig(
    num_experts=8,
    top_k=2,
    hidden_size=4096,
    intermediate_size=14336,
    use_grouped_topk=False,
    scoring_func="softmax",
    has_shared_expert=False,
    name="Mixtral-8x7B",
)


# DeepSeek-V2-Lite uses 64 routed experts in groups, with shared experts.
# We use the lite-scale numbers here to keep the demo runtime sane; full V3
# is E=256 with the same shapes.
# REFERENCE: vllm/model_executor/models/deepseek_v2.py:L319-L341
DEEPSEEK_V2_LITE = MoEConfig(
    num_experts=64,
    top_k=6,
    hidden_size=2048,
    intermediate_size=1408,
    use_grouped_topk=True,
    num_expert_group=8,
    topk_group=3,
    scoring_func="softmax",
    has_shared_expert=True,
    name="DeepSeek-V2-Lite",
)


def build_block(cfg: MoEConfig, ep_size: int = 1, seed: int = 0) -> FusedMoEBlock:
    """Construct a FusedMoEBlock matching the requested config."""
    return FusedMoEBlock(
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        ep_size=ep_size,
        use_grouped_topk=cfg.use_grouped_topk,
        num_expert_group=cfg.num_expert_group,
        topk_group=cfg.topk_group,
        scoring_func=cfg.scoring_func,
        seed=seed,
    )


def routing_fingerprint(
    block: FusedMoEBlock, num_tokens: int, seed: int = 0
) -> dict:
    """Run a fixed-seed routing pass and return a numerics fingerprint.

    Returns a dict the writer can quote verbatim:
        - ``E``, ``K``: chapter-headline expert and topk counts
        - ``per_expert_count``: list of int64 counts (length E)
        - ``max_count``, ``min_count``, ``mean_count``: aggregates
        - ``coverage``: fraction of experts that received at least one token
        - ``weight_sum_min``, ``weight_sum_max``: per-token weight sum range
          — when ``renormalize=True`` this should be 1.0 ± 1e-5.
    """
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(num_tokens, block.hidden_size, generator=g)
    logits = h @ block.gate_weight.T
    tw, ti = block._route(h, logits)
    counts = torch.zeros(block.num_experts, dtype=torch.int64)
    counts.index_add_(
        0, ti.reshape(-1).to(torch.int64), torch.ones_like(ti.reshape(-1).to(torch.int64))
    )
    weight_sums = tw.sum(dim=-1)
    return {
        "model": "fingerprint",
        "E": block.num_experts,
        "K": block.top_k,
        "num_tokens": num_tokens,
        "per_expert_count": counts.tolist(),
        "max_count": int(counts.max().item()),
        "min_count": int(counts.min().item()),
        "mean_count": float(counts.float().mean().item()),
        "coverage": float((counts > 0).float().mean().item()),
        "weight_sum_min": float(weight_sums.min().item()),
        "weight_sum_max": float(weight_sums.max().item()),
        "weight_sum_mean": float(weight_sums.mean().item()),
    }
