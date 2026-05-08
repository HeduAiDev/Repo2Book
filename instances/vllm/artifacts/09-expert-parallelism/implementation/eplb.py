"""Toy EplbState — runtime expert-parallel load-balancing.

Pedagogical mirror of ``vllm/distributed/eplb/eplb_state.py:L210 class EplbState``.
The production class is 1166 lines; we keep the *invariants* and skip the
async-worker plumbing.

Crucially, this is a **runtime statistical rebalancer**. It is NOT an aux
loss. vLLM is inference-only — there is no gradient anywhere in this code,
no backprop, no Switch Transformer ``L_balance``. EPLB's only knob is *which
physical rank gets which logical expert* — it shuffles the placement when
load skews.

Two ingredients:

1. **Redundant experts** — when ``num_redundant_experts > 0``, several
   physical slots can map to the same logical expert. A hot logical expert
   that lives on many physical slots gets parallelized automatically.
2. **Periodic reshuffle** — every ``rearrangement_step_interval`` forwards,
   compute the running per-expert load, decide a new
   logical→physical map, broadcast it (in real EP via the ``_EPLB`` group),
   and atomically swap weight pointers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence

import torch

# REFERENCE: instances/vllm/source/vllm/distributed/eplb/eplb_state.py:L210-L944


@dataclass
class EplbState:
    """Toy state machine for EPLB.

    Tracks a sliding window of per-logical-expert token counts. When the
    load imbalance exceeds a threshold (or every ``interval`` steps), we
    redistribute the logical→physical map.
    """

    num_logical_experts: int
    num_redundant_experts: int = 0
    ep_size: int = 1
    rearrangement_step_interval: int = 50
    window_size: int = 50

    # Internal bookkeeping (don't mutate from outside).
    _step: int = 0
    _last_rearrangement_step: int = 0
    _load_history: List[torch.Tensor] = field(default_factory=list)
    _physical_to_logical: torch.Tensor | None = None

    def __post_init__(self) -> None:
        # REFERENCE: vllm/distributed/eplb/eplb_state.py:L286-L304
        # Initial layout: [original logical experts, redundant experts]
        # where redundant ones round-robin re-use logical IDs.
        n_phys = self.num_logical_experts + self.num_redundant_experts
        layout = list(range(self.num_logical_experts))
        layout += [
            i % self.num_logical_experts for i in range(self.num_redundant_experts)
        ]
        self._physical_to_logical = torch.tensor(layout, dtype=torch.int64)
        assert self._physical_to_logical.numel() == n_phys

    @property
    def num_physical_experts(self) -> int:
        return self.num_logical_experts + self.num_redundant_experts

    @property
    def physical_to_logical(self) -> torch.Tensor:
        """Read-only view: physical slot → logical expert it currently serves."""
        assert self._physical_to_logical is not None
        return self._physical_to_logical

    # ------------------------------------------------------------------
    # Step — record load + maybe rearrange.
    # ------------------------------------------------------------------

    def record_step(self, per_expert_load: torch.Tensor) -> bool:
        """Record one forward pass; return True if a rearrangement happened.

        Args:
            per_expert_load: int64 tensor of shape ``[num_logical_experts]``
                with the per-logical-expert token count for this step.

        Returns:
            True if EPLB rearranged the layout this step, False otherwise.
        """
        # REFERENCE: vllm/distributed/eplb/eplb_state.py:L223-L249
        assert per_expert_load.shape == (self.num_logical_experts,)
        self._step += 1
        self._load_history.append(per_expert_load.detach().clone())
        if len(self._load_history) > self.window_size:
            self._load_history.pop(0)

        if self._step - self._last_rearrangement_step < self.rearrangement_step_interval:
            return False
        self._rearrange()
        self._last_rearrangement_step = self._step
        return True

    # ------------------------------------------------------------------
    # Rearrange — pick a new physical→logical map.
    # ------------------------------------------------------------------

    def _rearrange(self) -> None:
        """Recompute physical→logical based on the windowed load.

        Greedy heuristic (the production policy is more sophisticated and
        lives in ``vllm/distributed/eplb/policies.py``):

        1. Average the per-logical-expert load over the window.
        2. Sort logical experts by load (descending).
        3. Assign hot experts to redundant physical slots first so they
           appear on multiple ranks.
        4. Place remaining experts to fill the layout.
        """
        if not self._load_history:
            return
        # REFERENCE: vllm/distributed/eplb/eplb_state.py:L920+ rearrangement entrypoint.
        windowed = torch.stack(self._load_history, dim=0).float().mean(dim=0)
        n_log = self.num_logical_experts
        n_phys = self.num_physical_experts

        # Hottest expert duplicates go first into redundant slots.
        order = torch.argsort(windowed, descending=True).tolist()
        layout: list[int] = []
        # Each logical expert gets at least one physical slot.
        layout.extend(range(n_log))
        # Remaining redundant slots go to hot experts in round-robin order.
        redundant = n_phys - n_log
        for i in range(redundant):
            layout.append(order[i % len(order)])
        self._physical_to_logical = torch.tensor(layout, dtype=torch.int64)

    # ------------------------------------------------------------------
    # Imbalance metric — what we report in §3 demo.
    # ------------------------------------------------------------------

    def imbalance_ratio(self, per_rank_load: torch.Tensor) -> float:
        """``max / mean`` of per-rank load. 1.0 == perfect balance.

        The standard imbalance metric used in the EPLB literature. Low =
        good; high = a few ranks are doing all the work.
        """
        per_rank_load = per_rank_load.float()
        if per_rank_load.numel() == 0:
            return 1.0
        mean = per_rank_load.mean().clamp_min(1.0)
        return float(per_rank_load.max() / mean)


def per_rank_load_from_logical_load(
    per_logical_load: torch.Tensor, ep_size: int, placement_strategy: str = "linear"
) -> torch.Tensor:
    """Map per-logical-expert load to per-rank load under a given placement.

    Helper for the placement demo. Returns an ``[ep_size]`` int64 tensor.
    """
    # REFERENCE: same expert-map logic as expert_map.determine_expert_map.
    n_log = per_logical_load.numel()
    base = n_log // ep_size
    rem = n_log % ep_size
    out = torch.zeros(ep_size, dtype=torch.int64)
    if placement_strategy == "linear":
        for r in range(ep_size):
            local = base + (1 if r < rem else 0)
            start = r * base + min(r, rem)
            out[r] = per_logical_load[start : start + local].sum()
    elif placement_strategy == "round_robin":
        for r in range(ep_size):
            idx = torch.arange(r, n_log, ep_size)
            out[r] = per_logical_load[idx].sum()
    else:
        raise ValueError(placement_strategy)
    return out


def make_skewed_routing(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    hot_fraction: float = 0.2,
    hot_load_fraction: float = 0.6,
    seed: int = 0,
) -> torch.Tensor:
    """Generate fake topk_ids whose distribution is Pareto-skewed.

    Top ``hot_fraction`` of experts receive ``hot_load_fraction`` of tokens.
    Used for §3 placement and §5 EPLB demos.
    """
    g = torch.Generator().manual_seed(seed)
    n_hot = max(1, int(num_experts * hot_fraction))
    hot_experts = torch.arange(n_hot)
    cold_experts = torch.arange(n_hot, num_experts)
    hot_picks = int(top_k * num_tokens * hot_load_fraction)
    cold_picks = top_k * num_tokens - hot_picks

    hot_choices = hot_experts[
        torch.randint(0, n_hot, (hot_picks,), generator=g, dtype=torch.int64)
    ]
    cold_choices = cold_experts[
        torch.randint(
            0, max(1, num_experts - n_hot), (cold_picks,), generator=g, dtype=torch.int64
        )
    ]
    flat = torch.cat([hot_choices, cold_choices])
    flat = flat[torch.randperm(flat.numel(), generator=g)]
    return flat.view(num_tokens, top_k).to(torch.int32)
