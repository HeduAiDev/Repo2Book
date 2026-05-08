"""Composition layer — gate → routing → dispatch → expert exec → combine.

Mirror of ``vllm/model_executor/layers/fused_moe/layer.py:L219+ class FusedMoE``
at the algorithmic level. The production class is 1300+ lines because it
also has to handle:

- 7 quant methods (each with their own ``create_weights``)
- 7 all2all backends
- ROCm aiter shared-experts fusion
- EPLB integration with quant methods
- ``MoERunner`` indirection so monolithic kernels can be plugged in.

We strip all of that. What remains is the *core* MoE loop, which fits in
~150 lines and is faithful to the data flow shown in
``vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .all2all_baseline import AgRsAll2AllManager
from .expert_map import determine_expert_map
from .routing import expert_load_counts, fused_topk, grouped_topk

# REFERENCE: instances/vllm/source/vllm/model_executor/layers/fused_moe/layer.py:L219-L290
# REFERENCE: instances/vllm/source/vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:L71-L168
# REFERENCE: instances/vllm/source/vllm/model_executor/layers/fused_moe/router/router_factory.py


@dataclass
class ExpertFFNWeights:
    """Per-expert fused-MoE weight pair.

    Mirrors the (w13, w2) layout in
    ``vllm/model_executor/layers/fused_moe/layer.py:L222-L223``:

        w13 = MergedColumnParallelLinear (gate_proj | up_proj fused)
        w2  = RowParallelLinear           (down_proj)

    Stored shapes:
        w13: ``[E_local, 2 * intermediate, hidden]``  (gate concat up)
        w2 : ``[E_local, hidden, intermediate]``
    """

    w13: torch.Tensor
    w2: torch.Tensor


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """SwiGLU: split last dim in half, silu(left) * right.

    Same as ``vllm.model_executor.layers.activation.SiluAndMul``.
    """
    half = x.shape[-1] // 2
    gate, up = x[..., :half], x[..., half:]
    return torch.nn.functional.silu(gate) * up


class FusedMoEBlock:
    """One MoE block — replicated gate, EP-sharded experts.

    Pedagogical re-impl of ``FusedMoE.forward``. We do everything in a
    single process; "EP rank" is simulated by holding all P expert maps
    and running the per-rank expert FFN P times.

    The key teaching invariant (5-step rhythm anchor):

        gate(h) → softmax/topk → all-to-all dispatch → local expert FFN
                                                     → all-to-all combine

    For each token, only ``top_k`` experts fire. The all-to-alls move
    routed tokens to whichever rank owns their experts and bring the
    weighted partial outputs back.
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        ep_size: int = 1,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: int = 0,
        topk_group: int = 0,
        scoring_func: str = "softmax",
        expert_placement_strategy: str = "linear",
        seed: int = 0,
    ):
        # REFERENCE: vllm/model_executor/layers/fused_moe/layer.py:L251-L290 __init__ args
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.ep_size = ep_size
        self.renormalize = renormalize
        self.use_grouped_topk = use_grouped_topk
        self.num_expert_group = num_expert_group
        self.topk_group = topk_group
        self.scoring_func = scoring_func
        self.expert_placement_strategy = expert_placement_strategy

        torch.manual_seed(seed)
        # REFERENCE: vllm/model_executor/layers/fused_moe/layer.py:L590 quant_method.create_weights
        # gate weight is REPLICATED across all EP ranks.
        self.gate_weight = torch.randn(num_experts, hidden_size) * 0.02
        # All experts' weights live here (the EP fan-out is logical).
        self.experts = ExpertFFNWeights(
            w13=torch.randn(num_experts, 2 * intermediate_size, hidden_size) * 0.02,
            w2=torch.randn(num_experts, hidden_size, intermediate_size) * 0.02,
        )

        # Per-rank expert maps. Index ``r`` gives ``(local_E, expert_map_r)``.
        # REFERENCE: vllm/model_executor/layers/fused_moe/layer.py:L395-L406
        self._all_rank_maps = [
            determine_expert_map(ep_size, r, num_experts, expert_placement_strategy)
            for r in range(ep_size)
        ]
        self.local_num_experts_per_rank = [m[0] for m in self._all_rank_maps]
        self._expert_maps = [m[1] for m in self._all_rank_maps]
        self.a2a = AgRsAll2AllManager(ep_size=ep_size)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _route(
        self, hidden_states: torch.Tensor, router_logits: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Dispatch to the correct router family."""
        if self.use_grouped_topk:
            # REFERENCE: vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py:L341-L351
            tw, ti = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=self.top_k,
                renormalize=self.renormalize,
                num_expert_group=self.num_expert_group,
                topk_group=self.topk_group,
                scoring_func=self.scoring_func,
            )
            return tw, ti
        # REFERENCE: vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:L158-L165
        tw, ti, _ = fused_topk(
            hidden_states=hidden_states,
            gating_output=router_logits,
            topk=self.top_k,
            renormalize=self.renormalize,
            scoring_func=self.scoring_func,
        )
        return tw, ti

    # ------------------------------------------------------------------
    # Local expert FFN — runs on a single rank's experts only
    # ------------------------------------------------------------------

    def _run_local_experts(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        rank: int,
    ) -> torch.Tensor:
        """Run the experts owned by ``rank`` on the gathered tokens.

        Output is the WEIGHTED, SUMMED contribution from this rank's
        experts to each token. Tokens whose top-K all live on other ranks
        contribute zero from this rank.

        Equivalent to the ``apply`` path inside
        ``MoEPrepareAndFinalizeNaiveDPEPModular.prepare`` →
        ``UnquantizedFusedMoEMethod.apply`` → ``finalize``.
        """
        # REFERENCE: vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:L104-L168
        M = hidden_states.shape[0]
        H = hidden_states.shape[1]
        out = torch.zeros(M, H, dtype=hidden_states.dtype)

        expert_map = self._expert_maps[rank]
        if expert_map is None:
            # ep_size==1: every expert is local.
            local_mask = torch.ones(M, self.top_k, dtype=torch.bool)
            local_indices = topk_ids.to(torch.int64)
        else:
            # local_mask[i, k] is True when token i's k-th choice lives on this rank.
            global_ids = topk_ids.to(torch.int64)
            local_indices = expert_map[global_ids]
            local_mask = local_indices != -1

        for slot in range(self.top_k):
            slot_mask = local_mask[:, slot]
            if not slot_mask.any():
                continue
            # Token indices that route through this slot to a local expert.
            tok_idx = torch.where(slot_mask)[0]
            global_eid = topk_ids[tok_idx, slot].to(torch.int64)
            weight = topk_weights[tok_idx, slot].unsqueeze(-1).to(hidden_states.dtype)
            tok_h = hidden_states[tok_idx]

            # REFERENCE: per-expert FFN — w13 is the merged gate|up, w2 is down_proj.
            # The vLLM kernel batches across experts via grouped GEMM; we loop
            # for clarity (this is an educational impl, not a kernel).
            for eid in torch.unique(global_eid).tolist():
                row_mask = global_eid == eid
                rows = torch.where(row_mask)[0]
                w13 = self.experts.w13[eid]
                w2 = self.experts.w2[eid]
                # gate|up projection then SwiGLU then down_proj.
                gateup = tok_h[rows] @ w13.T  # [n, 2*intermediate]
                act = silu_and_mul(gateup)  # [n, intermediate]
                down = act @ w2.T  # [n, hidden]
                contribution = down * weight[rows]
                # Accumulate back at the original token indices.
                out.index_add_(0, tok_idx[rows], contribution)
        return out

    # ------------------------------------------------------------------
    # Forward — the full MoE block.
    # ------------------------------------------------------------------

    def forward(
        self, hidden_states: torch.Tensor, router_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """One MoE block evaluation, single-process EP simulation.

        The full data flow:

        1. (per-token, per-rank) router logits = h @ gate_weight^T.
           Gate weight is REPLICATED on all EP ranks; in our sim every rank
           reproduces the same logits.
        2. Top-K routing.
        3. ``dispatch``: in real EP, all-to-all sends each token to the
           rank owning the expert it picked. In-process, every rank sees
           every token (we mimic this by running the local-experts pass
           on the FULL set of tokens, masking out off-rank picks).
        4. Local expert FFN.
        5. ``combine``: sum the per-rank contributions.
        """
        # REFERENCE: vllm/model_executor/layers/fused_moe/layer.py:L1543-L1649 forward
        if router_logits is None:
            # REFERENCE: vllm/model_executor/models/mixtral.py:L152 router_logits, _ = gate(h)
            router_logits = hidden_states @ self.gate_weight.T

        # REFERENCE: vllm/model_executor/layers/fused_moe/router/fused_topk_router.py:L149-L167
        topk_weights, topk_ids = self._route(hidden_states, router_logits)

        # In a real run, dispatch would all-gather hidden_states + topk meta
        # across the EP group. Single-process: every "rank" already sees
        # everything; we just do the per-rank expert pass and sum.
        # REFERENCE: vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:L127-L132
        contributions = []
        for r in range(self.ep_size):
            contributions.append(
                self._run_local_experts(hidden_states, topk_weights, topk_ids, rank=r)
            )

        # REFERENCE: vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py:L166-L168
        # Combine = sum across ranks. Real path uses reduce_scatter to also
        # split per-rank ownership of the output; in-process the sum is the
        # full output and every rank sees it.
        combined = torch.stack(contributions, dim=0).sum(dim=0)
        return combined

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def expert_load(
        self, hidden_states: torch.Tensor, router_logits: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return the per-expert token count for the given input.

        Helper for the EPLB/placement demos.
        """
        if router_logits is None:
            router_logits = hidden_states @ self.gate_weight.T
        topk_weights, topk_ids = self._route(hidden_states, router_logits)
        return expert_load_counts(topk_ids, self.num_experts)


def memory_per_rank_MiB(
    num_experts: int,
    hidden: int,
    intermediate: int,
    ep_size: int,
    tp_size: int,
    bytes_per_param: int = 2,
) -> float:
    """Closed-form weight memory per rank.

    Mirrors the §9.5 invariant ``mem_per_rank ≈ E * intermediate * hidden /
    (ep × tp)``. Each expert has w13 (gate|up = 2 * intermediate * hidden)
    and w2 (down = hidden * intermediate), so per-expert params =
    3 * intermediate * hidden. Across E experts, sharded by ep_size on the
    expert axis and tp_size on the inside-the-expert axis.
    """
    params_per_expert = 3 * intermediate * hidden
    total = num_experts * params_per_expert * bytes_per_param
    return total / (ep_size * tp_size) / (1024**2)
