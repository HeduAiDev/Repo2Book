"""All-to-all dispatch+combine — the AgRs (allgather + reduce-scatter) baseline.

This is the *pedagogical* all-to-all — the simplest backend in vLLM, found at
``vllm/distributed/device_communicators/all2all.py:L40 AgRsAll2AllManager``.
The other six managers (``DeepEPHTAll2AllManager``,
``DeepEPLLAll2AllManager``, ``NixlEPAll2AllManager``,
``FlashInferNVLink{One,Two}SidedManager``, ``MoriAll2AllManager``) all share
the SAME interface (``dispatch``/``combine``) but use platform-specific
fused kernels for cross-node IB or NVLink P2P.

Two important things this file teaches:

1. **AgRs is NOT a true symmetric all-to-all.** ``dispatch`` is an
   ``all_gatherv`` (with per-rank chunk sizes) and ``combine`` is a
   ``reduce_scatterv``. The end state is the same as a hand-written
   all-to-all with reduction, but the kernel sequence is allgather + add.
2. **Cost model.** alpha-beta latency for both ops, comparing all-to-all
   to all-reduce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch

# REFERENCE: instances/vllm/source/vllm/distributed/device_communicators/all2all.py:L40-L139


# ---------------------------------------------------------------------------
# Single-process simulators of the two collectives we need.
# ---------------------------------------------------------------------------


def all_gatherv(
    per_rank_tensors: List[torch.Tensor], dim: int = 0
) -> torch.Tensor:
    """Single-process all_gatherv simulation.

    Each entry of ``per_rank_tensors`` is what one rank contributes (variable
    chunk size along ``dim``). Result is the concatenation along ``dim``,
    visible to every rank. Real ``GroupCoordinator.all_gatherv`` does the
    same end-state via NCCL when chunk sizes differ across ranks.
    """
    # REFERENCE: vllm/distributed/device_communicators/all2all.py:L73-L77
    return torch.cat(per_rank_tensors, dim=dim)


def reduce_scatterv(
    full_tensor: torch.Tensor, sizes: List[int], dim: int = 0
) -> List[torch.Tensor]:
    """Single-process reduce_scatterv simulation.

    Input is the concatenation of P partial-sum chunks. We split along
    ``dim`` according to ``sizes``. Real ``reduce_scatterv`` would first
    sum partial copies then split; here our caller passes the *already
    summed* tensor (each rank computed its expert contribution and we
    add them in the test harness).
    """
    # REFERENCE: vllm/distributed/device_communicators/all2all.py:L130-L135
    return list(torch.split(full_tensor, sizes, dim=dim))


# ---------------------------------------------------------------------------
# AgRsAll2AllManager mirror.
# ---------------------------------------------------------------------------


@dataclass
class AgRsAll2AllManager:
    """Pedagogical mirror of ``AgRsAll2AllManager``.

    We expose ``dispatch_router_logits``, ``dispatch``, and ``combine`` —
    the three methods the FusedMoE forward path uses.
    """

    ep_size: int

    def dispatch(
        self,
        per_rank_hidden_states: List[torch.Tensor],
        per_rank_topk_weights: List[torch.Tensor],
        per_rank_topk_ids: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """All-gather hidden states and topk metadata across the EP group.

        Returns the concatenated tensors as seen by EVERY rank. Mirrors
        ``AgRsAll2AllManager.dispatch`` (all2all.py:L83-L121) which does
        ``dist_group.all_gatherv([hidden_states, topk_weights, topk_ids],
        dim=0, sizes=...)``.
        """
        # REFERENCE: vllm/distributed/device_communicators/all2all.py:L108-L121
        hs = all_gatherv(per_rank_hidden_states, dim=0)
        tw = all_gatherv(per_rank_topk_weights, dim=0)
        ti = all_gatherv(per_rank_topk_ids, dim=0)
        return hs, tw, ti

    def combine(
        self, summed_full: torch.Tensor, sizes: List[int]
    ) -> List[torch.Tensor]:
        """Reduce-scatter expert outputs back to the originating ranks.

        Mirrors ``AgRsAll2AllManager.combine`` (all2all.py:L123-L136). In
        a real run, each EP rank produces its local-expert outputs for
        ALL tokens that route through it; the reduce-scatter sums and
        splits so each rank gets back ONLY the tokens it originally owned.

        For the single-process simulation, ``summed_full`` is the
        already-element-wise-summed tensor of shape
        ``[total_tokens, hidden]``; we split it back into per-rank
        tensors of size ``sizes[r]``.
        """
        # REFERENCE: vllm/distributed/device_communicators/all2all.py:L130-L135
        return reduce_scatterv(summed_full, sizes, dim=0)


# ---------------------------------------------------------------------------
# Alpha-beta cost model.
# ---------------------------------------------------------------------------


def alpha_beta_cost(
    payload_bytes: int, alpha_us: float, beta_GBps: float, p: int, op: str
) -> float:
    """Latency model for collective ops on a ring.

    Args:
        payload_bytes: total bytes per rank that need to traverse the ring.
            For all-reduce the convention is "input size" (each element
            ends up on every rank).
        alpha_us: per-message latency floor (microseconds).
        beta_GBps: per-rank network bandwidth (GB/s).
        p: number of ranks.
        op: ``"all_reduce"``, ``"all_to_all"``, ``"all_gather"`` — picks
            the constant in front.

    Returns:
        Predicted latency in microseconds.

    The standard alpha-beta formulas for a ring:
        T_AR    = 2 * (P-1)/P * (alpha + payload * beta_inv)
        T_A2A   = (P-1)/P     * (alpha + payload * beta_inv)
        T_AG    = (P-1)/P     * (alpha + payload * beta_inv)

    so all-to-all costs roughly *half* of all-reduce — that's Trap B's
    headline number, with the caveat that real all-to-all is
    imbalance-sensitive.
    """
    # REFERENCE: classic Hockney-style alpha-beta model for ring algorithms.
    if p <= 1:
        return 0.0
    beta_us_per_byte = 1e-3 / (beta_GBps)  # GB/s → μs per byte
    one_hop = alpha_us + payload_bytes * beta_us_per_byte
    factor = (p - 1) / p
    if op == "all_reduce":
        return 2.0 * factor * one_hop
    if op == "all_to_all":
        return factor * one_hop
    if op == "all_gather":
        return factor * one_hop
    raise ValueError(f"unknown op {op}")


def measure_dispatch_combine(
    per_rank_hs: List[torch.Tensor],
    per_rank_tw: List[torch.Tensor],
    per_rank_ti: List[torch.Tensor],
    sizes: List[int],
) -> dict:
    """Run dispatch + (toy) combine in-process, capturing tensor sizes.

    Real all-to-all timing is network-bound; in-process this is just
    memcopy. The function returns shapes and dtypes so the writer can quote
    them — we explicitly DO NOT report wall-clock here. (Honest demo
    caveat: see impl-notes §"Demo numerics".)
    """
    p = len(per_rank_hs)
    mgr = AgRsAll2AllManager(ep_size=p)
    hs, tw, ti = mgr.dispatch(per_rank_hs, per_rank_tw, per_rank_ti)
    info = {
        "ep_size": p,
        "hidden_after_dispatch_shape": tuple(hs.shape),
        "topk_weights_after_dispatch_shape": tuple(tw.shape),
        "topk_ids_after_dispatch_shape": tuple(ti.shape),
        "per_rank_input_sizes": sizes,
    }
    # toy "combine": sum all rank contributions then reduce_scatter.
    # In real EP, expert outputs from different ranks get *summed* per
    # token then split. For shape demonstration we just split the gathered
    # tensor back the same way.
    out_chunks = mgr.combine(hs, sizes)
    info["per_rank_output_shapes"] = [tuple(c.shape) for c in out_chunks]
    return info
