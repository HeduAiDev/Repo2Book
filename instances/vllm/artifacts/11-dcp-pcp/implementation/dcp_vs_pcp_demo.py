"""DCP vs PCP separability demonstration.

This module exists primarily for **language trap D** ("DCP and PCP must
match"): they are **separable axes**.

The two axes have entirely different roles:

  - **DCP** (Decode Context Parallel) — folded INSIDE TP. Shards the
    KV cache across GPUs that are already in the same TP group. Does
    NOT expand world_size. Constraint: ``tp_size % dcp_size == 0``.

  - **PCP** (Prefill Context Parallel) — INDEPENDENT axis. Expands
    world_size by ``pcp_size``. Shards the prefill input sequence so
    each rank computes Q, K, V on ``seq_len/pcp`` tokens.

REFERENCE: vllm/distributed/parallel_state.py:L1593-L1614 (DCP folds inside TP)
REFERENCE: vllm/distributed/parallel_state.py:L1616-L1633 (PCP independent axis)
REFERENCE: vllm/v1/executor/multiproc_executor.py:L116-L121 (world_size = tp x pp x pcp, no dcp)
"""

from __future__ import annotations

from dataclasses import dataclass

from .world_topology import MeshConfig


@dataclass(frozen=True)
class CPRoles:
    """Description of each axis's role in the system."""

    dcp_role: str = "Shards KV cache across decode ranks. Folded inside TP."
    pcp_role: str = "Shards prefill input sequence. Independent axis, expands world_size."

    @staticmethod
    def both_match_required() -> bool:
        """Trap D: the answer is **False**.

        The only hard constraint is ``tp % dcp == 0``. PCP and DCP are
        otherwise free to differ. Production configs commonly use
        ``(tp=8, dcp=2, pcp=4)`` — DCP=2 within each TP-group of 8,
        PCP=4 as a separate axis multiplying world_size.

        REFERENCE: vllm/config/parallel.py:L474-L478 (only tp % dcp == 0)
        """
        return False


# REFERENCE: vllm/distributed/parallel_state.py:L1593-L1614 (DCP folds inside TP)
# REFERENCE: vllm/distributed/parallel_state.py:L1616-L1633 (PCP independent axis via transpose 3,4)
def world_size_for(mesh: MeshConfig) -> int:
    """``world_size = ext_dp * dp * pp * pcp * tp`` — DCP excluded.

    REFERENCE: vllm/v1/executor/multiproc_executor.py:L116-L121
    REFERENCE: vllm/config/parallel.py:L765 (* self.prefill_context_parallel_size in world_size product)
    """
    return mesh.world_size


def per_rank_kv_chunk(seq_len: int, mesh: MeshConfig) -> int:
    """KV chunk size per rank under (DCP, PCP).

    Total per-rank KV chunk = ``seq_len / (dcp * pcp)`` because:
      - DCP shards across TP-internal ranks (each rank holds
        ``seq_len / dcp`` of the KV).
      - PCP shards the prefill input across independent ranks (each
        rank handles ``seq_len / pcp`` input tokens).

    Composed: ``seq_len / (dcp * pcp)``.

    REFERENCE: vllm/v1/kv_cache_interface.py:L195-L205
    REFERENCE: vllm/v1/attention/backend.py:L751 (total_cp_world_size = pcp * dcp)
    REFERENCE: vllm/v1/attention/backend.py:L752 (total_cp_rank = pcp_rank * dcp_world + dcp_rank)
    """
    return seq_len // (mesh.dcp * mesh.pcp)


def explain_separability() -> str:
    """Human-readable explanation, used by the demo and the writer."""
    return (
        "DCP and PCP are SEPARABLE axes:\n"
        "  - DCP folds inside TP: tp % dcp == 0 is the only constraint.\n"
        "  - PCP is independent: world_size = tp x pp x pcp x dp (NOT x dcp).\n"
        "Production: (tp=8, dcp=2, pcp=4) is valid — DCP=2 inside each\n"
        "TP-group of 8 GPUs (so 4 DCP sub-groups per TP), PCP=4 as a\n"
        "separate axis multiplying world_size by 4."
    )


def explain_axis_difference() -> dict[str, str]:
    """Per-axis role table — used in §11.4 narrative."""
    return {
        "DCP": (
            "Decode Context Parallel. Shards KV cache across decode "
            "ranks. Folded inside TP. world_size unchanged. Q is "
            "replicated; K, V are sharded; output combined via LSE."
        ),
        "PCP": (
            "Prefill Context Parallel. Shards prefill input sequence "
            "across ranks. Independent axis. world_size *= pcp. Each "
            "rank computes Q, K, V on its sequence shard, then runs "
            "CP attention to combine."
        ),
    }
