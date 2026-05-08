"""EP / EPLB process-group machinery and FusedMoEParallelConfig math.

Pedagogical mirror of three pieces of vLLM:

1. ``_EP`` / ``_EPLB`` module-level singletons in
   ``vllm/distributed/parallel_state.py`` (L1261-L1283, L1670-L1719).
2. ``FusedMoEParallelConfig`` dataclass + ``.make()`` math in
   ``vllm/model_executor/layers/fused_moe/config.py`` (L999-L1209).
3. The 4D mesh construction:
   ``all_ranks.transpose(1, 2).reshape(-1, dp * pcp * tp).unbind(0)``
   from ``parallel_state.py:L1670-L1696``.

We do NOT import torch.distributed; everything runs in-process with a
toy ``EPGroup`` that knows its rank list. That is enough to demonstrate
the orthogonality invariant (EP is the *complement* of TP×DP×PCP, not
a sub-axis of any one of them).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import torch

# REFERENCE: instances/vllm/source/vllm/distributed/parallel_state.py:L1261-L1283
# REFERENCE: instances/vllm/source/vllm/model_executor/layers/fused_moe/config.py:L998-L1209
# REFERENCE: instances/vllm/source/vllm/distributed/parallel_state.py:L1670-L1719


# ---------------------------------------------------------------------------
# FusedMoEParallelConfig — dataclass mirror of config.py:L999
# ---------------------------------------------------------------------------


@dataclass
class FusedMoEParallelConfig:
    """Per-FusedMoE parallel sizes, mirroring vLLM's dataclass.

    The *invariant* this class encodes is:

        ep_size = (tp_size_outer * dp_size * pcp_size)   if use_ep else 1
        tp_size_inner = 1                                 if use_ep else tp_size_outer

    Where ``tp_size_outer`` is what the operator passed to FusedMoE; under
    EP=True, that whole budget collapses into the EP axis and inside-the-expert
    TP becomes 1 (each rank holds whole experts, not slices). Under EP=False,
    we fall back to plain TP — each rank holds the same E experts but each
    expert's matmul is row/col sharded by ``tp_size_outer``.
    """

    tp_size: int
    pcp_size: int
    dp_size: int
    ep_size: int
    tp_rank: int
    pcp_rank: int
    dp_rank: int
    ep_rank: int
    sp_size: int
    use_ep: bool
    all2all_backend: str = "allgather_reducescatter"
    enable_eplb: bool = False

    @property
    def is_sequence_parallel(self) -> bool:
        return self.sp_size > 1

    @property
    def use_all2all_kernels(self) -> bool:
        # REFERENCE: vllm/model_executor/layers/fused_moe/config.py:L1019-L1020
        return self.dp_size > 1 and self.use_ep

    @staticmethod
    def flatten_tp_across_dp_and_pcp(
        tp_size: int, dp_size: int, dp_rank: int, pcp_size: int, pcp_rank: int
    ) -> tuple[int, int]:
        """Mirror of ``flatten_tp_across_dp_and_pcp`` (config.py:L1071).

        DP and PCP are treated as outer axes that *multiply* the TP budget
        when EP is on — the EP group has size ``tp × dp × pcp``.
        """
        # REFERENCE: vllm/model_executor/layers/fused_moe/config.py:L1071-L1079
        flatten_tp_size = dp_size * pcp_size * tp_size
        flatten_tp_rank = dp_rank * pcp_size * tp_size + pcp_rank * tp_size + 0
        return flatten_tp_size, flatten_tp_rank

    @staticmethod
    def make(
        tp_size_: int,
        pcp_size_: int,
        dp_size_: int,
        sp_size_: int,
        enable_expert_parallel: bool,
        all2all_backend: str = "allgather_reducescatter",
        enable_eplb: bool = False,
        dp_rank: int = 0,
        pcp_rank: int = 0,
    ) -> "FusedMoEParallelConfig":
        """Reproduce the EP-vs-TP collapse rule from config.py:L1082-L1209.

        The crux is at L1192-L1208: ``ep_size = tp_size; tp_size = 1`` when
        EP is on. Without this collapse, EP would be a *sub-axis* of TP and
        each token would need a TP-internal all-reduce *before* the EP
        all-to-all — defeating the purpose.
        """
        # REFERENCE: vllm/model_executor/layers/fused_moe/config.py:L1162-L1165
        use_ep = (dp_size_ * pcp_size_ * tp_size_ > 1) and enable_expert_parallel

        flatten_tp_size, flatten_tp_rank = (
            FusedMoEParallelConfig.flatten_tp_across_dp_and_pcp(
                tp_size_, dp_size_, dp_rank, pcp_size_, pcp_rank
            )
        )

        if not use_ep:
            # REFERENCE: vllm/model_executor/layers/fused_moe/config.py:L1175-L1189
            # No EP: TP stays at the flattened budget; EP is a no-op of size 1.
            return FusedMoEParallelConfig(
                tp_size=flatten_tp_size,
                tp_rank=flatten_tp_rank,
                pcp_size=pcp_size_,
                pcp_rank=pcp_rank,
                dp_size=dp_size_,
                dp_rank=dp_rank,
                ep_size=1,
                ep_rank=0,
                sp_size=sp_size_,
                use_ep=False,
                all2all_backend=all2all_backend,
                enable_eplb=enable_eplb,
            )
        # REFERENCE: vllm/model_executor/layers/fused_moe/config.py:L1192-L1208
        # EP=True: collapse the whole TP×DP×PCP budget into EP. tp_size→1.
        ep_size = flatten_tp_size
        ep_rank = flatten_tp_rank
        return FusedMoEParallelConfig(
            tp_size=1,
            tp_rank=0,
            pcp_size=pcp_size_,
            pcp_rank=pcp_rank,
            dp_size=dp_size_,
            dp_rank=dp_rank,
            ep_size=ep_size,
            ep_rank=ep_rank,
            sp_size=sp_size_,
            use_ep=True,
            all2all_backend=all2all_backend,
            enable_eplb=enable_eplb,
        )


# ---------------------------------------------------------------------------
# EPGroup — toy GroupCoordinator. No real torch.distributed.
# ---------------------------------------------------------------------------


@dataclass
class EPGroup:
    """Toy GroupCoordinator for the EP axis.

    Replaces ``vllm.distributed.parallel_state.GroupCoordinator`` for our
    single-process demos. Every rank has the same ``world_size`` view of the
    rank list; ``rank_in_group`` differs.
    """

    world_size: int
    rank_in_group: int
    group_name: str = "ep"
    rank_list: List[int] = field(default_factory=list)

    @property
    def size(self) -> int:
        return self.world_size

    def __repr__(self) -> str:
        return (
            f"EPGroup(name={self.group_name!r}, world_size={self.world_size}, "
            f"rank_in_group={self.rank_in_group}, ranks={self.rank_list})"
        )


# ---------------------------------------------------------------------------
# Module-level singletons. Deliberately mirror parallel_state.py.
# ---------------------------------------------------------------------------

# REFERENCE: vllm/distributed/parallel_state.py:L1261 _EP module-level singleton
_EP: Optional[EPGroup] = None
# REFERENCE: vllm/distributed/parallel_state.py:L1273 _EPLB module-level singleton
_EPLB: Optional[EPGroup] = None


def init_ep_group(
    world_size: int,
    rank: int,
    tp_size: int,
    dp_size: int,
    pcp_size: int,
    is_moe: bool = True,
    enable_eplb: bool = False,
) -> EPGroup:
    """Construct the EP (and optionally EPLB) groups.

    Mirrors the construction at parallel_state.py:L1670-L1719.

    The mesh ``all_ranks`` is shape ``(pp, pcp, dp, tp)``; for our demos we
    fix ``pp=1`` so the layout is ``(pcp, dp, tp)`` and EP groups are formed
    as the *complement* of TP × DP × PCP — i.e. all ranks that share the
    same (pp, pcp, dp, tp) coordinates EXCEPT TP collapsed.

    For pedagogical clarity we use the simplification that EP groups equal
    the flattened TP×DP×PCP block at fixed PP — which matches vLLM's mesh
    math when there is one DP/PCP replica per node.
    """
    global _EP, _EPLB
    assert _EP is None, "EP group already initialized"

    if not is_moe:
        # REFERENCE: vllm/distributed/parallel_state.py:L1672-L1673
        # _EP is left None for dense models.
        _EP = None
        _EPLB = None
        return None  # type: ignore[return-value]

    # Mesh: world_size == pp * pcp * dp * tp; we assume pp == 1.
    pp = 1
    assert world_size == pp * pcp_size * dp_size * tp_size, (
        f"world_size={world_size} != pp*pcp*dp*tp={pp * pcp_size * dp_size * tp_size}"
    )

    # all_ranks shape: (pp, pcp, dp, tp).
    # REFERENCE: vllm/distributed/parallel_state.py:L1674-L1683
    all_ranks = torch.arange(world_size, dtype=torch.int64).view(
        pp, pcp_size, dp_size, tp_size
    )
    # The EP group ranks for a given (pp,) are obtained by transposing
    # axes (1, 2) — i.e. swapping pcp and dp — then flattening into
    # ``dp * pcp * tp`` chunks. Reproduces the mesh formula in source.
    group_ranks = (
        all_ranks.transpose(1, 2).reshape(-1, dp_size * pcp_size * tp_size).unbind(0)
    )
    group_ranks = [g.tolist() for g in group_ranks]

    # Pick the group containing this rank.
    my_group = next(g for g in group_ranks if rank in g)
    rank_in_group = my_group.index(rank)
    _EP = EPGroup(
        world_size=len(my_group),
        rank_in_group=rank_in_group,
        group_name="ep",
        rank_list=my_group,
    )

    # REFERENCE: vllm/distributed/parallel_state.py:L1698-L1719
    # EPLB is a SEPARATE process group with the SAME rank list. The
    # separation prevents a deadlock: EPLB rebalance comm runs concurrently
    # with MoE forward all-to-all, and sharing a single group would let
    # them block each other.
    if enable_eplb:
        _EPLB = EPGroup(
            world_size=len(my_group),
            rank_in_group=rank_in_group,
            group_name="eplb",
            rank_list=list(my_group),
        )
    else:
        _EPLB = None

    return _EP


def get_ep_group() -> EPGroup:
    """Mirror of ``vllm.distributed.get_ep_group`` (parallel_state.py:L1264)."""
    # REFERENCE: vllm/distributed/parallel_state.py:L1264-L1270
    assert _EP is not None, (
        "expert parallel group is not initialized. "
        "EP group is only created for MoE models with num_experts > 0."
    )
    return _EP


def get_eplb_group() -> EPGroup:
    """Mirror of ``vllm.distributed.get_eplb_group`` (parallel_state.py:L1276)."""
    # REFERENCE: vllm/distributed/parallel_state.py:L1276-L1282
    assert _EPLB is not None, (
        "EPLB group is not initialized. "
        "EPLB group is only created for MoE models when EPLB is enabled."
    )
    return _EPLB


def reset_groups() -> None:
    """Test helper — re-init the module singletons between demos."""
    global _EP, _EPLB
    _EP = None
    _EPLB = None
