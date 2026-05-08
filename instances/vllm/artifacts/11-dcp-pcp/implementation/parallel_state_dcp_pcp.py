"""Pedagogical mirror of vLLM's _DCP / _PCP GroupCoordinator singletons.

This module reproduces the *structure* of the DCP / PCP machinery in
``vllm/distributed/parallel_state.py`` without the NCCL plumbing.

Why singletons? Every attention backend in ``vllm/v1/attention/backends/``
calls ``get_dcp_group()`` or ``get_pcp_group()`` from its ``__new__`` method
to discover its rank in the CP groups (see
``vllm/v1/attention/backend.py:L725-L753``). A singleton is the simplest
way to expose "the one and only DCP group for this process" without
threading the group through every constructor argument.

Reframe (5th instance after Ch07/Ch08/Ch09/Ch10):
  vLLM ships NO ``class RingAttention`` / ``class StripedAttention`` /
  ``class ContextParallel`` / ``class DecodeContextParallel`` /
  ``class PrefillContextParallel``. The DCP/PCP feature is a pair of
  module-level GroupCoordinator singletons + per-backend ``__new__``
  discovery. Verified: ``grep -rE '^class (RingAttention|StripedAttention|
  ContextParallel|DecodeContextParallel|PrefillContextParallel)' vllm/``
  returns 0 matches at commit 98661fe.
"""

from __future__ import annotations

from dataclasses import dataclass


# REFERENCE: vllm/distributed/parallel_state.py:L1234-L1243
# Source uses ``_DCP: GroupCoordinator | None = None`` and
# ``get_dcp_group()``. We mirror with a typed dataclass.
@dataclass
class CPGroupCoordinator:
    """Pedagogical stand-in for vLLM's ``GroupCoordinator``.

    The real ``GroupCoordinator`` in
    ``vllm/distributed/parallel_state.py`` wraps a torch.distributed
    ``ProcessGroup`` plus device communicator state. We only need the
    ranks list and per-rank role for the chapter's demos.
    """

    group_name: str
    ranks: list[int]  # global ranks that belong to this group, in order
    rank_in_group: int  # this process's index inside ``ranks``

    @property
    def world_size(self) -> int:
        return len(self.ranks)


# Module-level singletons mirror the source pattern.
# REFERENCE: vllm/distributed/parallel_state.py:L1234 (_DCP)
# REFERENCE: vllm/distributed/parallel_state.py:L1285 (_PCP)
_DCP: CPGroupCoordinator | None = None
_PCP: CPGroupCoordinator | None = None


def get_dcp_group() -> CPGroupCoordinator:
    # REFERENCE: vllm/distributed/parallel_state.py:L1237-L1239
    # Source raises AssertionError; we mirror exactly so attention
    # backend ``__new__`` can ``except AssertionError`` for testing.
    assert _DCP is not None, "decode context model parallel group is not initialized"
    return _DCP


def get_pcp_group() -> CPGroupCoordinator:
    # REFERENCE: vllm/distributed/parallel_state.py:L1288-L1290
    assert _PCP is not None, "prefill context parallel group is not initialized"
    return _PCP


# REFERENCE: vllm/distributed/parallel_state.py:L1242-L1243
# Backward-compat alias retained in source so old callers keep working.
get_context_model_parallel_group = get_dcp_group


def get_decode_context_model_parallel_world_size() -> int:
    # REFERENCE: vllm/distributed/parallel_state.py:L1847-L1849
    return get_dcp_group().world_size


def get_decode_context_model_parallel_rank() -> int:
    # REFERENCE: vllm/distributed/parallel_state.py:L1852-L1854
    return get_dcp_group().rank_in_group


def reset_cp_singletons() -> None:
    """Test helper — no source equivalent. Reset _DCP/_PCP between tests."""
    global _DCP, _PCP
    _DCP = None
    _PCP = None


def initialize_model_parallel(
    rank: int,
    world_size: int,
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    prefill_context_model_parallel_size: int = 1,
    decode_context_model_parallel_size: int = 1,
    data_parallel_size: int = 1,
) -> dict[str, list[list[int]]]:
    """Build all parallel groups by reshaping ``all_ranks``.

    Mirrors ``vllm/distributed/parallel_state.py:L1494-L1633``. Returns
    a dict mapping group-name to the list of group-rank-lists, plus
    populates the ``_DCP`` and ``_PCP`` singletons for ``rank``.

    The 5D mesh layout:

    .. code-block::

        all_ranks.shape = (-1,                                  # external_dp
                           data_parallel_size,                  # in-model dp
                           pipeline_model_parallel_size,        # pp
                           prefill_context_model_parallel_size, # pcp
                           tensor_model_parallel_size)          # tp

    World-size product (note: NOT ``x dcp_size``):

    .. code-block::

        world_size == external_dp x dp x pp x pcp x tp

    DCP folds *inside* the TP axis: each TP-group of ``tp_size`` GPUs
    splits into ``tp_size // dcp_size`` DCP sub-groups.

    Args:
        rank: this process's global rank.
        world_size: total number of ranks.
        tensor_model_parallel_size: TP shard count.
        pipeline_model_parallel_size: PP stage count.
        prefill_context_model_parallel_size: PCP shard count
            (independent axis, expands world_size).
        decode_context_model_parallel_size: DCP shard count
            (folded inside TP, does NOT expand world_size).
        data_parallel_size: in-model DP count.

    Returns:
        Dict with keys 'tp', 'dcp', 'pcp', 'pp', 'dp' mapping to the
        list of group rank lists.
    """
    global _DCP, _PCP

    # REFERENCE: vllm/config/parallel.py:L474-L478
    # Hard constraint: tp must be divisible by dcp.
    if tensor_model_parallel_size % decode_context_model_parallel_size != 0:
        raise ValueError(
            f"tp_size={tensor_model_parallel_size} must be divisible by "
            f"dcp_size={decode_context_model_parallel_size}."
        )

    # REFERENCE: vllm/v1/executor/multiproc_executor.py:L116-L121
    # World-size assertion in source: tp x pp x pcp (no dcp).
    expected = (
        tensor_model_parallel_size
        * pipeline_model_parallel_size
        * prefill_context_model_parallel_size
        * data_parallel_size
    )
    if world_size != expected:
        raise ValueError(
            f"world_size ({world_size}) must equal tp x pp x pcp x dp = "
            f"{tensor_model_parallel_size} x {pipeline_model_parallel_size} x "
            f"{prefill_context_model_parallel_size} x {data_parallel_size} "
            f"= {expected}. Note DCP is NOT in the product (it folds inside TP)."
        )

    # REFERENCE: vllm/distributed/parallel_state.py:L1569-L1575
    # all_ranks = torch.arange(world_size).reshape(
    #     -1, dp, pp, pcp, tp
    # )
    # We use plain Python lists indexed by (ext, d, p, c, t).
    ext_dp = world_size // (
        data_parallel_size
        * pipeline_model_parallel_size
        * prefill_context_model_parallel_size
        * tensor_model_parallel_size
    )

    def at(ext: int, d: int, p: int, c: int, t: int) -> int:
        # row-major flatten of the 5D mesh
        return (
            ((ext * data_parallel_size + d) * pipeline_model_parallel_size + p)
            * prefill_context_model_parallel_size
            + c
        ) * tensor_model_parallel_size + t

    # REFERENCE: vllm/distributed/parallel_state.py:L1577-L1592 (TP groups)
    # group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
    tp_groups: list[list[int]] = []
    for ext in range(ext_dp):
        for d in range(data_parallel_size):
            for p in range(pipeline_model_parallel_size):
                for c in range(prefill_context_model_parallel_size):
                    grp = [
                        at(ext, d, p, c, t)
                        for t in range(tensor_model_parallel_size)
                    ]
                    tp_groups.append(grp)

    # REFERENCE: vllm/distributed/parallel_state.py:L1594-L1614 (DCP groups)
    # group_ranks = all_ranks.reshape(-1, dcp_size).unbind(0)
    # DCP folds inside TP: each TP-group of tp_size GPUs splits into
    # tp_size // dcp_size DCP sub-groups. We rebuild this by chunking
    # each TP group into contiguous dcp_size pieces.
    dcp_groups: list[list[int]] = []
    for tp_grp in tp_groups:
        for chunk_start in range(0, tensor_model_parallel_size, decode_context_model_parallel_size):
            dcp_groups.append(
                tp_grp[chunk_start : chunk_start + decode_context_model_parallel_size]
            )

    # REFERENCE: vllm/distributed/parallel_state.py:L1616-L1633 (PCP groups)
    # group_ranks = all_ranks.transpose(3, 4).reshape(-1, pcp_size).unbind(0)
    # transpose(3, 4) swaps the pcp and tp axes, so each PCP group
    # connects ranks across the pcp axis at fixed (ext, d, p, t).
    pcp_groups: list[list[int]] = []
    for ext in range(ext_dp):
        for d in range(data_parallel_size):
            for p in range(pipeline_model_parallel_size):
                for t in range(tensor_model_parallel_size):
                    grp = [
                        at(ext, d, p, c, t)
                        for c in range(prefill_context_model_parallel_size)
                    ]
                    pcp_groups.append(grp)

    # REFERENCE: vllm/distributed/parallel_state.py:L1635-L1651 (PP groups)
    # transpose(2, 4) — PP groups span the pp axis at fixed (ext, d, c, t).
    pp_groups: list[list[int]] = []
    for ext in range(ext_dp):
        for d in range(data_parallel_size):
            for c in range(prefill_context_model_parallel_size):
                for t in range(tensor_model_parallel_size):
                    grp = [
                        at(ext, d, p, c, t)
                        for p in range(pipeline_model_parallel_size)
                    ]
                    pp_groups.append(grp)

    # REFERENCE: vllm/distributed/parallel_state.py:L1653-L1668 (DP groups)
    # transpose(1, 4) — DP groups span the dp axis at fixed (ext, p, c, t).
    dp_groups: list[list[int]] = []
    for ext in range(ext_dp):
        for p in range(pipeline_model_parallel_size):
            for c in range(prefill_context_model_parallel_size):
                for t in range(tensor_model_parallel_size):
                    grp = [
                        at(ext, d, p, c, t)
                        for d in range(data_parallel_size)
                    ]
                    dp_groups.append(grp)

    # Populate the singletons for this rank.
    for grp in dcp_groups:
        if rank in grp:
            _DCP = CPGroupCoordinator(
                group_name="dcp",
                ranks=list(grp),
                rank_in_group=grp.index(rank),
            )
            break
    for grp in pcp_groups:
        if rank in grp:
            _PCP = CPGroupCoordinator(
                group_name="pcp",
                ranks=list(grp),
                rank_in_group=grp.index(rank),
            )
            break

    return {
        "tp": tp_groups,
        "dcp": dcp_groups,
        "pcp": pcp_groups,
        "pp": pp_groups,
        "dp": dp_groups,
    }


# REFERENCE: vllm/distributed/parallel_state.py:L1670-L1696 (EP group built via transpose 1,2)
# REFERENCE: vllm/distributed/parallel_state.py:L1698-L1719 (EPLB group same ranks as EP)
# REFERENCE: vllm/distributed/parallel_state.py:L1723-L1735 (per-rank logger summary listing PCP rank)
# REFERENCE: vllm/distributed/parallel_state.py:L1741-L1782 (ensure_model_parallel_initialized PCP world-size assertion)
# REFERENCE: vllm/distributed/parallel_state.py:L1791-L1797 (prepare_communication_buffer_for_model PCP)
# REFERENCE: vllm/model_executor/layers/fused_moe/runner/moe_runner.py (PCP-EP integration: pcp_size>1 branch)
