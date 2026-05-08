"""5D device mesh: external_dp x dp x pp x pcp x tp.

Pedagogical reproduction of the world-size product and per-rank tag
naming at ``vllm/v1/executor/multiproc_executor.py:L116-L121, L985-L1001``.

Reframe (outline correction): outline subsection §11.5 says "3D
parallelism (CP+TP)". The actual production mesh is **5D** — five
distinct axes. DCP is **folded inside TP** (sub-group of size
``dcp_size`` within each TP group of size ``tp_size``), so DCP does NOT
appear in the world-size product.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MeshConfig:
    """5D device mesh layout for one vLLM deployment."""

    external_dp: int = 1
    """ExternalDP — verl integration; data-parallel groups outside the model.

    Defaults to 1 for non-verl deployments.
    """
    dp: int = 1
    """In-model DP — every rank in the same DP group must call ``generate``
    simultaneously, otherwise NCCL deadlocks.

    REFERENCE: vllm/distributed/parallel_state.py:L1561-L1568
    """
    pp: int = 1
    """Pipeline parallelism — model layers split across stages."""
    pcp: int = 1
    """Prefill Context Parallel — independent axis; expands world_size.

    REFERENCE: vllm/config/parallel.py:L115
    """
    tp: int = 1
    """Tensor parallel — heads / FFN sharding."""
    dcp: int = 1
    """Decode Context Parallel — folded INSIDE TP. tp % dcp == 0 required.

    REFERENCE: vllm/config/parallel.py:L310, L474-L478
    """

    def __post_init__(self) -> None:
        # REFERENCE: vllm/config/parallel.py:L474-L478
        if self.tp % self.dcp != 0:
            raise ValueError(
                f"tp_size={self.tp} must be divisible by dcp_size={self.dcp}."
            )

    @property
    def world_size(self) -> int:
        """World size product. **DCP is excluded** — it folds inside TP.

        REFERENCE: vllm/v1/executor/multiproc_executor.py:L116-L121
        ``world_size == tp_size * pp_size * pcp_size`` (with dp also folded
        when verl-style external DP is not used).
        """
        return self.external_dp * self.dp * self.pp * self.pcp * self.tp

    @property
    def total_cp_world_size(self) -> int:
        """Composed CP world size = pcp * dcp.

        REFERENCE: vllm/v1/attention/backend.py:L751
        ``self.total_cp_world_size = self.pcp_world_size * self.dcp_world_size``
        """
        return self.pcp * self.dcp

    @property
    def num_dcp_subgroups(self) -> int:
        """Number of DCP sub-groups inside each TP group.

        Each TP-group of ``tp_size`` GPUs splits into ``tp // dcp`` DCP
        sub-groups, all at intra-TP latency (NVLink within a node).
        """
        return self.tp // self.dcp


def process_name_for_rank(
    rank: int,
    mesh: MeshConfig,
    *,
    enable_ep: bool = False,
    ep_rank: int | None = None,
) -> str:
    """Reproduce the per-process name tag at multiproc_executor.py:L985-L1004.

    Source builds the name conditionally — only axes with size > 1 are
    appended. We mirror exactly so a reader can grep production logs and
    see the same shape.
    """
    # We compute per-axis ranks from the global rank by treating the
    # 5D mesh as a row-major tensor (external_dp, dp, pp, pcp, tp).
    # DCP rank is the position of ``tp_rank`` inside its DCP sub-group.
    n = mesh.tp
    tp_rank = rank % n
    rank //= n
    n = mesh.pcp
    pcp_rank = rank % n
    rank //= n
    n = mesh.pp
    pp_rank = rank % n
    rank //= n
    n = mesh.dp
    dp_rank = rank % n
    # external_dp_rank = rank // n  # not used in name tag
    dcp_rank = tp_rank % mesh.dcp

    # REFERENCE: vllm/v1/executor/multiproc_executor.py:L991-L1004
    parts = ["Worker"]
    if mesh.dp > 1:
        parts.append(f"DP{dp_rank}")
    if mesh.pp > 1:
        parts.append(f"PP{pp_rank}")
    if mesh.pcp > 1:
        parts.append(f"PCP{pcp_rank}")
    if mesh.tp > 1:
        parts.append(f"TP{tp_rank}")
    if mesh.dcp > 1:
        parts.append(f"DCP{dcp_rank}")
    if enable_ep:
        assert ep_rank is not None
        parts.append(f"EP{ep_rank}")
    return "_".join(parts)


# REFERENCE: vllm/v1/executor/multiproc_executor.py:L258-L259 (_get_parallel_sizes returns tp/pp/pcp)
# REFERENCE: vllm/v1/executor/multiproc_executor.py:L493 (* prefill_context_parallel_size in spawn product)
# REFERENCE: vllm/v1/executor/ray_executor_v2.py:L263-L268 (Ray executor: same world_size = tp*pp*pcp assertion)
def per_rank_kv_fraction(mesh: MeshConfig) -> float:
    """Fraction of total KV cache that each rank stores.

    Long-context KV cache is sharded along the sequence axis by the
    composed CP world size: ``total_cp = pcp * dcp``. So each rank
    stores ``1 / total_cp`` of the KV cache.

    REFERENCE: vllm/v1/kv_cache_interface.py:L195-L205
    ``max_model_len = cdiv(max_model_len, dcp_world_size * pcp_world_size)``
    """
    return 1.0 / (mesh.pcp * mesh.dcp)
