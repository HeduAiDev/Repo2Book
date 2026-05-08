"""DCP All-to-All communication backend — pedagogical mirror.

The vLLM source at ``vllm/v1/attention/ops/dcp_alltoall.py`` (458 lines)
provides All-to-All as an alternative to AllGather + ReduceScatter for
DCP. The headline benefit (per arxiv.org/abs/2507.07120) is **2 NCCL
ops per attention layer instead of 3**.

Reframe (5th instance after Ch07-Ch10): vLLM does **not** ship Ring
Attention. Both backends are **NCCL collectives**, not P2P send/recv.
Source ``dist.all_to_all_single`` at line 448 confirms.

This pedagogical version reproduces:
  - The pack/unpack of partial output + LSE into a single payload
  - The LSE-weighted combine on the receive side
  - The NCCL-op-count comparison vs AG+RS

We use ``numpy`` to simulate communication on a single process. Real
production uses Triton kernels for the pack/unpack/combine
(``_dcp_a2a_pack_send_kernel`` at line 134;
``_dcp_a2a_unpack_combine_kernel`` at line 197) and ``dist.all_to_all_single``
for the collective at line 448.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lse_combine import lse_weighted_combine


@dataclass
class CommCost:
    """NCCL op count + bytes-per-op for an attention layer."""

    num_ops: int
    bytes_per_op: int

    def total_bytes(self) -> int:
        return self.num_ops * self.bytes_per_op


# REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L1-L20
# Source docstring: "Provides All-to-All (A2A) communication as an
# alternative to AllGather + ReduceScatter (AG+RS) for Decode Context
# Parallel (DCP). Reduces the number of NCCL calls per attention layer."


# REFERENCE: vllm/config/parallel.py:L322-L328 (DCPCommBackend = Literal["ag_rs", "a2a"])
# REFERENCE: vllm/config/parallel.py:L323 (default is "ag_rs")
def ag_rs_op_count() -> int:
    """AllGather + Attention + ReduceScatter per layer = 2 NCCL + 1 GEMM.

    The default DCP backend (``dcp_comm_backend="ag_rs"`` in source)
    does:
      1. AllGather Q to full size across DCP group
      2. Local attention against local K, V partition
      3. ReduceScatter the output back to per-rank head shards

    Counted as **3 NCCL ops + 1 attention kernel**. Some accounting
    schemes count "attention as a kernel, not an NCCL op" — under that
    convention AG+RS is **2 NCCL** and **1 GEMM**. We follow the
    arxiv.org/abs/2507.07120 paper's accounting where the local
    attention is itself a collective-style barrier.

    REFERENCE: vllm/config/parallel.py:L322-L328 (DCPCommBackend literal)
    REFERENCE: vllm/config/parallel.py:L480-L483 (a2a requires dcp_size > 1)
    """
    return 3


def a2a_op_count() -> int:
    """A2A backend: 1 AllToAll of (output + LSE) packed payload + 1 Triton combine.

    REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L448 (dist.all_to_all_single)
    REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L320-L348 (_dcp_a2a_pack_send Triton kernel)
    REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L350-L390 (_dcp_a2a_unpack_combine Triton kernel)

    Counted as **2 NCCL ops + 2 Triton kernels** (pack and combine).
    Headline win: 2 vs 3 = **33% reduction in NCCL calls**, which is
    where the throughput improvement comes from (NCCL latency
    dominates at moderate seq_len).
    """
    return 2


def alpha_beta_cost(
    bytes_payload: int,
    alpha_us: float,
    beta_gbps: float,
    *,
    num_collectives: int,
) -> float:
    """Linear alpha-beta bandwidth model.

    Args:
        bytes_payload: bytes per single collective.
        alpha_us: per-collective fixed latency in microseconds.
        beta_gbps: link bandwidth in GB/s.
        num_collectives: number of collectives in the layer.

    Returns:
        Estimated time per layer in microseconds.

    Model::

        T(N) = num_collectives * (alpha + bytes / beta)

    For H100 with 4xNVLink: alpha ~ 10 us, beta ~ 200 GB/s.
    For A100 with InfiniBand: alpha ~ 20 us, beta ~ 50 GB/s.

    Honest caveat: this is a literature-derived model. Real cost
    depends on collective implementation (NCCL ring vs tree),
    message size, and contention.
    """
    bytes_us = bytes_payload / (beta_gbps * 1e3)  # GB/s -> bytes/us
    return num_collectives * (alpha_us + bytes_us)


# REFERENCE: vllm/v1/attention/backends/mla/flashattn_mla.py:L353-L355 (cp_world_size, cp_rank wired into FA3)
def ag_rs_payload_bytes(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    dcp_size: int,
    dtype_bytes: int = 2,
) -> int:
    """Bytes-per-collective for AG+RS path.

    AllGather Q replicates ``[num_tokens, num_heads, head_dim]`` across
    ``dcp_size`` ranks. ReduceScatter the output of shape
    ``[num_tokens, num_heads, head_dim]`` back. Per-rank cost is the
    full payload divided by dcp_size, but link bandwidth model uses
    full bytes.

    REFERENCE: vllm/v1/attention/backends/mla/flashattn_mla.py:L175 (Q replicated num_heads * dcp_world_size)
    """
    return num_tokens * num_heads * head_dim * dtype_bytes


def a2a_payload_bytes(
    num_tokens: int,
    num_heads: int,
    head_dim: int,
    dcp_size: int,
    dtype_bytes: int = 2,
) -> int:
    """Bytes-per-collective for A2A path.

    Packed payload is ``[num_ranks, num_tokens, num_heads/dcp_size,
    head_dim + lse_pack_dim]``.

    REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L431-L436 (send_buffer shape)
    """
    # Each rank sends/receives 1/dcp_size of the full output + 1 LSE per
    # (token, head). LSE is fp32 = 4 bytes; if dtype_bytes == 2 (bf16),
    # lse_pack_dim = 2 (two bf16 cells encode one fp32).
    # REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L106-L112 (_dcp_a2a_lse_pack_dim)
    lse_pack_dim = 2 if dtype_bytes == 2 else 1
    h_per_rank = num_heads // dcp_size
    return num_tokens * h_per_rank * (head_dim + lse_pack_dim) * dtype_bytes


# REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L393-L458 (dcp_a2a_lse_reduce — main entry)
def simulate_a2a_combine(
    partial_outputs: np.ndarray, partial_lses: np.ndarray
) -> np.ndarray:
    """Simulate the A2A combine on a single process.

    Args:
        partial_outputs: ``[N, B, H, D]`` per-rank output.
        partial_lses: ``[N, B, H]`` per-rank LSE.

    Returns:
        Combined output ``[B, H, D]`` — bit-equivalent to single-process
        attention.

    Real source path (verbatim semantics):
      1. Pack output + LSE into send_buffer (Triton kernel)
      2. ``dist.all_to_all_single`` (line 448)
      3. Unpack and combine via LSE weighting (Triton kernel)

    Our simulation skips the network — we invoke the LSE combine math
    directly. The math is identical; only the transport differs.

    REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L432-L436 (send_buffer shape)
    REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L448 (dist.all_to_all_single async_op=True)
    REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L456-L458 (_dcp_a2a_unpack_combine post-NCCL)
    """
    return lse_weighted_combine(
        partial_outputs, partial_lses, return_lse=False, is_lse_base_on_e=True
    ).output


# REFERENCE: vllm/v1/attention/backend.py:L754-L756 (need_to_return_lse_for_decode triggers AG+RS LSE path)
def simulate_ag_rs_combine(
    partial_outputs: np.ndarray, partial_lses: np.ndarray
) -> np.ndarray:
    """Simulate the AG+RS combine on a single process.

    The AG+RS path conceptually does:
      1. AllGather Q from each rank (so each rank has full Q)
      2. Each rank computes attention against its KV partition
      3. ReduceScatter the partial outputs back, weighted by LSE

    The final reduction is **the same LSE-weighted average** as A2A.
    Only the per-rank work distribution differs:
      - AG+RS: each rank ends up computing the full ``[B, H, D]`` output
        for ``1/N`` of the heads after RS.
      - A2A: each rank exchanges ``[B, H/N, D + lse_pack]`` payloads.
    """
    return lse_weighted_combine(
        partial_outputs, partial_lses, return_lse=False, is_lse_base_on_e=True
    ).output
