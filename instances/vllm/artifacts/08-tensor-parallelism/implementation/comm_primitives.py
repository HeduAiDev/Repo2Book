"""comm_primitives — α-β model for ring all-reduce, simulated locally.

vLLM's `tensor_model_parallel_all_reduce` is a one-line wrapper:

    REFERENCE: instances/vllm/source/vllm/distributed/communication_op.py:L12-L14
        def tensor_model_parallel_all_reduce(input_):
            return get_tp_group().all_reduce(input_)

Underneath, `GroupCoordinator.all_reduce` (parallel_state.py:L502-L530)
dispatches to the device communicator (NCCL on CUDA), which selects ring,
tree, or double-binary-tree based on payload size.

This module:
  1. Implements the α-β cost formula for ring all-reduce.
  2. Simulates ring all-reduce step-by-step (each rank's per-step send/recv
     is materialised as a numpy operation) so the reader sees the
     2*(P-1) communication steps and the per-step payload S/P.
  3. Provides `fit_alpha_beta` to recover (α, β) from measured timings.

We do NOT call torch.distributed; this is a single-process pedagogical
simulation. The number it predicts (T = 2(P-1)/P (α + S/P β)) matches the
classical Bandwidth-Optimal Ring all-reduce result and is what vLLM's
NCCL path achieves at large payloads.

References:
- parallel_state.py:L502-L530  GroupCoordinator.all_reduce
- communication_op.py:L12-L14  tensor_model_parallel_all_reduce
- device_communicators/cuda_communicator.py  (NCCL backend)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


# REFERENCE: instances/vllm/source/vllm/distributed/communication_op.py:L12-L14
# tensor_model_parallel_all_reduce dispatches to GroupCoordinator.all_reduce,
# which dispatches to device_communicator.all_reduce (NCCL on CUDA). The α-β
# model below predicts the time of that NCCL call.


@dataclass
class AlphaBetaModel:
    """Linear cost model for a collective operation.

        T(S) = α + β * S

    α has units of seconds per step (latency). β has units of seconds per byte
    (inverse bandwidth, so 1/β in bytes/sec is the achievable bandwidth).

    The classical "α-β" formulation; see Hockney 1994.
    """

    alpha_seconds: float
    beta_seconds_per_byte: float

    def predict(self, payload_bytes: float) -> float:
        return self.alpha_seconds + self.beta_seconds_per_byte * payload_bytes

    @property
    def bandwidth_GBps(self) -> float:
        if self.beta_seconds_per_byte <= 0:
            return float("inf")
        return 1.0 / self.beta_seconds_per_byte / 1e9


# REFERENCE: instances/vllm/source/vllm/distributed/parallel_state.py:L502-L530
def ring_all_reduce_cost(
    payload_bytes: float, num_ranks: int, ab: AlphaBetaModel
) -> float:
    """Predict ring all-reduce time using the bandwidth-optimal formula.

        T_ring = 2 * (P - 1) / P * (α + (S / P) * β)

    The factor 2(P-1) is the count of communication steps: (P-1) for the
    reduce-scatter half (each rank accumulates one chunk), plus (P-1) for the
    all-gather half (each rank propagates its accumulated chunk). Each step
    moves a chunk of size S/P bytes.

    This is what NCCL's ring path approaches asymptotically; vLLM hits this
    when the payload is large enough that the bandwidth term dominates.
    """
    P = num_ranks
    if P < 2:
        return 0.0
    chunk_bytes = payload_bytes / P
    return 2.0 * (P - 1) / P * (ab.alpha_seconds + chunk_bytes * ab.beta_seconds_per_byte)


# REFERENCE: instances/vllm/source/vllm/distributed/parallel_state.py:L502-L588
def simulate_all_reduce(
    per_rank_tensors: Sequence[np.ndarray],
) -> list[np.ndarray]:
    """Step-by-step ring all-reduce simulation.

    Input  : per_rank_tensors[r] = the local partial that rank r holds.
             All shapes must be identical (same as in real all-reduce).
    Output : per_rank_tensors_after[r] = the SAME tensor (sum over all r) on
             every rank.

    Algorithm (Bandwidth-Optimal Ring):
        Split each rank's tensor into P chunks along axis 0.
        Reduce-scatter half: P-1 steps. At step k, rank r SENDS chunk
            (r-k) mod P  to rank (r+1) mod P, RECEIVES chunk (r-k-1) mod P
            from rank (r-1) mod P, and ACCUMULATES it.
        After P-1 steps, rank r holds the fully-reduced version of chunk
            (r+1) mod P (by convention).
        All-gather half: P-1 steps. At step k, rank r SENDS its
            currently-final chunk forward, RECEIVES the chunk from behind,
            replacing its own copy.
        After P-1 more steps, every rank has all P fully-reduced chunks.

    We do this in a single process by mutating a list of arrays. The exact
    bookkeeping is what matters — this code mirrors the algorithm NCCL runs
    when vLLM calls `device_communicator.all_reduce(input_)`.
    """
    P = len(per_rank_tensors)
    if P == 1:
        # REFERENCE: parallel_state.py:L518-L519 — bypass when world_size==1
        return [per_rank_tensors[0].copy()]
    base_shape = per_rank_tensors[0].shape
    for t in per_rank_tensors:
        assert t.shape == base_shape, f"shape mismatch: {t.shape} vs {base_shape}"
    assert base_shape[0] % P == 0, (
        f"axis-0 size {base_shape[0]} must be divisible by P={P} for ring chunking"
    )
    # Build per-rank chunks: shape [P, chunk0, *rest]
    chunked = [np.split(t.copy(), P, axis=0) for t in per_rank_tensors]
    # Reduce-scatter: P-1 steps.
    for step in range(P - 1):
        new_chunked = [list(rank_chunks) for rank_chunks in chunked]
        for r in range(P):
            send_idx = (r - step) % P
            send_to = (r + 1) % P
            new_chunked[send_to][send_idx] = (
                new_chunked[send_to][send_idx] + chunked[r][send_idx]
            )
        chunked = new_chunked
    # After reduce-scatter, rank r owns the fully-reduced chunk at index (r+1) mod P.
    # All-gather: P-1 steps to broadcast each rank's owned chunk to everyone.
    for step in range(P - 1):
        new_chunked = [list(rank_chunks) for rank_chunks in chunked]
        for r in range(P):
            owned_idx = (r + 1 - step) % P
            send_to = (r + 1) % P
            new_chunked[send_to][owned_idx] = chunked[r][owned_idx].copy()
        chunked = new_chunked
    # Stitch chunks back into full tensors.
    return [np.concatenate(rank_chunks, axis=0) for rank_chunks in chunked]


# REFERENCE: instances/vllm/source/vllm/distributed/parallel_state.py:L585-L588
# `_reduce_scatter_out_place` similarly dispatches the collective; together with
# `_all_gather_out_place` (parallel_state.py:L547-L550) these compose the
# bandwidth-optimal ring all-reduce we simulate above (reduce-scatter + all-gather).
def fit_alpha_beta(
    payloads_bytes: Sequence[float], measured_seconds: Sequence[float]
) -> AlphaBetaModel:
    """Least-squares fit of T(S) = α + β·S given matched samples.

    Returns the AlphaBetaModel that explains the measurements. Used in
    Demo §2 to recover (α, β) from a microbench, then plugged into
    ring_all_reduce_cost for predictive comparison."""
    S = np.asarray(payloads_bytes, dtype=np.float64)
    T = np.asarray(measured_seconds, dtype=np.float64)
    assert S.shape == T.shape and S.ndim == 1
    # Design matrix [1, S], solve [α; β].
    A = np.stack([np.ones_like(S), S], axis=1)
    coeffs, *_ = np.linalg.lstsq(A, T, rcond=None)
    alpha, beta = float(coeffs[0]), float(coeffs[1])
    # Numerical floor: a least-squares fit may dip negative on pathological data;
    # a real link cannot have α < 0. Clamp to a representative tiny positive.
    alpha = max(alpha, 1e-12)
    beta = max(beta, 1e-15)
    return AlphaBetaModel(alpha_seconds=alpha, beta_seconds_per_byte=beta)


# Canonical interconnect calibrations (for plug-and-predict in §8.5 narrative).
# Numbers are realistic order-of-magnitudes the reader can compare against
# their own hardware. They are NOT measured here — they are the calibration
# the writer quotes when explaining "TP=2 doesn't give 2x" (Trap A).
HARDWARE_PROFILES = {
    # Tag                  : AlphaBetaModel(α seconds, β seconds/byte)
    "NVLink_HSXM4":         AlphaBetaModel(alpha_seconds=2.0e-6,  beta_seconds_per_byte=1.0/(300e9)),
    "NVLink_PCIe_DGX":      AlphaBetaModel(alpha_seconds=5.0e-6,  beta_seconds_per_byte=1.0/(150e9)),
    "PCIe_Gen4_x16":        AlphaBetaModel(alpha_seconds=10.0e-6, beta_seconds_per_byte=1.0/(32e9)),
    "InfiniBand_HDR_200":   AlphaBetaModel(alpha_seconds=2.0e-6,  beta_seconds_per_byte=1.0/(25e9)),
    "Ethernet_25G":         AlphaBetaModel(alpha_seconds=20.0e-6, beta_seconds_per_byte=1.0/(3.125e9)),
}


def predict_block_overhead(
    hidden: int, ffn: int, batch_seqs: int, dtype_bytes: int,
    tp_size: int, hardware: str = "NVLink_HSXM4",
) -> dict:
    """Predict the all-reduce overhead per Llama transformer block (1 attn + 1 mlp).

    Two all-reduces per block: one after o_proj (RowParallelLinear with
    reduce_results=True, linear.py:L1562-L1563), one after down_proj. Each
    reduces a [batch_seqs, hidden] tensor.
    """
    payload_bytes = batch_seqs * hidden * dtype_bytes
    ab = HARDWARE_PROFILES[hardware]
    cost_one = ring_all_reduce_cost(payload_bytes, num_ranks=tp_size, ab=ab)
    return {
        "hardware": hardware,
        "tp_size": tp_size,
        "payload_bytes_per_allreduce": payload_bytes,
        "predicted_seconds_per_allreduce": cost_one,
        "predicted_seconds_per_block": 2 * cost_one,  # attn-output + MLP-output
        "alpha_us": ab.alpha_seconds * 1e6,
        "beta_GBps": ab.bandwidth_GBps,
    }


if __name__ == "__main__":
    print("=" * 60)
    print("comm_primitives.py — ring all-reduce α-β model")
    print("=" * 60)

    # Equivalence proof: simulate_all_reduce really sums.
    rng = np.random.default_rng(0)
    P = 4
    base = rng.standard_normal((8, 4)).astype(np.float32)
    per_rank = [base + r for r in range(P)]
    target = sum(per_rank)
    out = simulate_all_reduce(per_rank)
    diff = max(np.max(np.abs(o - target)) for o in out)
    print(f"simulate_all_reduce: P={P}, max_abs_diff_to_naive_sum = {diff:.3e}")

    # Fit α-β from synthetic measurements (a known model adds white noise).
    true_ab = AlphaBetaModel(alpha_seconds=5e-6, beta_seconds_per_byte=1/(150e9))
    payloads = [1024, 16 * 1024, 256 * 1024, 4 * 1024 * 1024, 64 * 1024 * 1024]
    rng2 = np.random.default_rng(2)
    measured = [true_ab.predict(s) * (1 + rng2.normal(0, 0.02)) for s in payloads]
    fit = fit_alpha_beta(payloads, measured)
    print(f"fit_alpha_beta: α={fit.alpha_seconds*1e6:.2f}μs  β=1/{fit.bandwidth_GBps:.1f} GB/s")
    print(f"true:           α={true_ab.alpha_seconds*1e6:.2f}μs  β=1/{true_ab.bandwidth_GBps:.1f} GB/s")

    # Block-level prediction (Llama-7B-ish: hidden=4096, ffn=11008).
    for hw in ("NVLink_HSXM4", "PCIe_Gen4_x16"):
        for tp in (2, 4, 8):
            r = predict_block_overhead(
                hidden=4096, ffn=11008, batch_seqs=512, dtype_bytes=2,
                tp_size=tp, hardware=hw,
            )
            print(f"  {hw:18s} tp={tp}  per-block AR: "
                  f"{r['predicted_seconds_per_block']*1e3:.3f} ms  "
                  f"(α={r['alpha_us']:.1f}μs, β={r['beta_GBps']:.0f}GB/s)")
