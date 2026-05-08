"""LSE-weighted combine — the math behind every CP backend.

This module derives and implements the log-sum-exp (LSE) weighted
combination that takes per-rank partial attention outputs and produces
the global output, identical to running attention on a single GPU.

This is **the same algebra** as FlashAttention's online softmax
(Dao et al. 2022, §2.3) but applied across **ranks** instead of across
**KV tiles**. Whether the partial outputs were exchanged via Ring
P2P, AllGather + ReduceScatter, or All-to-All, the final reduction is
this LSE-weighted weighted average.

Reframe (5th instance after Ch07-Ch10): vLLM ships **no** ``class
RingAttention``. The ``_lse_weighted_combine`` function at
``vllm/v1/attention/ops/dcp_alltoall.py:L40-L100`` IS the algebraic
core; the CP transport (a2a or ag+rs) is a separate concern.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np


class CombineResult(NamedTuple):
    """Combined attention output and its global LSE."""

    output: np.ndarray  # [B, H, D]
    global_lse: np.ndarray  # [B, H]


# REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L39-L103
def lse_weighted_combine(
    partial_outputs: np.ndarray,
    partial_lses: np.ndarray,
    *,
    return_lse: bool = True,
    is_lse_base_on_e: bool = True,
) -> CombineResult:
    """Combine per-rank partial outputs with LSE weights.

    Args:
        partial_outputs: ``[N, B, H, D]`` — N CP ranks, B tokens, H heads,
            D head_dim. ``partial_outputs[i]`` is the output rank ``i``
            computed by attending its local Q against its local K, V.
        partial_lses: ``[N, B, H]`` — log-sum-exp of attention weights
            on rank ``i`` for each (token, head) pair.
        return_lse: also return the global LSE.
        is_lse_base_on_e: ``True`` means ``lse = log(sum_j exp(s_j))``;
            ``False`` means base 2.

    Returns:
        ``CombineResult(output[B,H,D], global_lse[B,H])``.

    Math derivation:

    Each rank ``i`` produces a partial softmax over its KV chunk::

        p_i,j = exp(s_i,j) / Z_i        with  Z_i = sum_j exp(s_i,j)
        O_i   = sum_j p_i,j * v_i,j     and  lse_i = log(Z_i)

    The global softmax over **all** KV positions (across all ranks) is::

        p_j  = exp(s_j) / Z              with  Z = sum_i Z_i
        O    = sum_j p_j * v_j

    Substituting ``Z_i = exp(lse_i)`` and ``O_i * Z_i = sum_j exp(s_i,j) * v_i,j``::

        O = (sum_i Z_i * O_i) / Z       (each rank's O_i scaled by its Z_i)

    For numerical stability we factor out ``lse_max = max_i(lse_i)``::

        weight_i = exp(lse_i - lse_max)
        O        = sum_i (weight_i * O_i) / sum_i weight_i
        lse      = log(sum_i weight_i) + lse_max

    The result is **bit-equivalent** to running attention on a single GPU.
    The transport is irrelevant — only the LSE algebra matters.
    """
    # REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L62-L94
    # Reproduce the exact CPU reference.
    N, B, H, D = partial_outputs.shape
    assert partial_lses.shape == (N, B, H), (
        f"partial_lses shape {partial_lses.shape} != ({N}, {B}, {H})"
    )

    # Sanitize NaN / +inf in LSEs (source: lines 66-70).
    lses = np.where(np.isnan(partial_lses) | np.isinf(partial_lses), -math.inf, partial_lses)

    # Stability shift: subtract max LSE per (token, head).
    # REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L72-L78
    lse_max = lses.max(axis=0)  # [B, H]
    lse_max = np.where(lse_max == -math.inf, 0.0, lse_max)

    # weight_i = exp(lse_i - lse_max) (or 2^(...) for base-2)
    # REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L81-L84
    diff = lses - lse_max[None, :, :]  # [N, B, H]
    if is_lse_base_on_e:
        weights = np.exp(diff)
    else:
        weights = np.power(2.0, diff)

    # Sanitize NaN weights.
    weights = np.where(np.isnan(weights), 0.0, weights)

    # Normalize.
    # REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L89-L91
    weight_sum = weights.sum(axis=0, keepdims=True)  # [1, B, H]
    weights_norm = weights / np.clip(weight_sum, a_min=1e-10, a_max=None)

    # Weighted combination across ranks.
    # REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L93-L94
    out = (partial_outputs * weights_norm[..., None]).sum(axis=0)  # [B, H, D]

    # Global LSE (matches the source's optional return).
    # REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L96-L101
    weight_sum_no_kd = weight_sum.squeeze(0)  # [B, H]
    if is_lse_base_on_e:
        global_lse = np.log(np.clip(weight_sum_no_kd, a_min=1e-30, a_max=None)) + lse_max
    else:
        global_lse = np.log2(np.clip(weight_sum_no_kd, a_min=1e-30, a_max=None)) + lse_max

    return CombineResult(output=out, global_lse=global_lse)


# REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L196-L313 (_dcp_a2a_unpack_combine_kernel — Triton equivalent)
# REFERENCE: vllm/v1/attention/ops/dcp_alltoall.py:L242-L271 (lse_max + lse_sum stability shift in Triton)
def reference_attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Single-process attention for ground-truth comparison.

    Computes ``softmax(Q @ K^T / sqrt(D)) @ V`` plus the per-(token, head)
    log-sum-exp of the unnormalized scores. Used to verify
    ``lse_weighted_combine`` is bit-equivalent.

    Args:
        q: ``[B, H, D]``
        k: ``[L, H, D]`` — full KV sequence
        v: ``[L, H, D]``
    Returns:
        ``(output [B, H, D], lse [B, H])``
    """
    B, H, D = q.shape
    L = k.shape[0]
    scale = 1.0 / math.sqrt(D)
    # scores: [B, H, L]
    scores = np.einsum("bhd,lhd->bhl", q, k) * scale
    lse = _logsumexp(scores, axis=-1)  # [B, H]
    p = np.exp(scores - lse[..., None])  # [B, H, L]
    out = np.einsum("bhl,lhd->bhd", p, v)
    return out, lse


# REFERENCE: vllm/v1/attention/backends/mla/flashattn_mla.py:L196-L250 (dcp_tot_seq_lens_device through MLA forward)
# REFERENCE: vllm/v1/attention/backends/mla/flashmla.py:L160-L200 (dcp_tot_seq_lens_device parameter in flashmla forward)
def split_attention(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    *,
    num_ranks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-rank partial outputs and LSEs.

    Splits the KV sequence axis into ``num_ranks`` contiguous chunks, runs
    attention on each chunk independently, and stacks results.

    Args:
        q: ``[B, H, D]`` — Q is replicated to every rank
        k: ``[L, H, D]`` — KV is sharded
        v: ``[L, H, D]``
        num_ranks: CP world size

    Returns:
        ``(partial_outputs [N, B, H, D], partial_lses [N, B, H])``
    """
    L = k.shape[0]
    assert L % num_ranks == 0, (
        f"L={L} must be divisible by num_ranks={num_ranks}"
    )
    chunk = L // num_ranks

    parts_o, parts_lse = [], []
    for i in range(num_ranks):
        k_i = k[i * chunk : (i + 1) * chunk]
        v_i = v[i * chunk : (i + 1) * chunk]
        o_i, lse_i = reference_attention(q, k_i, v_i)
        parts_o.append(o_i)
        parts_lse.append(lse_i)
    return np.stack(parts_o, axis=0), np.stack(parts_lse, axis=0)


def _logsumexp(x: np.ndarray, axis: int) -> np.ndarray:
    """Numerically stable log-sum-exp."""
    x_max = x.max(axis=axis, keepdims=True)
    return (np.log(np.exp(x - x_max).sum(axis=axis, keepdims=True)) + x_max).squeeze(axis)
