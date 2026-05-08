"""Sequence sharding under ``cp_kv_cache_interleave_size``.

Pedagogical mirror of ``vllm/v1/attention/backends/utils.py:L820-L857
get_dcp_local_seq_lens``.

The ``cp_kv_cache_interleave_size`` knob controls how the KV cache
sequence is sharded across CP ranks:

  - ``interleave_size=1`` — token-level striped: token ``i`` goes to
    rank ``i % cp_size``. Perfectly balanced under causal masking.
  - ``interleave_size=K`` — block-level striped: K-token chunks go
    round-robin. Trades load balance for cache-line-friendly access.
  - ``interleave_size=block_size`` — fully blocked striped: one full
    block per rank before moving on. Cache-friendly but most imbalanced.

The reframe: outline §11.4 says "striped vs balanced". Source has
**one knob with a continuous granularity**. Striped (interleave=1) IS
the load-balanced extreme; outline term "balanced" refers to this
limit, not to a separate algorithm.

REFERENCE: vllm/config/parallel.py:L330-L342 (knob declaration)
REFERENCE: vllm/v1/attention/backends/utils.py:L820-L857 (the helper)
"""

from __future__ import annotations

import numpy as np


# REFERENCE: vllm/v1/attention/backends/utils.py:L820-L857
def get_dcp_local_seq_lens(
    seq_lens: np.ndarray,
    dcp_size: int = 1,
    dcp_rank: int | None = None,
    cp_kv_cache_interleave_size: int = 1,
) -> np.ndarray:
    """Per-DCP-rank local sequence lengths under striped sharding.

    Args:
        seq_lens: ``[num_requests]`` global sequence length per request.
        dcp_size: DCP world size.
        dcp_rank: if given, return lengths only for this rank
            (shape ``[num_requests]``). If ``None``, return for all
            ranks (shape ``[num_requests, dcp_size]``).
        cp_kv_cache_interleave_size: striping granularity. Must
            satisfy ``block_size % interleave_size == 0`` (source
            constraint at parallel.py:L341-L342).

    Returns:
        Per-rank local seq_len. The sum across ranks equals the global
        ``seq_lens`` (modulo rounding).

    Math (from source, lines 844-856)::

        base       = seq_lens // interleave // dcp * interleave
        remainder  = seq_lens - base * dcp
        per_rank   = base + clip(remainder - rank * interleave, 0, interleave)

    Why this formula? Each rank receives ``base`` "full" interleave
    blocks, plus 0 or ``interleave_size`` of the remainder depending on
    where the remainder falls.
    """
    seq_lens = seq_lens.astype(np.int32)
    num_requests = seq_lens.shape[0]

    if dcp_rank is None:
        rank_offsets = np.tile(np.arange(dcp_size, dtype=np.int32), (num_requests, 1))
    else:
        rank_offsets = np.full((num_requests, 1), dcp_rank, dtype=np.int32)

    seq_lens_tiled = np.tile(seq_lens.reshape(-1, 1), (1, rank_offsets.shape[1]))

    # REFERENCE: vllm/v1/attention/backends/utils.py:L844-L849
    base = (
        seq_lens_tiled
        // cp_kv_cache_interleave_size
        // dcp_size
        * cp_kv_cache_interleave_size
    )
    remainder = seq_lens_tiled - base * dcp_size

    # REFERENCE: vllm/v1/attention/backends/utils.py:L851-L855
    remainder = np.clip(
        remainder - rank_offsets * cp_kv_cache_interleave_size,
        0,
        cp_kv_cache_interleave_size,
    )
    local = base + remainder
    if dcp_rank is not None:
        return local.squeeze(-1)
    return local


# REFERENCE: vllm/v1/attention/backends/mla/flashattn_mla.py:L125 (supports_dcp_with_varlen=(interleave_size==1))
# REFERENCE: vllm/config/parallel.py:L341-L342 (block_size must be divisible by interleave_size)
# REFERENCE: vllm/config/parallel.py:L315-L321 (dcp_kv_cache_interleave_size DEPRECATED alias)
def causal_attention_work_per_rank(
    seq_len: int,
    cp_size: int,
    cp_kv_cache_interleave_size: int,
) -> list[int]:
    """Count KV-attends per rank under causal masking.

    For each token ``i`` (0-indexed), it attends to KV positions
    ``[0, i]`` (``i+1`` positions). Each rank "owns" the tokens it is
    responsible for under the striped scheme; this returns the total
    work per rank (sum of ``i+1`` for tokens it owns).

    Used in §4 of demo.py to show the load-imbalance gap.
    """
    work = [0] * cp_size
    for i in range(seq_len):
        # token i goes to rank (i // interleave) % cp_size
        owner = (i // cp_kv_cache_interleave_size) % cp_size
        work[owner] += i + 1  # causal: token i attends i+1 KV positions
    return work


# REFERENCE: vllm/v1/attention/backends/utils.py:L827-L829 ("Only consider dcp now, we can extend the case of cp based on this")
def imbalance_ratio(work: list[int]) -> float:
    """Max work / min work — load imbalance metric."""
    mn = min(work)
    if mn == 0:
        return float("inf")
    return max(work) / mn
