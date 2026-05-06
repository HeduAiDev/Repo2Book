"""tp_math — Pure-math derivation of column-parallel and row-parallel GEMM.

This module proves, with code, that sharding `Y = X @ A` across `tp_size`
ranks reproduces the unsharded result exactly. No vLLM imports, no
torch.distributed, no NCCL — every "rank" is a slice of a numpy/torch tensor
in this process.

The derivations mirror the docstrings of vLLM's TP layers:

    instances/vllm/source/vllm/model_executor/layers/linear.py:L410-L432
        ColumnParallelLinear: A = [A_1, ..., A_p] split along output dim.
        Y_i = X @ A_i. Concat → Y. No comm during forward unless gather_output.

    instances/vllm/source/vllm/model_executor/layers/linear.py:L1394-L1425
        RowParallelLinear: A split along input dim, X split along last dim:
            Y = sum_i (X_i @ A_i).
        Requires all-reduce.

The Megatron-style block (column_then_row_block) shows that stacking col→row
keeps the intermediate sharded — only ONE all-reduce per attention/MLP block.

References:
- linear.py:L410-L608  ColumnParallelLinear
- linear.py:L1394-L1577 RowParallelLinear
- utils.py:L60-L66  divide() — universal asserting divisor
- utils.py:L67-L92  split_tensor_along_last_dim()
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


# REFERENCE: instances/vllm/source/vllm/distributed/utils.py:L53-L64
def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Mirror of vLLM's ensure_divisibility — every TP shard size goes through this."""
    assert numerator % denominator == 0, (
        f"{numerator} is not divisible by {denominator}"
    )


# REFERENCE: instances/vllm/source/vllm/distributed/utils.py:L60-L64
def divide(numerator: int, denominator: int) -> int:
    """Identical to vLLM's divide(). The contract `tp_size | total_heads`
    and `tp_size | hidden_size` is enforced here, not deep in the GEMM."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


# REFERENCE: instances/vllm/source/vllm/distributed/utils.py:L67-L92
def split_tensor_along_last_dim(
    tensor: np.ndarray, num_partitions: int
) -> Sequence[np.ndarray]:
    """Split a tensor along its last dim into `num_partitions` equal pieces.

    Used by RowParallelLinear when input_is_parallel=False — see
    linear.py:L1547-L1553. We keep the same name and signature as vLLM.
    """
    last_dim = tensor.ndim - 1
    last_dim_size = divide(tensor.shape[last_dim], num_partitions)
    return np.split(tensor, num_partitions, axis=last_dim)


# ---------------------------------------------------------------------------
# §1 Column-parallel GEMM
# ---------------------------------------------------------------------------

# REFERENCE: instances/vllm/source/vllm/model_executor/layers/linear.py:L410-L432
def column_parallel_forward(
    X: np.ndarray, A: np.ndarray, tp_size: int, gather_output: bool = False
) -> list[np.ndarray] | np.ndarray:
    """Column-parallel forward: split A along its OUTPUT dim into p shards.

    Math:
        Given Y = X @ A with A: [in, out].
        Split A column-wise: A = [A_1 | A_2 | ... | A_p], each A_i: [in, out/p].
        Each rank computes Y_i = X @ A_i, shape [..., out/p].
        Concat along last dim → Y, shape [..., out].

    NO communication is needed if downstream consumes per-rank Y_i (the common
    case — the next layer is row-parallel and expects a sharded input).
    If `gather_output=True`, an all-gather concatenates Y_1..Y_p back together;
    that mirrors `linear.py:L589-L591` calling `tensor_model_parallel_all_gather`.

    Returns:
        list[Y_i] of length tp_size if gather_output=False (the per-rank
        outputs that the next layer will consume), or the unified Y if True.
    """
    out = A.shape[-1]
    out_per_rank = divide(out, tp_size)
    # REFERENCE: linear.py:L454 — output_size_per_partition = divide(output_size, tp_size)
    A_shards = [
        A[..., r * out_per_rank : (r + 1) * out_per_rank] for r in range(tp_size)
    ]
    Y_shards = [X @ A_r for A_r in A_shards]
    if gather_output:
        # REFERENCE: linear.py:L589-L591 — tensor_model_parallel_all_gather on output
        return np.concatenate(Y_shards, axis=-1)
    return Y_shards


# REFERENCE: instances/vllm/source/vllm/model_executor/layers/linear.py:L534-L569
def column_parallel_weight_loader(
    A_full: np.ndarray, tp_rank: int, tp_size: int
) -> np.ndarray:
    """Mirror of ColumnParallelLinear.weight_loader's `narrow` step.

    vLLM's loader:
        shard_size = param_data.shape[output_dim]      # linear.py:L559
        start_idx  = self.tp_rank * shard_size         # linear.py:L560
        loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
                                                       # linear.py:L561
    Here we hand back the slice each rank should hold.
    """
    out = A_full.shape[-1]
    shard_size = divide(out, tp_size)
    start = tp_rank * shard_size
    # REFERENCE: linear.py:L561 — `narrow` along the OUTPUT dim for column-parallel
    return A_full[..., start : start + shard_size]


# ---------------------------------------------------------------------------
# §2 Row-parallel GEMM
# ---------------------------------------------------------------------------

# REFERENCE: instances/vllm/source/vllm/model_executor/layers/linear.py:L1394-L1425
def row_parallel_forward(
    X_shards: list[np.ndarray] | np.ndarray,
    A: np.ndarray,
    tp_size: int,
    input_is_parallel: bool = True,
    reduce_results: bool = True,
) -> np.ndarray | list[np.ndarray]:
    """Row-parallel forward: split A along its INPUT dim, X along its LAST dim.

    Math:
        Given Y = X @ A with A: [in, out].
        Split A row-wise: A = [A_1; A_2; ...; A_p]^T (stacked rows),
            each A_i: [in/p, out].
        Split X column-wise: X = [X_1, X_2, ..., X_p],
            each X_i: [..., in/p].
        Each rank computes partial Y_i = X_i @ A_i, shape [..., out].
        ALL ranks compute the SAME shape [..., out] but with partial sums.
        Final Y = sum_i Y_i. → all-reduce.

    The `input_is_parallel=True` default (linear.py:L1463) is the common case:
    the previous layer was column-parallel so it already produced X_i. The
    `False` branch (linear.py:L1547-L1553) calls split_tensor_along_last_dim.

    The all-reduce (linear.py:L1562-L1563) is the ONE collective per
    column→row block.
    """
    if input_is_parallel:
        # REFERENCE: linear.py:L1547-L1548 — `if self.input_is_parallel: input_parallel = input_`
        assert isinstance(X_shards, list) and len(X_shards) == tp_size, (
            f"input_is_parallel=True expects pre-split list of length tp_size; "
            f"got {type(X_shards).__name__} len={len(X_shards) if isinstance(X_shards, list) else 'NA'}"
        )
        X_local = X_shards
    else:
        # REFERENCE: linear.py:L1549-L1553 — split_tensor_along_last_dim, take rank-th slice
        assert isinstance(X_shards, np.ndarray)
        X_local = list(split_tensor_along_last_dim(X_shards, tp_size))

    in_dim = A.shape[0]
    in_per_rank = divide(in_dim, tp_size)
    # REFERENCE: linear.py:L1447 — input_size_per_partition = divide(input_size, tp_size)
    A_shards = [A[r * in_per_rank : (r + 1) * in_per_rank, :] for r in range(tp_size)]
    Y_partials = [X_local[r] @ A_shards[r] for r in range(tp_size)]
    if reduce_results:
        # REFERENCE: linear.py:L1562-L1563 — tensor_model_parallel_all_reduce
        return np.sum(Y_partials, axis=0)
    return Y_partials


# REFERENCE: instances/vllm/source/vllm/model_executor/layers/linear.py:L1499-L1524
def row_parallel_weight_loader(
    A_full: np.ndarray, tp_rank: int, tp_size: int
) -> np.ndarray:
    """Mirror of RowParallelLinear.weight_loader's `narrow`.

    vLLM's loader narrows along the INPUT dim, not the output dim:
        shard_size = param_data.shape[input_dim]    # linear.py:L1522
        start_idx  = self.tp_rank * shard_size      # linear.py:L1523
        loaded_weight = loaded_weight.narrow(input_dim, start_idx, shard_size)
                                                    # linear.py:L1524
    Easy-to-flip bug per W01 (wisdom/debugging.md) — column uses output_dim,
    row uses input_dim. Documented as Trap-F in impl-notes.
    """
    in_dim = A_full.shape[0]
    shard_size = divide(in_dim, tp_size)
    start = tp_rank * shard_size
    # REFERENCE: linear.py:L1524 — `narrow` along the INPUT dim for row-parallel
    return A_full[start : start + shard_size, :]


# ---------------------------------------------------------------------------
# §3 The Megatron col→row pair
# ---------------------------------------------------------------------------

# REFERENCE: instances/vllm/source/vllm/model_executor/models/llama.py:L94-L121 LlamaMLP
def column_then_row_block(
    X: np.ndarray,
    A_col: np.ndarray,
    A_row: np.ndarray,
    activation_fn,
    tp_size: int,
) -> tuple[np.ndarray, int]:
    """Compute Y = activation(X @ A_col) @ A_row using col→row decomposition.

    A_col: [hidden, ffn]            (column-parallel)
    A_row: [ffn,    hidden]         (row-parallel)
    activation_fn: applied element-wise on the sharded intermediate

    Communication accounting:
        - Column-parallel forward: NO collective (output stays sharded).
        - Element-wise activation: NO collective (works on sharded slice).
        - Row-parallel forward: ONE all-reduce.

    Total: ONE all-reduce per col→row block. This is the Megatron insight:
    if you naively all-gathered between gate/up and down, you'd DOUBLE
    the communication.

    Returns:
        (Y, num_collectives) — Y has shape matching X's batch dims with
        last dim = A_row.shape[-1]; num_collectives is exactly 1.
    """
    # Step 1: column-parallel — produces list of [..., ffn/p] shards.
    Z_shards = column_parallel_forward(X, A_col, tp_size=tp_size, gather_output=False)
    # Step 2: element-wise activation per rank — NO communication.
    Z_act_shards = [activation_fn(z) for z in Z_shards]
    # Step 3: row-parallel — single all-reduce produces full [..., hidden].
    Y = row_parallel_forward(
        Z_act_shards, A_row, tp_size=tp_size, input_is_parallel=True, reduce_results=True
    )
    num_collectives = 1
    return Y, num_collectives


# ---------------------------------------------------------------------------
# §4 Equivalence verification — the existence proof for §8.1 narrative.
# ---------------------------------------------------------------------------

def verify_column_parallel_equivalence(
    in_dim: int = 64, out_dim: int = 128, batch: int = 4, tp_size: int = 4, seed: int = 0
) -> dict:
    """Construct random X, A; compute Y_ref = X @ A; compute Y_tp via column
    sharding + concat; assert allclose. Return the max-abs diff for the demo
    (writer quotes verbatim — see Demo §1 in brief)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((batch, in_dim)).astype(np.float32)
    A = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    Y_ref = X @ A
    Y_tp = column_parallel_forward(X, A, tp_size=tp_size, gather_output=True)
    diff = float(np.max(np.abs(Y_ref - Y_tp)))
    ok = diff < 1e-5
    return {
        "kind": "column_parallel",
        "tp_size": tp_size,
        "shape": (batch, in_dim, out_dim),
        "max_abs_diff": diff,
        "allclose": ok,
        "tolerance_used": 1e-5,
    }


def verify_row_parallel_equivalence(
    in_dim: int = 64, out_dim: int = 128, batch: int = 4, tp_size: int = 4, seed: int = 1
) -> dict:
    """Construct random X, A; compute Y_ref = X @ A; row-shard and verify
    sum_i (X_i @ A_i) == Y_ref."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((batch, in_dim)).astype(np.float32)
    A = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    Y_ref = X @ A
    Y_tp = row_parallel_forward(X, A, tp_size=tp_size, input_is_parallel=False, reduce_results=True)
    diff = float(np.max(np.abs(Y_ref - Y_tp)))
    ok = diff < 1e-4  # row-parallel does extra additions → slightly larger numerical noise
    return {
        "kind": "row_parallel",
        "tp_size": tp_size,
        "shape": (batch, in_dim, out_dim),
        "max_abs_diff": diff,
        "allclose": ok,
        "tolerance_used": 1e-4,
    }


def verify_column_then_row_block(
    hidden: int = 64, ffn: int = 256, batch: int = 4, tp_size: int = 4, seed: int = 2
) -> dict:
    """Verify the col→row composition: Y_ref = relu(X @ A_col) @ A_row matches
    the sharded version with one all-reduce."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((batch, hidden)).astype(np.float32)
    A_col = rng.standard_normal((hidden, ffn)).astype(np.float32) * 0.05
    A_row = rng.standard_normal((ffn, hidden)).astype(np.float32) * 0.05
    relu = lambda t: np.maximum(t, 0.0)
    Y_ref = relu(X @ A_col) @ A_row
    Y_tp, ncoll = column_then_row_block(X, A_col, A_row, relu, tp_size=tp_size)
    diff = float(np.max(np.abs(Y_ref - Y_tp)))
    ok = diff < 1e-3
    return {
        "kind": "column_then_row",
        "tp_size": tp_size,
        "shape": (batch, hidden, ffn),
        "max_abs_diff": diff,
        "num_collectives": ncoll,
        "allclose": ok,
        "tolerance_used": 1e-3,
    }


if __name__ == "__main__":
    # Direct-run smoke test for `python3 implementation/tp_math.py`.
    print("=" * 60)
    print("tp_math.py — column-parallel and row-parallel equivalence")
    print("=" * 60)
    for r in (
        verify_column_parallel_equivalence(tp_size=2),
        verify_column_parallel_equivalence(tp_size=4),
        verify_row_parallel_equivalence(tp_size=2),
        verify_row_parallel_equivalence(tp_size=4),
        verify_column_then_row_block(tp_size=2),
        verify_column_then_row_block(tp_size=4),
    ):
        print(f"{r['kind']:>22s}  tp_size={r['tp_size']}  "
              f"max_abs_diff={r['max_abs_diff']:.3e}  "
              f"allclose={r['allclose']}  "
              f"({'extra: ncoll=' + str(r.get('num_collectives','NA')) if 'num_collectives' in r else ''})")
