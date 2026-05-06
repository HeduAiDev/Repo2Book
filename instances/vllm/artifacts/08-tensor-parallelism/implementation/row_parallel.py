"""row_parallel — pedagogical mirror of vLLM's RowParallelLinear.

Reproduces the class shape and forward semantics of:

    instances/vllm/source/vllm/model_executor/layers/linear.py:L1394-L1577
        class RowParallelLinear

Why "row" parallel: the weight matrix A is split along its INPUT (row) dim,
and the input X is split along its LAST dim. Per-rank product is partial:
    Y_partial_i = X_i @ A_i,   shape [..., out].
Final Y = sum_i Y_partial_i — that sum is the all-reduce.

Differences vs vLLM (ALL marked):

    - numpy arrays, no torch.Parameter, no quant.
    - Single-process simulation: all ranks held in `self.rank_states`.
    - All-reduce simulated by `np.sum` over the partial-output list.
    - The `input_is_parallel` flag is implemented exactly as in
      linear.py:L1547-L1553.
    - `reduce_results=True` returns the summed output (mirrors
      `tensor_model_parallel_all_reduce`); `reduce_results=False` returns
      the per-rank partial list (mirrors `output = output_parallel`).

References:
- linear.py:L1394-L1425 docstring (Megatron diagram)
- linear.py:L1429-L1497 __init__ (input_size_per_partition = divide(input_size, tp_size))
- linear.py:L1499-L1532 weight_loader (narrows along input_dim — opposite of column!)
- linear.py:L1543-L1577 forward (split_input → matmul → all-reduce)
"""

from __future__ import annotations

import numpy as np

from .tp_math import (
    divide,
    row_parallel_forward,
    row_parallel_weight_loader,
    split_tensor_along_last_dim,
)


# REFERENCE: instances/vllm/source/vllm/model_executor/layers/linear.py:L1394-L1577
class RowParallelLinear:
    """Linear layer with row parallelism.

    Y = X @ A + b, with A split row-wise, X split column-wise:
        A: [in, out]  -> A_i: [in/p, out]
        X: [..., in]  -> X_i: [..., in/p]
    Per-rank: Y_partial_i = X_i @ A_i + (b if rank == 0 else 0).
    Final:    Y = sum_i Y_partial_i  (all-reduce).

    Bias subtlety (matches linear.py:L1557-L1559):
        We add bias ONLY on rank 0, otherwise it would be added p times after
        the all-reduce sum. This is "fuse bias add into GEMM for rank 0".
    """

    # REFERENCE: linear.py:L1429-L1497 — __init__
    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_size: int = 1,
        bias: bool = False,
        input_is_parallel: bool = True,
        reduce_results: bool = True,
        params_dtype: np.dtype = np.float32,
        prefix: str = "",
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        # REFERENCE: linear.py:L1447-L1448 — input narrowed, output unsharded
        self.input_size_per_partition = divide(input_size, tp_size)
        self.output_size_per_partition = output_size
        self.output_partition_sizes = [output_size]
        self.input_is_parallel = input_is_parallel
        self.reduce_results = reduce_results
        self.params_dtype = params_dtype
        self.prefix = prefix
        self.has_bias = bias
        # REFERENCE: linear.py:L1480-L1483 — bias + reduce_results=False is invalid
        # (bias would not be summed correctly without an all-reduce).
        if bias and not reduce_results:
            raise ValueError(
                "When not reducing the results, adding bias to the results "
                "can lead to incorrect results"
            )
        self.rank_states: list[dict] = [
            {"weight": None, "bias": None, "tp_rank": r} for r in range(tp_size)
        ]
        self._loaded = False

    # REFERENCE: linear.py:L1499-L1532 — weight_loader (narrow along INPUT dim)
    def load_weight(self, A_full: np.ndarray, b_full: np.ndarray | None = None) -> None:
        """Populate per-rank shards from a full [in, out] matrix."""
        assert A_full.shape == (self.input_size, self.output_size), (
            f"Expected [in={self.input_size}, out={self.output_size}], got {A_full.shape}"
        )
        for r in range(self.tp_size):
            # REFERENCE: linear.py:L1521-L1524 — `narrow` along the INPUT dim
            # (vs ColumnParallelLinear which narrows along OUTPUT — Trap-F).
            self.rank_states[r]["weight"] = row_parallel_weight_loader(
                A_full, tp_rank=r, tp_size=self.tp_size
            ).astype(self.params_dtype)
            if b_full is not None:
                # REFERENCE: linear.py:L1486-L1487 — bias is FULL output_size, NOT sharded
                self.rank_states[r]["bias"] = b_full.astype(self.params_dtype).copy()
        self._loaded = True

    # REFERENCE: linear.py:L1543-L1577 — forward
    def forward(self, X) -> np.ndarray | list[np.ndarray]:
        """Per-rank forward + optional all-reduce.

        Args:
            X: if `input_is_parallel=True`, must be a list of `tp_size`
               per-rank tensors (the upstream column-parallel layer's output).
               If `input_is_parallel=False`, must be a single full-width tensor;
               we split it ourselves.
        """
        assert self._loaded, "Call load_weight() before forward()"
        # REFERENCE: linear.py:L1547-L1553 — input_is_parallel branch
        if self.input_is_parallel:
            assert isinstance(X, list) and len(X) == self.tp_size, (
                "input_is_parallel=True expects per-rank list of length tp_size"
            )
            X_local = X
        else:
            assert isinstance(X, np.ndarray), (
                "input_is_parallel=False expects a single full-width array"
            )
            X_local = list(split_tensor_along_last_dim(X, self.tp_size))

        Y_partials: list[np.ndarray] = []
        for r in range(self.tp_size):
            W_r = self.rank_states[r]["weight"]
            # REFERENCE: linear.py:L1557-L1560 — fuse bias on rank 0 ONLY
            bias_for_rank = (
                self.rank_states[r]["bias"] if (self.has_bias and r == 0) else None
            )
            Y_r = X_local[r] @ W_r
            if bias_for_rank is not None:
                Y_r = Y_r + bias_for_rank
            Y_partials.append(Y_r)
        if self.reduce_results and self.tp_size > 1:
            # REFERENCE: linear.py:L1562-L1563 — tensor_model_parallel_all_reduce
            return np.sum(Y_partials, axis=0)
        # `reduce_results=False`: caller is responsible for the eventual
        # collective. Useful for fused all-reduce-into-RMSNorm patterns.
        return Y_partials

    # REFERENCE: linear.py:L1572-L1578 — extra_repr
    def extra_repr(self) -> str:
        return (
            f"in_features={self.input_size_per_partition}, "
            f"output_features={self.output_size}, "
            f"tp_size={self.tp_size}, reduce_results={self.reduce_results}"
        )

    def __repr__(self) -> str:
        return f"RowParallelLinear({self.extra_repr()})"


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    in_dim, out_dim, batch = 32, 64, 8
    X = rng.standard_normal((batch, in_dim)).astype(np.float32)
    A = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
    Y_ref = X @ A

    # input_is_parallel=False path: pass full X, layer splits it.
    layer = RowParallelLinear(
        input_size=in_dim, output_size=out_dim, tp_size=4,
        input_is_parallel=False, reduce_results=True,
    )
    layer.load_weight(A_full=A)
    Y_tp = layer.forward(X)
    print(f"RowParallelLinear  tp=4  is_parallel=False  "
          f"max_abs_diff={np.max(np.abs(Y_ref - Y_tp)):.3e}")

    # input_is_parallel=True path: pre-split (the column→row composition case).
    layer2 = RowParallelLinear(
        input_size=in_dim, output_size=out_dim, tp_size=4,
        input_is_parallel=True, reduce_results=True,
    )
    layer2.load_weight(A_full=A)
    X_shards = list(np.split(X, 4, axis=-1))
    Y_tp2 = layer2.forward(X_shards)
    print(f"RowParallelLinear  tp=4  is_parallel=True   "
          f"max_abs_diff={np.max(np.abs(Y_ref - Y_tp2)):.3e}")

    # reduce_results=False — return per-rank partials, caller sums later.
    layer3 = RowParallelLinear(
        input_size=in_dim, output_size=out_dim, tp_size=4,
        input_is_parallel=True, reduce_results=False, bias=False,
    )
    layer3.load_weight(A_full=A)
    Y_partials = layer3.forward(X_shards)
    Y_caller = np.sum(Y_partials, axis=0)
    print(f"RowParallelLinear  tp=4  reduce_results=False  "
          f"len(partials)={len(Y_partials)}  max_abs_diff={np.max(np.abs(Y_ref - Y_caller)):.3e}")
