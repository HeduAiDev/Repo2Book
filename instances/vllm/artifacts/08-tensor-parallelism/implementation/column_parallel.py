"""column_parallel — pedagogical mirror of vLLM's ColumnParallelLinear suite.

Reproduces the class shape and weight-loader semantics of:

    instances/vllm/source/vllm/model_executor/layers/linear.py:L410-L608
        class ColumnParallelLinear

    instances/vllm/source/vllm/model_executor/layers/linear.py:L609-L976
        class MergedColumnParallelLinear  (fuses gate+up for SwiGLU MLP)

Differences vs vLLM (ALL marked):

    - Backed by numpy arrays, not torch.Parameter (no autograd, no quant).
    - All `tp_size` ranks held simultaneously by `self.rank_states[r]` —
      single-process simulation; production uses one process per rank.
    - No NCCL: forward returns the per-rank output list. The CALLER
      decides whether to all-gather (mirrors gather_output flag) or hand the
      shards directly to a row-parallel layer (the common case).

Class names, method names, attribute names match vLLM 1:1 so a reader can
diff our forward() against the source.

References:
- linear.py:L410-L608 ColumnParallelLinear
- linear.py:L451-L460 tp_rank/tp_size/output_size_per_partition init
- linear.py:L534-L569 weight_loader narrow on output_dim
- linear.py:L579-L598 forward + optional tensor_model_parallel_all_gather
- linear.py:L609-L976 MergedColumnParallelLinear (output_sizes list)
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .tp_math import (
    column_parallel_forward,
    column_parallel_weight_loader,
    divide,
)


# REFERENCE: instances/vllm/source/vllm/model_executor/layers/linear.py:L410-L608
class ColumnParallelLinear:
    """Linear layer with column parallelism.

    Y = X @ A + b, where A is split along its OUTPUT (column) dim:
        A = [A_1 | A_2 | ... | A_p],  A_i: [in, out/p]
    and per-rank Y_i = X @ A_i.

    If `gather_output=True`, an all-gather rebuilds Y across ranks.
    Otherwise the shards are returned directly (the next layer is row-parallel).

    Attributes (matching vLLM):
        input_size            : int
        output_size           : int  (the unsharded global output dim)
        tp_size               : int  (number of simulated ranks)
        output_size_per_partition : int  (out / tp_size)
        output_partition_sizes: list[int]  (for plain ColumnParallel: [out_per])
                                            (for MergedColumn:        [s/tp_size for s in output_sizes])
        gather_output         : bool
        rank_states           : list[dict] — one entry per simulated rank;
                                rank_states[r]["weight"] is the rank-r shard.
                                In real vLLM, each process holds only ONE.
    """

    # REFERENCE: linear.py:L436-L505 — __init__
    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_size: int = 1,
        bias: bool = False,
        gather_output: bool = False,
        params_dtype: np.dtype = np.float32,
        prefix: str = "",
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.tp_size = tp_size
        # REFERENCE: linear.py:L453-L454
        self.input_size_per_partition = input_size
        self.output_size_per_partition = divide(output_size, tp_size)
        # REFERENCE: linear.py:L455-L460 — output_partition_sizes is a LIST
        # (extended by MergedColumnParallelLinear via `output_sizes`).
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(s, tp_size) for s in self.output_sizes
            ]
        else:
            self.output_partition_sizes = [self.output_size_per_partition]
        self.gather_output = gather_output
        self.params_dtype = params_dtype
        self.prefix = prefix
        self.has_bias = bias
        # rank_states[r] holds the rank-r view; in real vLLM each process owns one.
        self.rank_states: list[dict] = [
            {"weight": None, "bias": None, "tp_rank": r} for r in range(tp_size)
        ]
        self._loaded = False

    # REFERENCE: linear.py:L534-L569 — weight_loader (narrows along output_dim)
    def load_weight(self, A_full: np.ndarray, b_full: np.ndarray | None = None) -> None:
        """Distribute a full [in, out] weight to the per-rank slots.

        Plain ColumnParallel: one segment, single narrow on output dim.
        MergedColumn (when `output_sizes` is set, len > 1): each segment is
        sharded INDEPENDENTLY along the output dim, then per-rank shards
        are concatenated. This mirrors linear.py:L767-L820 which loops over
        `output_sizes` and computes per-segment (shard_offset, shard_size).
        """
        assert A_full.shape == (self.input_size, self.output_size), (
            f"Expected [in={self.input_size}, out={self.output_size}], got {A_full.shape}"
        )
        # Build per-rank weight: for each output segment, narrow by tp_rank.
        # For plain ColumnParallel, output_sizes is the single global output.
        segment_sizes = (
            self.output_sizes if hasattr(self, "output_sizes")
            else [self.output_size]
        )
        for r in range(self.tp_size):
            # REFERENCE: linear.py:L767-L820 — per-segment narrow loop.
            shards = []
            running_offset = 0
            for seg_size in segment_sizes:
                seg = A_full[:, running_offset : running_offset + seg_size]
                shard_size = divide(seg_size, self.tp_size)
                start = r * shard_size
                shards.append(seg[:, start : start + shard_size])
                running_offset += seg_size
            self.rank_states[r]["weight"] = np.concatenate(
                shards, axis=-1
            ).astype(self.params_dtype)
            if b_full is not None:
                # Bias parallels the output dim — sharded the same way as the weight.
                # REFERENCE: linear.py:L492-L502
                bias_shards = []
                running_offset = 0
                for seg_size in segment_sizes:
                    seg = b_full[running_offset : running_offset + seg_size]
                    shard_size = divide(seg_size, self.tp_size)
                    start = r * shard_size
                    bias_shards.append(seg[start : start + shard_size])
                    running_offset += seg_size
                self.rank_states[r]["bias"] = np.concatenate(
                    bias_shards, axis=0
                ).astype(self.params_dtype)
        self._loaded = True

    # REFERENCE: linear.py:L579-L607 — forward
    def forward(self, X: np.ndarray) -> list[np.ndarray] | np.ndarray:
        """Per-rank forward. Returns list of Y_i (per-rank output shards).

        If `gather_output=True`, returns the concatenated full Y instead.
        """
        assert self._loaded, "Call load_weight() before forward()"
        Y_shards: list[np.ndarray] = []
        for r in range(self.tp_size):
            W_r = self.rank_states[r]["weight"]
            # REFERENCE: linear.py:L587 — output_parallel = quant_method.apply(self, input_, bias)
            Y_r = X @ W_r
            if self.has_bias:
                Y_r = Y_r + self.rank_states[r]["bias"]
            Y_shards.append(Y_r)
        if self.gather_output:
            # REFERENCE: linear.py:L589-L591 — tensor_model_parallel_all_gather across partitions
            return np.concatenate(Y_shards, axis=-1)
        return Y_shards

    # REFERENCE: linear.py:L600-L606 — extra_repr
    def extra_repr(self) -> str:
        return (
            f"in_features={self.input_size}, "
            f"output_features={self.output_size_per_partition}, "
            f"tp_size={self.tp_size}, gather_output={self.gather_output}"
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.extra_repr()})"


# REFERENCE: instances/vllm/source/vllm/model_executor/layers/linear.py:L609-L976
class MergedColumnParallelLinear(ColumnParallelLinear):
    """Column-parallel layer that fuses N output projections into one matmul.

    Used by Llama's MLP for `gate_up_proj`: rather than two separate
    column-parallel linears for `gate_proj` and `up_proj`, we stack them
    output-wise into a single weight of shape [hidden, 2*ffn], shard
    column-wise, do ONE matmul, and split the result.

    Source: linear.py:L609-L976 (vLLM) and llama.py:L94-L101 (usage).

    The key field is `output_sizes`, a list whose sum equals `output_size`.
    Each per-rank weight slot still holds a SINGLE matrix, but the
    `output_partition_sizes` records the (sharded) lengths so that the
    user can `np.split` the result. This is exactly what
    `linear.py:L457-L460` does:
        if hasattr(self, "output_sizes"):
            self.output_partition_sizes = [
                divide(s, tp_size) for s in self.output_sizes
            ]
    """

    # REFERENCE: linear.py:L609-L725
    def __init__(
        self,
        input_size: int,
        output_sizes: Sequence[int],
        tp_size: int = 1,
        bias: bool = False,
        gather_output: bool = False,
        params_dtype: np.dtype = np.float32,
        prefix: str = "",
    ) -> None:
        # Set output_sizes BEFORE super().__init__() so that the parent's
        # init can read it via hasattr (matches vLLM's MRO trick).
        self.output_sizes = list(output_sizes)
        super().__init__(
            input_size=input_size,
            output_size=sum(self.output_sizes),
            tp_size=tp_size,
            bias=bias,
            gather_output=gather_output,
            params_dtype=params_dtype,
            prefix=prefix,
        )

    def split_per_rank(self, Y_shards: list[np.ndarray]) -> list[list[np.ndarray]]:
        """Split each rank's fused output into the per-output-projection pieces.

        Returns a list-of-lists: split_per_rank[r][k] is the per-rank shard of
        the k-th output (e.g., [gate_per_rank, up_per_rank]).

        Mirrors how the caller would do `torch.split(qkv, [...], dim=-1)`
        — see llama.py:L228-L229 for the QKV equivalent.
        """
        per_rank = []
        for Y_r in Y_shards:
            # output_partition_sizes are the per-rank lengths of each output.
            offsets = np.cumsum([0] + self.output_partition_sizes)
            per_rank.append(
                [Y_r[..., offsets[k] : offsets[k + 1]]
                 for k in range(len(self.output_partition_sizes))]
            )
        return per_rank


if __name__ == "__main__":
    # Direct-run sanity check.
    rng = np.random.default_rng(0)
    in_dim, out_dim, batch = 32, 64, 8
    X = rng.standard_normal((batch, in_dim)).astype(np.float32)
    A = rng.standard_normal((in_dim, out_dim)).astype(np.float32)

    # Plain column-parallel.
    layer = ColumnParallelLinear(input_size=in_dim, output_size=out_dim, tp_size=4)
    layer.load_weight(A_full=A)
    Y_shards = layer.forward(X)
    Y_ref = X @ A
    Y_concat = np.concatenate(Y_shards, axis=-1)
    print(f"ColumnParallelLinear  tp=4  max_abs_diff={np.max(np.abs(Y_ref - Y_concat)):.3e}")

    # Merged column (fused gate+up).
    ffn = 32
    layer_m = MergedColumnParallelLinear(
        input_size=in_dim, output_sizes=[ffn, ffn], tp_size=4,
    )
    A_m = rng.standard_normal((in_dim, 2 * ffn)).astype(np.float32)
    layer_m.load_weight(A_full=A_m)
    Y_m = layer_m.forward(X)
    per_rank_split = layer_m.split_per_rank(Y_m)
    # per-rank gate and up shards together, when concatenated rank-wise, equal
    # what the unfused two-matmul version would produce.
    gate_per_rank = [t[0] for t in per_rank_split]
    up_per_rank   = [t[1] for t in per_rank_split]
    gate_ref = X @ A_m[:, :ffn]
    up_ref   = X @ A_m[:, ffn:]
    gate_concat = np.concatenate(gate_per_rank, axis=-1)
    up_concat   = np.concatenate(up_per_rank,   axis=-1)
    print(f"MergedColumn  tp=4  gate diff={np.max(np.abs(gate_ref - gate_concat)):.3e}  "
          f"up diff={np.max(np.abs(up_ref - up_concat)):.3e}")
    print(f"output_partition_sizes = {layer_m.output_partition_sizes}  (each rank holds [ffn/p, ffn/p])")
