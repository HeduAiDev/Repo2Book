"""Tests for `implementation.row_parallel` — RowParallelLinear.

Source claim (linear.py:L1394-L1577):
- input_size_per_partition = divide(input_size, tp_size); output is FULL.
- weight_loader narrows along the INPUT dim (linear.py:L1521-L1524).
- bias is FULL output_size, NOT sharded (linear.py:L1486-L1487).
- forward: if input_is_parallel=True (default), caller pre-splits X
  (linear.py:L1547-L1548); else split_tensor_along_last_dim (L1549-L1553).
- Bias added on rank 0 ONLY (T06 — linear.py:L1557-L1559); otherwise post-reduce
  output would have tp_size × bias.
- reduce_results=True ⇒ tensor_model_parallel_all_reduce (L1562-L1563).
- bias + reduce_results=False is INVALID (L1480-L1483).
"""

from __future__ import annotations

import numpy as np
import pytest

from implementation.row_parallel import RowParallelLinear


# ---------------------------------------------------------------------------
# §1 Init contracts
# ---------------------------------------------------------------------------

class TestRowParallelInit:
    def test_input_narrowed_output_full(self):
        """linear.py:L1447-L1448 — input narrowed by tp, output stays full."""
        layer = RowParallelLinear(input_size=64, output_size=128, tp_size=4)
        assert layer.input_size_per_partition == 16
        assert layer.output_size_per_partition == 128

    def test_default_input_is_parallel_true(self):
        """linear.py:L1463 — default is True. Trap-F evidence: this is the col→row
        composition pattern; mis-setting silently doubles communication."""
        layer = RowParallelLinear(input_size=8, output_size=16, tp_size=2)
        assert layer.input_is_parallel is True

    def test_default_reduce_results_true(self):
        """linear.py:L1463 — default is True (the all-reduce IS the row-parallel
        contract; if a caller wants partials they must opt out)."""
        layer = RowParallelLinear(input_size=8, output_size=16, tp_size=2)
        assert layer.reduce_results is True

    def test_bias_without_reduce_results_raises(self):
        """linear.py:L1480-L1483 — bias + reduce_results=False is invalid."""
        with pytest.raises(ValueError, match="bias"):
            RowParallelLinear(
                input_size=8, output_size=16, tp_size=2, bias=True, reduce_results=False,
            )

    def test_indivisible_input_asserts(self):
        with pytest.raises(AssertionError):
            RowParallelLinear(input_size=33, output_size=16, tp_size=4)


# ---------------------------------------------------------------------------
# §2 Loader: narrow along INPUT dim
# ---------------------------------------------------------------------------

class TestRowParallelLoader:
    def test_load_weight_narrows_along_input_dim(self):
        """W01 / Trap-F: row narrows along INPUT (the opposite of column).
        This is the easy-to-flip bug wisdom/debugging.md flags."""
        rng = np.random.default_rng(0)
        in_dim, out_dim, tp = 32, 16, 4
        A = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
        layer = RowParallelLinear(in_dim, out_dim, tp_size=tp)
        layer.load_weight(A_full=A)
        for r in range(tp):
            shard = layer.rank_states[r]["weight"]
            assert shard.shape == (in_dim // tp, out_dim)
            assert np.array_equal(shard, A[r * 8 : (r + 1) * 8, :])

    def test_bias_full_output_not_sharded(self):
        """linear.py:L1486-L1487 — bias is full output_size, replicated, NOT sharded.
        Sharding bias would give bias/p after the reduce — silent wrong outputs (T06)."""
        rng = np.random.default_rng(1)
        in_dim, out_dim, tp = 8, 16, 4
        A = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
        b = rng.standard_normal((out_dim,)).astype(np.float32)
        layer = RowParallelLinear(in_dim, out_dim, tp_size=tp, bias=True)
        layer.load_weight(A_full=A, b_full=b)
        for r in range(tp):
            assert layer.rank_states[r]["bias"].shape == (out_dim,)
            assert np.array_equal(layer.rank_states[r]["bias"], b)


# ---------------------------------------------------------------------------
# §3 Forward — input_is_parallel branches + all-reduce + bias-on-rank-0
# ---------------------------------------------------------------------------

class TestRowParallelForward:
    @pytest.mark.parametrize("tp", [2, 4, 8])
    def test_input_is_parallel_true_with_pre_split_input(self, tp):
        """The col→row composition path: caller hands per-rank shards directly."""
        rng = np.random.default_rng(0)
        in_dim, out_dim, batch = 32, 64, 4
        X = rng.standard_normal((batch, in_dim)).astype(np.float32)
        A = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
        Y_ref = X @ A
        layer = RowParallelLinear(in_dim, out_dim, tp_size=tp, input_is_parallel=True)
        layer.load_weight(A_full=A)
        X_shards = list(np.split(X, tp, axis=-1))
        Y = layer.forward(X_shards)
        assert Y.shape == (batch, out_dim)
        assert float(np.max(np.abs(Y_ref - Y))) < 1e-4

    def test_input_is_parallel_true_rejects_non_list(self):
        rng = np.random.default_rng(1)
        layer = RowParallelLinear(input_size=32, output_size=16, tp_size=4, input_is_parallel=True)
        layer.load_weight(A_full=rng.standard_normal((32, 16)).astype(np.float32))
        with pytest.raises(AssertionError):
            layer.forward(rng.standard_normal((4, 32)).astype(np.float32))

    @pytest.mark.parametrize("tp", [2, 4])
    def test_input_is_parallel_false_splits_internally(self, tp):
        """linear.py:L1549-L1553 — split_tensor_along_last_dim, take rank-th slice."""
        rng = np.random.default_rng(2)
        in_dim, out_dim, batch = 32, 64, 4
        X = rng.standard_normal((batch, in_dim)).astype(np.float32)
        A = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
        layer = RowParallelLinear(in_dim, out_dim, tp_size=tp, input_is_parallel=False)
        layer.load_weight(A_full=A)
        Y = layer.forward(X)
        assert float(np.max(np.abs(X @ A - Y))) < 1e-4

    def test_input_is_parallel_false_rejects_list(self):
        layer = RowParallelLinear(input_size=32, output_size=16, tp_size=4, input_is_parallel=False)
        layer.load_weight(A_full=np.zeros((32, 16), dtype=np.float32))
        with pytest.raises(AssertionError):
            layer.forward([np.zeros((4, 8), dtype=np.float32)] * 4)

    def test_bias_added_only_on_rank_zero(self):
        """T06 / linear.py:L1557-L1559 — bias on rank 0 only.
        We craft a case where reducing tp ranks of bias would 4× the bias."""
        rng = np.random.default_rng(3)
        in_dim, out_dim, tp = 8, 16, 4
        # Weight is zero so the GEMM contributes 0; only bias matters.
        A = np.zeros((in_dim, out_dim), dtype=np.float32)
        b = rng.standard_normal((out_dim,)).astype(np.float32)
        X = rng.standard_normal((4, in_dim)).astype(np.float32)
        layer = RowParallelLinear(in_dim, out_dim, tp_size=tp, bias=True)
        layer.load_weight(A_full=A, b_full=b)
        X_shards = list(np.split(X, tp, axis=-1))
        Y = layer.forward(X_shards)
        # Y should equal b (broadcast over batch). If bias were added on EVERY rank
        # before reduce, Y would equal 4*b (off by 4×).
        for i in range(4):
            assert np.allclose(Y[i], b, atol=1e-6)
        # And explicitly reject the wrong outcome:
        assert not np.allclose(Y[0], 4 * b, atol=1e-6)

    def test_reduce_results_false_returns_partials(self):
        """reduce_results=False: caller is responsible for the eventual sum."""
        rng = np.random.default_rng(4)
        in_dim, out_dim, tp = 8, 16, 4
        X = rng.standard_normal((4, in_dim)).astype(np.float32)
        A = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
        layer = RowParallelLinear(in_dim, out_dim, tp_size=tp, input_is_parallel=True, reduce_results=False)
        layer.load_weight(A_full=A)
        X_shards = list(np.split(X, tp, axis=-1))
        out = layer.forward(X_shards)
        assert isinstance(out, list) and len(out) == tp
        assert float(np.max(np.abs(X @ A - np.sum(out, axis=0)))) < 1e-4

    def test_tp_size_one_skips_all_reduce(self):
        """tp_size=1 ⇒ no addition needed; output equals X @ A bit-for-bit."""
        rng = np.random.default_rng(5)
        in_dim, out_dim = 8, 16
        X = rng.standard_normal((4, in_dim)).astype(np.float32)
        A = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
        layer = RowParallelLinear(in_dim, out_dim, tp_size=1, input_is_parallel=True)
        layer.load_weight(A_full=A)
        out = layer.forward([X])
        # tp_size=1 with reduce_results=True returns partials list (not summed because tp==1).
        # But the sum of a 1-element list is the same as the single element.
        assert isinstance(out, list)
        assert np.array_equal(out[0], X @ A)

    def test_forward_requires_load_first(self):
        layer = RowParallelLinear(input_size=8, output_size=16, tp_size=2)
        with pytest.raises(AssertionError):
            layer.forward([np.zeros((1, 4), dtype=np.float32), np.zeros((1, 4), dtype=np.float32)])
