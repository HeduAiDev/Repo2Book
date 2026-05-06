"""Tests for `implementation.tp_math` — column/row decomposition algebra.

Source claim: TP-sharded forward reproduces unsharded GEMM EXACTLY for
column-parallel (no reductions involved) and within fp32 numerical tolerance
for row-parallel (sum-of-partials adds noise).

We test:
- divide() / ensure_divisibility() honour the contract `numerator % denominator == 0`.
- split_tensor_along_last_dim() partitions exactly and reassembles to identity.
- column_parallel_forward()/_weight_loader() reproduce unsharded matmul EXACTLY
  (each rank does an independent slice; no addition).
- row_parallel_forward()/_weight_loader() reproduce unsharded matmul to fp32 tol.
- column_then_row_block() uses exactly ONE collective regardless of tp_size
  (Megatron pair invariant — Trap-E evidence).
- The verify_*() helpers used by demo §1 produce demo-numerics-grade diffs.
"""

from __future__ import annotations

import numpy as np
import pytest

from implementation.tp_math import (
    column_parallel_forward,
    column_parallel_weight_loader,
    column_then_row_block,
    divide,
    ensure_divisibility,
    row_parallel_forward,
    row_parallel_weight_loader,
    split_tensor_along_last_dim,
    verify_column_parallel_equivalence,
    verify_column_then_row_block,
    verify_row_parallel_equivalence,
)


# ---------------------------------------------------------------------------
# §1 divide / ensure_divisibility (T01 contract)
# ---------------------------------------------------------------------------

class TestDivide:
    def test_divides_evenly(self):
        """T01: divide(64, 4) == 16."""
        assert divide(64, 4) == 16
        assert divide(11008, 4) == 2752

    def test_raises_on_remainder(self):
        """T01: divide(7, 2) must AssertionError — model config incompatible with tp_size."""
        with pytest.raises(AssertionError):
            divide(7, 2)

    def test_ensure_divisibility_passes(self):
        ensure_divisibility(32, 4)

    def test_ensure_divisibility_fails(self):
        with pytest.raises(AssertionError):
            ensure_divisibility(33, 4)


# ---------------------------------------------------------------------------
# §2 split_tensor_along_last_dim (linear.py:L1547-L1553 input_is_parallel=False)
# ---------------------------------------------------------------------------

class TestSplitAlongLastDim:
    def test_split_count(self):
        x = np.arange(2 * 16, dtype=np.float32).reshape(2, 16)
        parts = list(split_tensor_along_last_dim(x, 4))
        assert len(parts) == 4
        for p in parts:
            assert p.shape == (2, 4)

    def test_concat_round_trips(self):
        """Reassembling preserves identity bit-for-bit."""
        x = np.arange(3 * 12, dtype=np.float32).reshape(3, 12)
        parts = list(split_tensor_along_last_dim(x, 3))
        recon = np.concatenate(parts, axis=-1)
        assert np.array_equal(recon, x)

    def test_split_asserts_divisibility(self):
        x = np.zeros((2, 7), dtype=np.float32)
        with pytest.raises(AssertionError):
            list(split_tensor_along_last_dim(x, 4))


# ---------------------------------------------------------------------------
# §3 column-parallel — exact mathematical equivalence
# ---------------------------------------------------------------------------

class TestColumnParallelMath:
    @pytest.mark.parametrize("tp", [2, 4, 8])
    def test_concat_equals_unsharded_exactly(self, tp):
        """Column-parallel does NO reductions: concatenated shards must EQUAL
        the unsharded matmul bit-for-bit (no fp summation noise)."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 64)).astype(np.float32)
        A = rng.standard_normal((64, 128)).astype(np.float32)
        Y_ref = X @ A
        Y_full = column_parallel_forward(X, A, tp_size=tp, gather_output=True)
        assert np.array_equal(Y_ref, Y_full), (
            f"column-parallel must reproduce unsharded matmul EXACTLY for tp={tp}"
        )

    @pytest.mark.parametrize("tp", [2, 4])
    def test_per_rank_shards_have_correct_shape(self, tp):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((4, 32)).astype(np.float32)
        A = rng.standard_normal((32, 64)).astype(np.float32)
        Y_shards = column_parallel_forward(X, A, tp_size=tp, gather_output=False)
        assert isinstance(Y_shards, list) and len(Y_shards) == tp
        for s in Y_shards:
            assert s.shape == (4, divide(64, tp))

    def test_weight_loader_narrow_along_output_dim(self):
        """T08 / Trap-F: column loader must narrow along the OUTPUT dim, not input."""
        rng = np.random.default_rng(2)
        A = rng.standard_normal((16, 32)).astype(np.float32)
        tp = 4
        for r in range(tp):
            shard = column_parallel_weight_loader(A, tp_rank=r, tp_size=tp)
            # Output dim narrowed to 32/4 = 8; input dim full.
            assert shard.shape == (16, 8), shard.shape
            # Slice equals the narrow on the OUTPUT dim.
            assert np.array_equal(shard, A[:, r * 8 : (r + 1) * 8])


# ---------------------------------------------------------------------------
# §4 row-parallel — sum-of-partials equivalence
# ---------------------------------------------------------------------------

class TestRowParallelMath:
    @pytest.mark.parametrize("tp", [2, 4, 8])
    def test_sum_of_partials_within_fp32_tolerance(self, tp):
        """Row-parallel does p-way addition: result equals reference within fp32 noise."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 64)).astype(np.float32)
        A = rng.standard_normal((64, 128)).astype(np.float32)
        Y_ref = X @ A
        Y_tp = row_parallel_forward(
            X, A, tp_size=tp, input_is_parallel=False, reduce_results=True
        )
        diff = float(np.max(np.abs(Y_ref - Y_tp)))
        # Demo §1 pinned: row_tp{2,4,8}_max_abs_diff in [7.6e-6, 9.6e-6]. Bound generously.
        assert diff < 1e-4, f"row tp={tp}: diff={diff:.3e}"

    def test_input_is_parallel_true_branch(self):
        """input_is_parallel=True: caller pre-splits X. Default path for col→row composition."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((4, 32)).astype(np.float32)
        A = rng.standard_normal((32, 64)).astype(np.float32)
        Y_ref = X @ A
        X_shards = list(np.split(X, 4, axis=-1))
        Y_tp = row_parallel_forward(
            X_shards, A, tp_size=4, input_is_parallel=True, reduce_results=True
        )
        diff = float(np.max(np.abs(Y_ref - Y_tp)))
        assert diff < 1e-4

    def test_input_is_parallel_false_branch(self):
        """input_is_parallel=False: layer splits X internally (linear.py:L1549-L1553)."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((4, 32)).astype(np.float32)
        A = rng.standard_normal((32, 64)).astype(np.float32)
        Y_ref = X @ A
        Y_tp = row_parallel_forward(
            X, A, tp_size=4, input_is_parallel=False, reduce_results=True
        )
        assert float(np.max(np.abs(Y_ref - Y_tp))) < 1e-4

    def test_reduce_results_false_returns_partials_list(self):
        """reduce_results=False: caller is responsible for the eventual collective."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((4, 32)).astype(np.float32)
        A = rng.standard_normal((32, 64)).astype(np.float32)
        Y_ref = X @ A
        partials = row_parallel_forward(
            X, A, tp_size=4, input_is_parallel=False, reduce_results=False
        )
        assert isinstance(partials, list) and len(partials) == 4
        # Each partial has FULL output shape — they SUM to Y_ref, not concat.
        for p in partials:
            assert p.shape == Y_ref.shape
        Y_caller_sum = np.sum(partials, axis=0)
        assert float(np.max(np.abs(Y_ref - Y_caller_sum))) < 1e-4

    def test_weight_loader_narrow_along_input_dim(self):
        """T-trap (Trap-F + W01): row loader narrows along the INPUT dim, not output.
        This is the classic flip that wisdom/debugging.md warns about."""
        rng = np.random.default_rng(4)
        A = rng.standard_normal((32, 16)).astype(np.float32)
        tp = 4
        for r in range(tp):
            shard = row_parallel_weight_loader(A, tp_rank=r, tp_size=tp)
            # Input dim narrowed to 32/4 = 8; output dim full.
            assert shard.shape == (8, 16), shard.shape
            assert np.array_equal(shard, A[r * 8 : (r + 1) * 8, :])


# ---------------------------------------------------------------------------
# §5 col→row composition — Megatron pair invariant
# ---------------------------------------------------------------------------

class TestColumnThenRowBlock:
    @pytest.mark.parametrize("tp", [2, 4, 8])
    def test_one_collective_per_block_regardless_of_tp(self, tp):
        """Trap-E evidence: ONE all-reduce per col→row pair, never two, never zero."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((4, 64)).astype(np.float32)
        A_col = rng.standard_normal((64, 256)).astype(np.float32) * 0.05
        A_row = rng.standard_normal((256, 64)).astype(np.float32) * 0.05
        relu = lambda t: np.maximum(t, 0.0)
        _, ncoll = column_then_row_block(X, A_col, A_row, relu, tp_size=tp)
        assert ncoll == 1, f"expected 1 collective per block, got {ncoll}"

    @pytest.mark.parametrize("tp", [2, 4, 8])
    def test_block_output_matches_reference(self, tp):
        """Composition with non-linear activation — sharded ≡ unsharded within tol."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((4, 64)).astype(np.float32)
        A_col = rng.standard_normal((64, 256)).astype(np.float32) * 0.05
        A_row = rng.standard_normal((256, 64)).astype(np.float32) * 0.05
        relu = lambda t: np.maximum(t, 0.0)
        Y_ref = relu(X @ A_col) @ A_row
        Y_tp, _ = column_then_row_block(X, A_col, A_row, relu, tp_size=tp)
        diff = float(np.max(np.abs(Y_ref - Y_tp)))
        # Demo §1 pinned: colrow_tp{2,4,8}_max_abs_diff in [0, 2.98e-7]. Bound generously.
        assert diff < 1e-3, f"col→row tp={tp}: diff={diff:.3e}"


# ---------------------------------------------------------------------------
# §6 Demo §1 verbatim diffs (writer cites these character-for-character)
# ---------------------------------------------------------------------------

class TestDemoSection1Numerics:
    """The exact verifiers that produce the §1 numerics quoted in chapter.md."""

    def test_verify_column_parallel_returns_zero_diff(self):
        """Demo §1: col_tp{2,4,8}_max_abs_diff ALL zero (column does no addition)."""
        for tp in (2, 4, 8):
            r = verify_column_parallel_equivalence(
                in_dim=128, out_dim=512, batch=4, tp_size=tp
            )
            assert r["max_abs_diff"] == 0.0, (
                f"verify_column tp={tp} expected zero diff, got {r['max_abs_diff']}"
            )
            assert r["allclose"] is True

    def test_verify_row_parallel_within_tolerance(self):
        """Demo §1: row_tp{2,4,8}_max_abs_diff ≈ 7.6e-6 to 9.5e-6."""
        for tp in (2, 4, 8):
            r = verify_row_parallel_equivalence(
                in_dim=128, out_dim=512, batch=4, tp_size=tp
            )
            assert r["max_abs_diff"] < 1e-4
            assert r["allclose"] is True

    def test_verify_col_then_row_uses_exactly_one_collective(self):
        """Demo §1: every col→row block has num_collectives = 1."""
        for tp in (2, 4, 8):
            r = verify_column_then_row_block(
                hidden=128, ffn=512, batch=4, tp_size=tp
            )
            assert r["num_collectives"] == 1
            assert r["allclose"] is True
