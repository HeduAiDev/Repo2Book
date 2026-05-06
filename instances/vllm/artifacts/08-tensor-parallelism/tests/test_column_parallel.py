"""Tests for `implementation.column_parallel` — ColumnParallelLinear + Merged.

Source claim (linear.py:L410-L608, L609-L976):
- ColumnParallelLinear narrows along the OUTPUT dim (linear.py:L561).
- gather_output=True triggers a tensor_model_parallel_all_gather (L589-L591).
- MergedColumnParallelLinear sets `output_sizes` BEFORE super().__init__ so the
  parent can read it via hasattr (linear.py:L455-L460 + L609-L725).
- Each segment in `output_sizes` is sharded INDEPENDENTLY along output_dim
  (linear.py:L767-L820 — T08 per-segment loop).
- Bias parallels the output dim and is sharded the same way as the weight.

We verify each one. The MergedColumn fidelity test directly demonstrates Trap-E /
T08: a naive narrow on the fused output puts wrong data in each rank — we show
that the per-segment loop produces correct shards.
"""

from __future__ import annotations

import numpy as np
import pytest

from implementation.column_parallel import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
)


# ---------------------------------------------------------------------------
# §1 ColumnParallelLinear init / loader / forward
# ---------------------------------------------------------------------------

class TestColumnParallelInit:
    def test_output_size_per_partition_uses_divide(self):
        """linear.py:L454 — output_size_per_partition = divide(output_size, tp_size)."""
        layer = ColumnParallelLinear(input_size=64, output_size=128, tp_size=4)
        assert layer.output_size_per_partition == 32
        # Plain ColumnParallel: output_partition_sizes is single-element list.
        assert layer.output_partition_sizes == [32]

    def test_indivisible_output_size_asserts(self):
        """T01: tp_size must divide output_size."""
        with pytest.raises(AssertionError):
            ColumnParallelLinear(input_size=64, output_size=33, tp_size=4)

    def test_input_size_per_partition_unchanged_for_column(self):
        """linear.py:L453 — input is NOT narrowed for column-parallel."""
        layer = ColumnParallelLinear(input_size=64, output_size=128, tp_size=4)
        assert layer.input_size_per_partition == 64

    def test_rank_states_pre_loaded_with_none(self):
        layer = ColumnParallelLinear(input_size=8, output_size=16, tp_size=4)
        assert len(layer.rank_states) == 4
        for r, st in enumerate(layer.rank_states):
            assert st["weight"] is None
            assert st["bias"] is None
            assert st["tp_rank"] == r


class TestColumnParallelLoader:
    def test_load_weight_narrows_along_output_dim(self):
        """T08 / Trap-F: loader's narrow is on output_dim, not input_dim."""
        rng = np.random.default_rng(0)
        in_dim, out_dim = 32, 16
        A = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
        layer = ColumnParallelLinear(input_size=in_dim, output_size=out_dim, tp_size=4)
        layer.load_weight(A_full=A)
        # Each rank's weight: shape [in_dim, out_dim/4] = [32, 4].
        for r in range(4):
            shard = layer.rank_states[r]["weight"]
            assert shard.shape == (32, 4)
            assert np.array_equal(shard, A[:, r * 4 : (r + 1) * 4])

    def test_load_weight_shape_assertion(self):
        rng = np.random.default_rng(0)
        layer = ColumnParallelLinear(input_size=32, output_size=16, tp_size=4)
        with pytest.raises(AssertionError):
            layer.load_weight(A_full=rng.standard_normal((32, 17)).astype(np.float32))

    def test_load_bias_shards_same_way_as_weight(self):
        """linear.py:L492-L502 — bias is parallel to the output dim."""
        rng = np.random.default_rng(0)
        in_dim, out_dim, tp = 8, 16, 4
        A = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
        b = rng.standard_normal((out_dim,)).astype(np.float32)
        layer = ColumnParallelLinear(in_dim, out_dim, tp_size=tp, bias=True)
        layer.load_weight(A_full=A, b_full=b)
        for r in range(tp):
            assert layer.rank_states[r]["bias"].shape == (out_dim // tp,)
            assert np.array_equal(layer.rank_states[r]["bias"], b[r * 4 : (r + 1) * 4])


class TestColumnParallelForward:
    def test_forward_requires_load_first(self):
        layer = ColumnParallelLinear(input_size=8, output_size=16, tp_size=2)
        with pytest.raises(AssertionError):
            layer.forward(np.zeros((1, 8), dtype=np.float32))

    @pytest.mark.parametrize("tp", [2, 4])
    def test_forward_shards_concat_to_unsharded_exactly(self, tp):
        """Forward returns per-rank list; concat must EQUAL X @ A bit-for-bit
        (column-parallel does no addition)."""
        rng = np.random.default_rng(0)
        in_dim, out_dim, batch = 32, 64, 8
        X = rng.standard_normal((batch, in_dim)).astype(np.float32)
        A = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
        layer = ColumnParallelLinear(in_dim, out_dim, tp_size=tp)
        layer.load_weight(A_full=A)
        Y_shards = layer.forward(X)
        assert isinstance(Y_shards, list) and len(Y_shards) == tp
        Y_concat = np.concatenate(Y_shards, axis=-1)
        assert np.array_equal(Y_concat, X @ A)

    @pytest.mark.parametrize("tp", [2, 4])
    def test_gather_output_returns_full_tensor(self, tp):
        """linear.py:L589-L591 — gather_output=True ⇒ all-gather concatenates back."""
        rng = np.random.default_rng(1)
        in_dim, out_dim, batch = 32, 64, 4
        X = rng.standard_normal((batch, in_dim)).astype(np.float32)
        A = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
        layer = ColumnParallelLinear(in_dim, out_dim, tp_size=tp, gather_output=True)
        layer.load_weight(A_full=A)
        Y = layer.forward(X)
        # Single ndarray, full output dim.
        assert isinstance(Y, np.ndarray)
        assert Y.shape == (batch, out_dim)
        assert np.array_equal(Y, X @ A)

    def test_forward_with_bias(self):
        rng = np.random.default_rng(2)
        in_dim, out_dim, tp = 16, 32, 4
        X = rng.standard_normal((4, in_dim)).astype(np.float32)
        A = rng.standard_normal((in_dim, out_dim)).astype(np.float32)
        b = rng.standard_normal((out_dim,)).astype(np.float32)
        layer = ColumnParallelLinear(in_dim, out_dim, tp_size=tp, bias=True, gather_output=True)
        layer.load_weight(A_full=A, b_full=b)
        Y = layer.forward(X)
        Y_ref = X @ A + b
        assert np.array_equal(Y, Y_ref)


# ---------------------------------------------------------------------------
# §2 MergedColumnParallelLinear — fused gate+up + per-segment narrow loop
# ---------------------------------------------------------------------------

class TestMergedColumnParallel:
    def test_output_partition_sizes_per_segment_per_rank(self):
        """linear.py:L455-L460 + L609-L725 — output_partition_sizes lists each
        segment's per-rank length."""
        layer = MergedColumnParallelLinear(
            input_size=16, output_sizes=[64, 64], tp_size=4,
        )
        # 64 / 4 = 16 per rank, per segment.
        assert layer.output_partition_sizes == [16, 16]
        assert layer.output_size == 128

    def test_per_segment_loader_avoids_naive_narrow_bug(self):
        """T08 / Trap-E: the per-segment loop narrows EACH segment independently.
        A naive narrow on the fused [hidden, 2*ffn] would put `[gate_rank0,
        gate_rank1, gate_rank2, gate_rank3]` (the whole gate output) in rank 0
        — the WRONG shards. We test that each rank holds [gate_rank_r | up_rank_r]."""
        rng = np.random.default_rng(0)
        in_dim, ffn, tp = 8, 16, 4
        # Build a recognizable fused weight: gate columns 0..15, up columns 16..31.
        W_gate = np.tile(np.arange(ffn, dtype=np.float32) + 100.0, (in_dim, 1))
        W_up = np.tile(np.arange(ffn, dtype=np.float32) + 200.0, (in_dim, 1))
        A_fused = np.concatenate([W_gate, W_up], axis=-1)  # [in, 2*ffn]
        layer = MergedColumnParallelLinear(in_dim, output_sizes=[ffn, ffn], tp_size=tp)
        layer.load_weight(A_full=A_fused)
        # rank 0 should hold gate columns 0..3 + up columns 0..3, NOT gate columns 0..7.
        rank0 = layer.rank_states[0]["weight"]
        # The per-segment loader concatenates per-rank shards → shape [in, ffn/p + ffn/p].
        assert rank0.shape == (in_dim, 2 * (ffn // tp))
        # First half (gate-per-rank) should be values 100..103; second half 200..203.
        gate_part = rank0[:, : ffn // tp]
        up_part = rank0[:, ffn // tp :]
        # Each row of gate_part is [100, 101, 102, 103]; each row of up_part is [200, 201, 202, 203].
        assert np.array_equal(gate_part[0], np.array([100.0, 101.0, 102.0, 103.0]))
        assert np.array_equal(up_part[0], np.array([200.0, 201.0, 202.0, 203.0]))

    def test_naive_uniform_narrow_would_be_wrong(self):
        """Negative test: prove that a naive narrow on the fused output is INCORRECT.
        This documents the bug T08 caught and the fix prevents."""
        rng = np.random.default_rng(0)
        in_dim, ffn, tp = 8, 16, 4
        W_gate = np.tile(np.arange(ffn, dtype=np.float32) + 100.0, (in_dim, 1))
        W_up = np.tile(np.arange(ffn, dtype=np.float32) + 200.0, (in_dim, 1))
        A_fused = np.concatenate([W_gate, W_up], axis=-1)
        # Naive narrow: rank 0 gets the FIRST 2*(ffn/p) = 8 columns of A_fused.
        naive_rank0 = A_fused[:, : 2 * (ffn // tp)]
        # That puts gate columns 0..7 in rank 0, but rank 0 should hold gate cols 0..3 + up cols 0..3.
        assert not np.array_equal(naive_rank0[0, : ffn // tp], np.array([100.0, 101.0, 102.0, 103.0])) is False
        # Concretely: naive's first 4 cols are gate 0..3 (correct by coincidence), but cols 4..7 are gate 4..7 (WRONG, should be up 0..3).
        assert np.array_equal(naive_rank0[0, 4:8], np.array([104.0, 105.0, 106.0, 107.0]))
        # Up-half on rank 0 under correct sharding is up cols 0..3 = [200..203].
        # If we used naive rank 0, the "up half" would actually be gate 4..7 = [104..107] — wrong.
        # The proper loader gives [200..203]:
        layer = MergedColumnParallelLinear(in_dim, output_sizes=[ffn, ffn], tp_size=tp)
        layer.load_weight(A_full=A_fused)
        proper_rank0 = layer.rank_states[0]["weight"]
        assert np.array_equal(proper_rank0[0, 4:8], np.array([200.0, 201.0, 202.0, 203.0]))

    def test_split_per_rank_separates_segments(self):
        """split_per_rank returns list-of-lists where [r][k] is rank r's k-th output."""
        rng = np.random.default_rng(2)
        in_dim, ffn, tp = 8, 16, 4
        X = rng.standard_normal((4, in_dim)).astype(np.float32)
        A_g = rng.standard_normal((in_dim, ffn)).astype(np.float32)
        A_u = rng.standard_normal((in_dim, ffn)).astype(np.float32)
        A_fused = np.concatenate([A_g, A_u], axis=-1)
        layer = MergedColumnParallelLinear(in_dim, output_sizes=[ffn, ffn], tp_size=tp)
        layer.load_weight(A_full=A_fused)
        Y_shards = layer.forward(X)
        per_rank = layer.split_per_rank(Y_shards)
        assert len(per_rank) == tp
        # Each rank r has 2 outputs, each of width ffn/p.
        for r in range(tp):
            assert len(per_rank[r]) == 2
            for k in range(2):
                assert per_rank[r][k].shape == (4, ffn // tp)
        # Concatenated per-rank gate must equal X @ A_g; same for up.
        gate_concat = np.concatenate([per_rank[r][0] for r in range(tp)], axis=-1)
        up_concat = np.concatenate([per_rank[r][1] for r in range(tp)], axis=-1)
        assert np.allclose(gate_concat, X @ A_g, atol=1e-5)
        assert np.allclose(up_concat, X @ A_u, atol=1e-5)

    @pytest.mark.parametrize("tp", [2, 4])
    def test_fused_forward_equals_two_separate_matmuls(self, tp):
        """Equivalence: MergedColumn(gate, up) === two separate ColumnParallel(gate) + ColumnParallel(up)."""
        rng = np.random.default_rng(3)
        in_dim, ffn = 16, 32
        X = rng.standard_normal((4, in_dim)).astype(np.float32)
        W_gate = rng.standard_normal((in_dim, ffn)).astype(np.float32)
        W_up = rng.standard_normal((in_dim, ffn)).astype(np.float32)

        # Reference: two separate matmuls.
        gate_ref = X @ W_gate
        up_ref = X @ W_up

        # Fused via MergedColumn.
        layer = MergedColumnParallelLinear(in_dim, output_sizes=[ffn, ffn], tp_size=tp)
        layer.load_weight(A_full=np.concatenate([W_gate, W_up], axis=-1))
        Y_shards = layer.forward(X)
        per_rank = layer.split_per_rank(Y_shards)
        gate_concat = np.concatenate([per_rank[r][0] for r in range(tp)], axis=-1)
        up_concat = np.concatenate([per_rank[r][1] for r in range(tp)], axis=-1)
        assert np.allclose(gate_concat, gate_ref, atol=1e-5)
        assert np.allclose(up_concat, up_ref, atol=1e-5)
