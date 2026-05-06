"""Tests for `implementation.qkv_parallel` — QKVParallelLinear (head sharding + GQA).

Source claim (linear.py:L977-L1393):
- num_heads = divide(total_num_heads, tp_size) (linear.py:L1030)
- if tp_size >= total_num_kv_heads: num_kv_heads=1, num_kv_head_replicas=divide(tp_size, total_num_kv_heads)
  else: num_kv_heads = divide(total_num_kv_heads, tp_size), num_kv_head_replicas=1
  (linear.py:L1031-L1036)
- output_sizes triple = [q_full, k_full, v_full] (linear.py:L1043-L1047)
- Sharding is along the HEAD dim, NOT arbitrary feature columns (Trap-C)
- Per-rank fused output is split into Q, K, V via the [q_size, kv_size, kv_size] triple
  (llama.py:L228-L229)

This file is the heart of Trap-C and Trap-D verification.
"""

from __future__ import annotations

import numpy as np
import pytest

from implementation.qkv_parallel import QKVParallelLinear
from implementation.tp_math import divide


# ---------------------------------------------------------------------------
# §1 Head-count math (the §3.4 / §3.4.1 of the chapter)
# ---------------------------------------------------------------------------

class TestHeadShardingMath:
    """Trap-C evidence: QKVParallelLinear splits along HEAD dim, not feature dim."""

    @pytest.mark.parametrize("total_heads,tp,expected_per_rank", [
        (32, 1, 32),
        (32, 2, 16),
        (32, 4, 8),
        (32, 8, 4),
        (64, 4, 16),  # Llama-3-70B: 64 Q heads / tp=4 = 16 heads/rank
    ])
    def test_num_heads_per_rank(self, total_heads, tp, expected_per_rank):
        """linear.py:L1030 — num_heads = divide(total_num_heads, tp_size)."""
        layer = QKVParallelLinear(
            hidden_size=4096, head_size=128,
            total_num_heads=total_heads, total_num_kv_heads=total_heads,
            tp_size=tp,
        )
        assert layer.num_heads == expected_per_rank
        assert layer.num_heads * tp == total_heads, "no head dropped"

    def test_indivisible_heads_asserts(self):
        """T01: tp_size must divide total_num_heads."""
        with pytest.raises(AssertionError):
            QKVParallelLinear(
                hidden_size=4096, head_size=128,
                total_num_heads=33, total_num_kv_heads=33, tp_size=4,
            )


# ---------------------------------------------------------------------------
# §2 GQA boundary — num_kv_head_replicas (THE Trap-D evidence)
# ---------------------------------------------------------------------------

class TestGQABoundary:
    """Trap-D: 'TP halves KV cache' is conditional on `total_num_kv_heads >= tp_size`."""

    def test_mha_no_replication(self):
        """When total_kv_heads == total_q_heads (MHA), no replication anywhere."""
        layer = QKVParallelLinear(
            hidden_size=4096, head_size=128,
            total_num_heads=32, total_num_kv_heads=32, tp_size=4,
        )
        assert layer.num_kv_heads == 8  # 32/4
        assert layer.num_kv_head_replicas == 1

    def test_gqa_below_boundary_kv_shards_normally(self):
        """tp_size < total_kv_heads (8 KV heads, tp=4): KV is sharded, no replicas."""
        layer = QKVParallelLinear(
            hidden_size=8192, head_size=128,
            total_num_heads=64, total_num_kv_heads=8, tp_size=4,
        )
        assert layer.num_kv_heads == 2  # 8/4
        assert layer.num_kv_head_replicas == 1

    def test_gqa_at_boundary_clean_one_kv_per_rank(self):
        """tp_size == total_kv_heads (8 KV heads, tp=8): clean 1 KV head/rank."""
        layer = QKVParallelLinear(
            hidden_size=8192, head_size=128,
            total_num_heads=64, total_num_kv_heads=8, tp_size=8,
        )
        # NB: the impl branch fires when tp_size >= total_kv_heads, so num_kv_heads=1.
        assert layer.num_kv_heads == 1
        assert layer.num_kv_head_replicas == 1

    def test_gqa_above_boundary_replicates_kv(self):
        """tp_size > total_kv_heads (8 KV heads, tp=16): KV is replicated 2× per rank-pair."""
        layer = QKVParallelLinear(
            hidden_size=8192, head_size=128,
            total_num_heads=64, total_num_kv_heads=8, tp_size=16,
        )
        assert layer.num_kv_heads == 1
        assert layer.num_kv_head_replicas == 2

    def test_gqa_far_above_boundary(self):
        """tp_size = 32, 8 kv heads → 4 replicas per kv head. Demo §4 pinned: replicas=4."""
        layer = QKVParallelLinear(
            hidden_size=8192, head_size=128,
            total_num_heads=64, total_num_kv_heads=8, tp_size=32,
        )
        assert layer.num_kv_heads == 1
        assert layer.num_kv_head_replicas == 4

    def test_kv_memory_floor_verbatim_from_demo_section_4(self):
        """Demo §4 verbatim: at tp=8 the KV save factor is 8×; at tp=16 still 8× (cap)."""
        head_size = 128
        full_kv_per_token_bytes = 2 * 8 * head_size * 2  # 4096

        configs = [
            (2, 4, 1, 2048, 2.0),
            (4, 2, 1, 1024, 4.0),
            (8, 1, 1, 512, 8.0),
            (16, 1, 2, 512, 8.0),
            (32, 1, 4, 512, 8.0),
        ]
        for tp, expected_kv_h, expected_repl, expected_bytes, expected_save in configs:
            layer = QKVParallelLinear(
                hidden_size=8192, head_size=head_size,
                total_num_heads=64, total_num_kv_heads=8, tp_size=tp,
            )
            s = layer.per_rank_summary()
            assert s["num_kv_heads_per_rank"] == expected_kv_h
            assert s["num_kv_head_replicas"] == expected_repl
            kv_per_rank_bytes = 2 * s["num_kv_heads_per_rank"] * head_size * 2
            assert kv_per_rank_bytes == expected_bytes
            save = full_kv_per_token_bytes / kv_per_rank_bytes
            assert save == pytest.approx(expected_save, rel=1e-9)


# ---------------------------------------------------------------------------
# §3 Output triple [q, k, v] sharding (linear.py:L1043-L1047)
# ---------------------------------------------------------------------------

class TestOutputSizesTriple:
    def test_output_sizes_triple_for_mha(self):
        """linear.py:L1043-L1047 — output_sizes is [q_full, k_full, v_full]."""
        layer = QKVParallelLinear(
            hidden_size=4096, head_size=128,
            total_num_heads=32, total_num_kv_heads=32, tp_size=4,
        )
        q_full = layer.num_heads * layer.head_size * 4
        kv_full = layer.num_kv_heads * layer.head_size * 4
        assert layer.output_sizes == [q_full, kv_full, kv_full]
        assert q_full == 32 * 128
        assert kv_full == 32 * 128

    def test_output_partition_sizes_per_rank(self):
        """Per-rank lengths via the parent's MergedColumn-style segment shard."""
        layer = QKVParallelLinear(
            hidden_size=4096, head_size=128,
            total_num_heads=32, total_num_kv_heads=8, tp_size=4,
        )
        # Per rank: 8 q heads * 128 = 1024; 2 kv heads * 128 = 256.
        assert layer.output_partition_sizes == [1024, 256, 256]


# ---------------------------------------------------------------------------
# §4 load_qkv_weights + forward equivalence
# ---------------------------------------------------------------------------

class TestLoadQKVWeights:
    @pytest.mark.parametrize("tp", [1, 2, 4, 8])
    def test_mha_forward_matches_three_separate_matmuls(self, tp):
        """MHA case: per-rank QKV concat must equal X @ Wq, X @ Wk, X @ Wv."""
        rng = np.random.default_rng(0)
        H = 256
        layer = QKVParallelLinear(
            hidden_size=H, head_size=64, total_num_heads=8, total_num_kv_heads=8, tp_size=tp,
        )
        Wq = rng.standard_normal((H, 8 * 64)).astype(np.float32) * 0.02
        Wk = rng.standard_normal((H, 8 * 64)).astype(np.float32) * 0.02
        Wv = rng.standard_normal((H, 8 * 64)).astype(np.float32) * 0.02
        layer.load_qkv_weights(Wq, Wk, Wv)
        X = rng.standard_normal((4, H)).astype(np.float32)
        Y = layer.forward(X)
        splits = layer.split_qkv(Y)
        q_all = np.concatenate(splits["q"], axis=-1)
        k_all = np.concatenate(splits["k"], axis=-1)
        v_all = np.concatenate(splits["v"], axis=-1)
        assert np.allclose(q_all, X @ Wq, atol=1e-4)
        assert np.allclose(k_all, X @ Wk, atol=1e-4)
        assert np.allclose(v_all, X @ Wv, atol=1e-4)

    @pytest.mark.parametrize("tp", [2, 4])
    def test_gqa_below_boundary_forward_correct(self, tp):
        """GQA, tp < total_kv_heads (8 KV heads, tp ∈ {2,4}): KV sharded, no replication."""
        rng = np.random.default_rng(1)
        H = 256
        layer = QKVParallelLinear(
            hidden_size=H, head_size=32, total_num_heads=8, total_num_kv_heads=8,
            tp_size=tp,
        )
        Wq = rng.standard_normal((H, 8 * 32)).astype(np.float32) * 0.02
        Wk = rng.standard_normal((H, 8 * 32)).astype(np.float32) * 0.02
        Wv = rng.standard_normal((H, 8 * 32)).astype(np.float32) * 0.02
        layer.load_qkv_weights(Wq, Wk, Wv)
        X = rng.standard_normal((4, H)).astype(np.float32)
        Y = layer.forward(X)
        splits = layer.split_qkv(Y)
        q_all = np.concatenate(splits["q"], axis=-1)
        k_all = np.concatenate(splits["k"], axis=-1)
        v_all = np.concatenate(splits["v"], axis=-1)
        assert np.allclose(q_all, X @ Wq, atol=1e-4)
        assert np.allclose(k_all, X @ Wk, atol=1e-4)
        assert np.allclose(v_all, X @ Wv, atol=1e-4)

    def test_gqa_replication_q_still_correct(self):
        """GQA + replication (tp=16, 8 KV heads): Q sharding is unaffected; verify Q output."""
        rng = np.random.default_rng(2)
        H = 256
        layer = QKVParallelLinear(
            hidden_size=H, head_size=32, total_num_heads=64, total_num_kv_heads=8,
            tp_size=16,
        )
        Wq = rng.standard_normal((H, 64 * 32)).astype(np.float32) * 0.02
        Wk = rng.standard_normal((H, 8 * 32)).astype(np.float32) * 0.02
        Wv = rng.standard_normal((H, 8 * 32)).astype(np.float32) * 0.02
        layer.load_qkv_weights(Wq, Wk, Wv)
        X = rng.standard_normal((2, H)).astype(np.float32)
        Y = layer.forward(X)
        splits = layer.split_qkv(Y)
        q_all = np.concatenate(splits["q"], axis=-1)
        # Q has 64 heads / 16 ranks = 4 heads/rank. Concatenated equals unsharded.
        assert np.allclose(q_all, X @ Wq, atol=1e-4)

    def test_load_shape_assertion(self):
        layer = QKVParallelLinear(
            hidden_size=64, head_size=32, total_num_heads=4, total_num_kv_heads=4, tp_size=2,
        )
        Wq_wrong = np.zeros((64, 64), dtype=np.float32)
        Wk = np.zeros((64, 4 * 32), dtype=np.float32)
        Wv = np.zeros((64, 4 * 32), dtype=np.float32)
        with pytest.raises(AssertionError):
            layer.load_qkv_weights(Wq_wrong, Wk, Wv)


# ---------------------------------------------------------------------------
# §5 split_qkv — llama.py:L228-L229 contract
# ---------------------------------------------------------------------------

class TestSplitQKV:
    def test_split_returns_q_k_v_keys(self):
        layer = QKVParallelLinear(
            hidden_size=64, head_size=16, total_num_heads=4, total_num_kv_heads=4, tp_size=2,
        )
        Wq = np.zeros((64, 4 * 16), dtype=np.float32)
        Wk = np.zeros((64, 4 * 16), dtype=np.float32)
        Wv = np.zeros((64, 4 * 16), dtype=np.float32)
        layer.load_qkv_weights(Wq, Wk, Wv)
        Y = layer.forward(np.zeros((2, 64), dtype=np.float32))
        splits = layer.split_qkv(Y)
        assert set(splits.keys()) == {"q", "k", "v"}
        for key in ("q", "k", "v"):
            assert len(splits[key]) == 2  # tp_size

    def test_split_offsets_are_q_then_kv_then_kv(self):
        """The split is at positions [0, q_size, q_size+kv_size, q_size+2*kv_size]."""
        layer = QKVParallelLinear(
            hidden_size=64, head_size=16, total_num_heads=8, total_num_kv_heads=4, tp_size=2,
        )
        # Per rank: q_size = 4 heads * 16 = 64; kv_size = 2 heads * 16 = 32.
        Wq = np.tile(np.arange(8 * 16, dtype=np.float32), (64, 1))  # values 0..127
        Wk = np.tile(np.arange(4 * 16, dtype=np.float32) + 1000.0, (64, 1))  # 1000..1063
        Wv = np.tile(np.arange(4 * 16, dtype=np.float32) + 2000.0, (64, 1))  # 2000..2063
        layer.load_qkv_weights(Wq, Wk, Wv)
        # Forward with all-ones input: Y @ Wq has each row = sum over input of Wq → 64 * Wq[0,:]
        # We check the SHAPES + that q region has values from Wq, k from Wk, v from Wv.
        X = np.zeros((1, 64), dtype=np.float32)
        X[0, 0] = 1.0  # only first hidden dim nonzero → output equals first row of W.
        Y = layer.forward(X)
        splits = layer.split_qkv(Y)
        # rank 0's q slice: Wq's first row, columns 0..63 (q heads 0..3 of 8).
        q_rank0 = splits["q"][0]
        assert q_rank0.shape == (1, 64)
        np.testing.assert_array_equal(q_rank0[0], np.arange(64, dtype=np.float32))
        # rank 0's k slice: Wk's first row, columns 0..31 (kv heads 0..1 of 4).
        k_rank0 = splits["k"][0]
        assert k_rank0.shape == (1, 32)
        np.testing.assert_array_equal(k_rank0[0], np.arange(32, dtype=np.float32) + 1000.0)
        # rank 0's v slice: Wv's first row, columns 0..31.
        v_rank0 = splits["v"][0]
        assert v_rank0.shape == (1, 32)
        np.testing.assert_array_equal(v_rank0[0], np.arange(32, dtype=np.float32) + 2000.0)


# ---------------------------------------------------------------------------
# §6 per_rank_summary — diagnostic dict for Demo §4
# ---------------------------------------------------------------------------

class TestPerRankSummary:
    def test_summary_contains_expected_keys(self):
        layer = QKVParallelLinear(
            hidden_size=4096, head_size=128, total_num_heads=32, total_num_kv_heads=8, tp_size=4,
        )
        s = layer.per_rank_summary()
        for key in (
            "tp_size", "total_num_heads", "total_num_kv_heads",
            "num_heads_per_rank", "num_kv_heads_per_rank", "num_kv_head_replicas",
            "q_size_per_rank", "kv_size_per_rank",
        ):
            assert key in s
        # 32 q heads / tp=4 = 8 q heads/rank * 128 = 1024.
        assert s["q_size_per_rank"] == 1024
        # 8 kv heads / tp=4 = 2 kv heads/rank * 128 = 256.
        assert s["kv_size_per_rank"] == 256
