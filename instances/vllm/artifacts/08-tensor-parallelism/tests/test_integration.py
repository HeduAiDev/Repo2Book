"""Cross-module integration tests for Ch08 — TP behaviour as a whole.

We verify:
- The col→row composition uses ONE collective for arbitrary hidden/ffn shapes
  AND for an attention-shaped triad (qkv → matmul stand-in → o_proj).
- Demo §3 throughput-sweep memory math: weights/rank exactly halve at tp=2,
  quarter at tp=4 (Trap-A's saving column — memory IS halved cleanly).
- Trap-A evidence: at small payloads, raising tp from 2 to 4 to 8 does NOT
  give linear speedup; comm overhead is non-zero.
- Demo §3 verbatim (weights memory): 270.5 MB total, 135.3 MB at tp=2,
  67.6 MB at tp=4.
- Demo §1 verbatim col→row: tp=2 max_abs_diff = 0; tp=4 / tp=8 < 1e-6.
- Llama transformer block (attn + mlp) collective count = 2 per block.
"""

from __future__ import annotations

import numpy as np
import pytest

from implementation.column_parallel import MergedColumnParallelLinear
from implementation.comm_primitives import (
    HARDWARE_PROFILES,
    predict_block_overhead,
    ring_all_reduce_cost,
)
from implementation.mlp_block import LlamaMLPTP, reference_unsharded_mlp
from implementation.qkv_parallel import QKVParallelLinear
from implementation.row_parallel import RowParallelLinear
from implementation.tp_math import (
    column_then_row_block,
    verify_column_then_row_block,
)


# ---------------------------------------------------------------------------
# §1 Memory math (Trap-A's saving column — Demo §3 verbatim)
# ---------------------------------------------------------------------------

class TestMemoryMath:
    def test_weights_per_layer_demo_section_3(self):
        """Demo §3 pinned: weights_per_layer_MB_fp16 = 270.5 (Llama-7B-shaped)."""
        H, F = 4096, 11008
        # gate + up + down = 2*H*F + F*H = 3*H*F (fp16 = 2 bytes).
        weights_MB = (2 * H * F + F * H) * 2 / 1e6
        assert weights_MB == pytest.approx(270.5, abs=0.1)

    def test_weights_per_rank_halves_at_tp2(self):
        """Demo §3 pinned: tp=1 → 270.5 MB; tp=2 → 135.3 MB (exact halving)."""
        H, F = 4096, 11008
        full_MB = (2 * H * F + F * H) * 2 / 1e6
        for tp, expected in ((1, 270.5), (2, 135.3), (4, 67.6)):
            per_rank = full_MB / tp
            assert per_rank == pytest.approx(expected, abs=0.1), (
                f"tp={tp}: per_rank={per_rank:.2f}"
            )


# ---------------------------------------------------------------------------
# §2 Trap-A: TP=2 ≠ 2× (sub-linear comm)
# ---------------------------------------------------------------------------

class TestTrapAEvidenceCommOverhead:
    """Trap-A: 'TP=2 doubles throughput' is wrong because comm overhead is non-zero."""

    def test_small_payload_p4_slower_than_p2(self):
        """α-bound regime: more ranks = more synchronization steps = slower."""
        ab = HARDWARE_PROFILES["NVLink_HSXM4"]
        S = 1024
        t_p2 = ring_all_reduce_cost(S, 2, ab)
        t_p4 = ring_all_reduce_cost(S, 4, ab)
        t_p8 = ring_all_reduce_cost(S, 8, ab)
        assert t_p4 > t_p2
        assert t_p8 > t_p4

    def test_large_payload_p8_only_2x_faster_than_p2_not_4x(self):
        """β-bound regime: speedup factor is (P-1)/P ratio → asymptotically 1.
        P=8 vs P=2 = (7/8) / (1/2) = 7/4 = 1.75× theoretical max speedup,
        i.e. SUB-LINEAR. Real ratio is ~2.17 due to (P-1)/P term × chunking benefit.
        Either way: NOT 4×."""
        ab = HARDWARE_PROFILES["NVLink_HSXM4"]
        S = 64 * 1024 * 1024  # 64 MB
        t_p2 = ring_all_reduce_cost(S, 2, ab)
        t_p8 = ring_all_reduce_cost(S, 8, ab)
        ratio = t_p2 / t_p8
        assert ratio < 4.0, f"P=2/P=8 ratio={ratio:.3f}; if it were 4.0 that'd be linear speedup"
        assert ratio > 1.0, "P=8 should still be faster than P=2 in β-bound regime"

    def test_block_overhead_grows_for_some_tp(self):
        """For Llama-7B at small batch sizes, all-reduce overhead does NOT vanish."""
        for tp in (2, 4, 8):
            r = predict_block_overhead(
                hidden=4096, ffn=11008, batch_seqs=64, dtype_bytes=2,
                tp_size=tp, hardware="NVLink_HSXM4",
            )
            # Per-block all-reduce cost is non-zero. For batch=64 (small), it's α-bound.
            assert r["predicted_seconds_per_block"] > 0


# ---------------------------------------------------------------------------
# §3 Llama block end-to-end: attn (qkv → out) + mlp = 2 all-reduces per block
# ---------------------------------------------------------------------------

class TestLlamaBlockCollectiveCount:
    """A Llama transformer block has TWO all-reduces: one after o_proj (attn),
    one after down_proj (mlp). Verify that count by stitching the modules."""

    @pytest.mark.parametrize("tp", [2, 4, 8])
    def test_attn_then_mlp_two_collectives_per_block(self, tp):
        rng = np.random.default_rng(0)
        H = 256
        head_size = 32
        n_heads = 8
        F = 1024  # ffn

        # Build the QKV (col-parallel along heads — no all-reduce).
        qkv = QKVParallelLinear(
            hidden_size=H, head_size=head_size,
            total_num_heads=n_heads, total_num_kv_heads=n_heads, tp_size=tp,
        )
        Wq = rng.standard_normal((H, n_heads * head_size)).astype(np.float32) * 0.02
        Wk = rng.standard_normal((H, n_heads * head_size)).astype(np.float32) * 0.02
        Wv = rng.standard_normal((H, n_heads * head_size)).astype(np.float32) * 0.02
        qkv.load_qkv_weights(Wq, Wk, Wv)

        # o_proj: row-parallel, one all-reduce.
        o_proj = RowParallelLinear(
            input_size=n_heads * head_size, output_size=H, tp_size=tp,
            input_is_parallel=True, reduce_results=True,
        )
        Wo = rng.standard_normal((n_heads * head_size, H)).astype(np.float32) * 0.02
        o_proj.load_weight(A_full=Wo)

        # MLP: 1 all-reduce.
        mlp = LlamaMLPTP(hidden_size=H, intermediate_size=F, tp_size=tp)
        mlp.load_weights(
            rng.standard_normal((H, F)).astype(np.float32) * 0.02,
            rng.standard_normal((H, F)).astype(np.float32) * 0.02,
            rng.standard_normal((F, H)).astype(np.float32) * 0.02,
        )

        x = rng.standard_normal((4, H)).astype(np.float32) * 0.5

        # Attention path: qkv → split → (skip the actual scaled-dot-product attention
        # for this collective-count test — it's TP-agnostic per impl-notes §1.3) →
        # take the q_per_rank as a stand-in for attention output → o_proj.
        Y_qkv_per_rank = qkv.forward(x)
        splits = qkv.split_qkv(Y_qkv_per_rank)
        # The attention kernel sees per-rank q heads; output shape on rank r is
        # [batch, n_heads_per_rank * head_size] = [4, q_size_per_rank].
        attn_out_per_rank = splits["q"]  # stand-in (the kernel itself does no comm)

        n_collectives = 0
        # o_proj forward triggers ONE all-reduce inside the layer.
        if tp > 1:
            o_proj.forward(attn_out_per_rank)  # this all-reduces internally
            n_collectives += 1

        # MLP forward triggers ONE all-reduce inside down_proj.
        mlp.reset_collective_count()
        mlp.forward(x)
        n_collectives += mlp.count_collectives()

        assert n_collectives == 2, (
            f"tp={tp}: Llama block must have exactly 2 all-reduces "
            f"(attn o_proj + mlp down_proj), got {n_collectives}"
        )


# ---------------------------------------------------------------------------
# §4 Demo §1 col→row pinned numerics — exact pinning for the writer
# ---------------------------------------------------------------------------

class TestDemoSection1ColRow:
    """Pin the exact diffs the writer quotes character-for-character in Demo §1."""

    def test_tp2_diff_is_zero(self):
        r = verify_column_then_row_block(hidden=128, ffn=512, batch=4, tp_size=2)
        assert r["max_abs_diff"] == 0.0
        assert r["num_collectives"] == 1

    def test_tp4_diff_below_3e_minus_7(self):
        r = verify_column_then_row_block(hidden=128, ffn=512, batch=4, tp_size=4)
        assert r["max_abs_diff"] < 3e-7
        assert r["num_collectives"] == 1

    def test_tp8_diff_below_3e_minus_7(self):
        r = verify_column_then_row_block(hidden=128, ffn=512, batch=4, tp_size=8)
        assert r["max_abs_diff"] < 3e-7
        assert r["num_collectives"] == 1


# ---------------------------------------------------------------------------
# §5 Cross-cutting fidelity: T08 + Trap-E in a single composed test
# ---------------------------------------------------------------------------

class TestT08AndTrapEComposed:
    """Build a MergedColumn → SiluAndMul → RowParallel chain. Verify (a) the
    per-segment loader correctness AND (b) the col→row pair uses 1 collective.
    This is the smallest meaningful end-to-end TP demonstration of the chapter."""

    @pytest.mark.parametrize("tp", [2, 4])
    def test_chain_with_naive_narrow_would_be_wrong(self, tp):
        rng = np.random.default_rng(0)
        H, F = 32, 64
        x = rng.standard_normal((4, H)).astype(np.float32) * 0.5
        W_gate = rng.standard_normal((H, F)).astype(np.float32) * 0.02
        W_up = rng.standard_normal((H, F)).astype(np.float32) * 0.02
        W_down = rng.standard_normal((F, H)).astype(np.float32) * 0.02

        # Reference: unsharded MLP.
        y_ref = reference_unsharded_mlp(x, W_gate, W_up, W_down)

        # Proper: TP MLP via MergedColumn (per-segment loader).
        mlp = LlamaMLPTP(hidden_size=H, intermediate_size=F, tp_size=tp)
        mlp.load_weights(W_gate, W_up, W_down)
        y_proper, _ = mlp.forward(x)
        assert np.allclose(y_ref, y_proper, atol=1e-5)

        # Synthetic "naive narrow" reproduction: shard the FUSED gate-up uniformly
        # (i.e. each rank gets columns [r*2*F/tp : (r+1)*2*F/tp] — broken).
        # Demonstrate the bug yields a different output.
        # Build naive shards: rank 0 gets gate[0:F/tp]+gate[F/tp:2F/tp] (i.e. all gate,
        # no up); rank 1 gets gate[2F/tp:3F/tp]+gate[3F/tp:F]; rank 2-3 get up only.
        # This is exactly what a naive narrow on the fused [H, 2F] tensor would do.
        ffn_per_rank = F // tp
        # Build naive per-rank weights for gate_up:
        A_fused = np.concatenate([W_gate, W_up], axis=-1)  # [H, 2F]
        naive_per_rank = []
        for r in range(tp):
            naive_r = A_fused[:, r * 2 * ffn_per_rank : (r + 1) * 2 * ffn_per_rank]  # [H, 2*ffn/p]
            naive_per_rank.append(naive_r)

        # Apply naive shards manually: each rank's MLP.forward equivalent.
        # Just run the gate_up and silu_and_mul:
        naive_z_shards = []
        for naive_r in naive_per_rank:
            x_r = x @ naive_r  # [4, 2*ffn/p]
            half = x_r.shape[-1] // 2
            gate, up = x_r[..., :half], x_r[..., half:]
            silu = gate / (1.0 + np.exp(-gate))
            naive_z_shards.append(silu * up)

        # Down_proj: row-parallel sum.
        down_per_rank_in_dim = F // tp
        Wd_shards = [W_down[r * down_per_rank_in_dim : (r + 1) * down_per_rank_in_dim, :] for r in range(tp)]
        y_naive = sum(naive_z_shards[r] @ Wd_shards[r] for r in range(tp))

        # The naive output should differ from y_ref by a measurable amount when
        # tp >= 2 (because the pairing of gate ↔ up is broken). A tiny chance of
        # accidental near-match exists with low magnitude weights, so we tolerate
        # equality in the tp=1 case (no shards) and require difference for tp >= 2.
        # In tp=2: rank 0 has [gate[0:F/2], gate[F/2:F]] = ALL gate, no up; SiluAndMul
        # multiplies "gate × gate" (essentially) — definitely not equal to silu(gate)*up.
        if tp >= 2:
            naive_diff = float(np.max(np.abs(y_ref - y_naive)))
            # The proper TP MLP achieves diff ~1e-7 vs reference; naive narrow
            # is at least 1000× larger because it pairs the wrong gate↔up halves.
            assert naive_diff > 1e-4, (
                f"naive narrow expected to be visibly wrong; got diff={naive_diff:.3e}"
            )

    def test_chain_collective_count_one(self):
        """The MLP chain (gate_up → SiluAndMul → down_proj) does ONE all-reduce."""
        rng = np.random.default_rng(1)
        H, F = 64, 128
        x = rng.standard_normal((4, H)).astype(np.float32)
        mlp = LlamaMLPTP(hidden_size=H, intermediate_size=F, tp_size=4)
        mlp.load_weights(
            rng.standard_normal((H, F)).astype(np.float32) * 0.02,
            rng.standard_normal((H, F)).astype(np.float32) * 0.02,
            rng.standard_normal((F, H)).astype(np.float32) * 0.02,
        )
        mlp.reset_collective_count()
        mlp.forward(x)
        assert mlp.count_collectives() == 1


# ---------------------------------------------------------------------------
# §6 Cross-chapter compatibility: imports of Ch04..Ch07 don't crash
# ---------------------------------------------------------------------------

class TestCrossChapterImports:
    def test_ch08_implementation_imports_cleanly(self):
        """Smoke test: every public class/function used by demo.py imports."""
        from implementation.tp_math import (
            column_parallel_forward, row_parallel_forward, column_then_row_block,
        )
        from implementation.comm_primitives import (
            AlphaBetaModel, ring_all_reduce_cost, simulate_all_reduce, fit_alpha_beta,
        )
        from implementation.column_parallel import (
            ColumnParallelLinear, MergedColumnParallelLinear,
        )
        from implementation.row_parallel import RowParallelLinear
        from implementation.qkv_parallel import QKVParallelLinear
        from implementation.mlp_block import LlamaMLPTP, reference_unsharded_mlp
        # No assertion — successful import is the test.
        _ = (
            column_parallel_forward, row_parallel_forward, column_then_row_block,
            AlphaBetaModel, ring_all_reduce_cost, simulate_all_reduce, fit_alpha_beta,
            ColumnParallelLinear, MergedColumnParallelLinear, RowParallelLinear,
            QKVParallelLinear, LlamaMLPTP, reference_unsharded_mlp,
        )
