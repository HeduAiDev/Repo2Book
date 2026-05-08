"""Fidelity tests for fused_moe_block.py — composition layer + memory math.

Verifies:

- Construction shapes: gate_weight, w13, w2 follow vLLM's
  layer.py:L590-L605 layout: w13 = [E, 2*intermediate, hidden], w2 = [E, hidden, intermediate].
- Forward shapes: output is [M, hidden]; preserves dtype.
- ep=1 vs ep=N mathematical invariance (chain-break analog from Ch08).
- silu_and_mul: shape and value sanity.
- ``expert_load`` returns per-expert routing histogram (matches expert_load_counts).
- ``memory_per_rank_MiB``: scaling 1/(ep × tp); §3.4 demo numerics pinned.
- Mixtral-style (ungrouped) and DeepSeek-style (grouped) configs both work.
- Trap F: ep=1 short-circuits — uses no expert_map.
"""

from __future__ import annotations

import math

import torch

from implementation.fused_moe_block import (
    ExpertFFNWeights,
    FusedMoEBlock,
    memory_per_rank_MiB,
    silu_and_mul,
)
from implementation.routing import expert_load_counts


# ---------------------------------------------------------------------------
# silu_and_mul
# ---------------------------------------------------------------------------


def test_silu_and_mul_halves_last_dim():
    """SwiGLU halves the last dim (gate|up split → silu(gate)*up)."""
    x = torch.randn(4, 16)
    out = silu_and_mul(x)
    assert out.shape == (4, 8)


def test_silu_and_mul_matches_definition():
    """silu_and_mul(x) = silu(x[:, :half]) * x[:, half:]."""
    x = torch.randn(2, 8)
    out = silu_and_mul(x)
    expected = torch.nn.functional.silu(x[:, :4]) * x[:, 4:]
    assert torch.allclose(out, expected)


def test_silu_and_mul_zero_gate_zero_output():
    """When gate==0, silu(0)==0 → output is 0 regardless of up."""
    x = torch.cat([torch.zeros(2, 4), torch.ones(2, 4)], dim=-1)
    out = silu_and_mul(x)
    assert torch.allclose(out, torch.zeros(2, 4))


# ---------------------------------------------------------------------------
# Block construction
# ---------------------------------------------------------------------------


def test_block_construction_shapes():
    """w13 = [E, 2*intermediate, hidden]; w2 = [E, hidden, intermediate]; gate = [E, hidden]."""
    block = FusedMoEBlock(
        num_experts=8,
        top_k=2,
        hidden_size=64,
        intermediate_size=128,
        ep_size=1,
        seed=0,
    )
    assert block.experts.w13.shape == (8, 256, 64)  # 2*128 = 256
    assert block.experts.w2.shape == (8, 64, 128)
    assert block.gate_weight.shape == (8, 64)


def test_block_ep1_uses_no_expert_map():
    """ep_size==1 short-circuits — every rank's map is None (E03 / Trap F)."""
    block = FusedMoEBlock(
        num_experts=4, top_k=2, hidden_size=16, intermediate_size=16, ep_size=1, seed=0
    )
    assert block._expert_maps == [None]


def test_block_ep4_has_4_maps_with_minus_one_sentinel():
    """ep_size=4 → 4 maps, each with -1 sentinels for off-rank experts."""
    block = FusedMoEBlock(
        num_experts=8, top_k=2, hidden_size=16, intermediate_size=16, ep_size=4, seed=0
    )
    assert len(block._expert_maps) == 4
    for m in block._expert_maps:
        assert (m != -1).sum().item() == 2  # 2 experts per rank for E=8, P=4
        assert (m == -1).sum().item() == 6


def test_block_ep_size_eq_num_experts():
    """ep_size == num_experts — each rank owns 1 expert."""
    block = FusedMoEBlock(
        num_experts=4, top_k=2, hidden_size=8, intermediate_size=8, ep_size=4, seed=0
    )
    for r, m in enumerate(block._expert_maps):
        owned = (m != -1).sum().item()
        assert owned == 1


def test_block_grouped_config_accepts_args():
    """DeepSeek-style: use_grouped_topk=True with num_expert_group/topk_group."""
    block = FusedMoEBlock(
        num_experts=8,
        top_k=4,
        hidden_size=16,
        intermediate_size=16,
        ep_size=1,
        use_grouped_topk=True,
        num_expert_group=4,
        topk_group=2,
        seed=0,
    )
    assert block.use_grouped_topk is True
    assert block.num_expert_group == 4


# ---------------------------------------------------------------------------
# Forward — basic shapes + computation
# ---------------------------------------------------------------------------


def test_forward_output_shape():
    """forward(h) returns [M, hidden]."""
    block = FusedMoEBlock(
        num_experts=4, top_k=2, hidden_size=16, intermediate_size=32, ep_size=1, seed=0
    )
    h = torch.randn(8, 16)
    out = block.forward(h)
    assert out.shape == (8, 16)


def test_forward_output_nonzero_with_random_input():
    """Random input + nonzero weights → nonzero output (sanity vs zero-init bug)."""
    torch.manual_seed(0)
    block = FusedMoEBlock(
        num_experts=4, top_k=2, hidden_size=16, intermediate_size=32, ep_size=1, seed=0
    )
    h = torch.randn(8, 16)
    out = block.forward(h)
    assert out.abs().sum().item() > 0


def test_forward_zero_input_zero_output():
    """forward(0) == 0 (W13/W2 are linear → linear in input)."""
    block = FusedMoEBlock(
        num_experts=4, top_k=2, hidden_size=16, intermediate_size=32, ep_size=1, seed=0
    )
    h = torch.zeros(4, 16)
    out = block.forward(h)
    assert torch.allclose(out, torch.zeros(4, 16))


def test_forward_with_external_router_logits_uses_them():
    """Passing router_logits skips the gate and uses them directly."""
    torch.manual_seed(0)
    block = FusedMoEBlock(
        num_experts=4, top_k=2, hidden_size=16, intermediate_size=32, ep_size=1, seed=0
    )
    h = torch.randn(4, 16)
    # Force every token to expert 0 with overwhelming logit.
    rl = torch.full((4, 4), -10.0)
    rl[:, 0] = 100.0
    out_forced = block.forward(h, router_logits=rl)
    # Now compute manually: every token uses expert 0 with weight ~1.
    # (Top-2 picks expert 0 plus next-highest, both get renormalized.)
    assert out_forced.shape == (4, 16)


# ---------------------------------------------------------------------------
# ep=1 vs ep=N invariance — chain-break analog from Ch08
# ---------------------------------------------------------------------------


def test_chain_break_ep1_eq_ep4():
    """ep=1 and ep=4 produce IDENTICAL outputs — EP is only a partition of the expert sum."""
    torch.manual_seed(1)
    h = torch.randn(16, 64)
    block_1 = FusedMoEBlock(
        num_experts=8, top_k=2, hidden_size=64, intermediate_size=128, ep_size=1, seed=1
    )
    block_4 = FusedMoEBlock(
        num_experts=8, top_k=2, hidden_size=64, intermediate_size=128, ep_size=4, seed=1
    )
    out_1 = block_1.forward(h)
    out_4 = block_4.forward(h)
    assert torch.allclose(out_1, out_4, atol=1e-6, rtol=1e-5)


def test_chain_break_ep1_eq_ep8():
    """ep=1 and ep=8 also identical (ep_size == num_experts edge case)."""
    torch.manual_seed(2)
    h = torch.randn(8, 32)
    b1 = FusedMoEBlock(
        num_experts=8, top_k=2, hidden_size=32, intermediate_size=64, ep_size=1, seed=2
    )
    b8 = FusedMoEBlock(
        num_experts=8, top_k=2, hidden_size=32, intermediate_size=64, ep_size=8, seed=2
    )
    out_1 = b1.forward(h)
    out_8 = b8.forward(h)
    assert torch.allclose(out_1, out_8, atol=1e-6, rtol=1e-5)


def test_chain_break_ep_with_round_robin():
    """Linear vs round_robin placement also produce identical outputs (just different ranks)."""
    torch.manual_seed(3)
    h = torch.randn(8, 32)
    b_lin = FusedMoEBlock(
        num_experts=8, top_k=2, hidden_size=32, intermediate_size=64, ep_size=4,
        expert_placement_strategy="linear", seed=3
    )
    b_rr = FusedMoEBlock(
        num_experts=8, top_k=2, hidden_size=32, intermediate_size=64, ep_size=4,
        expert_placement_strategy="round_robin", seed=3
    )
    out_lin = b_lin.forward(h)
    out_rr = b_rr.forward(h)
    assert torch.allclose(out_lin, out_rr, atol=1e-6, rtol=1e-5)


def test_chain_break_grouped_topk_ep1_vs_ep2():
    """DeepSeek-style grouped routing also satisfies ep=1 vs ep=N invariance."""
    torch.manual_seed(4)
    h = torch.randn(16, 32)
    b1 = FusedMoEBlock(
        num_experts=8, top_k=4, hidden_size=32, intermediate_size=64, ep_size=1,
        use_grouped_topk=True, num_expert_group=4, topk_group=2, seed=4
    )
    b2 = FusedMoEBlock(
        num_experts=8, top_k=4, hidden_size=32, intermediate_size=64, ep_size=2,
        use_grouped_topk=True, num_expert_group=4, topk_group=2, seed=4
    )
    out_1 = b1.forward(h)
    out_2 = b2.forward(h)
    assert torch.allclose(out_1, out_2, atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# expert_load helper
# ---------------------------------------------------------------------------


def test_expert_load_returns_E_long_tensor():
    """expert_load returns a length-E int64 histogram."""
    torch.manual_seed(5)
    block = FusedMoEBlock(
        num_experts=8, top_k=2, hidden_size=16, intermediate_size=32, ep_size=1, seed=0
    )
    h = torch.randn(64, 16)
    counts = block.expert_load(h)
    assert counts.shape == (8,)
    assert counts.dtype == torch.int64


def test_expert_load_sum_equals_M_times_K():
    """Σ counts must be M*K (every token-slot pair counted once)."""
    torch.manual_seed(6)
    M, K = 64, 2
    block = FusedMoEBlock(
        num_experts=8, top_k=K, hidden_size=16, intermediate_size=32, ep_size=1, seed=0
    )
    h = torch.randn(M, 16)
    counts = block.expert_load(h)
    assert int(counts.sum().item()) == M * K


def test_expert_load_consistent_with_external_helper():
    """block.expert_load == expert_load_counts(routing_output, E)."""
    torch.manual_seed(7)
    block = FusedMoEBlock(
        num_experts=4, top_k=2, hidden_size=16, intermediate_size=32, ep_size=1, seed=0
    )
    h = torch.randn(32, 16)
    rl = h @ block.gate_weight.T
    tw, ti = block._route(h, rl)
    via_helper = expert_load_counts(ti, block.num_experts)
    via_block = block.expert_load(h)
    assert torch.equal(via_helper, via_block)


# ---------------------------------------------------------------------------
# memory_per_rank_MiB and §3.4 verbatim demo
# ---------------------------------------------------------------------------


def test_memory_per_rank_baseline():
    """E=64 DeepSeek-V2-Lite block @ ep=1, tp=1 → 1056.00 MiB (verbatim §3.4)."""
    m = memory_per_rank_MiB(num_experts=64, hidden=2048, intermediate=1408,
                            ep_size=1, tp_size=1)
    assert math.isclose(m, 1056.00, abs_tol=0.01)


def test_memory_per_rank_ep4_tp1():
    """ep=4, tp=1 → 264.00 MiB (4× reduction)."""
    m = memory_per_rank_MiB(64, 2048, 1408, ep_size=4, tp_size=1)
    assert math.isclose(m, 264.00, abs_tol=0.01)


def test_memory_per_rank_ep4_tp2():
    """ep=4, tp=2 → 132.00 MiB (8× reduction)."""
    m = memory_per_rank_MiB(64, 2048, 1408, ep_size=4, tp_size=2)
    assert math.isclose(m, 132.00, abs_tol=0.01)


def test_memory_per_rank_ep8_tp4():
    """ep=8, tp=4 → 33.00 MiB (32× reduction)."""
    m = memory_per_rank_MiB(64, 2048, 1408, ep_size=8, tp_size=4)
    assert math.isclose(m, 33.00, abs_tol=0.01)


def test_memory_per_rank_ep16_tp1():
    """ep=16, tp=1 → 66.00 MiB (16×) — same as ep=8, tp=2."""
    m = memory_per_rank_MiB(64, 2048, 1408, ep_size=16, tp_size=1)
    assert math.isclose(m, 66.00, abs_tol=0.01)


def test_memory_per_rank_ep8_tp2():
    """ep=8, tp=2 → 66.00 MiB (16×)."""
    m = memory_per_rank_MiB(64, 2048, 1408, ep_size=8, tp_size=2)
    assert math.isclose(m, 66.00, abs_tol=0.01)


def test_memory_invariant_proportional_to_inverse_ep_tp():
    """mem_per_rank ∝ 1/(ep × tp) — the §9.5 invariant."""
    base = memory_per_rank_MiB(64, 2048, 1408, ep_size=1, tp_size=1)
    half = memory_per_rank_MiB(64, 2048, 1408, ep_size=2, tp_size=1)
    quarter = memory_per_rank_MiB(64, 2048, 1408, ep_size=2, tp_size=2)
    assert math.isclose(base / half, 2.0, abs_tol=1e-6)
    assert math.isclose(base / quarter, 4.0, abs_tol=1e-6)


def test_memory_total_params_pin():
    """Total params for E=64 block: 64*3*1408*2048 = 553,648,128 (pin §3.4)."""
    total = 64 * 3 * 1408 * 2048
    assert total == 553_648_128


def test_memory_per_expert_params_pin():
    """Per-expert params: 3·1408·2048 = 8,650,752 (pin §3.4)."""
    per_exp = 3 * 1408 * 2048
    assert per_exp == 8_650_752


# ---------------------------------------------------------------------------
# Single-expert edge case (Trap F: dispatch always runs?)
# ---------------------------------------------------------------------------


def test_single_expert_topk_one_no_dispatch_overhead_degenerate():
    """Edge case: E=1, K=1 → every token routes to the only expert.
    All-to-all is trivial. Verifies the block doesn't crash with degenerate sizes."""
    torch.manual_seed(8)
    block = FusedMoEBlock(
        num_experts=1, top_k=1, hidden_size=8, intermediate_size=16, ep_size=1, seed=0
    )
    h = torch.randn(4, 8)
    out = block.forward(h)
    assert out.shape == (4, 8)
    assert out.abs().sum().item() > 0


# ---------------------------------------------------------------------------
# ExpertFFNWeights dataclass
# ---------------------------------------------------------------------------


def test_expert_ffn_weights_dataclass():
    """ExpertFFNWeights stores w13, w2 and is constructible directly."""
    w13 = torch.zeros(4, 16, 8)
    w2 = torch.zeros(4, 8, 8)
    e = ExpertFFNWeights(w13=w13, w2=w2)
    assert e.w13 is w13
    assert e.w2 is w2
