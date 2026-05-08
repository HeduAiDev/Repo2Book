"""Smoke tests — confirm the chapter's core invariants hold.

These are NOT the tester's full test suite; they're a minimal sanity check
the implementer ran before handoff. The tester writes the full pinned-numerics
suite (per the v6 protocol).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make ``implementation`` importable when tests run from chapter root.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import torch  # noqa: E402

from implementation.expert_map import determine_expert_map  # noqa: E402
from implementation.fused_moe_block import (  # noqa: E402
    FusedMoEBlock,
    memory_per_rank_MiB,
)
from implementation.routing import expert_load_counts, fused_topk, grouped_topk  # noqa: E402


def test_fused_topk_renormalize():
    """When renormalize=True, K weights sum to ~1 per token."""
    torch.manual_seed(0)
    h = torch.randn(32, 16)
    logits = torch.randn(32, 8)
    tw, ti, _ = fused_topk(h, logits, topk=2, renormalize=True)
    sums = tw.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert ti.shape == (32, 2)


def test_fused_topk_no_renormalize():
    """Without renormalize, weights sum to <=1 (softmax tail)."""
    torch.manual_seed(0)
    h = torch.randn(32, 16)
    logits = torch.randn(32, 8)
    tw, _, _ = fused_topk(h, logits, topk=2, renormalize=False)
    sums = tw.sum(dim=-1)
    assert (sums <= 1.0 + 1e-5).all()
    assert (sums >= 2.0 / 8.0 - 1e-5).all()  # at least 2 / E uniform mass


def test_grouped_topk_shapes():
    """DeepSeek grouped path returns (M,K) weights and ids."""
    torch.manual_seed(0)
    h = torch.randn(64, 32)
    logits = torch.randn(64, 64)
    tw, ti = grouped_topk(
        h, logits, topk=6, renormalize=True, num_expert_group=8, topk_group=3
    )
    assert tw.shape == (64, 6)
    assert ti.shape == (64, 6)
    assert torch.allclose(tw.sum(dim=-1), torch.ones(64), atol=1e-5)


def test_determine_expert_map_linear():
    """Linear placement: rank r owns experts [r*E/P, (r+1)*E/P)."""
    local, m = determine_expert_map(ep_size=4, ep_rank=1, global_num_experts=8)
    assert local == 2
    assert m.tolist() == [-1, -1, 0, 1, -1, -1, -1, -1]


def test_determine_expert_map_round_robin():
    """Round-robin: rank r owns experts r, r+P, r+2P..."""
    local, m = determine_expert_map(
        ep_size=2, ep_rank=0, global_num_experts=8, expert_placement_strategy="round_robin"
    )
    assert local == 4
    assert m.tolist() == [0, -1, 1, -1, 2, -1, 3, -1]


def test_determine_expert_map_remainder():
    """When E % P != 0, the first ranks get one extra expert."""
    local0, m0 = determine_expert_map(ep_size=3, ep_rank=0, global_num_experts=7)
    local2, m2 = determine_expert_map(ep_size=3, ep_rank=2, global_num_experts=7)
    assert local0 == 3
    assert local2 == 2


def test_ep1_eq_ep4_forward():
    """ep_size=1 and ep_size=4 must produce identical outputs.

    This is the chain-break invariant: EP is just a partition of the
    expert sum; the math doesn't change. If ep=4 disagreed with ep=1,
    something is broken.
    """
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


def test_memory_scaling():
    """Memory per rank scales as 1 / (ep × tp)."""
    base = memory_per_rank_MiB(64, 2048, 1408, ep_size=1, tp_size=1)
    quad = memory_per_rank_MiB(64, 2048, 1408, ep_size=4, tp_size=1)
    octa = memory_per_rank_MiB(64, 2048, 1408, ep_size=4, tp_size=2)
    assert abs(base / quad - 4.0) < 1e-6
    assert abs(base / octa - 8.0) < 1e-6


def test_load_counts_consistency():
    """expert_load_counts must sum to M*K."""
    ti = torch.randint(0, 8, (32, 2), dtype=torch.int32)
    counts = expert_load_counts(ti, 8)
    assert int(counts.sum().item()) == 32 * 2


if __name__ == "__main__":
    # tiny in-process runner so the implementer can sanity-check without pytest
    fns = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"smoke: {len(fns)} tests passed")
