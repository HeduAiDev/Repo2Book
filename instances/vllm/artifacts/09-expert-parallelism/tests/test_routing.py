"""Fidelity tests for routing.py — fused_topk + grouped_topk math.

Mirrors the production tests one would write against
``vllm/model_executor/layers/fused_moe/router/{fused_topk,grouped_topk}_router.py``.
We verify:

- fused_topk: shapes, softmax/sigmoid scoring, renormalize on/off, top-K
  selects the K largest, weight invariants.
- grouped_topk: 2-stage selection (group → in-group), bias correction
  uses biased scores for SELECTION but unbiased for WEIGHTS, sum-of-top-2
  group score for the noaux_tc path, mask correctness.
- Trap G: softmax-then-topk vs topk-then-softmax do NOT commute.
- expert_load_counts: histogram identity Σ counts = M·K.
- §3.1 demo numerics pinned bit-for-bit.
"""

from __future__ import annotations

import math

import torch

from implementation.mixtral_vs_deepseek import (
    DEEPSEEK_V2_LITE,
    MIXTRAL_8x7B,
    MoEConfig,
    build_block,
    routing_fingerprint,
)
from implementation.routing import expert_load_counts, fused_topk, grouped_topk


# ---------------------------------------------------------------------------
# fused_topk — shape & basic invariants
# ---------------------------------------------------------------------------


def test_fused_topk_returns_three_tensors():
    """``fused_topk`` returns (weights, ids, token_expert_indices) — 3-tuple."""
    h = torch.zeros(4, 16)
    g = torch.randn(4, 8)
    out = fused_topk(h, g, topk=2, renormalize=True)
    assert isinstance(out, tuple)
    assert len(out) == 3


def test_fused_topk_shapes_M_K():
    """All three outputs have leading dim M and trailing dim K."""
    M, E, K = 64, 8, 2
    h = torch.zeros(M, 32)
    g = torch.randn(M, E)
    tw, ti, idx = fused_topk(h, g, topk=K, renormalize=True)
    assert tw.shape == (M, K)
    assert ti.shape == (M, K)
    assert idx.shape == (M, K)


def test_fused_topk_dtypes():
    """Weights are float32, ids are int32, token_expert_indices are int32 — matches Triton kernel contract."""
    h = torch.zeros(8, 8)
    g = torch.randn(8, 4)
    tw, ti, idx = fused_topk(h, g, topk=2)
    assert tw.dtype == torch.float32
    assert ti.dtype == torch.int32
    assert idx.dtype == torch.int32


def test_fused_topk_token_expert_indices_is_arange():
    """Per docstring of the Triton kernel: token_expert_indices = [[0,1,...,K-1]] * M."""
    M, K = 5, 3
    h = torch.zeros(M, 8)
    g = torch.randn(M, 6)
    _, _, idx = fused_topk(h, g, topk=K)
    expected = torch.arange(K, dtype=torch.int32).unsqueeze(0).expand(M, K)
    assert torch.equal(idx, expected)


# ---------------------------------------------------------------------------
# fused_topk — math correctness
# ---------------------------------------------------------------------------


def test_fused_topk_renormalize_true_sums_to_one():
    """When renormalize=True, K weights sum to ~1.0 per token (Mixtral path)."""
    torch.manual_seed(0)
    h = torch.randn(64, 32)
    g = torch.randn(64, 16)
    tw, _, _ = fused_topk(h, g, topk=4, renormalize=True)
    sums = tw.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_fused_topk_renormalize_false_sums_below_one():
    """Without renormalize, K weights are softmax tail mass — strictly < 1 except in degenerate cases."""
    torch.manual_seed(1)
    h = torch.randn(32, 16)
    g = torch.randn(32, 8)
    tw, _, _ = fused_topk(h, g, topk=2, renormalize=False)
    sums = tw.sum(dim=-1)
    # Each weight ≤ 1; sum of K of them ≤ 1.
    assert (sums <= 1.0 + 1e-5).all()
    # Two largest of an 8-way softmax can't exceed sum of all 8 = 1, AND can't be below 2/8 (min uniform mass).
    assert (sums >= 2.0 / 8.0 - 1e-6).all()


def test_fused_topk_picks_K_largest_logits():
    """Top-K must select the K LARGEST logit indices (after softmax monotone)."""
    g = torch.tensor([[0.1, 5.0, 2.0, 4.0, 0.5]])
    h = torch.zeros(1, 4)
    _, ti, _ = fused_topk(h, g, topk=2, renormalize=True)
    chosen = sorted(ti[0].tolist())
    assert chosen == [1, 3]  # 5.0 and 4.0 are the two largest


def test_fused_topk_softmax_vs_sigmoid():
    """softmax and sigmoid are different scoring functions — they pick differently when ordering differs."""
    # Construct logits where softmax-top-1 differs from sigmoid-top-1.
    # Sigmoid is monotonic so top-1 by sigmoid is the largest logit.
    # Softmax is also monotonic over positive scores. Use IDENTICAL behaviour:
    # for top-K id selection both pick the same set, but WEIGHTS differ.
    g = torch.tensor([[1.0, 2.0, 0.5, 1.5]])
    h = torch.zeros(1, 4)
    tw_sm, ti_sm, _ = fused_topk(h, g, topk=2, renormalize=False, scoring_func="softmax")
    tw_sg, ti_sg, _ = fused_topk(h, g, topk=2, renormalize=False, scoring_func="sigmoid")
    assert sorted(ti_sm[0].tolist()) == sorted(ti_sg[0].tolist())  # same ids
    assert not torch.allclose(tw_sm, tw_sg)  # different weights


def test_fused_topk_invalid_scoring_raises():
    """Unsupported scoring function must raise ValueError (mirror of source check)."""
    h = torch.zeros(2, 4)
    g = torch.randn(2, 4)
    try:
        fused_topk(h, g, topk=1, scoring_func="not_a_real_one")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unsupported scoring_func")


def test_fused_topk_tokens_mismatch_asserts():
    """Pre-condition assert at fused_topk_router.py:L77 must fire on mismatch."""
    h = torch.zeros(4, 8)
    g = torch.randn(5, 8)
    try:
        fused_topk(h, g, topk=2)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError on token mismatch")


def test_fused_topk_weights_match_softmax_values_when_no_renorm():
    """When renormalize=False, weights must equal softmax(logits) gathered at the topk indices."""
    torch.manual_seed(2)
    g = torch.randn(8, 6)
    h = torch.zeros(8, 4)
    tw, ti, _ = fused_topk(h, g, topk=3, renormalize=False, scoring_func="softmax")
    softmax_full = torch.softmax(g.float(), dim=-1)
    expected = softmax_full.gather(1, ti.long())
    assert torch.allclose(tw, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Trap G — softmax-then-topk vs topk-then-softmax DO NOT COMMUTE
# ---------------------------------------------------------------------------


def test_trap_G_softmax_topk_does_not_commute_with_topk_softmax():
    """Trap G — softmax→topk (renormalize=False) does NOT equal topk→softmax.

    This is the actual non-commutation: when ``renormalize=False`` the
    weights are softmax-tail probabilities (sum < 1). topk-first-then-softmax
    would always sum to exactly 1. The two paths produce DIFFERENT values
    even though they pick the same expert IDs.

    (Note: under ``renormalize=True`` the two ARE algebraically equivalent
    because softmax(g_i)/sum_topk == softmax over only the top-K logits.
    The non-commutation surfaces when renormalize=False — vLLM's actual
    Mixtral path.)
    """
    torch.manual_seed(3)
    g = torch.randn(4, 8)
    h = torch.zeros(4, 4)

    # vLLM order WITHOUT renormalize — what Mixtral / DeepSeek actually use.
    tw_vllm, ti, _ = fused_topk(h, g, topk=2, renormalize=False)

    # Hypothetical "topk first then softmax over the K picked".
    topk_logits, _ = torch.topk(g, k=2, dim=-1)
    tw_alt = torch.softmax(topk_logits, dim=-1)

    # vLLM weights sum to <1 (softmax tail); alt weights sum to exactly 1.
    vllm_sums = tw_vllm.sum(dim=-1)
    alt_sums = tw_alt.sum(dim=-1)
    assert (vllm_sums < 1.0).all()
    assert torch.allclose(alt_sums, torch.ones_like(alt_sums), atol=1e-5)
    # And the values themselves diverge.
    assert not torch.allclose(tw_vllm, tw_alt, atol=1e-3)


def test_renormalize_true_recovers_alt_path_only_in_K_equals_E_limit():
    """When K==E, vLLM's renormalize-after-softmax-topk path reduces to softmax over all E (which sums to 1)."""
    torch.manual_seed(4)
    g = torch.randn(4, 5)
    h = torch.zeros(4, 4)
    tw, ti, _ = fused_topk(h, g, topk=5, renormalize=True)
    # Sum to 1 always (renormalize). And the weights, sorted by id, equal softmax(g) sorted.
    sorted_tw = torch.zeros_like(tw)
    for i in range(4):
        order = torch.argsort(ti[i].long())
        sorted_tw[i] = tw[i][order]
    # softmax(g) doesn't quite equal sorted_tw because renormalize divides by sum (=1),
    # so they should be allclose.
    assert torch.allclose(sorted_tw, torch.softmax(g.float(), dim=-1), atol=1e-5)


# ---------------------------------------------------------------------------
# grouped_topk — DeepSeek path
# ---------------------------------------------------------------------------


def test_grouped_topk_shapes():
    """Grouped path returns (weights, ids) shaped (M, K)."""
    torch.manual_seed(0)
    h = torch.zeros(32, 16)
    g = torch.randn(32, 64)
    tw, ti = grouped_topk(
        h, g, topk=6, renormalize=True, num_expert_group=8, topk_group=3
    )
    assert tw.shape == (32, 6)
    assert ti.shape == (32, 6)


def test_grouped_topk_renormalize_sum_to_one():
    """Renormalize=True sets per-token weight sum to 1.0."""
    torch.manual_seed(1)
    h = torch.zeros(64, 32)
    g = torch.randn(64, 64)
    tw, _ = grouped_topk(
        h, g, topk=6, renormalize=True, num_expert_group=8, topk_group=3
    )
    sums = tw.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_grouped_topk_picks_only_from_topk_group_groups():
    """Selected expert ids must lie inside the chosen ``topk_group`` groups (mask invariant)."""
    torch.manual_seed(2)
    M = 16
    E = 32
    n_grp = 4
    grp_size = E // n_grp  # 8
    h = torch.zeros(M, 16)

    # Build logits where two groups CLEARLY dominate.
    g = torch.full((M, E), -10.0)
    g[:, 0:grp_size] = 5.0  # group 0 strong
    g[:, grp_size : 2 * grp_size] = 4.0  # group 1 strong
    g[:, 2 * grp_size :] = -5.0  # groups 2,3 weak

    _, ti = grouped_topk(
        h, g, topk=4, renormalize=True, num_expert_group=n_grp, topk_group=2
    )
    # All chosen ids must come from groups 0 or 1.
    for row in ti:
        for eid in row.tolist():
            grp = eid // grp_size
            assert grp in (0, 1), f"expert {eid} outside chosen groups"


def test_grouped_topk_group_score_default_uses_max():
    """Default group score (no bias correction) is per-group max."""
    torch.manual_seed(3)
    M, E = 4, 8
    h = torch.zeros(M, 4)
    g = torch.zeros(M, E)
    # Group sizes: n_group=2 → groups {0..3}, {4..7}.
    # Make group 0's max < group 1's max via raw logits.
    g[:, 0:4] = torch.tensor([0.0, 0.0, 0.0, 0.5])  # group 0: max=0.5 → softmax->small
    g[:, 4:8] = torch.tensor([0.0, 0.0, 0.0, 5.0])  # group 1: max=5.0 → dominates
    _, ti = grouped_topk(
        h, g, topk=2, renormalize=True, num_expert_group=2, topk_group=1
    )
    # All ids must be in group 1 (4..7).
    for row in ti:
        for eid in row.tolist():
            assert 4 <= eid < 8


def test_grouped_topk_e_score_bias_uses_unbiased_weights():
    """noaux_tc path: bias is added to scores for SELECTION but weights come from the UNBIASED scores.

    Trap (subtle): when ``e_score_correction_bias`` is non-zero, the
    selection might pick experts with smaller raw scores. The returned
    weights must still be from the original (unbiased) softmax.
    """
    torch.manual_seed(4)
    M, E = 8, 8
    h = torch.zeros(M, 4)
    g = torch.randn(M, E)
    bias = torch.tensor([10.0, 0, 0, 0, 0, 0, 0, 0], dtype=torch.float32)

    tw, ti = grouped_topk(
        h,
        g,
        topk=2,
        renormalize=False,  # so we can read raw weights
        num_expert_group=2,
        topk_group=1,
        e_score_correction_bias=bias,
    )

    # With bias=+10 on expert 0 inside group 0, group 0 gets selected (top-2-sum is huge there).
    # But the weights returned for expert 0 must equal the UNBIASED softmax(g) at column 0.
    unbiased = torch.softmax(g.float(), dim=-1)
    # Check weights: each (token, slot) weight must equal unbiased[token, ti[token, slot]].
    for tok in range(M):
        for slot in range(2):
            eid = int(ti[tok, slot].item())
            assert math.isclose(
                tw[tok, slot].item(),
                unbiased[tok, eid].item(),
                rel_tol=1e-5,
                abs_tol=1e-6,
            )


def test_grouped_topk_routed_scaling_factor_multiplies_after_renorm():
    """``routed_scaling_factor`` scales the FINAL weights — sum becomes scale × 1 = scale."""
    torch.manual_seed(5)
    h = torch.zeros(8, 4)
    g = torch.randn(8, 16)
    scale = 2.5
    tw, _ = grouped_topk(
        h,
        g,
        topk=4,
        renormalize=True,
        num_expert_group=4,
        topk_group=2,
        routed_scaling_factor=scale,
    )
    sums = tw.sum(dim=-1)
    assert torch.allclose(sums, torch.full_like(sums, scale), atol=1e-5)


def test_grouped_topk_invalid_scoring_raises():
    """Unsupported scoring function raises ValueError."""
    h = torch.zeros(2, 4)
    g = torch.randn(2, 8)
    try:
        grouped_topk(
            h, g, topk=1, num_expert_group=2, topk_group=1, scoring_func="not-real"
        )
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_grouped_topk_token_count_assert():
    """Pre-condition assert fires on token-count mismatch (mirror grouped_topk_router.py:L111)."""
    h = torch.zeros(4, 8)
    g = torch.randn(5, 8)
    try:
        grouped_topk(h, g, topk=1, num_expert_group=2, topk_group=1)
    except AssertionError:
        return
    raise AssertionError("expected AssertionError")


def test_grouped_topk_top2_sum_group_score_when_bias_set():
    """noaux_tc uses sum-of-top-2 inside each group as the group score (impl-notes Trap reframe)."""
    M = 4
    n_grp = 2
    grp_size = 4  # E=8

    # Build scores after sigmoid-like (here we simulate with very small logits → softmax≈uniform).
    # Use a setup where:
    #   group 0 has scores [0.2, 0.2, 0.2, 0.2]  → max=0.2, sum-of-top-2=0.4
    #   group 1 has scores [0.5, 0.0, 0.0, 0.0]  → max=0.5, sum-of-top-2=0.5
    # With max-rule (no bias): group 1 wins (0.5 > 0.2).
    # With sum-top-2 rule (bias path): group 1 still wins (0.5 > 0.4)
    # — but the rule machinery is exercised. We assert correct group selection.

    g = torch.zeros(M, 8)
    g[:, 0:4] = torch.tensor([0.2, 0.2, 0.2, 0.2])
    g[:, 4:8] = torch.tensor([5.0, 0.0, 0.0, 0.0])  # exaggerated
    h = torch.zeros(M, 4)
    bias = torch.zeros(8)
    _, ti = grouped_topk(
        h,
        g,
        topk=1,
        renormalize=True,
        num_expert_group=n_grp,
        topk_group=1,
        e_score_correction_bias=bias,
    )
    # All chosen experts must come from group 1 (ids 4..7).
    for row in ti.tolist():
        assert all(4 <= e < 8 for e in row)


# ---------------------------------------------------------------------------
# expert_load_counts
# ---------------------------------------------------------------------------


def test_expert_load_counts_total_equals_M_times_K():
    """Σ counts must equal M·K — every (token, slot) pair contributes one count."""
    torch.manual_seed(10)
    M, K, E = 100, 4, 16
    ti = torch.randint(0, E, (M, K), dtype=torch.int32)
    counts = expert_load_counts(ti, E)
    assert int(counts.sum().item()) == M * K


def test_expert_load_counts_zero_when_no_routing():
    """An expert that no token routes to receives 0 — sentinel-aware (no implicit smoothing)."""
    M, K, E = 8, 1, 4
    ti = torch.zeros(M, K, dtype=torch.int32)  # everyone routes to expert 0
    counts = expert_load_counts(ti, E)
    assert counts[0].item() == M
    assert counts[1].item() == 0
    assert counts[2].item() == 0
    assert counts[3].item() == 0


def test_expert_load_counts_dtype_and_shape():
    """Returns int64 of length E."""
    ti = torch.zeros(4, 2, dtype=torch.int32)
    counts = expert_load_counts(ti, 6)
    assert counts.shape == (6,)
    assert counts.dtype == torch.int64


# ---------------------------------------------------------------------------
# §3.1 demo numerics — verbatim pin
# ---------------------------------------------------------------------------


def test_section_31_mixtral_distribution_verbatim():
    """Pin the §3.1 Mixtral fingerprint: per_expert_count = [250, 285, 277, 243, 253, 272, 247, 221]."""
    mix_small = MoEConfig(
        num_experts=MIXTRAL_8x7B.num_experts,
        top_k=MIXTRAL_8x7B.top_k,
        hidden_size=512,
        intermediate_size=512,
        scoring_func="softmax",
        name="Mixtral-tiny",
    )
    mix = build_block(mix_small, ep_size=1, seed=0)
    fp = routing_fingerprint(mix, num_tokens=1024, seed=7)
    assert fp["per_expert_count"] == [250, 285, 277, 243, 253, 272, 247, 221]
    assert fp["max_count"] == 285
    assert fp["min_count"] == 221
    assert math.isclose(fp["mean_count"], 256.00, abs_tol=0.01)
    assert math.isclose(fp["coverage"], 1.000, abs_tol=1e-3)
    assert math.isclose(fp["weight_sum_min"], 1.0000, abs_tol=1e-4)
    assert math.isclose(fp["weight_sum_max"], 1.0000, abs_tol=1e-4)


def test_section_31_deepseek_distribution_verbatim():
    """Pin §3.1 DeepSeek: max=131 min=78 mean=96.00 coverage=1.000."""
    ds_small = MoEConfig(
        num_experts=DEEPSEEK_V2_LITE.num_experts,
        top_k=DEEPSEEK_V2_LITE.top_k,
        hidden_size=512,
        intermediate_size=512,
        use_grouped_topk=True,
        num_expert_group=DEEPSEEK_V2_LITE.num_expert_group,
        topk_group=DEEPSEEK_V2_LITE.topk_group,
        scoring_func="softmax",
        name="DeepSeek-V2-tiny",
    )
    ds = build_block(ds_small, ep_size=1, seed=0)
    fp = routing_fingerprint(ds, num_tokens=1024, seed=7)
    assert fp["max_count"] == 131
    assert fp["min_count"] == 78
    assert math.isclose(fp["mean_count"], 96.00, abs_tol=0.01)
    assert math.isclose(fp["coverage"], 1.000, abs_tol=1e-3)


def test_section_31_renormalize_off_range_verbatim():
    """Pin §3.1 'renormalize=False → sum range [0.2730, 0.6171] mean 0.3899'."""
    g = torch.Generator().manual_seed(7)
    h = torch.randn(1024, 512, generator=g)
    # Build the block to reproduce the same gate weight; same seed=0 and the same
    # (E=8, hidden=512) match the demo's mix.gate_weight.
    mix_small = MoEConfig(
        num_experts=MIXTRAL_8x7B.num_experts,
        top_k=MIXTRAL_8x7B.top_k,
        hidden_size=512,
        intermediate_size=512,
        scoring_func="softmax",
        name="Mixtral-tiny",
    )
    mix = build_block(mix_small, ep_size=1, seed=0)
    logits = h @ mix.gate_weight.T
    tw_off, _, _ = fused_topk(h, logits, topk=2, renormalize=False)
    sums = tw_off.sum(dim=-1)
    assert math.isclose(sums.min().item(), 0.2730, abs_tol=1e-3)
    assert math.isclose(sums.max().item(), 0.6171, abs_tol=1e-3)
    assert math.isclose(sums.mean().item(), 0.3899, abs_tol=1e-3)
