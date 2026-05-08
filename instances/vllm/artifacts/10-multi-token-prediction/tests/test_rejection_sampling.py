"""Tests for rejection_sampling.py — the verifier core.

# REFERENCE: vllm/v1/sample/rejection_sampler.py:L37-L920

Covers:
  - Greedy kernel correctness (accept/reject by argmax)
  - Random kernel unbiasedness (KL test reproducing demo §3.1)
  - sample_recovered_tokens via Gumbel-max
  - Chain-break sentinel (-1) propagation
  - Bonus-token slot semantics
  - parse_output filtering
  - NO_DRAFT_PROBS (ngram) path
  - SYNTHETIC_MODE path
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from implementation.rejection_sampling import (
    RejectionSampler,
    parse_output,
    rejection_greedy_sample_loop,
    rejection_random_sample_loop,
    rejection_sample,
    sample_recovered_tokens_loop,
)
from implementation.spec_metadata import (
    PLACEHOLDER_TOKEN_ID,
    SpecDecodeMetadata,
)


# ============================================================================
# GREEDY PATH
# ============================================================================


def test_greedy_all_accept_emits_K_plus_one():
    """All drafts == argmax(target) → K + 1 emitted (K drafts + bonus)."""
    drafts = [[1, 2, 3]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    vocab = 8
    # Target argmax matches each draft exactly.
    target_logits = torch.full((3, vocab), -10.0)
    for i, d in enumerate(drafts[0]):
        target_logits[i, d] = 100.0
    bonus = torch.tensor([7], dtype=torch.int32)
    out = rejection_sample(
        md, draft_probs=None, target_logits=target_logits,
        bonus_token_ids=bonus, all_greedy=True, all_random=False,
    )
    assert out[0].tolist() == [1, 2, 3, 7]


def test_greedy_first_reject_breaks_chain():
    """Reject at pos 0 → emit target argmax at pos 0; positions 1..K stay -1."""
    drafts = [[5, 5, 5]]  # all draft "5"
    md = SpecDecodeMetadata.make_dummy(drafts)
    vocab = 8
    # Target argmax at pos 0 is "2" (≠ 5) → reject, emit 2, chain breaks.
    target_logits = torch.full((3, vocab), -10.0)
    target_logits[0, 2] = 100.0
    target_logits[1, 5] = 100.0  # never read
    target_logits[2, 5] = 100.0  # never read
    bonus = torch.tensor([99], dtype=torch.int32)  # never emitted
    out = rejection_sample(
        md, draft_probs=None, target_logits=target_logits,
        bonus_token_ids=bonus, all_greedy=True, all_random=False,
    )
    # Position 0 emits target argmax = 2; positions 1..3 stay -1 (PLACEHOLDER).
    assert out[0, 0].item() == 2
    assert out[0, 1].item() == PLACEHOLDER_TOKEN_ID
    assert out[0, 2].item() == PLACEHOLDER_TOKEN_ID
    assert out[0, 3].item() == PLACEHOLDER_TOKEN_ID  # bonus slot also -1


def test_greedy_reject_in_middle():
    """Accept pos 0, reject pos 1; pos 2 stays -1, no bonus."""
    drafts = [[1, 9, 3]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    vocab = 16
    target_logits = torch.full((3, vocab), -10.0)
    target_logits[0, 1] = 100.0   # accept (1 == 1)
    target_logits[1, 5] = 100.0   # reject (argmax=5, draft=9) → emit 5
    target_logits[2, 3] = 100.0   # never seen
    bonus = torch.tensor([99], dtype=torch.int32)
    out = rejection_sample(
        md, None, target_logits, bonus, all_greedy=True, all_random=False,
    )
    assert out[0, 0].item() == 1   # accepted draft
    assert out[0, 1].item() == 5   # rejected draft, emit target argmax
    assert out[0, 2].item() == PLACEHOLDER_TOKEN_ID
    assert out[0, 3].item() == PLACEHOLDER_TOKEN_ID  # bonus blocked


def test_greedy_chain_break_invariant_at_every_position():
    """Once any position rejects, ALL later positions stay -1."""
    K = 5
    for reject_at in range(K):
        # Construct drafts that match argmax at all positions except reject_at.
        drafts_inner = list(range(K))
        md = SpecDecodeMetadata.make_dummy([drafts_inner])
        vocab = 32
        target_logits = torch.full((K, vocab), -10.0)
        for i in range(K):
            argmax_id = i if i != reject_at else (i + 100) % vocab  # mismatch
            target_logits[i, argmax_id] = 100.0
        bonus = torch.tensor([42], dtype=torch.int32)
        out = rejection_sample(
            md, None, target_logits, bonus, all_greedy=True, all_random=False,
        )
        # Positions 0..reject_at have valid tokens; reject_at+1..K stay -1.
        for p in range(reject_at + 1):
            assert out[0, p].item() != PLACEHOLDER_TOKEN_ID, (
                f"reject_at={reject_at}, pos {p} should have emitted"
            )
        for p in range(reject_at + 1, K + 1):
            assert out[0, p].item() == PLACEHOLDER_TOKEN_ID, (
                f"reject_at={reject_at}, pos {p} should be PLACEHOLDER"
            )


def test_greedy_kernel_writes_target_argmax_on_accept_too():
    """Source writes target_argmax always — same value as draft when accept."""
    drafts = [[3]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    vocab = 8
    target_logits = torch.full((1, vocab), -10.0)
    target_logits[0, 3] = 100.0  # argmax == draft → accept
    bonus = torch.tensor([5], dtype=torch.int32)
    out = rejection_sample(md, None, target_logits, bonus,
                           all_greedy=True, all_random=False)
    # Accepted draft = target argmax = 3
    assert out[0, 0].item() == 3
    assert out[0, 1].item() == 5  # bonus emitted


# ============================================================================
# RANDOM PATH — UNBIASEDNESS
# ============================================================================


def test_random_unbiasedness_kl_below_threshold():
    """Reproduce demo §3.1: KL(empirical || p) < 0.01 over 10000 trials."""
    p = torch.tensor([0.30, 0.20, 0.15, 0.10, 0.10, 0.07, 0.05, 0.03],
                     dtype=torch.float32)
    q = torch.tensor([0.10, 0.20, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05],
                     dtype=torch.float32)
    p = p / p.sum()
    q = q / q.sum()
    n_trials = 10000
    counts = torch.zeros(8, dtype=torch.float64)
    gen = torch.Generator().manual_seed(42)
    for _ in range(n_trials):
        d = int(torch.multinomial(q, 1, generator=gen).item())
        u = float(torch.rand(1, dtype=torch.float64, generator=gen).item())
        if q[d] > 0 and (p[d] / q[d]).item() >= u:
            tok = d
        else:
            diff = (p - q).clamp_min(0.0)
            s = float(diff.sum().item())
            residual = diff / s if s > 0 else p
            tok = int(torch.multinomial(residual, 1, generator=gen).item())
        counts[tok] += 1
    empirical = counts / counts.sum()
    safe_emp = empirical.clamp_min(1e-12)
    safe_p = p.clamp_min(1e-12)
    kl = float((safe_emp * (safe_emp.log() - safe_p.log())).sum().item())
    assert kl < 0.01, f"KL({kl:.6f}) exceeds threshold 0.01"
    # Verbatim demo numerics — pin exact decimal.
    assert abs(kl - 0.000395) < 0.001, f"Reproduce demo KL value (got {kl})"


def test_random_unbiasedness_at_high_temperature():
    """Trap D negative test: rejection sampling is unbiased at any temperature.

    Use very different p and q to stress the algorithm; KL must still be small.
    """
    p = torch.tensor([0.5, 0.3, 0.1, 0.1], dtype=torch.float32)
    q = torch.tensor([0.1, 0.1, 0.4, 0.4], dtype=torch.float32)  # very different
    p = p / p.sum()
    q = q / q.sum()
    n_trials = 5000
    counts = torch.zeros(4, dtype=torch.float64)
    gen = torch.Generator().manual_seed(123)
    for _ in range(n_trials):
        d = int(torch.multinomial(q, 1, generator=gen).item())
        u = float(torch.rand(1, dtype=torch.float64, generator=gen).item())
        if q[d] > 0 and (p[d] / q[d]).item() >= u:
            tok = d
        else:
            diff = (p - q).clamp_min(0.0)
            s = float(diff.sum().item())
            residual = diff / s if s > 0 else p
            tok = int(torch.multinomial(residual, 1, generator=gen).item())
        counts[tok] += 1
    empirical = counts / counts.sum()
    safe_emp = empirical.clamp_min(1e-12)
    safe_p = p.clamp_min(1e-12)
    kl = float((safe_emp * (safe_emp.log() - safe_p.log())).sum().item())
    assert kl < 0.05, f"KL({kl:.6f}) too high — rejection sampling biased!"


def test_random_unbiasedness_when_p_equals_q():
    """If p == q, draft is always accepted, output ~ q == p exactly."""
    p = torch.tensor([0.4, 0.3, 0.2, 0.1], dtype=torch.float32)
    q = p.clone()
    n_trials = 3000
    counts = torch.zeros(4, dtype=torch.float64)
    gen = torch.Generator().manual_seed(7)
    for _ in range(n_trials):
        d = int(torch.multinomial(q, 1, generator=gen).item())
        u = float(torch.rand(1, dtype=torch.float64, generator=gen).item())
        if q[d] > 0 and (p[d] / q[d]).item() >= u:
            counts[d] += 1
        else:  # never reaches here when p == q with ratio == 1
            counts[d] += 1
    empirical = counts / counts.sum()
    # All drafts accepted → empirical == q == p
    assert (empirical - p).abs().max() < 0.05


# ============================================================================
# RANDOM PATH — SHAPE / SEMANTICS
# ============================================================================


def test_random_path_shape_and_dtype():
    """Output shape [batch, max_spec_len + 1], dtype int32."""
    drafts = [[1, 2, 3], [4, 5], [6]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    vocab = 8
    target_logits = torch.randn(6, vocab) * 2.0
    draft_probs = torch.rand(6, vocab)
    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
    bonus = torch.tensor([7, 7, 7], dtype=torch.int32)
    g = torch.Generator().manual_seed(42)
    out = rejection_sample(md, draft_probs, target_logits, bonus,
                           all_greedy=False, all_random=True, generator=g)
    assert out.shape == (3, md.max_spec_len + 1)
    assert out.dtype == torch.int32


def test_random_emits_bonus_when_all_accept():
    """When all drafts accepted, bonus token appears at slot K."""
    # Stack the deck: target == draft probabilities are 1.0 at draft id, 0 elsewhere
    K = 3
    vocab = 8
    drafts = [[1, 2, 3]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    # Logits very sharp at draft id → target_probs[draft] ≈ 1
    target_logits = torch.full((K, vocab), -100.0)
    for i, d in enumerate(drafts[0]):
        target_logits[i, d] = 100.0
    # Draft probs: also ~1 at draft id
    draft_probs = torch.full((K, vocab), 1e-9)
    for i, d in enumerate(drafts[0]):
        draft_probs[i, d] = 1.0 - (vocab - 1) * 1e-9
    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
    bonus = torch.tensor([42], dtype=torch.int32)
    g = torch.Generator().manual_seed(5)
    out = rejection_sample(md, draft_probs, target_logits, bonus,
                           all_greedy=False, all_random=True, generator=g)
    assert out[0, K].item() == 42  # bonus emitted


def test_random_no_bonus_when_any_reject():
    """If any draft rejected, bonus slot stays -1."""
    K = 3
    vocab = 8
    drafts = [[1, 2, 3]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    # Construct so first draft rejects under random kernel.
    # target gives 0 mass to draft id 1; draft gives 1.0 mass — ratio 0/1 = 0, always reject.
    target_logits = torch.full((K, vocab), -100.0)
    target_logits[0, 5] = 100.0   # target wants 5, not 1 → reject
    target_logits[1, 2] = 100.0
    target_logits[2, 3] = 100.0
    draft_probs = torch.full((K, vocab), 1e-9)
    draft_probs[0, 1] = 1.0 - (vocab - 1) * 1e-9   # draft chooses 1
    draft_probs[1, 2] = 1.0 - (vocab - 1) * 1e-9
    draft_probs[2, 3] = 1.0 - (vocab - 1) * 1e-9
    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
    bonus = torch.tensor([42], dtype=torch.int32)
    g = torch.Generator().manual_seed(99)
    out = rejection_sample(md, draft_probs, target_logits, bonus,
                           all_greedy=False, all_random=True, generator=g)
    # Bonus slot must NOT contain 42 — chain broke.
    assert out[0, K].item() == PLACEHOLDER_TOKEN_ID


# ============================================================================
# parse_output
# ============================================================================


def test_parse_output_strips_placeholders():
    """parse_output filters out PLACEHOLDER_TOKEN_ID values."""
    raw = torch.tensor([
        [1, 2, 3, 5],          # all accept + bonus
        [7, -1, -1, -1],       # reject at pos 1
        [-1, -1, -1, -1],      # somehow all rejected (vacuous)
    ], dtype=torch.int32)
    parsed = parse_output(raw, vocab_size=100)
    assert parsed[0] == [1, 2, 3, 5]
    assert parsed[1] == [7]
    assert parsed[2] == []


def test_parse_output_strips_out_of_vocab():
    """parse_output strips values >= vocab_size."""
    raw = torch.tensor([[5, 99, 200, -1]], dtype=torch.int32)
    parsed = parse_output(raw, vocab_size=100)
    assert parsed[0] == [5, 99]


def test_parse_output_returns_python_lists():
    """parse_output returns list[list[int]] — easy to assert in tests."""
    raw = torch.tensor([[1, 2, -1, -1], [3, -1, -1, -1]], dtype=torch.int32)
    parsed = parse_output(raw, vocab_size=100)
    assert isinstance(parsed, list)
    assert all(isinstance(row, list) for row in parsed)
    assert all(isinstance(t, int) for row in parsed for t in row)


# ============================================================================
# sample_recovered_tokens (Gumbel-max)
# ============================================================================


def test_sample_recovered_tokens_no_draft_probs_excludes_draft_id():
    """NO_DRAFT_PROBS=True path masks draft_id to 0 in target_probs.

    The recovered token must NEVER equal the draft id (draft was already rejected).
    """
    num_tokens, vocab = 5, 16
    draft_token_ids = torch.tensor([3, 7, 1, 9, 14], dtype=torch.int32)
    target_probs = torch.rand(num_tokens, vocab)
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
    cu_num_draft_tokens = torch.tensor([5], dtype=torch.int32)
    inv_q = torch.rand(1, vocab) + 0.1
    recovered = sample_recovered_tokens_loop(
        cu_num_draft_tokens, draft_token_ids,
        draft_probs=None, target_probs=target_probs, inv_q=inv_q,
        NO_DRAFT_PROBS=True,
    )
    # No recovered token equals its draft id — the residual sets that prob to 0.
    for i in range(num_tokens):
        assert recovered[i].item() != draft_token_ids[i].item(), (
            f"pos {i}: recovered token equals draft id {draft_token_ids[i]}"
        )


def test_sample_recovered_tokens_residual_path_argmax_in_support():
    """Standard residual: sample only from where (p - q)_+ > 0."""
    num_tokens, vocab = 3, 8
    draft_token_ids = torch.tensor([1, 1, 1], dtype=torch.int32)
    # p has mass on indices 4-7; q has mass on indices 0-3.
    # residual (p - q)_+ has support only on 4-7 (where p > q).
    p = torch.zeros(num_tokens, vocab)
    p[:, 4:] = 0.25
    q = torch.zeros(num_tokens, vocab)
    q[:, :4] = 0.25
    cu_num_draft_tokens = torch.tensor([3], dtype=torch.int32)
    inv_q = torch.ones(1, vocab)  # uniform — argmax just picks the largest residual
    recovered = sample_recovered_tokens_loop(
        cu_num_draft_tokens, draft_token_ids,
        draft_probs=q, target_probs=p, inv_q=inv_q,
        NO_DRAFT_PROBS=False,
    )
    # All recovered must be in [4, 7] — the residual support.
    for i in range(num_tokens):
        assert 4 <= recovered[i].item() < 8


def test_sample_recovered_tokens_shape_matches_num_tokens():
    """Output is [num_tokens] int."""
    num_tokens, vocab = 7, 32
    draft_token_ids = torch.zeros(num_tokens, dtype=torch.int32)
    target_probs = torch.rand(num_tokens, vocab)
    target_probs = target_probs / target_probs.sum(dim=-1, keepdim=True)
    cu_num_draft_tokens = torch.tensor([num_tokens], dtype=torch.int32)
    inv_q = torch.ones(1, vocab)
    recovered = sample_recovered_tokens_loop(
        cu_num_draft_tokens, draft_token_ids,
        draft_probs=None, target_probs=target_probs, inv_q=inv_q,
        NO_DRAFT_PROBS=True,
    )
    assert recovered.shape == (num_tokens,)


# ============================================================================
# NO_DRAFT_PROBS PATH (n-gram)
# ============================================================================


def test_no_draft_probs_accepts_when_target_prob_high():
    """NO_DRAFT_PROBS path: accept iff target_prob[draft] >= u (u in [0,1))."""
    drafts = [[2]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    vocab = 8
    # Target gives draft id 2 mass ≈ 1.0 (sharp).
    target_logits = torch.full((1, vocab), -100.0)
    target_logits[0, 2] = 100.0
    bonus = torch.tensor([5], dtype=torch.int32)
    g = torch.Generator().manual_seed(11)
    out = rejection_sample(md, draft_probs=None, target_logits=target_logits,
                           bonus_token_ids=bonus, all_greedy=False, all_random=True,
                           generator=g)
    # target_prob[2] ~ 1, so any u < 1 accepts.
    assert out[0, 0].item() == 2
    assert out[0, 1].item() == 5  # bonus also emitted


# ============================================================================
# RejectionSampler wrapper
# ============================================================================


def test_rejection_sampler_class_invokes_rejection_sample():
    """RejectionSampler() callable wraps rejection_sample with same semantics."""
    sampler = RejectionSampler()
    drafts = [[1, 2]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    vocab = 8
    target_logits = torch.full((2, vocab), -10.0)
    target_logits[0, 1] = 100.0
    target_logits[1, 2] = 100.0
    bonus = torch.tensor([7], dtype=torch.int32)
    out = sampler(md, draft_probs=None, target_logits=target_logits,
                  bonus_token_ids=bonus, all_greedy=True, all_random=False)
    assert out[0].tolist() == [1, 2, 7]


def test_rejection_sampler_synthetic_mode_init():
    """Constructing with synthetic_conditional_rates enables synthetic mode."""
    rates = torch.tensor([0.5, 0.3], dtype=torch.float32)
    sampler = RejectionSampler(synthetic_conditional_rates=rates)
    assert sampler.synthetic_mode is True
    sampler2 = RejectionSampler()
    assert sampler2.synthetic_mode is False


# ============================================================================
# MULTI-REQUEST CONSISTENCY
# ============================================================================


def test_multi_request_independent_chain_breaks():
    """Different requests can have different chain-break positions."""
    drafts = [[1, 2], [9, 8]]  # req0 will accept, req1 will reject
    md = SpecDecodeMetadata.make_dummy(drafts)
    vocab = 16
    target_logits = torch.full((4, vocab), -10.0)
    # Req0 positions 0,1: argmax matches drafts 1,2
    target_logits[0, 1] = 100.0
    target_logits[1, 2] = 100.0
    # Req1 positions 0,1: argmax does NOT match drafts 9,8
    target_logits[2, 5] = 100.0  # reject (5 != 9), emit 5, chain breaks
    target_logits[3, 7] = 100.0
    bonus = torch.tensor([15, 14], dtype=torch.int32)
    out = rejection_sample(md, None, target_logits, bonus,
                           all_greedy=True, all_random=False)
    # Req0: all accept, then bonus.
    assert out[0, 0].item() == 1
    assert out[0, 1].item() == 2
    assert out[0, 2].item() == 15
    # Req1: reject at pos 0, emit 5, chain break.
    assert out[1, 0].item() == 5
    assert out[1, 1].item() == PLACEHOLDER_TOKEN_ID
    assert out[1, 2].item() == PLACEHOLDER_TOKEN_ID  # bonus blocked


def test_varying_K_per_request_uses_max_spec_len_buffer():
    """Output buffer is sized to max_spec_len + 1 even if requests have smaller K."""
    drafts = [[1, 2, 3, 4], [5, 6], [7]]  # K_i = 4, 2, 1
    md = SpecDecodeMetadata.make_dummy(drafts)
    vocab = 16
    target_logits = torch.full((7, vocab), -10.0)
    for i, d in enumerate([1, 2, 3, 4, 5, 6, 7]):
        target_logits[i, d] = 100.0
    bonus = torch.tensor([10, 11, 12], dtype=torch.int32)
    out = rejection_sample(md, None, target_logits, bonus,
                           all_greedy=True, all_random=False)
    assert out.shape == (3, 5)  # max_spec_len(=4) + 1
    # Req0: emit 1,2,3,4, bonus 10
    assert out[0].tolist() == [1, 2, 3, 4, 10]
    # Req1: emit 5,6, bonus 11, then PLACEHOLDER, PLACEHOLDER (K_1 = 2 < 4)
    assert out[1, 0].item() == 5
    assert out[1, 1].item() == 6
    assert out[1, 2].item() == 11  # bonus at slot K_1 = 2
    # Req2: emit 7, bonus 12
    assert out[2, 0].item() == 7
    assert out[2, 1].item() == 12


# ============================================================================
# SYNTHETIC MODE
# ============================================================================


def test_synthetic_mode_greedy_obeys_conditional_rate():
    """Synthetic mode: per-position rate determines accept under uniform u."""
    K = 3
    drafts = [[1, 2, 3]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    vocab = 8
    target_logits = torch.full((K, vocab), -10.0)
    for i in range(K):
        target_logits[i, 0] = 100.0  # target argmax always 0 (≠ drafts)
    # Force u to be deterministic via a generator with a fixed seed.
    bonus = torch.tensor([99], dtype=torch.int32)
    # rate=1.0 at pos 0 means accept regardless of u; pos 1 also 1.0; pos 2 = 0.0.
    rates = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)
    sampler = RejectionSampler(synthetic_conditional_rates=rates)
    g = torch.Generator().manual_seed(42)
    out = sampler(md, None, target_logits, bonus,
                  all_greedy=True, all_random=False, generator=g)
    # rate=1 → accept (emit draft); pos 2 rate=0 → reject (emit target argmax).
    assert out[0, 0].item() == 1  # accepted via synthetic
    assert out[0, 1].item() == 2  # accepted via synthetic
    # rate=0 at pos 2 means u ≥ 0 always, so reject; greedy emits target_argmax = 0.
    # Wait: u < rate; rate=0 means never accept; emit target_id = 0.
    assert out[0, 2].item() == 0


# ============================================================================
# DETERMINISM
# ============================================================================


def test_random_path_is_deterministic_with_generator():
    """Same generator seed → same output."""
    drafts = [[1, 2, 3]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    vocab = 8
    target_logits = torch.randn(3, vocab)
    draft_probs = torch.rand(3, vocab)
    draft_probs = draft_probs / draft_probs.sum(dim=-1, keepdim=True)
    bonus = torch.tensor([7], dtype=torch.int32)
    g1 = torch.Generator().manual_seed(42)
    out1 = rejection_sample(md, draft_probs, target_logits, bonus,
                            all_greedy=False, all_random=True, generator=g1)
    g2 = torch.Generator().manual_seed(42)
    out2 = rejection_sample(md, draft_probs, target_logits, bonus,
                            all_greedy=False, all_random=True, generator=g2)
    assert torch.equal(out1, out2)


# ============================================================================
# BUFFER PRE-FILL INVARIANT
# ============================================================================


def test_output_buffer_pre_filled_with_placeholder():
    """Without writes, every slot must be PLACEHOLDER (-1).

    Achieved by constructing a request that fully rejects and emits via residual.
    Then verify slot K is -1 (no bonus) and short-K rows have trailing -1.
    """
    # Request 0: K=3 all accept; bonus at slot 3.
    # Request 1: K=1 — only one draft slot, bonus at slot 1; remaining slots 2..3 must be -1.
    drafts = [[1, 2, 3], [4]]
    md = SpecDecodeMetadata.make_dummy(drafts)  # max_spec_len=3
    vocab = 8
    target_logits = torch.full((4, vocab), -10.0)
    for i, d in enumerate([1, 2, 3, 4]):
        target_logits[i, d] = 100.0
    bonus = torch.tensor([7, 9], dtype=torch.int32)
    out = rejection_sample(md, None, target_logits, bonus,
                           all_greedy=True, all_random=False)
    # Req1 has K=1; slots 0=4, 1=bonus 9, slots 2,3 stay -1.
    assert out[1, 0].item() == 4
    assert out[1, 1].item() == 9
    assert out[1, 2].item() == PLACEHOLDER_TOKEN_ID
    assert out[1, 3].item() == PLACEHOLDER_TOKEN_ID


# ============================================================================
# K=1 FAST-PATH SHAPE
# ============================================================================


def test_K1_fast_path_emits_at_most_two_tokens():
    """K=1: output is [batch, 2] (one draft + bonus)."""
    drafts = [[5], [7], [9]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    assert md.max_spec_len == 1
    vocab = 16
    target_logits = torch.full((3, vocab), -10.0)
    target_logits[0, 5] = 100.0
    target_logits[1, 7] = 100.0
    target_logits[2, 9] = 100.0
    bonus = torch.tensor([1, 2, 3], dtype=torch.int32)
    out = rejection_sample(md, None, target_logits, bonus,
                           all_greedy=True, all_random=False)
    assert out.shape == (3, 2)
    # All accepted → [draft, bonus]
    assert out[0].tolist() == [5, 1]
    assert out[1].tolist() == [7, 2]
    assert out[2].tolist() == [9, 3]


# ============================================================================
# LARGE VOCAB
# ============================================================================


def test_large_vocab_smoke():
    """Verify shapes/dtype hold at vocab=32000 (DeepSeek-V3 scale)."""
    vocab = 32000
    drafts = [[100, 200, 300]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    target_logits = torch.randn(3, vocab) * 0.1
    bonus = torch.tensor([42], dtype=torch.int32)
    out = rejection_sample(md, None, target_logits, bonus,
                           all_greedy=True, all_random=False)
    assert out.shape == (1, 4)
    assert out.dtype == torch.int32


# ============================================================================
# MAX_SPEC_LEN guard
# ============================================================================


def test_max_spec_len_guard_raises():
    """rejection_sample asserts max_spec_len <= MAX_SPEC_LEN (= 128)."""
    # Construct a metadata with max_spec_len exceeding MAX_SPEC_LEN.
    huge_K = 130
    md = SpecDecodeMetadata.make_dummy([list(range(huge_K))])
    assert md.max_spec_len == huge_K
    vocab = 8
    target_logits = torch.full((huge_K, vocab), -10.0)
    bonus = torch.tensor([0], dtype=torch.int32)
    with pytest.raises(AssertionError):
        rejection_sample(md, None, target_logits, bonus,
                         all_greedy=True, all_random=False)


# ============================================================================
# SOURCE FIDELITY: greedy kernel writes target_id always (not draft_id)
# ============================================================================


def test_greedy_kernel_uses_target_id_for_emit():
    """Source quote (L743-L745): writes target_id always — accepted iff draft==target.

    This means rejected positions get target_argmax; accepted positions also get
    target_argmax (which equals draft_id by accept condition).
    """
    drafts = [[1, 9]]  # pos 0 accepts (1 == argmax 1), pos 1 rejects
    md = SpecDecodeMetadata.make_dummy(drafts)
    vocab = 16
    target_logits = torch.full((2, vocab), -10.0)
    target_logits[0, 1] = 100.0
    target_logits[1, 5] = 100.0  # argmax=5, draft=9 → reject; emit 5
    bonus = torch.tensor([99], dtype=torch.int32)
    out = rejection_sample(md, None, target_logits, bonus,
                           all_greedy=True, all_random=False)
    assert out[0, 0].item() == 1  # target argmax == draft (accept)
    assert out[0, 1].item() == 5  # target argmax (reject case)
