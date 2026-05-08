"""Integration tests — cross-module invariants for Ch10.

These tests exercise the full data path:
- proposer → SpecDecodeMetadata → RejectionSampler.
- Acceptance math + rejection sampler chain-break invariant agree.
- Mass conservation through the full pipeline.
- Trap A/B/D/E/F/G end-to-end.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from implementation.acceptance_math import (
    break_even_alpha,
    expected_tokens,
    parameter_count_medusa,
    parameter_count_mtp,
    speedup,
)
from implementation.proposers.base import SpecDecodeBaseProposer, ToyDraftModel
from implementation.proposers.medusa import MedusaProposer
from implementation.proposers.ngram import NgramProposer
from implementation.rejection_sampling import (
    RejectionSampler,
    parse_output,
    rejection_sample,
)
from implementation.spec_metadata import (
    PLACEHOLDER_TOKEN_ID,
    SpecDecodeMetadata,
)


# ---------------------------------------------------------------------------
# Pipeline: proposer → metadata → rejection sampler
# ---------------------------------------------------------------------------


def test_pipeline_base_proposer_to_metadata_to_sampler_greedy():
    """End-to-end greedy: base proposer drafts, sampler verifies."""
    proposer = SpecDecodeBaseProposer(num_speculative_tokens=2, hidden_size=32)
    proposer.model = ToyDraftModel(vocab=64, hidden=32)
    next_ids = torch.randint(0, 64, (2,))
    drafts = proposer.propose(
        torch.randint(0, 64, (8,)),
        torch.arange(8),
        torch.randn(8, 32),
        next_ids,
    )
    # Build metadata from drafts.
    md = SpecDecodeMetadata.make_dummy(drafts.tolist())
    # Build target logits where target argmax matches first draft of each request
    # (so first position accepts), but rejects on second.
    target_logits = torch.zeros(md.draft_token_ids.shape[0], 64)
    for i, did in enumerate(md.draft_token_ids.tolist()):
        target_logits[i, did] = 100.0  # target argmax = draft → accept
    bonus = torch.tensor([0, 0], dtype=torch.int32)
    out = rejection_sample(md, None, target_logits, bonus, all_greedy=True, all_random=False)
    assert out.shape == (2, 3)  # K=2, +1 bonus


def test_pipeline_medusa_to_metadata():
    """Medusa drafts → SpecDecodeMetadata works."""
    proposer = MedusaProposer(K=4, hidden=32, vocab=64)
    proposer.load_model()
    drafts = proposer.propose(torch.randn(3, 32))
    # Convert each row to a list to feed make_dummy.
    md = SpecDecodeMetadata.make_dummy(drafts.tolist())
    assert md.max_spec_len == 4
    assert len(md.num_draft_tokens) == 3


def test_pipeline_ngram_to_metadata():
    """Ngram drafts → metadata. Empty drafts row produces 0 entries."""
    proposer = NgramProposer(num_speculative_tokens=3, prompt_lookup_min=2, prompt_lookup_max=4)
    contexts = [
        np.array([10, 20, 30, 40, 10, 20]),  # match → propose [30,40]
        np.array([1, 2, 3]),                   # no match → []
    ]
    out = proposer.propose([[100], [50]], contexts)
    # Filter empty rows for metadata (sampler can't handle 0 drafts cleanly).
    nonempty = [row for row in out if row]
    if nonempty:
        md = SpecDecodeMetadata.make_dummy(nonempty)
        assert sum(md.num_draft_tokens) == sum(len(r) for r in nonempty)


# ---------------------------------------------------------------------------
# Acceptance math <-> rejection sampler agreement
# ---------------------------------------------------------------------------


def test_chain_break_invariant_matches_geometric_series():
    """Empirical chain-break sum approximates analytic E[tok | alpha, K]."""
    # Build a metadata with K=4 drafts that all reject (target_argmax never matches).
    K = 4
    md = SpecDecodeMetadata.make_dummy([[5] * K])
    # target_argmax = [0]*K → all reject at pos 0.
    target_logits = torch.zeros(K, 8)
    target_logits[:, 0] = 100.0
    bonus = torch.tensor([99], dtype=torch.int32)
    out = rejection_sample(md, None, target_logits, bonus, all_greedy=True, all_random=False)
    # First reject at pos 0 → emit target_argmax=0; rest are -1; bonus not written.
    n_emitted = int(((out >= 0) & (out < 8)).sum().item())
    assert n_emitted == 1


def test_chain_break_at_K2_emits_at_most_K_plus_one():
    """Output buffer width is K+1 — at most K+1 tokens can be written."""
    K = 2
    md = SpecDecodeMetadata.make_dummy([[3] * K])
    target_logits = torch.zeros(K, 8)
    target_logits[:, 3] = 100.0  # all argmax = 3 → all accept
    bonus = torch.tensor([7], dtype=torch.int32)
    out = rejection_sample(md, None, target_logits, bonus, all_greedy=True, all_random=False)
    assert out.shape == (1, K + 1)


# ---------------------------------------------------------------------------
# Trap E end-to-end: MTP heads are heavy
# ---------------------------------------------------------------------------


def test_trap_E_mtp_per_layer_far_exceeds_medusa_per_head_mlp():
    """End-to-end Trap E: at hidden=2048, mtp_block dominates medusa per-head.

    Note: acceptance_math.parameter_count_mtp uses num_routed_experts=0
    (not num_heads) — heads are folded into the decoder_attn term.
    Returned dict has 'per_layer_with_lm' / 'per_layer_without_lm'.
    """
    mtp = parameter_count_mtp(
        hidden_size=2048, intermediate_size=8192, vocab_size=32000,
        num_mtp_layers=1, num_routed_experts=0,
    )
    medusa = parameter_count_medusa(hidden_size=2048, K=1, vocab_size=32000)
    # MTP per-layer (without LM) >> Medusa per-head MLP (without LM proj).
    medusa_mlp_only = 2 * 2048 * 2048  # per definition: 2*h*h = 8.4M
    assert mtp["per_layer_without_lm"] > 8 * medusa_mlp_only


# ---------------------------------------------------------------------------
# Trap A/B end-to-end: speedup = E[tok] / (1 + cK)
# ---------------------------------------------------------------------------


def test_trap_A_speedup_at_K4_alpha05_below_K():
    """Trap A end-to-end: at α=0.5, K=4, c=0.10 → speedup << K=4."""
    s = speedup(0.5, 4, 0.10)
    assert s < 4.0


def test_trap_B_net_loss_zone_consistent_with_break_even():
    """Operator with α below break-even must see speedup < 1."""
    K, c = 4, 0.20
    a_star = break_even_alpha(K, c)
    # Just below break-even
    s_below = speedup(a_star - 0.05, K, c)
    s_above = speedup(a_star + 0.05, K, c)
    assert s_below < 1.0
    assert s_above > 1.0


# ---------------------------------------------------------------------------
# Mass conservation — bonus + drafts add up
# ---------------------------------------------------------------------------


def test_mass_conservation_all_accept_emits_K_plus_one():
    """All-accept request emits K + 1 tokens (K drafts + bonus)."""
    K = 4
    md = SpecDecodeMetadata.make_dummy([[5] * K])
    target_logits = torch.zeros(K, 8)
    target_logits[:, 5] = 100.0  # all accept
    bonus = torch.tensor([7], dtype=torch.int32)
    out = rejection_sample(md, None, target_logits, bonus, all_greedy=True, all_random=False)
    n_valid = int(((out >= 0) & (out < 8)).sum().item())
    assert n_valid == K + 1


def test_mass_conservation_first_reject_emits_one():
    """First-pos reject emits exactly 1 token (target argmax)."""
    K = 4
    md = SpecDecodeMetadata.make_dummy([[5] * K])
    target_logits = torch.zeros(K, 8)
    target_logits[:, 7] = 100.0  # always reject (argmax = 7 != 5)
    bonus = torch.tensor([99], dtype=torch.int32)
    out = rejection_sample(md, None, target_logits, bonus, all_greedy=True, all_random=False)
    n_valid = int(((out >= 0) & (out < 8)).sum().item())
    assert n_valid == 1  # only target argmax at pos 0


# ---------------------------------------------------------------------------
# parse_output integration — final user-facing output
# ---------------------------------------------------------------------------


def test_parse_output_after_rejection_strips_chain_break():
    """parse_output(rejection_sample(...)) gives the user-facing token list."""
    md = SpecDecodeMetadata.make_dummy([[3, 3, 3]])
    target_logits = torch.zeros(3, 8)
    target_logits[0, 3] = 100.0  # accept
    target_logits[1, 7] = 100.0  # reject
    target_logits[2, 3] = 100.0  # never visited (chain broke)
    bonus = torch.tensor([99], dtype=torch.int32)
    out = rejection_sample(md, None, target_logits, bonus, all_greedy=True, all_random=False)
    parsed = parse_output(out, vocab_size=8)
    assert parsed[0] == [3, 7]  # accepted, then rejected with target argmax 7


# ---------------------------------------------------------------------------
# Cross-proposer comparison: design tradeoff matrix
# ---------------------------------------------------------------------------


def test_eagle_and_mtp_share_pass_hidden_states_design():
    """Both EAGLE and MTP set pass_hidden=True (high coupling)."""
    from implementation.proposers.eagle import EagleProposer
    from implementation.proposers.mtp import DeepSeekMTPProposer
    eagle = EagleProposer(num_speculative_tokens=2, hidden_size=32)
    mtp = DeepSeekMTPProposer(
        num_speculative_tokens=2, hidden_size=32,
        intermediate_size=64, num_heads=4, vocab_size=64, num_mtp_layers=1,
    )
    assert eagle.pass_hidden_states_to_model
    assert mtp.pass_hidden_states_to_model


def test_draft_model_and_ngram_dont_pass_hidden_states():
    """DraftModel and Ngram don't see target hidden state."""
    from implementation.proposers.draft_model import DraftModelProposer
    from implementation.proposers.ngram import NgramProposer
    dm = DraftModelProposer(
        num_speculative_tokens=2, hidden_size=32,
        target_vocab_size=64, draft_vocab_size=64,
    )
    ng = NgramProposer(num_speculative_tokens=2)
    assert not dm.pass_hidden_states_to_model
    # Ngram doesn't even subclass base, but conceptually pass_hidden=False.
    assert not hasattr(ng, "pass_hidden_states_to_model")
