"""Tests for proposers/* — the 6 proposer family.

Pin the architectural diversity:
  - SpecDecodeBaseProposer scaffolding (K=1 fast-path, K>1 sequential)
  - EagleProposer pure inheritance (pass_hidden_states_to_model=True)
  - MedusaProposer NOT inheriting; K independent MLP heads
  - DraftModelProposer with vocab/TP guards
  - NgramProposer with prompt-lookup matching
  - ExtractHiddenStatesProposer asserts num_speculative_tokens == 1
  - DeepSeekMTPProposer integration
  - ProposerOutput dataclass
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from implementation.proposers.base import (
    ProposerOutput,
    SpecDecodeBaseProposer,
    ToyDraftModel,
)
from implementation.proposers.draft_model import DraftModelProposer
from implementation.proposers.eagle import EagleProposer
from implementation.proposers.extract_hidden import ExtractHiddenStatesProposer
from implementation.proposers.medusa import MedusaHeads, MedusaProposer
from implementation.proposers.mtp import DeepSeekMTPProposer
from implementation.proposers.ngram import NgramProposer


# ============================================================================
# ProposerOutput
# ============================================================================


def test_proposer_output_default_draft_probs_none():
    """ProposerOutput.draft_probs defaults to None (greedy proposers)."""
    drafts = torch.tensor([[1, 2, 3]])
    out = ProposerOutput(draft_token_ids=drafts)
    assert out.draft_probs is None


def test_proposer_output_with_probs():
    drafts = torch.tensor([[1, 2]])
    probs = torch.zeros(2, 8)
    out = ProposerOutput(draft_token_ids=drafts, draft_probs=probs)
    assert out.draft_probs is probs


# ============================================================================
# SpecDecodeBaseProposer
# ============================================================================


def test_base_proposer_init_defaults():
    """num_speculative_tokens, hidden_size stored; pass_hidden_states defaults False."""
    p = SpecDecodeBaseProposer(num_speculative_tokens=4, hidden_size=64)
    assert p.num_speculative_tokens == 4
    assert p.hidden_size == 64
    assert p.pass_hidden_states_to_model is False
    assert p.parallel_drafting is False


def test_base_proposer_init_with_pass_hidden_states():
    p = SpecDecodeBaseProposer(num_speculative_tokens=4, hidden_size=64,
                               pass_hidden_states_to_model=True)
    assert p.pass_hidden_states_to_model is True


def test_base_proposer_K1_returns_batch_K_shape():
    """K=1 fast-path: drafts shape == [batch, 1]."""
    p = SpecDecodeBaseProposer(num_speculative_tokens=1, hidden_size=64)
    p.model = ToyDraftModel(vocab=128, hidden=64)
    batch = 3
    target_token_ids = torch.randint(0, 128, (12,))
    target_positions = torch.arange(12)
    target_hidden = torch.randn(12, 64)
    next_token_ids = torch.randint(0, 128, (batch,))
    drafts = p.propose(target_token_ids, target_positions, target_hidden, next_token_ids)
    assert drafts.shape == (batch, 1)


def test_base_proposer_K4_sequential_returns_batch_K_shape():
    """K=4 sequential path: drafts shape == [batch, 4]."""
    p = SpecDecodeBaseProposer(num_speculative_tokens=4, hidden_size=64)
    p.model = ToyDraftModel(vocab=128, hidden=64)
    batch = 3
    target_token_ids = torch.randint(0, 128, (12,))
    target_positions = torch.arange(12)
    target_hidden = torch.randn(12, 64)
    next_token_ids = torch.randint(0, 128, (batch,))
    drafts = p.propose(target_token_ids, target_positions, target_hidden, next_token_ids)
    assert drafts.shape == (batch, 4)


def test_base_proposer_parallel_drafting_flag_set():
    """parallel_drafting flag stored on init.

    Note: actually running the parallel-drafting path requires a draft model
    that emits batch*K hidden states in one forward pass — out of scope for
    the toy ToyDraftModel here.
    """
    p = SpecDecodeBaseProposer(num_speculative_tokens=4, hidden_size=64,
                               parallel_drafting=True)
    assert p.parallel_drafting is True
    assert p.num_speculative_tokens == 4


def test_base_proposer_drafts_in_vocab_range():
    """Drafts must be valid vocab ids (argmax over LM head logits)."""
    p = SpecDecodeBaseProposer(num_speculative_tokens=3, hidden_size=64)
    p.model = ToyDraftModel(vocab=128, hidden=64)
    batch = 2
    drafts = p.propose(
        torch.randint(0, 128, (8,)), torch.arange(8),
        torch.randn(8, 64), torch.randint(0, 128, (batch,)),
    )
    assert drafts.min().item() >= 0
    assert drafts.max().item() < 128


# ============================================================================
# EagleProposer — pure inheritance
# ============================================================================


def test_eagle_inherits_base():
    """EagleProposer subclasses SpecDecodeBaseProposer."""
    assert issubclass(EagleProposer, SpecDecodeBaseProposer)


def test_eagle_pass_hidden_states_True():
    """EAGLE always sets pass_hidden_states_to_model=True (the defining trait)."""
    p = EagleProposer(num_speculative_tokens=4, hidden_size=64)
    assert p.pass_hidden_states_to_model is True


def test_eagle_no_propose_override():
    """EAGLE has no propose() of its own — algorithm IS the base class."""
    # Source signal: eagle.py is 22 lines, no propose() definition
    assert "propose" not in EagleProposer.__dict__


def test_eagle_no_greedy_sample_override():
    """EAGLE doesn't override _greedy_sample either."""
    assert "_greedy_sample" not in EagleProposer.__dict__


def test_eagle_propose_runs_through_base():
    """EAGLE.propose actually calls the base class."""
    p = EagleProposer(num_speculative_tokens=2, hidden_size=64)
    p.model = ToyDraftModel(vocab=128, hidden=64)
    batch = 2
    drafts = p.propose(
        torch.randint(0, 128, (8,)), torch.arange(8),
        torch.randn(8, 64), torch.randint(0, 128, (batch,)),
    )
    assert drafts.shape == (batch, 2)


# ============================================================================
# MedusaProposer — does NOT inherit
# ============================================================================


def test_medusa_does_NOT_inherit_from_base():
    """MedusaProposer does NOT inherit from SpecDecodeBaseProposer."""
    assert not issubclass(MedusaProposer, SpecDecodeBaseProposer)


def test_medusa_proposer_init_K_hidden_vocab():
    p = MedusaProposer(K=4, hidden=64, vocab=128)
    assert p.K == 4
    assert p.hidden_size == 64
    assert p.vocab_size == 128


def test_medusa_load_model_creates_K_heads():
    p = MedusaProposer(K=4, hidden=64, vocab=128)
    p.load_model()
    assert isinstance(p.model, MedusaHeads)
    assert len(p.model.heads) == 4


def test_medusa_propose_returns_batch_K_shape():
    """MedusaProposer.propose: target_hidden[batch, hidden] → drafts[batch, K]."""
    p = MedusaProposer(K=4, hidden=64, vocab=128)
    p.load_model()
    target_hidden = torch.randn(3, 64)
    drafts = p.propose(target_hidden)
    assert drafts.shape == (3, 4)


def test_medusa_propose_drafts_in_vocab():
    p = MedusaProposer(K=4, hidden=64, vocab=128)
    p.load_model()
    drafts = p.propose(torch.randn(3, 64))
    assert drafts.min().item() >= 0
    assert drafts.max().item() < 128


def test_medusa_heads_are_independent():
    """K heads share NO weights — independent MLPs."""
    heads = MedusaHeads(K=4, hidden=64, vocab=128)
    weights = [id(h.weight) for h in heads.heads]
    assert len(set(weights)) == 4  # 4 distinct Parameter objects


# ============================================================================
# DraftModelProposer — vocab + TP guards
# ============================================================================


def test_draft_model_proposer_inherits_base():
    assert issubclass(DraftModelProposer, SpecDecodeBaseProposer)


def test_draft_model_pass_hidden_states_False():
    """DraftModel has its own model — no shared hidden states."""
    p = DraftModelProposer(
        num_speculative_tokens=4, hidden_size=64,
        target_vocab_size=128, draft_vocab_size=128,
    )
    assert p.pass_hidden_states_to_model is False


def test_draft_model_vocab_mismatch_raises():
    """target_vocab != draft_vocab → ValueError."""
    with pytest.raises(ValueError, match="vocab_size"):
        DraftModelProposer(
            num_speculative_tokens=4, hidden_size=64,
            target_vocab_size=128, draft_vocab_size=64,
        )


def test_draft_model_tp_mismatch_raises():
    """draft_tp != target_tp → ValueError (Tomas Ruiz issue)."""
    with pytest.raises(ValueError, match="tensor_parallel_size"):
        DraftModelProposer(
            num_speculative_tokens=4, hidden_size=64,
            target_vocab_size=128, draft_vocab_size=128,
            target_tp=2, draft_tp=1,
        )


def test_draft_model_share_methods_pass():
    """DraftModel has its own embed and lm_head — share methods are no-ops."""
    p = DraftModelProposer(
        num_speculative_tokens=4, hidden_size=64,
        target_vocab_size=128, draft_vocab_size=128,
    )
    p._maybe_share_embeddings(target_language_model=None)  # no-op
    p._maybe_share_lm_head(target_language_model=None)  # no-op


# ============================================================================
# NgramProposer — n-gram lookup
# ============================================================================


def test_ngram_proposer_init():
    p = NgramProposer(num_speculative_tokens=3, prompt_lookup_min=2, prompt_lookup_max=4)
    assert p.k == 3
    assert p.min_n == 2
    assert p.max_n == 4


def test_ngram_finds_repeated_prefix():
    """Context ends with sequence that appeared earlier → emit next K tokens."""
    p = NgramProposer(num_speculative_tokens=3, prompt_lookup_min=2, prompt_lookup_max=4)
    # Context: "the cat sat on the mat ... the cat" — match "the cat" → next [sat, on, the]
    context = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 10, 20])
    drafts = p.propose([[100]], [context])
    assert drafts == [[30, 40, 50]]


def test_ngram_no_match_returns_empty_list():
    """No suffix match → empty draft list."""
    p = NgramProposer(num_speculative_tokens=3)
    context = np.array([1, 2, 3, 4, 5])  # no repeated suffix
    drafts = p.propose([[6]], [context])
    assert drafts == [[]]


def test_ngram_skips_finished_request():
    """Request with empty sampled returns []."""
    p = NgramProposer(num_speculative_tokens=3)
    drafts = p.propose([[]], [np.array([1, 2, 3])])
    assert drafts == [[]]


def test_ngram_short_context_returns_empty():
    """Context shorter than min_n returns []."""
    p = NgramProposer(num_speculative_tokens=3, prompt_lookup_min=4, prompt_lookup_max=8)
    context = np.array([1, 2, 3])  # 3 tokens, min_n=4
    drafts = p.propose([[100]], [context])
    assert drafts == [[]]


# ============================================================================
# ExtractHiddenStatesProposer — asserts K==1
# ============================================================================


def test_extract_hidden_K1_constructs():
    """K==1 is the only allowed value."""
    p = ExtractHiddenStatesProposer(num_speculative_tokens=1, hidden_size=64)
    assert p.hidden_size == 64


def test_extract_hidden_K_gt_1_asserts():
    """K>1 → AssertionError at construction."""
    with pytest.raises(AssertionError, match="num_speculative_tokens == 1"):
        ExtractHiddenStatesProposer(num_speculative_tokens=4, hidden_size=64)


def test_extract_hidden_K_gt_1_asserts_K2():
    with pytest.raises(AssertionError):
        ExtractHiddenStatesProposer(num_speculative_tokens=2, hidden_size=64)


def test_extract_hidden_propose_returns_sampled_unchanged():
    """Drafts == sampled_token_ids (always trivially verify)."""
    p = ExtractHiddenStatesProposer(num_speculative_tokens=1, hidden_size=64,
                                    num_hidden_states=2)
    sampled = torch.tensor([7, 13, 21], dtype=torch.int32)
    target_hidden = [torch.randn(3, 64), torch.randn(3, 64)]
    drafts = p.propose(sampled, target_hidden)
    assert drafts.shape == (3, 1)
    assert drafts[:, 0].tolist() == [7, 13, 21]


def test_extract_hidden_caches_target_hidden_states():
    """The hidden states buffer is populated."""
    p = ExtractHiddenStatesProposer(num_speculative_tokens=1, hidden_size=64,
                                    num_hidden_states=2)
    sampled = torch.tensor([1, 2, 3], dtype=torch.int32)
    h1 = torch.ones(3, 64) * 7.0
    h2 = torch.ones(3, 64) * 9.0
    p.propose(sampled, [h1, h2])
    # Buffer index 0..3 should match stacked input.
    assert torch.allclose(p.hidden_states_buffer[:3, 0], h1)
    assert torch.allclose(p.hidden_states_buffer[:3, 1], h2)


# ============================================================================
# DeepSeekMTPProposer — MTP integration
# ============================================================================


def test_mtp_proposer_init():
    p = DeepSeekMTPProposer(
        num_speculative_tokens=2, hidden_size=32,
        intermediate_size=64, num_heads=4, vocab_size=128, num_mtp_layers=1,
    )
    assert p.num_speculative_tokens == 2
    assert p.hidden_size == 32
    assert p.vocab_size == 128
    assert p.pass_hidden_states_to_model is True


def test_mtp_proposer_inherits_base():
    """DeepSeekMTPProposer inherits SpecDecodeBaseProposer."""
    assert issubclass(DeepSeekMTPProposer, SpecDecodeBaseProposer)


def test_mtp_proposer_propose_returns_proposer_output():
    """propose returns ProposerOutput with draft_token_ids[batch, K] and draft_probs=None."""
    p = DeepSeekMTPProposer(
        num_speculative_tokens=3, hidden_size=32,
        intermediate_size=64, num_heads=4, vocab_size=128, num_mtp_layers=1,
    )
    T = 4
    target_hidden = torch.randn(T, 32)
    last_token = torch.randint(0, 128, (T,))
    out = p.propose(target_hidden, last_token)
    assert isinstance(out, ProposerOutput)
    assert out.draft_token_ids.shape == (T, 3)
    assert out.draft_probs is None


def test_mtp_proposer_drafts_in_vocab():
    p = DeepSeekMTPProposer(
        num_speculative_tokens=2, hidden_size=32,
        intermediate_size=64, num_heads=4, vocab_size=128, num_mtp_layers=1,
    )
    out = p.propose(torch.randn(3, 32), torch.randint(0, 128, (3,)))
    assert out.draft_token_ids.min().item() >= 0
    assert out.draft_token_ids.max().item() < 128


def test_mtp_proposer_predictor_has_mtp_layers():
    """The predictor has K_layer MTP layers + ONE shared head."""
    p = DeepSeekMTPProposer(
        num_speculative_tokens=2, hidden_size=32,
        intermediate_size=64, num_heads=4, vocab_size=128, num_mtp_layers=2,
    )
    assert len(p.predictor.layers) == 2


# ============================================================================
# Cross-proposer family pin: 5 distinct architectural shapes
# ============================================================================


def test_proposer_family_inheritance_topology():
    """Pin the source-fidelity topology:
        EAGLE inherits base; Medusa does NOT; DraftModel inherits;
        Ngram is its own class (no model); ExtractHidden asserts K==1;
        DeepSeekMTPProposer inherits base.
    """
    assert issubclass(EagleProposer, SpecDecodeBaseProposer)
    assert not issubclass(MedusaProposer, SpecDecodeBaseProposer)
    assert issubclass(DraftModelProposer, SpecDecodeBaseProposer)
    assert not issubclass(NgramProposer, SpecDecodeBaseProposer)
    assert not issubclass(ExtractHiddenStatesProposer, SpecDecodeBaseProposer)
    assert issubclass(DeepSeekMTPProposer, SpecDecodeBaseProposer)
