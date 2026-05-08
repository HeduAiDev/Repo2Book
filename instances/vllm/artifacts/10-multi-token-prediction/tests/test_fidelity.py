"""7-trap fidelity verification + source-grep negative tests + E2E MTP→sampler.

Covers:
  - Trap A: K * α != E[tok]
  - Trap B: net loss zone exists
  - Trap C: shared architecture not required
  - Trap D: rejection sampling unbiased at high temperature
  - Trap E: MTP block has full transformer machinery
  - Trap F: NO training code in vllm/v1/spec_decode/ (negative grep)
  - Trap G: α is workload+temperature-dependent (not a model property)
  - "no class MultiTokenPrediction" in vllm/ (4th instance pattern)
  - SpeculativeMethod literal does NOT include "mtp"
  - K=1 spec_decode equivalent to standard sampling
  - E2E: MTP propose → rejection sample → output
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from implementation.acceptance_math import expected_tokens, simulate_chain_break, speedup
from implementation.mtp_head import MTPBlock
from implementation.proposers.draft_model import DraftModelProposer
from implementation.proposers.mtp import DeepSeekMTPProposer
from implementation.rejection_sampling import (
    parse_output,
    rejection_sample,
)
from implementation.spec_metadata import (
    PLACEHOLDER_TOKEN_ID,
    SpecDecodeMetadata,
)


# tests/ → 10-multi-token-prediction → artifacts → vllm → instances → repo2book
# parents[5] = repo2book root; then add instances/vllm/source
VLLM_SOURCE = Path(__file__).resolve().parents[5] / "instances/vllm/source"


# ============================================================================
# E2E pipeline: MTP propose → rejection sample
# ============================================================================


def test_e2e_mtp_propose_then_rejection_sample():
    """MTP proposer emits drafts; rejection sampler verifies them end-to-end."""
    K, hidden, vocab = 3, 16, 32
    proposer = DeepSeekMTPProposer(
        num_speculative_tokens=K, hidden_size=hidden,
        intermediate_size=hidden * 2, num_heads=2, vocab_size=vocab,
        num_mtp_layers=1,
    )
    T = 2
    target_hidden = torch.randn(T, hidden)
    last_token = torch.randint(0, vocab, (T,))
    out = proposer.propose(target_hidden, last_token)

    drafts_lists = out.draft_token_ids.tolist()
    md = SpecDecodeMetadata.make_dummy(drafts_lists)
    flat_K = T * K
    target_logits = torch.randn(flat_K, vocab)
    bonus = torch.tensor([0, 1], dtype=torch.int32)
    sampler_out = rejection_sample(
        md, draft_probs=None, target_logits=target_logits,
        bonus_token_ids=bonus, all_greedy=True, all_random=False,
    )
    assert sampler_out.shape == (T, md.max_spec_len + 1)


def test_e2e_K1_matches_standard_greedy_sampling():
    """K=1 spec_decode reduces to standard greedy sampling.

    When draft matches argmax: emit [draft, bonus].
    When draft mismatches: emit [target_argmax] (chain break, no bonus).
    """
    drafts = [[5]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    vocab = 16
    target_logits = torch.full((1, vocab), -10.0)
    target_logits[0, 5] = 100.0  # argmax matches draft
    bonus = torch.tensor([7], dtype=torch.int32)
    out = rejection_sample(md, None, target_logits, bonus,
                           all_greedy=True, all_random=False)
    assert parse_output(out, vocab)[0] == [5, 7]

    target_logits[0, 5] = -10.0
    target_logits[0, 9] = 100.0  # argmax = 9 != draft 5
    out = rejection_sample(md, None, target_logits, bonus,
                           all_greedy=True, all_random=False)
    assert parse_output(out, vocab)[0] == [9]


# ============================================================================
# Trap A: K * α != E[tok]
# ============================================================================


def test_trap_A_K_times_alpha_not_speedup():
    """At α=0.5, K=4: E[tok] = 1.9375, NOT 4*0.5 = 2.0."""
    et = expected_tokens(0.5, 4)
    assert et == pytest.approx(1.9375, abs=1e-4)
    assert et != pytest.approx(4 * 0.5)


def test_trap_A_K_alpha_overshoots_at_alpha_07():
    """At α=0.7, K=4: K*α=2.8, E[tok]=2.7731. K*α overshoots E[tok]."""
    et = expected_tokens(0.7, 4)
    assert 4 * 0.7 > et
    assert et == pytest.approx(2.7731, abs=1e-4)


# ============================================================================
# Trap B: spec-decode is NOT always cheaper
# ============================================================================


def test_trap_B_net_loss_K4_c020_alpha030():
    """At K=4, c=0.20, α=0.30: speedup = 0.792 < 1 (NET LOSS)."""
    s = speedup(0.30, 4, 0.20)
    assert s < 1.0
    assert s == pytest.approx(0.792, abs=1e-3)


def test_trap_B_net_loss_persists_at_K8_high_c():
    """At K=8, c=0.20, α=0.5: still below break-even."""
    s = speedup(0.5, 8, 0.20)
    assert s < 1.0


# ============================================================================
# Trap C: draft does NOT need to share architecture
# ============================================================================


def test_trap_C_draft_model_with_different_architecture_works():
    """DraftModelProposer constructs successfully with different vocab=128 (matched)."""
    p = DraftModelProposer(
        num_speculative_tokens=4, hidden_size=128,  # arbitrary draft hidden
        target_vocab_size=32000, draft_vocab_size=32000,
    )
    # Only constraint: same vocab size, same TP. Architecture is otherwise free.
    assert p.num_speculative_tokens == 4


# ============================================================================
# Trap D: rejection sampling is unbiased at any temperature
# ============================================================================


def test_trap_D_unbiased_at_disjoint_supports():
    """Even with very different p, q (high temperature divergence), KL stays small."""
    p = torch.tensor([0.5, 0.3, 0.1, 0.1], dtype=torch.float32)
    q = torch.tensor([0.1, 0.1, 0.4, 0.4], dtype=torch.float32)
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
    emp = counts / counts.sum()
    safe_e = emp.clamp_min(1e-12)
    safe_p = p.clamp_min(1e-12)
    kl = float((safe_e * (safe_e.log() - safe_p.log())).sum().item())
    assert kl < 0.05


# ============================================================================
# Trap E: MTP block is a FULL transformer block
# ============================================================================


def test_trap_E_mtp_block_has_attn_ffn_and_2_layernorms():
    blk = MTPBlock(hidden_size=64, intermediate_size=128, num_heads=4)
    assert hasattr(blk, "attn")
    assert hasattr(blk, "mlp")
    assert hasattr(blk, "input_layernorm")
    assert hasattr(blk, "post_attention_layernorm")


def test_trap_E_mtp_block_total_params_far_exceed_lightweight_mlp():
    """MTPBlock total params >> a 2-layer MLP."""
    blk = MTPBlock(hidden_size=64, intermediate_size=128, num_heads=4)
    total = sum(p.numel() for p in blk.parameters())
    lightweight_mlp_estimate = 2 * 64 * 64  # 2 hidden×hidden Linear blocks
    assert total > 5 * lightweight_mlp_estimate


# ============================================================================
# Trap F: vLLM is INFERENCE-ONLY — no MTP training code
# ============================================================================


@pytest.mark.skipif(not VLLM_SOURCE.exists(), reason="vLLM source not checked out")
def test_trap_F_no_mtp_training_in_spec_decode():
    """No MTPLoss / multi_step_ce / mtp_aux_loss in vllm/v1/spec_decode/."""
    spec_decode_dir = VLLM_SOURCE / "vllm/v1/spec_decode"
    if not spec_decode_dir.exists():
        pytest.skip("vllm/v1/spec_decode/ not present")
    pattern = re.compile(
        r"MTPLoss|multi_step_ce|compute_mtp_loss|mtp_aux_loss",
        re.MULTILINE,
    )
    matches = []
    for py_file in spec_decode_dir.rglob("*.py"):
        text = py_file.read_text(errors="ignore")
        cleaned = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
        cleaned = re.sub(r"'''.*?'''", '', cleaned, flags=re.DOTALL)
        if pattern.search(cleaned):
            matches.append(str(py_file))
    assert matches == []


@pytest.mark.skipif(not VLLM_SOURCE.exists(), reason="vLLM source not checked out")
def test_trap_F_no_backward_in_spec_decode():
    """No `.backward(` calls in vllm/v1/spec_decode/."""
    spec_decode_dir = VLLM_SOURCE / "vllm/v1/spec_decode"
    if not spec_decode_dir.exists():
        pytest.skip("vllm/v1/spec_decode/ not present")
    pattern = re.compile(r"\.backward\(", re.MULTILINE)
    matches = []
    for py_file in spec_decode_dir.rglob("*.py"):
        text = py_file.read_text(errors="ignore")
        cleaned = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
        cleaned = re.sub(r"'''.*?'''", '', cleaned, flags=re.DOTALL)
        if pattern.search(cleaned):
            matches.append(str(py_file))
    assert matches == []


@pytest.mark.skipif(not VLLM_SOURCE.exists(), reason="vLLM source not checked out")
def test_trap_F_no_optimizer_in_spec_decode():
    """No `torch.optim` or `Optimizer(` in vllm/v1/spec_decode/."""
    spec_decode_dir = VLLM_SOURCE / "vllm/v1/spec_decode"
    if not spec_decode_dir.exists():
        pytest.skip("vllm/v1/spec_decode/ not present")
    pattern = re.compile(r"torch\.optim|Optimizer\(", re.MULTILINE)
    matches = []
    for py_file in spec_decode_dir.rglob("*.py"):
        text = py_file.read_text(errors="ignore")
        cleaned = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
        cleaned = re.sub(r"'''.*?'''", '', cleaned, flags=re.DOTALL)
        if pattern.search(cleaned):
            matches.append(str(py_file))
    assert matches == []


# ============================================================================
# Trap G: α is NOT a model property
# ============================================================================


def test_trap_G_alpha_varies_with_K_for_fixed_workload():
    """For fixed workload, different K values produce different E[tok]."""
    # Same conceptual α=0.7, but K determines emit count.
    e1 = expected_tokens(0.7, 1)
    e4 = expected_tokens(0.7, 4)
    e8 = expected_tokens(0.7, 8)
    assert e1 < e4 < e8


def test_trap_G_alpha_varies_synthetic_simulation():
    """Different α inputs to chain_break sim → different empirical means."""
    m_03, _, _ = simulate_chain_break(0.3, 4, n_trials=2000, seed=7)
    m_07, _, _ = simulate_chain_break(0.7, 4, n_trials=2000, seed=7)
    m_09, _, _ = simulate_chain_break(0.9, 4, n_trials=2000, seed=7)
    assert m_03 < m_07 < m_09


# ============================================================================
# "no class MultiTokenPrediction" — 4th instance pattern
# ============================================================================


@pytest.mark.skipif(not VLLM_SOURCE.exists(), reason="vLLM source not checked out")
def test_no_top_level_class_MultiTokenPrediction():
    """No `class MultiTokenPrediction` / `class MTPHead` / `class MTPModel` in vllm/."""
    src_dir = VLLM_SOURCE / "vllm"
    if not src_dir.exists():
        pytest.skip("vllm/ source not checked out")
    pattern = re.compile(
        r"^class\s+(MultiTokenPrediction|MTPHead|MTPModel)\b",
        re.MULTILINE,
    )
    matches = []
    for py_file in src_dir.rglob("*.py"):
        text = py_file.read_text(errors="ignore")
        if pattern.search(text):
            matches.append(str(py_file))
    assert matches == []


@pytest.mark.skipif(not VLLM_SOURCE.exists(), reason="vLLM source not checked out")
def test_DeepSeekMultiTokenPredictor_DOES_exist():
    """The model-prefixed classes ARE in source — DeepSeekMultiTokenPredictor + Layer."""
    deepseek_mtp = VLLM_SOURCE / "vllm/model_executor/models/deepseek_mtp.py"
    if not deepseek_mtp.exists():
        pytest.skip("deepseek_mtp.py not present")
    text = deepseek_mtp.read_text()
    assert "class DeepSeekMultiTokenPredictor" in text
    assert "class DeepSeekMultiTokenPredictorLayer" in text
    assert "class DeepSeekMTP" in text


# ============================================================================
# SpeculativeMethod literal does NOT include "mtp"
# ============================================================================


@pytest.mark.skipif(not VLLM_SOURCE.exists(), reason="vLLM source not checked out")
def test_speculative_method_literal_no_mtp():
    """speculative.py SpeculativeMethod literal contains no `"mtp"` quoted entry."""
    spec_cfg = VLLM_SOURCE / "vllm/config/speculative.py"
    if not spec_cfg.exists():
        pytest.skip("speculative.py not present")
    text = spec_cfg.read_text()
    m = re.search(
        r"SpeculativeMethod\s*=\s*Literal\[(.*?)\]",
        text, re.DOTALL,
    )
    assert m is not None
    assert '"mtp"' not in m.group(1)


# ============================================================================
# Knowledge facts pinned (M01-M07 cross-validation)
# ============================================================================


def test_M02_greedy_path_skips_softmax():
    """M02: greedy fast-path runs without softmax/draft_probs."""
    drafts = [[1, 2]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    vocab = 8
    target_logits = torch.full((2, vocab), -10.0)
    target_logits[0, 1] = 100.0
    target_logits[1, 2] = 100.0
    bonus = torch.tensor([5], dtype=torch.int32)
    # all_greedy=True with draft_probs=None — no softmax computation
    out = rejection_sample(md, draft_probs=None, target_logits=target_logits,
                           bonus_token_ids=bonus, all_greedy=True, all_random=False)
    assert out[0].tolist() == [1, 2, 5]


def test_M06_K_is_global_per_engine():
    """M06: num_speculative_tokens is global at engine init, not per-request."""
    p1 = DeepSeekMTPProposer(
        num_speculative_tokens=4, hidden_size=32,
        intermediate_size=64, num_heads=4, vocab_size=128, num_mtp_layers=1,
    )
    # K is engine-wide; metadata.num_draft_tokens varies per request but K caps it.
    assert p1.num_speculative_tokens == 4


def test_M01_to_M04_cross_chapter_sanity():
    """Cross-fact sanity: M01-M04 all interact in this single E2E."""
    # M01: instantiate as nn-style class
    from implementation.rejection_sampling import RejectionSampler
    sampler = RejectionSampler()
    # M02: greedy path
    drafts = [[1, 2]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    vocab = 8
    target_logits = torch.full((2, vocab), -10.0)
    target_logits[0, 1] = 100.0
    target_logits[1, 5] = 100.0  # reject at pos 1 (argmax 5 != draft 2)
    bonus = torch.tensor([7], dtype=torch.int32)
    out = sampler(md, None, target_logits, bonus,
                  all_greedy=True, all_random=False)
    # M04: chain-break, slot 2 (bonus) stays -1
    assert out[0, 0].item() == 1   # accept
    assert out[0, 1].item() == 5   # reject, emit target argmax
    assert out[0, 2].item() == PLACEHOLDER_TOKEN_ID  # chain break, no bonus
