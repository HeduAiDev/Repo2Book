"""
Tests for the tiny end-to-end DFlash draft model (arXiv:2602.06036 §3.2
Eq.(2)/(3), §4.1, §4.2): a stack of KV-injected draft layers that turns one
forward pass over (bonus token + masked block) into logits for the whole
block at once -- the runnable form of "T_draft = t_parallel, independent of
block size" (Eq.3), contrasted with an autoregressive drafter that needs one
forward call per token (Eq.2).
"""
import torch

from dflash_draft_model import (
    TinyDflashDraftModel,
    count_forward_calls_autoregressive,
    count_forward_calls_diffusion,
)

torch.manual_seed(0)

HIDDEN = 8
TARGET_HIDDEN = 8
VOCAB = 16
NUM_LAYERS = 3
NUM_HEADS = 2
NUM_KV_HEADS = 2
HEAD_DIM = 4


def _make_model():
    return TinyDflashDraftModel(
        hidden_size=HIDDEN,
        target_hidden_size=TARGET_HIDDEN,
        vocab_size=VOCAB,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
    )


def _make_inputs(num_ctx, block_size):
    selected_target_layers = [torch.randn(num_ctx, TARGET_HIDDEN) for _ in range(5)]
    context_positions = torch.arange(num_ctx, dtype=torch.float32)
    block_input_ids = torch.randint(0, VOCAB, (block_size,))
    query_positions = num_ctx + torch.arange(block_size, dtype=torch.float32)
    return selected_target_layers, context_positions, block_input_ids, query_positions


class TestSingleForwardProducesWholeBlock:
    def test_one_call_returns_logits_for_every_masked_position(self):
        model = _make_model()
        for block_size in (1, 4, 8, 16):
            selected, ctx_pos, ids, q_pos = _make_inputs(num_ctx=6, block_size=block_size)
            logits = model(selected, ctx_pos, ids, q_pos)
            assert logits.shape == (block_size, VOCAB)

    def test_forward_call_count_independent_of_block_size(self):
        # Eq.(3): T_draft = t_parallel regardless of gamma -- structurally,
        # this means exactly one model call produces the entire block.
        assert count_forward_calls_diffusion(block_size=1) == 1
        assert count_forward_calls_diffusion(block_size=16) == 1

    def test_autoregressive_call_count_scales_with_block_size(self):
        # Eq.(2): T_draft = gamma * t_step -- one call per token.
        assert count_forward_calls_autoregressive(block_size=1) == 1
        assert count_forward_calls_autoregressive(block_size=16) == 16


class TestBlockIsNonCausal:
    def test_perturbing_one_mask_position_changes_another_position_logits(self):
        model = _make_model()
        selected, ctx_pos, ids, q_pos = _make_inputs(num_ctx=5, block_size=6)
        logits_base = model(selected, ctx_pos, ids, q_pos)

        ids_perturbed = ids.clone()
        ids_perturbed[1] = (ids_perturbed[1] + 1) % VOCAB
        logits_perturbed = model(selected, ctx_pos, ids_perturbed, q_pos)

        # position 0's (bonus token's) logits should be affected too, since
        # the draft's forward pass is fully non-causal across the query span
        # (cad.causal = False in the real proposer).
        assert not torch.allclose(logits_base[0], logits_perturbed[0])


class TestContextConditioning:
    def test_output_depends_on_target_context_features(self):
        # Swapping out the target's hidden features (the whole point of KV
        # injection) must change the draft's predictions.
        model = _make_model()
        selected, ctx_pos, ids, q_pos = _make_inputs(num_ctx=5, block_size=4)
        logits_base = model(selected, ctx_pos, ids, q_pos)

        selected_perturbed = [h + 5.0 for h in selected]
        logits_perturbed = model(selected_perturbed, ctx_pos, ids, q_pos)
        assert not torch.allclose(logits_base, logits_perturbed)
