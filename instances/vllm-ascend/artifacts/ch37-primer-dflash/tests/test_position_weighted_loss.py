"""
Tests for the DFlash position-weighted training loss and anchor-block
sampling (arXiv:2602.06036 §4.2 Eq.(4) and the surrounding prose).
"""
import math

import numpy as np
import pytest
import torch

from position_weighted_loss import (
    build_training_attention_mask,
    position_weighted_cross_entropy,
    position_weights,
    sample_anchor_blocks,
)


class TestPositionWeightsEq4:
    def test_first_position_weight_is_one(self):
        # w_k = exp(-(k-1)/gamma); at k=1 this is exp(0) = 1 regardless of gamma.
        w = position_weights(block_size=8, gamma=4)
        assert w[0].item() == pytest.approx(1.0)

    def test_matches_closed_form(self):
        block_size, gamma = 6, 3
        w = position_weights(block_size, gamma)
        expected = [math.exp(-(k - 1) / gamma) for k in range(1, block_size + 1)]
        assert torch.allclose(w, torch.tensor(expected, dtype=torch.float32), atol=1e-6)

    def test_monotonically_decreasing(self):
        w = position_weights(block_size=16, gamma=8)
        diffs = w[1:] - w[:-1]
        assert torch.all(diffs < 0)


class TestPositionWeightedCrossEntropy:
    def test_early_position_error_costs_more_than_late_position_error(self):
        # Same-magnitude prediction error, but at position 1 (early) vs the
        # last position (late) within the block. Because acceptance is a
        # longest-prefix match, an error at position 1 invalidates the whole
        # block while a late error only costs one token -- Eq.(4)'s
        # exponential decay is supposed to reflect exactly this asymmetry.
        block_size, gamma, vocab = 8, 4, 5
        torch.manual_seed(0)

        def make_logits_wrong_at(pos):
            targets = torch.zeros(block_size, dtype=torch.long)
            logits = torch.zeros(block_size, vocab)
            logits[:, 0] = 10.0  # every position confidently (and correctly) predicts token 0...
            logits[pos, 0] = -10.0  # ...except `pos`, which confidently predicts something else.
            logits[pos, 1] = 10.0
            return logits, targets

        logits_early, targets = make_logits_wrong_at(0)
        logits_late, _ = make_logits_wrong_at(block_size - 1)

        loss_early = position_weighted_cross_entropy(logits_early, targets, gamma)
        loss_late = position_weighted_cross_entropy(logits_late, targets, gamma)
        assert loss_early.item() > loss_late.item()

    def test_all_correct_gives_near_zero_loss(self):
        block_size, gamma, vocab = 5, 3, 4
        targets = torch.zeros(block_size, dtype=torch.long)
        logits = torch.full((block_size, vocab), -10.0)
        logits[:, 0] = 10.0
        loss = position_weighted_cross_entropy(logits, targets, gamma)
        assert loss.item() < 1e-3


class TestSampleAnchorBlocks:
    def test_anchors_leave_room_for_a_full_block(self):
        rng = np.random.default_rng(0)
        response_len, block_size, num_blocks = 40, 8, 200
        anchors = sample_anchor_blocks(response_len, block_size, num_blocks, rng)
        assert anchors.min() >= 0
        assert anchors.max() <= response_len - block_size

    def test_rejects_response_shorter_than_block(self):
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            sample_anchor_blocks(response_len=3, block_size=8, num_blocks=1, rng=rng)


class TestTrainingAttentionMask:
    def test_context_columns_always_visible(self):
        mask = build_training_attention_mask(block_boundaries=[(0, 3), (3, 5)], num_context=4)
        assert torch.all(mask[:, :4])

    def test_blocks_are_block_diagonal_and_isolated(self):
        # Two blocks of length 3 and 2: tokens inside a block see each other,
        # but never see the other block's tokens (Figure 4: "attention
        # across different blocks is disallowed").
        mask = build_training_attention_mask(block_boundaries=[(0, 3), (3, 5)], num_context=4)
        query_block1 = mask[0:3, 4:]  # query-axis columns (excluding context)
        query_block2 = mask[3:5, 4:]
        assert torch.all(query_block1[:, 0:3])
        assert torch.all(~query_block1[:, 3:5])
        assert torch.all(query_block2[:, 3:5])
        assert torch.all(~query_block2[:, 0:3])
