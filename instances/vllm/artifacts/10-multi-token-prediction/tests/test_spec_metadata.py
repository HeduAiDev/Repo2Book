"""Tests for SpecDecodeMetadata — the proposer↔sampler data contract.

# REFERENCE: vllm/v1/spec_decode/metadata.py:L1-L66
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from implementation.spec_metadata import (
    GREEDY_TEMPERATURE,
    MAX_SPEC_LEN,
    PLACEHOLDER_TOKEN_ID,
    SpecDecodeMetadata,
)


# --- constants ------------------------------------------------------------------------------------


def test_placeholder_is_minus_one():
    """PLACEHOLDER_TOKEN_ID must be -1 (matches source L30)."""
    assert PLACEHOLDER_TOKEN_ID == -1


def test_max_spec_len_is_128():
    """Source pin: MAX_SPEC_LEN = 128."""
    assert MAX_SPEC_LEN == 128


def test_greedy_temperature_is_zero():
    """GREEDY_TEMPERATURE sentinel = 0 per source L31."""
    assert GREEDY_TEMPERATURE == 0


# --- make_dummy basic shapes ----------------------------------------------------------------------


def test_make_dummy_uniform_K_3reqs():
    """batch=3, K=4 each → flat=12 tokens, max_spec_len=4."""
    drafts = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    assert md.draft_token_ids.shape == (12,)
    assert md.num_draft_tokens == [4, 4, 4]
    assert md.max_spec_len == 4


def test_make_dummy_varying_K():
    """batch=3, K_i ∈ {3, 2, 1} → max_spec_len = 3."""
    drafts = [[101, 102, 103], [201, 202], [301]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    assert md.num_draft_tokens == [3, 2, 1]
    assert md.max_spec_len == 3
    assert md.draft_token_ids.tolist() == [101, 102, 103, 201, 202, 301]


def test_make_dummy_empty_drafts_max_spec_len_zero():
    """Empty batch → max_spec_len = 0 (no IndexError)."""
    md = SpecDecodeMetadata.make_dummy([])
    assert md.max_spec_len == 0
    assert md.num_draft_tokens == []
    assert md.draft_token_ids.shape == (0,)


def test_make_dummy_single_draft():
    """K=1 fast-path: single draft per request."""
    md = SpecDecodeMetadata.make_dummy([[42]])
    assert md.num_draft_tokens == [1]
    assert md.max_spec_len == 1
    assert md.draft_token_ids.tolist() == [42]


# --- cumsum invariants ----------------------------------------------------------------------------


def test_cu_num_draft_tokens_is_cumulative_sum():
    """cu_num_draft_tokens[i] = sum(num_draft_tokens[0..i]) — exclusive at start in source.

    Our pedagogical impl uses inclusive cumsum (numpy default); each entry equals running total.
    """
    drafts = [[1, 2, 3], [4, 5], [6]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    assert md.cu_num_draft_tokens.tolist() == [3, 5, 6]


def test_cu_num_sampled_tokens_includes_bonus():
    """cu_num_sampled_tokens[i] = sum(K_j + 1 for j <= i)."""
    drafts = [[1, 2, 3], [4, 5], [6]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    # K_i + 1: [4, 3, 2] → cumsum [4, 7, 9]
    assert md.cu_num_sampled_tokens.tolist() == [4, 7, 9]


def test_cu_num_sampled_minus_cu_num_draft_equals_batch_index_plus_one():
    """Invariant: cu_sampled[i] - cu_draft[i] = i + 1 (one bonus per request 0..i)."""
    drafts = [[1, 2], [3, 4, 5], [6], [7, 8, 9, 10]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    diff = md.cu_num_sampled_tokens.numpy() - md.cu_num_draft_tokens.numpy()
    assert diff.tolist() == [1, 2, 3, 4]


# --- index tensors --------------------------------------------------------------------------------


def test_target_logits_indices_size_matches_num_tokens():
    """target_logits_indices size = total flat draft tokens."""
    md = SpecDecodeMetadata.make_dummy([[1, 2], [3, 4, 5]])
    assert md.target_logits_indices.shape == (5,)


def test_bonus_logits_indices_size_matches_batch():
    """bonus_logits_indices size = batch_size."""
    md = SpecDecodeMetadata.make_dummy([[1], [2, 3], [4]])
    assert md.bonus_logits_indices.shape == (3,)


def test_logits_indices_concatenates_target_and_bonus():
    """logits_indices size = num_tokens + batch_size."""
    drafts = [[1, 2], [3, 4, 5]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    assert md.logits_indices.shape == (5 + 2,)


def test_target_logits_indices_dtype_int32():
    """Source uses int32 for index tensors (kernel constraint)."""
    md = SpecDecodeMetadata.make_dummy([[1, 2]])
    assert md.target_logits_indices.dtype == torch.int32
    assert md.bonus_logits_indices.dtype == torch.int32
    assert md.logits_indices.dtype == torch.int32


def test_draft_token_ids_dtype_int32():
    """draft_token_ids tensor is int32."""
    md = SpecDecodeMetadata.make_dummy([[1, 2, 3]])
    assert md.draft_token_ids.dtype == torch.int32


# --- max_spec_len computation ---------------------------------------------------------------------


def test_max_spec_len_is_max_per_request():
    """max_spec_len = max(num_draft_tokens) — shapes the output buffer."""
    md = SpecDecodeMetadata.make_dummy([[1], [2, 3, 4, 5, 6], [7, 8]])
    assert md.max_spec_len == 5


def test_output_buffer_shape_implied():
    """Output buffer is [batch, max_spec_len + 1] per kernel L425-L430."""
    md = SpecDecodeMetadata.make_dummy([[1, 2], [3, 4, 5]])
    bs = len(md.num_draft_tokens)
    expected_shape = (bs, md.max_spec_len + 1)
    assert expected_shape == (2, 4)


# --- device routing -------------------------------------------------------------------------------


def test_make_dummy_device_cpu():
    """make_dummy honors device='cpu'."""
    md = SpecDecodeMetadata.make_dummy([[1, 2]], device="cpu")
    assert md.draft_token_ids.device.type == "cpu"
    assert md.cu_num_draft_tokens.device.type == "cpu"


# --- post_init -----------------------------------------------------------------------------------


def test_post_init_sets_max_spec_len_from_field():
    """__post_init__ fills max_spec_len from num_draft_tokens."""
    md = SpecDecodeMetadata(
        draft_token_ids=torch.zeros(5, dtype=torch.int32),
        num_draft_tokens=[2, 3],
        cu_num_draft_tokens=torch.zeros(2, dtype=torch.int32),
        cu_num_sampled_tokens=torch.zeros(2, dtype=torch.int32),
        target_logits_indices=torch.zeros(5, dtype=torch.int32),
        bonus_logits_indices=torch.zeros(2, dtype=torch.int32),
        logits_indices=torch.zeros(7, dtype=torch.int32),
    )
    assert md.max_spec_len == 3


# --- numerical sanity -----------------------------------------------------------------------------


@pytest.mark.parametrize("K", [1, 2, 4, 8, 16])
def test_make_dummy_uniform_K(K):
    """Test K-uniform batches; verify shapes scale linearly with K."""
    drafts = [list(range(K)), list(range(K, 2 * K))]
    md = SpecDecodeMetadata.make_dummy(drafts)
    assert md.draft_token_ids.shape == (2 * K,)
    assert md.max_spec_len == K
    assert md.num_draft_tokens == [K, K]


@pytest.mark.parametrize("batch", [1, 2, 4, 8])
def test_make_dummy_batch_sizes(batch):
    """Test batch sizes; verify bonus indices size."""
    drafts = [[i, i + 1] for i in range(batch)]
    md = SpecDecodeMetadata.make_dummy(drafts)
    assert md.bonus_logits_indices.shape == (batch,)


def test_flat_token_ordering_preserves_request_order():
    """Flat draft_token_ids is request-major (req0 tokens then req1 tokens)."""
    drafts = [[10, 20], [30, 40, 50]]
    md = SpecDecodeMetadata.make_dummy(drafts)
    assert md.draft_token_ids.tolist() == [10, 20, 30, 40, 50]


def test_cumsum_dtype_int32():
    """cu_* tensors are int32 (kernel-friendly)."""
    md = SpecDecodeMetadata.make_dummy([[1, 2], [3]])
    assert md.cu_num_draft_tokens.dtype == torch.int32
    assert md.cu_num_sampled_tokens.dtype == torch.int32
