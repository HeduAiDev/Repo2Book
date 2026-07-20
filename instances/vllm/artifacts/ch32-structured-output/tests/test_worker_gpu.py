"""m08/m09/m10 (GPU-only): the new gpu-worker payoff path -- StructuredOutputsWorker
copies the bitmask + a bitmask-row -> logits-row mapping to GPU on an
independent copy_stream, then launches the real _apply_grammar_bitmask_kernel,
which unpacks the 32-bit-packed bitmask and writes -inf into every logit
whose bit says "illegal".

This is the payoff of the *opt-in* V2 model-runner path
(VLLM_USE_V2_MODEL_RUNNER=1) -- structured_output/utils.py's
xgr.apply_token_bitmask_inplace (test_legacy_path.py) is what actually runs
by default.

Requires CUDA + Triton; on host without a GPU these are skipped -- run in
the vLLM container with: scripts/vllm_docker.sh -m pytest /work/.../tests/

SOURCE anchors: vllm/v1/worker/gpu/structured_outputs.py:L12-115
"""
import math
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "implementation")
)

import numpy as np
import pytest
import torch

from input_batch import InputBatch  # noqa: E402
from structured_outputs import StructuredOutputsWorker  # noqa: E402

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="requires CUDA + Triton")

VOCAB = 96  # 3 packed int32 columns


def _pack_allowed(allowed: set[int], vocab_size: int) -> np.ndarray:
    """Build one packed bitmask row: bit==1 means "legal" (kept), bit==0
    means "illegal" (kernel writes -inf there) -- this is the same
    convention as _full_mask == -1 (all bits set) meaning "fully allowed"
    for an unconstrained row (see structured_output_manager._fill_bitmasks)."""
    cols = math.ceil(vocab_size / 32)
    row = np.zeros(cols, dtype=np.int32)  # all bits clear = "all illegal"
    for tok in allowed:
        col, bit = tok // 32, tok % 32
        row[col] |= np.int32(1 << bit)
    return row


def test_kernel_masks_illegal_tokens_to_negative_infinity():
    dev = torch.device("cuda")
    worker = StructuredOutputsWorker(max_num_logits=4, vocab_size=VOCAB, device=dev)

    grammar_bitmask = np.stack([_pack_allowed({5, 40, 90}, VOCAB)])
    input_batch = InputBatch(
        req_ids=["r0"],
        logits_indices=torch.zeros(1, dtype=torch.int32, device=dev),
        cu_num_logits=torch.tensor([0, 1], dtype=torch.int32, device=dev),
        cu_num_logits_np=np.array([0, 1], dtype=np.int32),
    )
    logits = torch.zeros((1, VOCAB), device=dev)

    worker.apply_grammar_bitmask(logits, input_batch, ["r0"], grammar_bitmask)
    torch.cuda.synchronize()

    row = logits[0].cpu()
    legal = {5, 40, 90}
    for tok in range(VOCAB):
        if tok in legal:
            assert row[tok].item() == 0.0
        else:
            assert row[tok].item() == float("-inf")


def test_row_mapping_targets_correct_request_across_batch():
    # Batch has 2 requests: "a" (2 logits rows -- e.g. 1 spec token) then "b"
    # (1 row). Only "b" is structured -- its row (index 2) must be the only
    # one masked; "a"'s two rows must be untouched.
    dev = torch.device("cuda")
    worker = StructuredOutputsWorker(max_num_logits=4, vocab_size=VOCAB, device=dev)

    grammar_bitmask = np.stack([_pack_allowed({3}, VOCAB)])
    input_batch = InputBatch(
        req_ids=["a", "b"],
        logits_indices=torch.zeros(3, dtype=torch.int32, device=dev),
        cu_num_logits=torch.tensor([0, 2, 3], dtype=torch.int32, device=dev),
        cu_num_logits_np=np.array([0, 2, 3], dtype=np.int32),
    )
    logits = torch.zeros((3, VOCAB), device=dev)

    worker.apply_grammar_bitmask(logits, input_batch, ["b"], grammar_bitmask)
    torch.cuda.synchronize()

    cpu_logits = logits.cpu()
    # "a"'s rows (0, 1) are untouched.
    assert torch.all(cpu_logits[0] == 0.0)
    assert torch.all(cpu_logits[1] == 0.0)
    # "b"'s row (2) is masked: only token 3 survives.
    assert cpu_logits[2, 3].item() == 0.0
    assert cpu_logits[2, 0].item() == float("-inf")
    assert cpu_logits[2, VOCAB - 1].item() == float("-inf")


def test_no_grammar_req_ids_is_a_no_op():
    dev = torch.device("cuda")
    worker = StructuredOutputsWorker(max_num_logits=4, vocab_size=VOCAB, device=dev)
    input_batch = InputBatch(
        req_ids=["a"],
        logits_indices=torch.zeros(1, dtype=torch.int32, device=dev),
        cu_num_logits=torch.tensor([0, 1], dtype=torch.int32, device=dev),
        cu_num_logits_np=np.array([0, 1], dtype=np.int32),
    )
    logits = torch.ones((1, VOCAB), device=dev)
    worker.apply_grammar_bitmask(logits, input_batch, [], np.zeros((0, 3), dtype=np.int32))
    torch.cuda.synchronize()
    assert torch.all(logits.cpu() == 1.0)


def test_mapping_length_must_match_bitmask_row_count():
    # assert num_masks == len(mapping) -- if the caller (grammar_req_ids)
    # claims a request that spans more logits rows than the bitmask actually
    # provides, the kernel launch must never silently proceed.
    dev = torch.device("cuda")
    worker = StructuredOutputsWorker(max_num_logits=4, vocab_size=VOCAB, device=dev)
    grammar_bitmask = np.stack([_pack_allowed({1}, VOCAB)])  # 1 row
    input_batch = InputBatch(
        req_ids=["a"],
        logits_indices=torch.zeros(2, dtype=torch.int32, device=dev),
        cu_num_logits=torch.tensor([0, 2], dtype=torch.int32, device=dev),  # 2 rows
        cu_num_logits_np=np.array([0, 2], dtype=np.int32),
    )
    logits = torch.zeros((2, VOCAB), device=dev)
    with pytest.raises(AssertionError):
        worker.apply_grammar_bitmask(logits, input_batch, ["a"], grammar_bitmask)
