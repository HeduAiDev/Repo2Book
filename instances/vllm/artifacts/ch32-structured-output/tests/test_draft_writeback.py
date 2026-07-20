"""m14 (GPU-only): DraftTokensHandler is the D2H channel that lets the
scheduler validate speculative draft tokens against the grammar --
has_structured_output_reqs gates whether that transfer even happens (no
structured-output requests in this batch -> skip the D2H roundtrip
entirely).

Requires CUDA (torch.cuda.Stream/Event); on host without a GPU these are
skipped -- run in the vLLM container.

SOURCE anchors: vllm/v1/worker/gpu/spec_decode/utils.py:L11-47
"""
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "implementation")
)

import pytest
import torch

from input_batch import InputBatch  # noqa: E402
from spec_decode_utils import DraftTokensHandler  # noqa: E402

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="requires CUDA")


def _input_batch(req_ids, has_structured_output_reqs):
    n = len(req_ids)
    return InputBatch(
        req_ids=req_ids,
        logits_indices=torch.zeros(n, dtype=torch.int32),
        cu_num_logits=torch.arange(n + 1, dtype=torch.int32),
        cu_num_logits_np=None,
        has_structured_output_reqs=has_structured_output_reqs,
    )


def test_no_structured_output_reqs_skips_the_d2h_roundtrip():
    dev = torch.device("cuda")
    handler = DraftTokensHandler(device=dev)
    draft_tokens = torch.tensor([[1, 2], [3, 4]], device=dev, dtype=torch.int64)
    input_batch = _input_batch(["a", "b"], has_structured_output_reqs=False)

    handler.set_draft_tokens(input_batch, draft_tokens)

    assert handler.draft_tokens_np is None
    # get_draft_tokens returns None -- the caller (deferred chain) must not
    # proceed to update_draft_token_ids_in_output with garbage.
    assert handler.get_draft_tokens() is None


def test_structured_output_reqs_transfers_draft_tokens_to_cpu():
    dev = torch.device("cuda")
    handler = DraftTokensHandler(device=dev)
    draft_tokens = torch.tensor([[7, 8], [9, 10]], device=dev, dtype=torch.int64)
    input_batch = _input_batch(["a", "b"], has_structured_output_reqs=True)

    handler.set_draft_tokens(input_batch, draft_tokens)
    torch.cuda.synchronize()

    assert handler.draft_tokens_np is not None
    draft_ids = handler.get_draft_tokens()
    assert draft_ids is not None
    assert draft_ids.req_ids == ["a", "b"]
    assert draft_ids.draft_token_ids == [[7, 8], [9, 10]]
