"""m11: the legacy/default worker path -- structured_output/utils.py's
apply_grammar_bitmask re-sorts the compact scheduler-order bitmask into
batch/logits order (accounting for per-request speculative-token offsets)
and hands it to xgr.apply_token_bitmask_inplace. This is what actually runs
in a default deployment (VLLM_USE_V2_MODEL_RUNNER=0); the Triton kernel in
structured_outputs.py only runs when that env flag is explicitly set.

SOURCE anchors: vllm/v1/structured_output/utils.py:L44-105
"""
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "implementation")
)

import numpy as np
import torch

import utils  # noqa: E402
from input_batch import InputBatch  # noqa: E402
from output import GrammarOutput, SchedulerOutput  # noqa: E402


class FakeXgr:
    """Records the (logits, bitmask, indices) triple it was called with so
    the test can assert on the reordering utils.apply_grammar_bitmask did
    before handing off, without needing the real xgrammar C++ extension."""

    def __init__(self):
        self.calls = []

    def apply_token_bitmask_inplace(self, logits, bitmask, indices=None):
        self.calls.append(
            {
                "logits_shape": tuple(logits.shape),
                "bitmask": bitmask.clone(),
                "indices": None if indices is None else indices.clone(),
            }
        )


def _input_batch(req_ids):
    n = len(req_ids)
    return InputBatch(
        req_ids=req_ids,
        logits_indices=torch.arange(n, dtype=torch.int32),
        cu_num_logits=torch.arange(n + 1, dtype=torch.int32),
        cu_num_logits_np=np.arange(n + 1, dtype=np.int32),
    )


def test_legacy_path_reorders_scheduler_rows_into_batch_order(monkeypatch):
    fake_xgr = FakeXgr()
    monkeypatch.setattr(utils, "xgr", fake_xgr)

    # Batch order (gpu runner side): ["b", "a"]. Scheduler emitted the
    # bitmask in the *other* order: ["a", "b"] -- this is exactly the case
    # the re-sort exists for.
    input_batch = _input_batch(["b", "a"])
    grammar_bitmask = np.array(
        [[0b0010], [0b0100]], dtype=np.int32
    )  # row0 -> "a", row1 -> "b"
    grammar_output = GrammarOutput(
        structured_output_request_ids=["a", "b"], grammar_bitmask=grammar_bitmask
    )
    scheduler_output = SchedulerOutput(scheduled_spec_decode_tokens={})
    logits = torch.zeros((2, 32))

    utils.apply_grammar_bitmask(scheduler_output, grammar_output, input_batch, logits)

    assert len(fake_xgr.calls) == 1
    sorted_bitmask = fake_xgr.calls[0]["bitmask"]
    # Batch row 0 is "b" (originally scheduler row 1); batch row 1 is "a"
    # (originally scheduler row 0).
    assert sorted_bitmask[0, 0].item() == 0b0100
    assert sorted_bitmask[1, 0].item() == 0b0010


def test_legacy_path_skips_indices_when_bitmask_already_aligned():
    fake_xgr = FakeXgr()
    import utils as utils_mod

    utils_mod.xgr = fake_xgr

    input_batch = _input_batch(["a", "b"])
    grammar_bitmask = np.array([[0b01], [0b10]], dtype=np.int32)
    grammar_output = GrammarOutput(
        structured_output_request_ids=["a", "b"], grammar_bitmask=grammar_bitmask
    )
    scheduler_output = SchedulerOutput(scheduled_spec_decode_tokens={})
    logits = torch.zeros((2, 32))

    utils_mod.apply_grammar_bitmask(
        scheduler_output, grammar_output, input_batch, logits
    )

    assert fake_xgr.calls[0]["indices"] is None


def test_legacy_path_offsets_by_scheduled_spec_tokens():
    fake_xgr = FakeXgr()
    import utils as utils_mod

    utils_mod.xgr = fake_xgr

    # Batch order ["a", "b"]; "a" has 1 scheduled spec token, so it occupies
    # 2 logits rows (bonus + 1 draft) before "b"'s row starts at index 2.
    input_batch = _input_batch(["a", "b"])
    grammar_bitmask = np.array(
        [[0b01], [0b01], [0b10]], dtype=np.int32
    )  # "a" bonus, "a" draft, "b" bonus
    grammar_output = GrammarOutput(
        structured_output_request_ids=["a", "b"], grammar_bitmask=grammar_bitmask
    )
    scheduler_output = SchedulerOutput(
        scheduled_spec_decode_tokens={"a": [7]}
    )
    logits = torch.zeros((3, 32))

    utils_mod.apply_grammar_bitmask(
        scheduler_output, grammar_output, input_batch, logits
    )

    sorted_bitmask = fake_xgr.calls[0]["bitmask"]
    assert sorted_bitmask[0, 0].item() == 0b01  # a, bonus
    assert sorted_bitmask[1, 0].item() == 0b01  # a, draft
    assert sorted_bitmask[2, 0].item() == 0b10  # b, bonus
