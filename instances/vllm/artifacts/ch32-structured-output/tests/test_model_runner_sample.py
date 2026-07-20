"""m18: the landing site -- GPUModelRunner.sample() applies the grammar
bitmask to logits in-place right after compute_logits() and before any
sampler runs (ch30/ch34 own everything after this point).

SOURCE anchors: vllm/v1/worker/gpu/model_runner.py:L906-922
"""
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "implementation")
)

import numpy as np
import pytest
import torch

from model_runner import GPUModelRunner  # noqa: E402
from output import GrammarOutput  # noqa: E402


class FakeInputBatch:
    def __init__(self, logits_indices):
        self.logits_indices = logits_indices


class FakeModel:
    def __init__(self, logits: torch.Tensor):
        self._logits = logits
        self.calls = 0

    def compute_logits(self, hidden_states):
        self.calls += 1
        return self._logits


class RecordingWorker:
    def __init__(self):
        self.calls = []

    def apply_grammar_bitmask(self, logits, input_batch, req_ids, bitmask):
        self.calls.append((logits, input_batch, req_ids, bitmask))
        logits[0, 0] = float("-inf")  # simulate an in-place mask


def test_sample_applies_bitmask_after_compute_logits_when_present():
    logits = torch.zeros((1, 8))
    model = FakeModel(logits)
    worker = RecordingWorker()
    runner = GPUModelRunner(model=model, structured_outputs_worker=worker)

    hidden_states = torch.zeros((1, 4))
    input_batch = FakeInputBatch(logits_indices=torch.tensor([0]))
    grammar_output = GrammarOutput(
        structured_output_request_ids=["r0"],
        grammar_bitmask=np.zeros((1, 1), dtype=np.int32),
    )

    out = runner.sample(hidden_states, input_batch, grammar_output)

    assert model.calls == 1
    assert len(worker.calls) == 1
    called_logits, called_batch, req_ids, bitmask = worker.calls[0]
    assert called_logits is logits
    assert req_ids == ["r0"]
    assert out[0, 0].item() == float("-inf")  # mutated in place, then returned


def test_sample_skips_worker_when_grammar_output_is_none():
    logits = torch.ones((1, 8))
    model = FakeModel(logits)
    worker = RecordingWorker()
    runner = GPUModelRunner(model=model, structured_outputs_worker=worker)

    out = runner.sample(
        torch.zeros((1, 4)), FakeInputBatch(logits_indices=torch.tensor([0])), None
    )
    assert worker.calls == []
    assert torch.all(out == 1.0)


def test_sample_asserts_worker_present_when_grammar_output_given():
    logits = torch.zeros((1, 8))
    model = FakeModel(logits)
    runner = GPUModelRunner(model=model, structured_outputs_worker=None)

    grammar_output = GrammarOutput(
        structured_output_request_ids=["r0"],
        grammar_bitmask=np.zeros((1, 1), dtype=np.int32),
    )
    with pytest.raises(AssertionError):
        runner.sample(
            torch.zeros((1, 4)),
            FakeInputBatch(logits_indices=torch.tensor([0])),
            grammar_output,
        )
