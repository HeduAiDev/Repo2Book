"""TDD tests for the ch18 reduced companion.

These assert the *real vLLM observable behaviors* of the persistent batch and
input preparation, reproduced by the reduced companion:

  - slot reuse: add fills the smallest removed slot first (pop_removed)
  - remove only punches a hole; condense compacts holes so [0, num_reqs) is dense
  - the 2D->1D flat gather: token_indices = positions + req_index * max_model_len
  - positions = num_computed_tokens + per-request offset (unifies prefill/decode)
  - slot mapping: slot = block_table[req, pos//bs] * bs + pos % bs, tail PAD

CPU/numpy paths run on host; the Triton slot-mapping kernel needs CUDA and is
skipped when unavailable.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "implementation")
)

from _support import SamplingParams  # noqa: E402
from gpu_input_batch import CachedRequestState, InputBatch  # noqa: E402
from gpu_model_runner import (  # noqa: E402
    CachedRequestData,
    GPUModelRunner,
    NewRequestData,
    SchedulerOutput,
)

CUDA = torch.cuda.is_available()
BLOCK_SIZE = 16
MAX_MODEL_LEN = 128


def make_input_batch(max_num_reqs=8):
    return InputBatch(
        max_num_reqs=max_num_reqs,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=1024,
        device=torch.device("cpu"),
        pin_memory=False,
        vocab_size=1000,
        block_sizes=[BLOCK_SIZE],
        kernel_block_sizes=[BLOCK_SIZE],
    )


def make_req(req_id, prompt, computed=0, blocks=(0,), outputs=None):
    return CachedRequestState(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(temperature=1.0),
        generator=None,
        block_ids=([list(blocks)]),
        num_computed_tokens=computed,
        output_token_ids=list(outputs or []),
    )


# ---- persistent batch mechanics -------------------------------------------

def test_add_assigns_sequential_slots():
    ib = make_input_batch()
    ib.add_request(make_req("a", [1, 2, 3]))
    ib.add_request(make_req("b", [4, 5]))
    assert ib.req_id_to_index == {"a": 0, "b": 1}
    assert ib.num_reqs == 2
    # token_ids_cpu row holds that request's prompt at its slot.
    assert list(ib.token_ids_cpu[0, :3]) == [1, 2, 3]
    assert list(ib.token_ids_cpu[1, :2]) == [4, 5]


def test_remove_punches_hole_without_compacting():
    ib = make_input_batch()
    for rid, toks in [("a", [1]), ("b", [2]), ("c", [3]), ("d", [4])]:
        ib.add_request(make_req(rid, toks))
    ib.remove_request("b")  # slot 1
    ib.remove_request("d")  # slot 3
    # remove only unbinds the mapping + records the hole; no data shuffled yet.
    assert "b" not in ib.req_id_to_index and "d" not in ib.req_id_to_index
    assert ib.batch_update_builder.removed == [3, 1]  # descending order
    assert ib.num_reqs == 2


def test_add_reuses_smallest_empty_slot():
    ib = make_input_batch()
    for rid, toks in [("a", [1]), ("b", [2]), ("c", [3]), ("d", [4])]:
        ib.add_request(make_req(rid, toks))
    ib.remove_request("b")  # hole at slot 1
    ib.remove_request("d")  # hole at slot 3
    # New request must reuse the *smallest* empty slot (1), not append at 4.
    idx_e = ib.add_request(make_req("e", [7, 8]))
    assert idx_e == 1
    assert list(ib.token_ids_cpu[1, :2]) == [7, 8]
    # Next add reuses the next smallest hole (3).
    idx_f = ib.add_request(make_req("f", [9]))
    assert idx_f == 3


def test_condense_compacts_remaining_holes():
    ib = make_input_batch()
    for rid in ["a", "b", "c", "d"]:
        ib.add_request(make_req(rid, [ord(rid)]))
    ib.remove_request("b")  # hole at 1, no add to fill it
    ib.condense()
    # Tail-most live request (d @ slot 3) slides into the smallest hole (1).
    assert ib.req_id_to_index["d"] == 1
    assert list(ib.token_ids_cpu[1, :1]) == [ord("d")]
    # Indices are now dense [0, num_reqs).
    assert sorted(ib.req_id_to_index.values()) == [0, 1, 2]
    assert ib.num_reqs == 3


def test_condense_skipped_when_holes_all_refilled():
    ib = make_input_batch()
    for rid in ["a", "b", "c"]:
        ib.add_request(make_req(rid, [ord(rid)]))
    ib.remove_request("b")
    ib.add_request(make_req("x", [42]))  # refills slot 1 via pop_removed
    assert not ib.batch_update_builder.removed  # nothing left to condense
    ib.condense()  # no-op
    assert ib.req_id_to_index["x"] == 1


# ---- input preparation: gather index math ---------------------------------

def test_cumsum_and_arange():
    runner = _make_runner()
    out = np.zeros(32, dtype=np.int32)
    cu = runner._get_cumsum_and_arange(np.array([2, 5, 3], dtype=np.int32), out)
    assert list(cu) == [2, 7, 10]
    assert list(out[:10]) == [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]


def test_token_index_flattening():
    # The 2D (req, pos) -> 1D flat index used by torch.index_select.
    M = MAX_MODEL_LEN
    req_indices = np.array([0, 0, 1, 1, 1, 1, 1, 2, 2, 2])
    query_pos = np.array([0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
    num_computed = np.array([0, 0, 0])  # pure prefill
    positions = num_computed[req_indices] + query_pos
    token_indices = positions + req_indices * M
    assert list(token_indices[:2]) == [0, 1]
    assert list(token_indices[2:7]) == [M, M + 1, M + 2, M + 3, M + 4]
    assert list(token_indices[7:]) == [2 * M, 2 * M + 1, 2 * M + 2]


def test_positions_offset_by_num_computed():
    # Decode step: positions absolute = num_computed_tokens + per-req offset.
    req_indices = np.array([0, 1])
    query_pos = np.array([0, 0])
    num_computed = np.array([10, 25])
    positions = num_computed[req_indices] + query_pos
    assert list(positions) == [10, 25]


def _make_runner():
    return GPUModelRunner(
        max_num_reqs=8,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=1024,
        device=torch.device("cuda" if CUDA else "cpu"),
        vocab_size=1000,
        block_size=BLOCK_SIZE,
    )
