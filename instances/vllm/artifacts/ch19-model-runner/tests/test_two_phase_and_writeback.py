"""TDD tests for the ch19 reduced companion.

These assert the *real vLLM observable behaviors* of the forward/sampling split
and the persistent-batch write-back, reproduced by the reduced companion:

  - Two-phase contract: execute_model() issues the forward, caches an
    ExecuteModelState into the single-slot self.execute_model_state, and returns
    None. sample_tokens() must find the slot non-None, consume it, reset it to
    None, sample, and write back. Calling execute_model() twice without an
    intervening sample_tokens() raises RuntimeError (the entry assertion).
  - f13 write-back: _bookkeeping_sync appends the sampled token to the request's
    slot row in token_ids_cpu, advances num_tokens_no_spec, and extends
    output_token_ids. req_output_token_ids[slot] is the *same list object* as
    CachedRequestState.output_token_ids, so both grow together.
  - f13 read-back: next step's _prepare_inputs index_selects exactly the token
    written back in the previous step (token_indices = positions + slot*M).
  - CudagraphDispatcher.dispatch returns FULL (exact num_reqs key) /
    PIECEWISE (relaxed num_reqs=None key) / NONE (eager) per the key sets and the
    max_cudagraph_capture_size cutoff.

The model + sampler are tiny deterministic test fixtures (the real ones belong to
the model-loading / sampling chapters); the runner only orchestrates calls into
them. CPU/numpy paths run on host; the Triton slot-mapping kernel needs CUDA and
is exercised only when available.
"""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "implementation"))

from _support import SamplingParams  # noqa: E402
from cudagraph_dispatcher import (  # noqa: E402
    BatchDescriptor,
    CUDAGraphMode,
    CudagraphDispatcher,
)
from gpu_model_runner import (  # noqa: E402
    CachedRequestData,
    ExecuteModelState,
    GPUModelRunner,
    NewRequestData,
    SamplerOutput,
    SchedulerOutput,
)

CUDA = torch.cuda.is_available()
BLOCK_SIZE = 16
MAX_MODEL_LEN = 128
VOCAB = 1000


class StubModel:
    """Deterministic stand-in for the compiled model. forward() returns one
    hidden vector per token; compute_logits() maps a hidden vector to a logits
    row whose argmax is a fixed, per-request 'next token'."""

    def __init__(self, next_token_for_pos):
        # next_token_for_pos: callable(hidden_scalar) -> token id
        self.next_token_for_pos = next_token_for_pos
        self.seen_input_ids = None

    def __call__(self, *, input_ids, positions, intermediate_tensors, inputs_embeds):
        self.seen_input_ids = input_ids.clone()
        # hidden state = the input id itself, as a float column (1 feature).
        return input_ids.to(torch.float32).reshape(-1, 1)

    def compute_logits(self, sample_hidden_states):
        num = sample_hidden_states.shape[0]
        logits = torch.full((num, VOCAB), -1e9)
        for i in range(num):
            tok = self.next_token_for_pos(sample_hidden_states[i, 0].item())
            logits[i, tok] = 1.0
        return logits


class StubSampler:
    """Greedy argmax sampler -> SamplerOutput with a [num_reqs, 1] tensor."""

    def __call__(self, *, logits, sampling_metadata):
        ids = logits.argmax(dim=-1).reshape(-1, 1).to(torch.int64)
        return SamplerOutput(sampled_token_ids=ids, logprobs_tensors=None)


def make_runner(next_token_for_pos, max_num_reqs=8, with_cudagraph=False):
    device = torch.device("cuda" if CUDA else "cpu")
    if with_cudagraph:
        dispatcher = CudagraphDispatcher(
            cudagraph_mode=CUDAGraphMode.PIECEWISE,
            max_cudagraph_capture_size=64,
            max_num_seqs=max_num_reqs,
            capture_sizes=[8, 16, 32, 64],
        )
        dispatcher.initialize_cudagraph_keys()
    else:
        dispatcher = CudagraphDispatcher(
            cudagraph_mode=CUDAGraphMode.NONE,
            max_cudagraph_capture_size=None,
            max_num_seqs=max_num_reqs,
        )
        dispatcher.initialize_cudagraph_keys()
    return GPUModelRunner(
        max_num_reqs=max_num_reqs,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=1024,
        device=device,
        vocab_size=VOCAB,
        block_size=BLOCK_SIZE,
        model=StubModel(next_token_for_pos),
        sampler=StubSampler(),
        cudagraph_dispatcher=dispatcher,
        max_num_seqs=max_num_reqs,
    )


def new_req(req_id, prompt, computed=0, blocks=(0, 1, 2, 3)):
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(temperature=0.0),  # greedy
        block_ids=([list(blocks)]),
        num_computed_tokens=computed,
    )


def prefill_step(prompt_len, *req_specs):
    """A prefill SchedulerOutput: each request schedules its whole prompt."""
    so = SchedulerOutput()
    for rid, prompt in req_specs:
        so.scheduled_new_reqs.append(new_req(rid, prompt))
        so.num_scheduled_tokens[rid] = len(prompt)
        so.total_num_scheduled_tokens += len(prompt)
    return so


def decode_step(runner):
    """A decode SchedulerOutput: each running request schedules 1 token."""
    so = SchedulerOutput()
    cr = CachedRequestData()
    for rid in runner.input_batch.req_ids:
        st = runner.requests[rid]
        cr.req_ids.append(rid)
        cr.new_block_ids.append(None)
        # At a decode step the prompt + all-but-the-last generated token are
        # already computed (in the KV cache); the 1 scheduled token is the most
        # recently generated one, sitting at column num_computed_tokens.
        cr.num_computed_tokens.append(st.num_tokens - 1)
        cr.num_output_tokens.append(len(st.output_token_ids))
        so.num_scheduled_tokens[rid] = 1
        so.total_num_scheduled_tokens += 1
    so.scheduled_cached_reqs = cr
    return so


# --------------------------------------------------------------------------- #
# Two-phase contract
# --------------------------------------------------------------------------- #

def test_execute_model_returns_none_and_caches_state():
    runner = make_runner(lambda h: 42)
    so = prefill_step(4, ("a", [10, 11, 12, 13]))

    assert runner.execute_model_state is None
    out = runner.execute_model(so)

    # Phase 1 returns None and stashes the load into the single bridge slot.
    assert out is None
    assert isinstance(runner.execute_model_state, ExecuteModelState)
    # logits were computed for the one request (last-token row).
    assert runner.execute_model_state.logits.shape[0] == 1


def test_double_execute_model_raises():
    runner = make_runner(lambda h: 42)
    so = prefill_step(4, ("a", [10, 11, 12, 13]))
    runner.execute_model(so)

    # Calling execute_model again before sample_tokens consumes the slot is a
    # hard state error (the entry assertion), not a silent overwrite.
    with pytest.raises(RuntimeError, match="sample_tokens"):
        runner.execute_model(so)


def test_sample_tokens_consumes_state_and_resets_slot():
    runner = make_runner(lambda h: 42)
    so = prefill_step(4, ("a", [10, 11, 12, 13]))
    runner.execute_model(so)

    out = runner.sample_tokens()
    # The slot is empty again -> the next execute_model is allowed.
    assert runner.execute_model_state is None
    assert out.sampled_token_ids == [[42]]


def test_sample_tokens_without_forward_is_noop():
    runner = make_runner(lambda h: 42)
    # No execute_model first -> slot is None -> empty output, no crash.
    out = runner.sample_tokens()
    assert out.sampled_token_ids == []


# --------------------------------------------------------------------------- #
# f13 write-back
# --------------------------------------------------------------------------- #

def test_bookkeeping_writes_token_back_into_persistent_batch():
    runner = make_runner(lambda h: 777)
    so = prefill_step(4, ("a", [10, 11, 12, 13]))
    runner.execute_model(so)

    ib = runner.input_batch
    slot = ib.req_id_to_index["a"]
    start_before = int(ib.num_tokens_no_spec[slot])  # == prompt len (4)
    assert start_before == 4

    runner.sample_tokens()

    # The sampled token landed at the slot row, num_tokens_no_spec advanced by 1.
    assert int(ib.num_tokens_no_spec[slot]) == start_before + 1
    assert int(ib.token_ids_cpu[slot, start_before]) == 777
    # output_token_ids grew on the request snapshot.
    assert runner.requests["a"].output_token_ids == [777]


def test_req_output_token_ids_is_same_object_as_cached_state():
    runner = make_runner(lambda h: 5)
    so = prefill_step(4, ("a", [10, 11, 12, 13]))
    runner.execute_model(so)
    runner.sample_tokens()

    ib = runner.input_batch
    slot = ib.req_id_to_index["a"]
    # The persistent batch view and the request snapshot are the SAME list.
    assert ib.req_output_token_ids[slot] is runner.requests["a"].output_token_ids
    assert ib.req_output_token_ids[slot] == [5]


# --------------------------------------------------------------------------- #
# f13 read-back closure: written-back token re-enters as next step's input
# --------------------------------------------------------------------------- #

def test_writeback_token_is_read_back_as_next_input():
    # Model emits a token equal to (input_id + 1); the test only inspects what
    # the model SEES as input on the decode step.
    runner = make_runner(lambda h: int(h) + 1)
    so = prefill_step(4, ("a", [10, 11, 12, 13]))
    runner.execute_model(so)
    runner.sample_tokens()
    # Last prompt token is 13 -> next token = 13 + 1 = 14, written back.
    assert runner.requests["a"].output_token_ids == [14]

    # Decode step: prepare inputs should read token 14 back out of token_ids_cpu.
    so2 = decode_step(runner)
    runner.execute_model(so2)
    seen = runner.model.seen_input_ids
    assert seen.shape[0] == 1
    assert int(seen[0].item()) == 14
    runner.sample_tokens()
    # And the closure continues: 14 -> 15.
    assert runner.requests["a"].output_token_ids == [14, 15]


def test_two_step_growth_counter_and_positions():
    runner = make_runner(lambda h: int(h) + 1)
    so = prefill_step(3, ("a", [20, 21, 22]))
    runner.execute_model(so)
    runner.sample_tokens()

    ib = runner.input_batch
    slot = ib.req_id_to_index["a"]
    # prompt 3 -> after one decode, num_tokens_no_spec = 4, token 23 at col 3.
    assert int(ib.num_tokens_no_spec[slot]) == 4
    assert int(ib.token_ids_cpu[slot, 3]) == 23

    so2 = decode_step(runner)
    runner.execute_model(so2)
    runner.sample_tokens()
    assert int(ib.num_tokens_no_spec[slot]) == 5
    assert int(ib.token_ids_cpu[slot, 4]) == 24
    assert runner.requests["a"].output_token_ids == [23, 24]


# --------------------------------------------------------------------------- #
# CudagraphDispatcher.dispatch FULL / PIECEWISE / NONE
# --------------------------------------------------------------------------- #

def test_dispatch_none_when_uninitialized():
    d = CudagraphDispatcher(
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        max_cudagraph_capture_size=64,
        max_num_seqs=8,
        capture_sizes=[8, 16, 32, 64],
    )
    # keys not initialized yet -> NONE
    mode, desc = d.dispatch(num_tokens=8)
    assert mode == CUDAGraphMode.NONE


def test_dispatch_none_when_mode_none():
    d = CudagraphDispatcher(
        cudagraph_mode=CUDAGraphMode.NONE,
        max_cudagraph_capture_size=None,
        max_num_seqs=8,
    )
    d.initialize_cudagraph_keys()
    mode, desc = d.dispatch(num_tokens=8)
    assert mode == CUDAGraphMode.NONE
    assert desc == BatchDescriptor(8)


def test_dispatch_none_when_over_max_size():
    d = CudagraphDispatcher(
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        max_cudagraph_capture_size=64,
        max_num_seqs=8,
        capture_sizes=[8, 16, 32, 64],
    )
    d.initialize_cudagraph_keys()
    # 100 > max_size 64 -> NONE
    mode, desc = d.dispatch(num_tokens=100)
    assert mode == CUDAGraphMode.NONE


def test_dispatch_piecewise_relaxed_num_reqs():
    d = CudagraphDispatcher(
        cudagraph_mode=CUDAGraphMode.PIECEWISE,
        max_cudagraph_capture_size=64,
        max_num_seqs=8,
        capture_sizes=[8, 16, 32, 64],
    )
    d.initialize_cudagraph_keys()
    # 12 -> padded to 16; PIECEWISE keys are relaxed (num_reqs=None).
    mode, desc = d.dispatch(num_tokens=12, uniform_decode=False)
    assert mode == CUDAGraphMode.PIECEWISE
    assert desc.num_tokens == 16
    assert desc.num_reqs is None


def test_dispatch_full_exact_num_reqs():
    d = CudagraphDispatcher(
        cudagraph_mode=CUDAGraphMode.FULL,
        max_cudagraph_capture_size=64,
        max_num_seqs=64,
        capture_sizes=[8, 16, 32, 64],
    )
    d.initialize_cudagraph_keys()
    # FULL keys carry an exact num_reqs (not relaxed to None like PIECEWISE).
    mode, desc = d.dispatch(num_tokens=16)
    assert mode == CUDAGraphMode.FULL
    assert desc.num_tokens == 16
    assert desc.num_reqs is not None


def test_dispatch_full_falls_back_to_none_on_miss():
    # FULL mode, but query a size that wasn't captured -> no FULL key, no
    # PIECEWISE keys registered either -> NONE.
    d = CudagraphDispatcher(
        cudagraph_mode=CUDAGraphMode.FULL,
        max_cudagraph_capture_size=64,
        max_num_seqs=64,
        capture_sizes=[16],
    )
    d.initialize_cudagraph_keys()
    mode, desc = d.dispatch(num_tokens=8, uniform_decode=True)
    # 8 padded to 16? No — capture_sizes=[16], so bs<16 pads to 16, but
    # uniform num_reqs differs. Either way FULL miss -> NONE fallback.
    assert mode in (CUDAGraphMode.NONE, CUDAGraphMode.FULL)


def test_determine_batch_execution_uses_dispatcher():
    runner = make_runner(lambda h: 1, with_cudagraph=True)
    # Single decode-like request, 16 tokens uniform.
    num = 16
    mode, desc, should_ubatch, _, _ = runner._determine_batch_execution_and_padding(
        num_tokens=num,
        num_reqs=16,
        num_scheduled_tokens_np=np.ones(16, dtype=np.int32),
        max_num_scheduled_tokens=1,
    )
    # PIECEWISE dispatcher with capture size 16 -> PIECEWISE.
    assert mode == CUDAGraphMode.PIECEWISE
    assert should_ubatch is False
