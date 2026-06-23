"""GPU/Triton tests for ch18: slot-mapping kernel + end-to-end prepare_inputs.

These exercise the real Triton _compute_slot_mapping_kernel and the GPU paths of
_prepare_inputs, so they require CUDA. On host they are skipped; in the vLLM
container run with: scripts/vllm_docker.sh -m pytest /work/.../tests/
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
from block_table import PAD_SLOT_ID, BlockTable, CpuGpuBuffer  # noqa: E402
from gpu_input_batch import CachedRequestState  # noqa: E402
from gpu_model_runner import (  # noqa: E402
    CachedRequestData,
    GPUModelRunner,
    NewRequestData,
    SchedulerOutput,
)

CUDA = torch.cuda.is_available()
pytestmark = pytest.mark.skipif(not CUDA, reason="requires CUDA + Triton")

BLOCK_SIZE = 16
MAX_MODEL_LEN = 128


def test_slot_mapping_formula():
    # A single request whose logical blocks map to physical blocks [7, 3, 9].
    dev = torch.device("cuda")
    bt = BlockTable(
        block_size=BLOCK_SIZE,
        max_num_reqs=4,
        max_num_blocks_per_req=8,
        max_num_batched_tokens=64,
        pin_memory=False,
        device=dev,
        kernel_block_size=BLOCK_SIZE,
    )
    bt.add_row([7, 3, 9], row_idx=0)
    bt.commit_block_table(num_reqs=1)

    # Tokens at positions 16..18 (logical block 1 -> physical 3) and 33
    # (logical block 2 -> physical 9).
    positions = torch.tensor([16, 17, 18, 33], dtype=torch.int64, device=dev)
    qsl = CpuGpuBuffer(2, dtype=torch.int32, device=dev, pin_memory=False)
    qsl.np[0] = 0
    qsl.np[1] = 4
    qsl.copy_to_gpu()

    bt.compute_slot_mapping(1, qsl.gpu[:2], positions)
    slots = bt.slot_mapping.gpu[:4].cpu().tolist()
    # slot = block * block_size + pos % block_size
    assert slots == [3 * 16 + 0, 3 * 16 + 1, 3 * 16 + 2, 9 * 16 + 1]
    # Tail beyond num_tokens is padded with PAD_SLOT_ID for CUDA-graph capture.
    tail = bt.slot_mapping.gpu[4:64].cpu().tolist()
    assert set(tail) == {PAD_SLOT_ID}


def _new_req(req_id, prompt, blocks, computed=0):
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        sampling_params=SamplingParams(temperature=1.0),
        block_ids=([list(blocks)]),
        num_computed_tokens=computed,
    )


def test_end_to_end_prefill_gather():
    runner = GPUModelRunner(
        max_num_reqs=8,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=1024,
        device=torch.device("cuda"),
        vocab_size=1000,
        block_size=BLOCK_SIZE,
    )
    # Two new requests, full prefill.
    p0 = [11, 12, 13]
    p1 = [21, 22, 23, 24, 25]
    so = SchedulerOutput(
        scheduled_new_reqs=[_new_req("a", p0, [0]), _new_req("b", p1, [1])],
        scheduled_cached_reqs=CachedRequestData(),
        num_scheduled_tokens={"a": 3, "b": 5},
        total_num_scheduled_tokens=8,
    )
    runner._update_states(so)
    assert runner.input_batch.num_reqs == 2

    num_sched = np.array([3, 5], dtype=np.int32)
    logits_indices = runner._prepare_inputs(so, num_sched)

    # input_ids gathered in flat, request-major order = p0 ++ p1.
    got = runner.input_ids.gpu[:8].cpu().tolist()
    assert got == p0 + p1
    # positions: each request starts at its num_computed_tokens (0 here).
    pos = runner.positions[:8].cpu().tolist()
    assert pos == [0, 1, 2, 0, 1, 2, 3, 4]
    # logits_indices point at the last token of each request: cu=[3,8] -> [2,7].
    assert logits_indices.cpu().tolist() == [2, 7]
    # seq_lens = num_computed + num_scheduled.
    assert runner.seq_lens[:2].cpu().tolist() == [3, 5]


def test_decode_step_positions_advance():
    runner = GPUModelRunner(
        max_num_reqs=8,
        max_model_len=MAX_MODEL_LEN,
        max_num_batched_tokens=1024,
        device=torch.device("cuda"),
        vocab_size=1000,
        block_size=BLOCK_SIZE,
    )
    so = SchedulerOutput(
        scheduled_new_reqs=[_new_req("a", [11, 12, 13], [0])],
        scheduled_cached_reqs=CachedRequestData(),
        num_scheduled_tokens={"a": 3},
        total_num_scheduled_tokens=3,
    )
    runner._update_states(so)
    runner._prepare_inputs(so, np.array([3], dtype=np.int32))

    # Simulate one decode token cached on the runner side.
    runner.requests["a"].output_token_ids.append(99)
    runner.input_batch.token_ids_cpu[0, 3] = 99
    runner.input_batch.num_tokens_no_spec[0] = 4

    so2 = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=["a"],
            new_block_ids=[None],
            num_computed_tokens=[3],
            num_output_tokens=[1],
        ),
        num_scheduled_tokens={"a": 1},
        total_num_scheduled_tokens=1,
    )
    runner._update_states(so2)
    runner._prepare_inputs(so2, np.array([1], dtype=np.int32))
    # The single decode token is the one cached at position 3.
    assert runner.input_ids.gpu[:1].cpu().tolist() == [99]
    assert runner.positions[:1].cpu().tolist() == [3]
