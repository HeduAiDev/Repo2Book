# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Subtract-only reduced companion for ch18.
# SOURCE: vllm/v1/worker/gpu_model_runner.py (pin f3fef123)
# The per-step orchestrator: _update_states reconciles the persistent batch from
# the SchedulerOutput; _prepare_inputs gathers input_ids / positions / slot
# mapping; _build_attention_metadata assembles the attention inputs. Reduced to
# the single-rank, non-pooling, non-LoRA, non-spec, token-only main path per the
# approved subtraction_plan.

from dataclasses import dataclass, field

import numpy as np
import torch

from block_table import CpuGpuBuffer
from gpu_input_batch import CachedRequestState, InputBatch


# SOURCE: vllm/v1/core/sched/output.py:L31  NewRequestData (field subset)
@dataclass
class NewRequestData:  # SOURCE: vllm/v1/core/sched/output.py:L31
    req_id: str
    prompt_token_ids: list[int]
    sampling_params: object
    block_ids: tuple[list[int], ...]
    num_computed_tokens: int


# SOURCE: vllm/v1/core/sched/output.py:L112  CachedRequestData (field subset)
@dataclass
class CachedRequestData:  # SOURCE: vllm/v1/core/sched/output.py:L112
    req_ids: list[str] = field(default_factory=list)
    resumed_req_ids: set[str] = field(default_factory=set)
    new_token_ids: list[list[int]] = field(default_factory=list)
    all_token_ids: dict[str, list[int]] = field(default_factory=dict)
    new_block_ids: list[tuple[list[int], ...] | None] = field(default_factory=list)
    num_computed_tokens: list[int] = field(default_factory=list)
    num_output_tokens: list[int] = field(default_factory=list)


# SOURCE: vllm/v1/core/sched/output.py:L181  SchedulerOutput (field subset)
@dataclass
class SchedulerOutput:  # SOURCE: vllm/v1/core/sched/output.py:L181
    scheduled_new_reqs: list[NewRequestData] = field(default_factory=list)
    scheduled_cached_reqs: CachedRequestData = field(default_factory=CachedRequestData)
    num_scheduled_tokens: dict[str, int] = field(default_factory=dict)
    total_num_scheduled_tokens: int = 0
    scheduled_spec_decode_tokens: dict[str, list[int]] = field(default_factory=dict)
    finished_req_ids: set[str] = field(default_factory=set)


# SOURCE: vllm/v1/worker/gpu_model_runner.py  GPUModelRunner (ch18 slice)
class GPUModelRunner:
    """Reduced GPUModelRunner holding only the state and methods on the ch18
    spine: the cached request states, the persistent InputBatch, and the
    per-step buffers consumed by _prepare_inputs."""

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        vocab_size: int,
        block_size: int,
        pin_memory: bool = False,
    ):
        self.device = device
        self.max_model_len = max_model_len

        # req_id -> CachedRequestState: full worker-side snapshot of every request.
        self.requests: dict[str, CachedRequestState] = {}

        self.input_batch = InputBatch(
            max_num_reqs=max_num_reqs,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            device=device,
            pin_memory=pin_memory,
            vocab_size=vocab_size,
            block_sizes=[block_size],
            kernel_block_sizes=[block_size],
        )

        # SUBTRACTED: attention-backend reorder threshold discovery. Approved —
        #   reduced companion keeps the _may_reorder_batch hook as a no-op.
        self.reorder_batch_threshold = None

        # Pre-allocated per-step buffers (CpuGpuBuffer pairs / numpy aranges).
        # SOURCE: vllm/v1/worker/gpu_model_runner.py (buffer allocation in __init__)
        self.arange_np = np.arange(
            max(max_num_reqs + 1, max_num_batched_tokens), dtype=np.int32
        )
        self.query_pos = self._make_buffer(max_num_batched_tokens, dtype=torch.int32)
        self.req_indices = self._make_buffer(max_num_batched_tokens, dtype=torch.int32)
        self.query_start_loc = self._make_buffer(max_num_reqs + 1, dtype=torch.int32)
        self.num_scheduled_tokens = self._make_buffer(max_num_reqs, dtype=torch.int32)
        self.input_ids = self._make_buffer(max_num_batched_tokens, dtype=torch.int32)
        self.positions = torch.zeros(
            max_num_batched_tokens, dtype=torch.int64, device=device
        )
        self.num_computed_tokens = torch.zeros(
            max_num_reqs, dtype=torch.int32, device=device
        )
        self.seq_lens = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.optimistic_seq_lens_cpu = torch.zeros(max_num_reqs, dtype=torch.int32)

    def _make_buffer(self, *size, dtype) -> CpuGpuBuffer:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py (CpuGpuBuffer factory in __init__)
        return CpuGpuBuffer(*size, dtype=dtype, device=self.device, pin_memory=False)

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1003  _may_reorder_batch
    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the order of requests in the batch based on the attention
        backend's needs (e.g. MLA may separate compute- vs memory-bound)."""
        # SUBTRACTED: kv_cache_groups emptiness guard (attention-free models).
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L1018-L1019
        if self.reorder_batch_threshold is not None:
            reorder_batch_to_split_decodes_and_prefills(
                self.input_batch,
                scheduler_output,
                decode_threshold=self.reorder_batch_threshold,
            )

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1065  _update_states
    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output. The updated states are used by `_prepare_inputs`."""
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
        # SUBTRACTED: num_prompt_logprobs.pop / late_interaction_runner.on_requests_
        #   finished. Approved (prompt logprobs / late interaction pooling).
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L1078-L1081

        # Remove the finished requests from the persistent batch.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # SUBTRACTED: _zero_block_ids (fresh-block NaN zeroing) + encoder_cache
        #   free. Approved — orthogonal to the persistent-batch reconciliation.
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L1091-L1098

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap, this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        # SUBTRACTED: ngram_gpu tracking lists init. Approved (async spec decode).
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L1122-L1127

        reqs_to_add: list[CachedRequestState] = []

        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            # SUBTRACTED: streaming same-req_id reuse branch. Approved.
            #   Orig: vllm/v1/worker/gpu_model_runner.py:L1135-L1139

            sampling_params = new_req_data.sampling_params
            # SUBTRACTED: RANDOM_SEED generator construction + pooling pooler
            #   updates. Approved (seeded RNG plumbing / pooling).
            #   Orig: vllm/v1/worker/gpu_model_runner.py:L1144-L1160
            generator = None

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
            )
            self.requests[req_id] = req_state
            # SUBTRACTED: late_interaction_runner.register_request / num_prompt_logprobs
            #   / M-RoPE / XD-RoPE init / ngram_gpu tracking. Approved.
            #   Orig: vllm/v1/worker/gpu_model_runner.py:L1177-L1197
            reqs_to_add.append(req_state)

        # Update the states of the running/resumed requests.
        # SUBTRACTED: is_last_rank handling (PP) — reduced companion is last rank.
        req_data = scheduler_output.scheduled_cached_reqs
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        # SUBTRACTED: ngram_gpu original_num_spec_per_req / async spec-decode
        #   prev_num_draft_tokens bookkeeping. Approved.
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L1204-L1220

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            num_output_tokens = req_data.num_output_tokens[i]
            req_index = self.input_batch.req_id_to_index.get(req_id)

            # SUBTRACTED: async-scheduling prev_num_draft_len optimistic counting +
            #   deferred spec-decode corrections. Approved.
            #   Orig: vllm/v1/worker/gpu_model_runner.py:L1230-L1271

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            # SUBTRACTED: non-last-rank token backfill (PP). On the last rank the
            #   sampled tokens are already cached, so no token_ids update needed
            #   here. Approved (pipeline parallelism).
            #   Orig: vllm/v1/worker/gpu_model_runner.py:L1275-L1295
            if num_output_tokens < len(req_state.output_token_ids):
                # Some output tokens were discarded due to a sync-KV-load failure.
                # Align the cached state.
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = (
                        self.input_batch.num_prompt_tokens[req_index]
                        + num_output_tokens
                    )
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert req_index is None
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            if req_index is None:
                # The request is not in the persistent batch. It was either
                # preempted and resumed later, or was not scheduled in the
                # previous step and needs to be added again.
                # SUBTRACTED: async-scheduling output_token_ids recovery +
                #   ngram_gpu tracking. Approved.
                #   Orig: vllm/v1/worker/gpu_model_runner.py:L1326-L1335
                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # SUBTRACTED: non-last-rank token_ids_cpu backfill (PP). On the last
            #   rank the sampled tokens are written by bookkeeping, not here.
            #   Orig: vllm/v1/worker/gpu_model_runner.py:L1345-L1352

            # Add spec_token_ids to token_ids_cpu.
            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)
            # SUBTRACTED: ngram trimming restore of prev_num_draft_len. Approved.
            #   Orig: vllm/v1/worker/gpu_model_runner.py:L1356-L1360

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

        # SUBTRACTED: ngram_gpu incremental tensor update + deferred spec-decode
        #   correction closure return. Approved (async spec decode).
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L1375-L1419

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1572  _get_cumsum_and_arange
    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        arange_out: np.ndarray,
        cumsum_dtype=None,
    ) -> np.ndarray:
        """Get the cumulative sum and batched arange of the given array.
        E.g., [2, 5, 3] -> [2, 7, 10], arange written to
        arange_out[:10] as [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]."""
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        np.subtract(
            self.arange_np[:total_num_tokens],
            cumsums_offsets,
            out=arange_out[:total_num_tokens],
        )
        return cu_num_tokens

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1787  _prepare_inputs
    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> torch.Tensor:
        """Gather input_ids / positions / slot_mapping for the current batch.

        Returns logits_indices (the last-token index of each request).
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # self.query_pos.np[:10]: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )

        # Get positions.
        positions_np = (
            self.input_batch.num_computed_tokens_cpu[req_indices]
            + self.query_pos.np[: cu_num_tokens[-1]]
        )

        # SUBTRACTED: M-RoPE / XD-RoPE position calc. Approved (multimodal).
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L1825-L1833

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, ..., 2 * M, 2 * M + 1, 2 * M + 2] where M=max_model_len.
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        token_indices_tensor = torch.from_numpy(token_indices)

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster for large tensors.
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        # SUBTRACTED: prompt_embeds is_token_ids gather + per-request embeds fill.
        #   Approved (prompt_embeds). Orig: gpu_model_runner.py:L1853-L1898

        # Prepare the attention metadata.
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()

        # Optimistic seq_lens (CPU) used by _build_attention_metadata (max_seq_len).
        torch.add(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            torch.from_numpy(num_scheduled_tokens),
            out=self.optimistic_seq_lens_cpu[:num_reqs],
        )
        self.optimistic_seq_lens_cpu[num_reqs:].fill_(0)

        # SUBTRACTED: prev_positions / discard_request_mask / num_accepted_tokens
        #   sync + async spec-decode num_computed_tokens GPU correction. Approved.
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L1920-L1984
        self.num_computed_tokens[:num_reqs].copy_(
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs],
            non_blocking=True,
        )

        self.req_indices.np[:total_num_scheduled_tokens] = req_indices
        self.req_indices.copy_to_gpu(total_num_scheduled_tokens)
        req_indices_gpu = self.req_indices.gpu[:total_num_scheduled_tokens]

        self.query_pos.copy_to_gpu(total_num_scheduled_tokens)
        self.num_scheduled_tokens.np[:num_reqs] = num_scheduled_tokens
        self.num_scheduled_tokens.copy_to_gpu(num_reqs)
        num_scheduled_tokens_gpu = self.num_scheduled_tokens.gpu[:num_reqs]
        self.positions[:total_num_scheduled_tokens] = (
            self.num_computed_tokens[req_indices_gpu].to(torch.int64)
            + self.query_pos.gpu[:total_num_scheduled_tokens]
        )
        self.seq_lens[:num_reqs] = (
            self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
        )
        self.seq_lens[num_reqs:].fill_(0)

        self.input_batch.block_table.compute_slot_mapping(
            num_reqs,
            self.query_start_loc.gpu[: num_reqs + 1],
            self.positions[:total_num_scheduled_tokens],
        )

        # Copy the input ids to the GPU.
        # SUBTRACTED: _prepare_input_ids prev_sampled_token_ids backfill (async
        #   scheduling). Reduced companion does the normal-scheduling copy.
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L1613-L1636, L2015
        self.input_ids.copy_to_gpu(total_num_scheduled_tokens)

        # SUBTRACTED: spec_decode_metadata branch + LoRA hot-swap. Approved.
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L2043-L2091
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]
        logits_indices = query_start_loc[1:] - 1
        return logits_indices

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L2098  _build_attention_metadata
    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
    ) -> dict:
        """Assemble CommonAttentionMetadata from the persistent batch's block
        table GPU mirror + slot_mapping + seq_lens + query_start_loc."""
        # SUBTRACTED: padding (num_*_padded), ubatch_slices, cudagraph capture,
        #   EncoderOnlyAttentionSpec zero-table branch, routed_experts, DCP local
        #   seq lens, async spec-decode CPU-field nulling, hybrid KV-group cache.
        #   Approved. Orig: gpu_model_runner.py:L2103-末
        max_seq_len = self.optimistic_seq_lens_cpu.numpy()[:num_reqs].max().item()

        blk_table = self.input_batch.block_table[0]
        block_table_tensor = blk_table.get_device_tensor(num_reqs)
        slot_mapping = blk_table.slot_mapping.gpu[:num_tokens]

        num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[
            :num_reqs
        ]
        num_prompt_tokens_cpu = self.input_batch.num_prompt_tokens_cpu_tensor[:num_reqs]
        seq_lens_cpu = self.optimistic_seq_lens_cpu[:num_reqs]

        # is_prefilling: True if request is still in prefill phase.
        is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu

        # CommonAttentionMetadata is the attention-backend interface; reduced to a
        # plain dict here (the dataclass + per-backend build belong to the
        # attention chapters).
        cm_base = dict(
            query_start_loc=self.query_start_loc.gpu[: num_reqs + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs + 1],
            seq_lens=self.seq_lens[:num_reqs],
            seq_lens_cpu=seq_lens_cpu,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            block_table_tensor=block_table_tensor,
            slot_mapping=slot_mapping,
            causal=True,
            is_prefilling=is_prefilling,
            positions=self.positions[:num_tokens],
        )
        return cm_base
