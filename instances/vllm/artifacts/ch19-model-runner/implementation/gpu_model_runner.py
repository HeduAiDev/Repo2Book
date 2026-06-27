# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Subtract-only reduced companion for ch19.
# SOURCE: vllm/v1/worker/gpu_model_runner.py (pin f3fef123)
#
# The two-phase per-step orchestrator. execute_model() runs preprocess +
# forward, then *caches* logits/hidden_states/etc. into self.execute_model_state
# (an ExecuteModelState NamedTuple) and returns None — the forward has been
# issued to the GPU but its results are not yet consumed. sample_tokens() later
# unpacks that state, samples, and _bookkeeping_sync() writes the new tokens back
# into the persistent batch (token_ids_cpu slot rows + output_token_ids growth,
# f13). CUDA-graph dispatch (PIECEWISE / FULL / NONE) is chosen by
# _determine_batch_execution_and_padding -> CudagraphDispatcher.dispatch.
#
# Reduced to the single-rank, non-pooling, non-LoRA, non-spec-decode,
# non-PP, non-KV-connector, token-only main path per the approved
# subtraction_plan. _update_states / _prepare_inputs carry over the ch18
# persistent-batch reconciliation (their own subtractions are re-stated here).

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import NamedTuple

import numpy as np
import torch

from cudagraph_dispatcher import (
    BatchDescriptor,
    CUDAGraphMode,
    CudagraphDispatcher,
)
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
    scheduled_encoder_inputs: dict = field(default_factory=dict)
    finished_req_ids: set[str] = field(default_factory=set)


# SOURCE: vllm/v1/outputs.py  SamplerOutput (field subset used by ch19)
@dataclass
class SamplerOutput:  # SOURCE: vllm/v1/outputs.py SamplerOutput
    # 2D: [num_reqs, num_generated_tokens_per_req]. Without spec decode the
    # second dim is 1.
    sampled_token_ids: torch.Tensor
    logprobs_tensors: object | None = None


# SOURCE: vllm/v1/outputs.py:  ModelRunnerOutput (field subset)
@dataclass
class ModelRunnerOutput:  # SOURCE: vllm/v1/outputs.py ModelRunnerOutput
    req_ids: list[str]
    req_id_to_index: dict[str, int]
    sampled_token_ids: list[list[int]]
    logprobs: object | None = None
    prompt_logprobs_dict: dict | None = None
    num_nans_in_logits: dict | None = None
    cudagraph_stats: object | None = None


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L383  ExecuteModelState
class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: object | None
    spec_decode_common_attn_metadata: object | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None
    ec_connector_output: object | None
    cudagraph_stats: object | None
    slot_mappings: object | None


# SOURCE: vllm/forward_context.py  set_forward_context (reduced context manager)
@contextmanager
def set_forward_context(  # SOURCE: vllm/forward_context.py set_forward_context
    attn_metadata,
    *,
    num_tokens=None,
    cudagraph_runtime_mode: CUDAGraphMode = CUDAGraphMode.NONE,
    batch_descriptor: BatchDescriptor | None = None,
    slot_mapping=None,
):
    """Publish the cudagraph runtime mode + batch descriptor into a global
    forward context. The compiled model reads these inside its forward to decide
    whether to replay a captured CUDA graph (and which one). Reduced here to the
    fields ch19 cares about — the full implementation also carries DP/ubatch and
    attention metadata. SOURCE: vllm/forward_context.py set_forward_context."""
    global _forward_context
    prev = _forward_context
    _forward_context = {
        "attn_metadata": attn_metadata,
        "num_tokens": num_tokens,
        "cudagraph_runtime_mode": cudagraph_runtime_mode,
        "batch_descriptor": batch_descriptor,
        "slot_mapping": slot_mapping,
    }
    try:
        yield
    finally:
        _forward_context = prev


_forward_context: dict | None = None


def get_forward_context() -> dict | None:  # SOURCE: vllm/forward_context.py get_forward_context
    return _forward_context


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L399  GPUModelRunner (ch19 slice)
class GPUModelRunner:
    """Reduced GPUModelRunner exposing the ch19 two-phase spine:

      execute_model()  -> preprocess (_update_states / _prepare_inputs) ->
                          _determine_batch_execution_and_padding ->
                          set_forward_context(_model_forward) -> compute_logits ->
                          cache ExecuteModelState -> return None
      sample_tokens()  -> unpack state -> _sample -> _bookkeeping_sync (write the
                          new tokens back into the persistent batch) -> output
    """

    def __init__(
        self,
        max_num_reqs: int,
        max_model_len: int,
        max_num_batched_tokens: int,
        device: torch.device,
        vocab_size: int,
        block_size: int,
        model,
        sampler,
        cudagraph_dispatcher: "CudagraphDispatcher",
        max_num_seqs: int | None = None,
        pin_memory: bool = False,
    ):
        self.device = device
        self.max_model_len = max_model_len
        self.max_num_seqs = max_num_seqs if max_num_seqs is not None else max_num_reqs

        # The compiled model and the sampler. They are produced by other
        # subsystems (model loading / the sampler chapter); the runner only
        # orchestrates calls into them.
        self.model = model
        self.sampler = sampler
        self.cudagraph_dispatcher = cudagraph_dispatcher
        self.uniform_decode_query_len = 1

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

        # The single-slot bridge between the two phases. execute_model() fills
        # it and returns None; sample_tokens() must find it non-None, consume it,
        # and reset it to None.
        # SOURCE: vllm/v1/worker/gpu_model_runner.py (self.execute_model_state init)
        self.execute_model_state: ExecuteModelState | None = None

        # SUBTRACTED: async-scheduling toggle is fixed False in the reduced
        #   companion — bookkeeping writes the real sampled ids straight into
        #   token_ids_cpu instead of the -1 placeholder. Approved (async sched).
        self.use_async_scheduling = False

        # Pre-allocated per-step buffers (CpuGpuBuffer pairs / numpy aranges).
        # SOURCE: vllm/v1/worker/gpu_model_runner.py (buffer allocation in __init__)
        self.arange_np = np.arange(
            max(max_num_reqs + 1, max_num_batched_tokens), dtype=np.int32
        )
        self.query_pos = self._make_buffer(max_num_batched_tokens, dtype=torch.int32)
        self.query_start_loc = self._make_buffer(max_num_reqs + 1, dtype=torch.int32)
        self.input_ids = self._make_buffer(max_num_batched_tokens, dtype=torch.int32)
        self.positions = torch.zeros(
            max_num_batched_tokens, dtype=torch.int64, device=device
        )

        # discard_request_mask flags requests whose sampled token must be dropped
        # (e.g. structured-output / spec-decode rejects). Default: nothing dropped.
        self.discard_request_mask = self._make_buffer(max_num_reqs, dtype=torch.bool)
        self.discard_request_mask.np[:] = False

    def _make_buffer(self, *size, dtype) -> CpuGpuBuffer:
        # SOURCE: vllm/v1/worker/gpu_model_runner.py (CpuGpuBuffer factory in __init__)
        return CpuGpuBuffer(*size, dtype=dtype, device=self.device, pin_memory=False)

    # ------------------------------------------------------------------ #
    # Preprocess (carried over from the ch18 spine, reduced).
    # ------------------------------------------------------------------ #

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1073  _update_states
    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Reconcile the cached states and the persistent batch with the
        scheduler output (remove finished/unscheduled, add new/resumed,
        condense). The updated states feed _prepare_inputs."""
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.input_batch.remove_request(req_id)

        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                sampling_params=new_req_data.sampling_params,
                generator=None,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
            )
            self.requests[req_id] = req_state
            reqs_to_add.append(req_state)

        # SUBTRACTED: M-RoPE / LoRA / pooling / async-spec / PP token-backfill /
        #   ngram_gpu bookkeeping branches of _update_states. Approved — these
        #   are the same deletions made on the ch18 spine.
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L1136-L1447
        req_data = scheduler_output.scheduled_cached_reqs
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            req_index = self.input_batch.req_id_to_index.get(req_id)

            req_state.num_computed_tokens = num_computed_tokens

            if not resumed_from_preemption:
                if new_block_ids is not None:
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert req_index is None
                assert new_block_ids is not None
                req_state.block_ids = new_block_ids

            if req_index is None:
                reqs_to_add.append(req_state)
                continue

            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)
            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)

        for request in reqs_to_add:
            self.input_batch.add_request(request)
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

        self.input_batch.condense()
        self.input_batch.refresh_metadata()

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1600  _get_cumsum_and_arange
    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        arange_out: np.ndarray,
        cumsum_dtype=None,
    ) -> np.ndarray:
        """Cumulative sum + batched arange. E.g. [2, 5, 3] -> cumsum [2, 7, 10]
        and arange_out[:10] = [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]."""
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        np.subtract(
            self.arange_np[:total_num_tokens],
            cumsums_offsets,
            out=arange_out[:total_num_tokens],
        )
        return cu_num_tokens

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1815  _prepare_inputs
    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, object | None]:
        """Gather input_ids / positions / slot_mapping for the current batch.

        f13 read-back side: input_ids are pulled out of the persistent
        token_ids_cpu via torch.index_select using
        token_indices = positions + req_index * max_model_len, where
        positions = num_computed_tokens_cpu + per-request query offset. This is
        exactly where last step's written-back tokens re-enter as this step's
        inputs. Returns (logits_indices, spec_decode_metadata)."""
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: start copying the block table first to overlap with the
        # following CPU work.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # req_indices: [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # query_pos.np[:10] = [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )

        # f13 read-back: positions = num_computed_tokens + per-request offset.
        positions_np = (
            self.input_batch.num_computed_tokens_cpu[req_indices]
            + self.query_pos.np[: cu_num_tokens[-1]]
        )

        # SUBTRACTED: M-RoPE / XD-RoPE position calc. Approved (multimodal).
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L1853-L1861

        # token_indices = positions + req_index * max_model_len (2D -> flat 1D).
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        token_indices_tensor = torch.from_numpy(token_indices)

        # NOTE(woosuk): torch.index_select is much faster than np.take here.
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )
        # SUBTRACTED: prompt_embeds is_token_ids gather + per-request embeds fill.
        #   Approved (prompt_embeds). Orig: gpu_model_runner.py:L1853-L1898

        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()

        self.positions[:total_num_scheduled_tokens] = torch.from_numpy(
            positions_np
        ).to(torch.int64)

        # SUBTRACTED: prev_positions / async spec-decode num_computed_tokens GPU
        #   correction + _prepare_input_ids prev_sampled_token_ids backfill
        #   (async scheduling). Reduced companion does the normal-scheduling copy.
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L1641-L1664, L1920-L2015
        self.input_ids.copy_to_gpu(total_num_scheduled_tokens)

        # slot_mapping via the block-table Triton kernel (CUDA only).
        if self.device.type == "cuda":
            self.input_batch.block_table.compute_slot_mapping(
                num_reqs,
                self.query_start_loc.gpu[: num_reqs + 1],
                self.positions[:total_num_scheduled_tokens],
            )

        # SUBTRACTED: spec_decode_metadata branch + LoRA hot-swap. Approved.
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L2071-L2119
        spec_decode_metadata = None
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]
        logits_indices = query_start_loc[1:] - 1
        return logits_indices, spec_decode_metadata

    # ------------------------------------------------------------------ #
    # CUDA graph dispatch.
    # ------------------------------------------------------------------ #

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3600  _is_uniform_decode
    @staticmethod
    def _is_uniform_decode(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3600
        max_num_scheduled_tokens: int,
        uniform_decode_query_len: int,
        num_tokens: int,
        num_reqs: int,
    ) -> bool:
        """A decode batch with the same number of scheduled tokens per request."""
        return (max_num_scheduled_tokens == uniform_decode_query_len) and (
            num_tokens == max_num_scheduled_tokens * num_reqs
        )

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3621  _determine_batch_execution_and_padding
    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        num_reqs: int,
        num_scheduled_tokens_np: np.ndarray,
        max_num_scheduled_tokens: int,
        use_cascade_attn: bool = False,
        num_encoder_reqs: int = 0,
    ) -> tuple[CUDAGraphMode, BatchDescriptor, bool, object | None, object | None]:
        """Decide the CUDA-graph runtime mode and the (padded) batch descriptor
        for this step by querying CudagraphDispatcher.dispatch."""
        uniform_decode = self._is_uniform_decode(
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            uniform_decode_query_len=self.uniform_decode_query_len,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
        )

        # SUBTRACTED: LoRA num_active_loras count, SP/DP padding, ubatching
        #   (DBO) coordination across data-parallel ranks, encoder-decoder
        #   full-graph disabling. Approved — single-rank, no-LoRA, no-DP path.
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L3656-末
        has_lora = False
        num_active_loras = 0

        cudagraph_mode, batch_descriptor = self.cudagraph_dispatcher.dispatch(
            num_tokens=num_tokens,
            has_lora=has_lora,
            uniform_decode=uniform_decode,
            num_active_loras=num_active_loras,
            # cascade attention is not supported by full cudagraphs.
            invalid_modes={CUDAGraphMode.FULL} if use_cascade_attn else None,
        )

        should_ubatch = False
        num_tokens_across_dp = None
        cudagraph_stats = None
        return (
            cudagraph_mode,
            batch_descriptor,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        )

    # ------------------------------------------------------------------ #
    # Forward.
    # ------------------------------------------------------------------ #

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3568  _model_forward
    def _model_forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **model_kwargs,
    ):
        """Call the model forward pass. Isolated so it can be inspected without
        the surrounding execute_model logic."""
        return self.model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Phase 1: execute_model — issue the forward, cache state, return None.
    # ------------------------------------------------------------------ #

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3855  execute_model
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors=None,
    ) -> "ModelRunnerOutput | None":
        # The two-phase contract: execute_model() must run with the bridge slot
        # empty (the previous step's state must already have been consumed by
        # sample_tokens). A non-empty slot means execute_model was called twice
        # without an intervening sample_tokens -> hard error rather than a silent
        # leak of the previous step's GPU tensors.
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        # SUBTRACTED: RoutedExpertsCapturer clear, ngram_gpu scheduler_output
        #   shallow-copy, KV-transfer handle_preemptions. Approved (MoE capture /
        #   ngram spec decode / PD-disaggregation).
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L3866-L3891

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        # Preprocess: reconcile the persistent batch, then build the per-step
        # input tensors. (synchronize_input_prep() is a no-op without async copy.)
        self._update_states(scheduler_output)

        if not num_scheduled_tokens:
            # SUBTRACTED: external-launcher+DP dummy run / KV-transfer no-forward.
            #   Approved. Orig: vllm/v1/worker/gpu_model_runner.py:L3910-L3925
            # Return an empty output if there is no work to do.
            return ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])

        # SUBTRACTED: EC connector consumer mm-encoder early return,
        #   kv_sharing_fast_prefill assert. Approved.
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L3901-L3932
        num_reqs = self.input_batch.num_reqs
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
        num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

        logits_indices, spec_decode_metadata = self._prepare_inputs(
            scheduler_output,
            num_scheduled_tokens_np,
        )

        # SUBTRACTED: cascade attention prefix pre-computation (disables FULL).
        #   Approved (cascade attn). Orig: gpu_model_runner.py:L3918-L3926
        cascade_attn_prefix_lens = None

        (
            cudagraph_mode,
            batch_desc,
            should_ubatch,
            num_tokens_across_dp,
            cudagraph_stats,
        ) = self._determine_batch_execution_and_padding(
            num_tokens=num_tokens_unpadded,
            num_reqs=num_reqs,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
            use_cascade_attn=cascade_attn_prefix_lens is not None,
            num_encoder_reqs=len(scheduler_output.scheduled_encoder_inputs),
        )

        num_tokens_padded = batch_desc.num_tokens

        # SUBTRACTED: ubatch slice creation, separate-KV-update padding, mamba
        #   preprocess, _build_attention_metadata, _preprocess (input_ids /
        #   inputs_embeds / model_kwargs assembly). Reduced companion feeds the
        #   prepared input_ids/positions straight in; the attention metadata and
        #   per-backend build belong to the attention chapters. Approved.
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L3984-L4077
        input_ids = self.input_ids.gpu[:num_tokens_unpadded]
        positions = self.positions[:num_tokens_unpadded]
        attn_metadata = None
        slot_mappings = None
        aux_hidden_states = None
        ec_connector_output = None

        # Run the model. set_forward_context publishes cudagraph_runtime_mode +
        # batch_descriptor so the compiled model replays the matching CUDA graph.
        with set_forward_context(
            attn_metadata,
            num_tokens=num_tokens_padded,
            cudagraph_runtime_mode=cudagraph_mode,
            batch_descriptor=batch_desc,
            slot_mapping=slot_mappings,
        ):
            model_output = self._model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=None,
            )

        # Postprocess: gather the last-token hidden states, compute logits.
        # SUBTRACTED: EAGLE-3 aux_hidden_states unpacking, PP non-last-rank early
        #   return of IntermediateTensors, pooling-model _pool, broadcast_pp_output
        #   logits broadcast. Approved (spec decode / PP / pooling).
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L4126-L4182
        hidden_states = model_output
        sample_hidden_states = hidden_states[logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)

        # Phase boundary: cache everything sample_tokens() needs and return None.
        # The forward has been issued; its results live in `logits` /
        # `hidden_states`, not yet consumed. Returning None lets the caller
        # overlap "issue next step's forward" with "sample this step".
        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            None,  # spec_decode_common_attn_metadata
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        )

        # SUBTRACTED: kv_connector_output stashing + deferred_state_corrections_fn
        #   (async scheduling waits for the previous forward's corrections here).
        #   Approved. Orig: vllm/v1/worker/gpu_model_runner.py:L4196-L4201
        return None

    # ------------------------------------------------------------------ #
    # Phase 2: sample_tokens — unpack state, sample, write back.
    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L4206  sample_tokens
    def sample_tokens(self, grammar_output=None) -> "ModelRunnerOutput":
        if self.execute_model_state is None:
            # SUBTRACTED: PP non-last-rank receive of sampled ids + pure
            #   KV-transfer pass-through early returns. In the reduced companion
            #   sample_tokens is always called right after a forward, so the
            #   bridge slot is non-None. Approved (PP / KV connector).
            #   Orig: vllm/v1/worker/gpu_model_runner.py:L4209-L4225
            return ModelRunnerOutput(req_ids=[], req_id_to_index={}, sampled_token_ids=[])

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            ec_connector_output,
            cudagraph_stats,
            slot_mappings,
        ) = self.execute_model_state
        # Clear ephemeral state — the bridge slot is now empty for the next
        # execute_model().
        self.execute_model_state = None

        # SUBTRACTED: apply_grammar_bitmask (structured output). Approved.
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L4244-L4247

        sampler_output = self._sample(logits, spec_decode_metadata)

        self._update_states_after_model_execute(
            sampler_output.sampled_token_ids, scheduler_output
        )

        # SUBTRACTED: async-scheduling PP broadcast of sampled ids, draft-token
        #   proposal (EAGLE / ngram / DraftModel), KV connector finalize, eplb,
        #   RoutedExpertsCapturer save. Approved (spec decode / PP / connectors).
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L4255-L4415

        (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        ) = self._bookkeeping_sync(
            scheduler_output,
            sampler_output,
            logits,
            hidden_states,
            scheduler_output.total_num_scheduled_tokens,
        )

        output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            num_nans_in_logits=num_nans_in_logits,
            cudagraph_stats=cudagraph_stats,
        )

        # SUBTRACTED: AsyncGPUModelRunnerOutput wrapping + set_async_sampled_token_ids
        #   (async scheduling keeps sampled ids on the GPU and registers a deferred
        #   CPU copy). Reduced companion is synchronous. Approved.
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L4434-L4455
        return output

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3397  _sample
    def _sample(self, logits, spec_decode_metadata) -> SamplerOutput:
        """Sample the next token (and logprobs if needed)."""
        sampling_metadata = self.input_batch.sampling_metadata
        # SUBTRACTED: update_async_output_token_ids (async sched), rejection
        #   sampler / spec-decode path. Approved — reduced companion never has
        #   spec_decode_metadata, so the plain sampler branch always runs.
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L3406, L3383-L3395
        return self.sampler(logits=logits, sampling_metadata=sampling_metadata)

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L1449  _update_states_after_model_execute
    def _update_states_after_model_execute(
        self, output_token_ids: torch.Tensor, scheduler_output: "SchedulerOutput"
    ) -> None:
        """Post-execute cached-state update for MTP/EAGLE hybrid models (counts
        accepted draft tokens). No-op for plain Transformers."""
        # SUBTRACTED: hybrid mamba accepted-token counting + mamba postprocess.
        #   Reduced companion has speculative_config=None / is_hybrid=False, so
        #   the real method returns immediately here too. The call site is kept to
        #   show where it sits in the flow. Approved.
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L1460-L1492
        return

    # SOURCE: vllm/v1/worker/gpu_model_runner.py:L3427  _bookkeeping_sync
    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits,
        hidden_states,
        num_scheduled_tokens: int,
    ):
        """f13 write-back side. For each request that produced a token, append it
        to that request's slot row in the persistent token_ids_cpu, advance
        num_tokens_no_spec, and extend the request's output_token_ids list. Since
        req_output_token_ids[slot] *is* CachedRequestState.output_token_ids (same
        list object), the persistent-batch view and the request snapshot grow
        together — the batch carries the new token into the next step."""
        num_nans_in_logits: dict = {}
        # SUBTRACTED: VLLM_COMPUTE_NANS_IN_LOGITS NaN scan. Approved (diagnostic).
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L3443-L3445

        num_reqs = self.input_batch.num_reqs
        discard_sampled_tokens_req_indices = np.nonzero(
            self.discard_request_mask.np[:num_reqs]
        )[0]
        # SUBTRACTED: seeded-generator offset rewind for discarded requests
        #   (RANDOM_SEED). Approved. Orig: gpu_model_runner.py:L3421-L3424

        # Copy these so they aren't mutated after returning (matters for async
        # scheduling).
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        invalid_req_indices: list[int] = []
        logprobs_lists = None
        # SUBTRACTED: the use_async_scheduling branch (GPU-cached sampled ids +
        #   prev_req_id_to_index). Reduced companion is synchronous, so we take
        #   the eager CPU path. Approved.
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L3486-L3502
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
            # Mask out the sampled tokens that should not be sampled.
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[int(i)].clear()
            if logprobs_tensors is not None:
                logprobs_lists = logprobs_tensors.tolists()
        else:
            # SUBTRACTED: RejectionSampler.parse_output (spec decode multi-token).
            #   Approved. Orig: vllm/v1/worker/gpu_model_runner.py:L3478-L3485
            raise AssertionError("spec decode is subtracted in the reduced companion")

        # Write the sampled tokens back into the persistent batch so the
        # scheduler doesn't need to send them back.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            sampled_ids = valid_sampled_token_ids[req_idx]
            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            # f13 write-back: append to the slot row, advance the counter, grow
            # the (aliased) output_token_ids list.
            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        # SUBTRACTED: _get_prompt_logprobs_dict (prompt logprobs). Approved.
        #   Orig: vllm/v1/worker/gpu_model_runner.py:L3537-L3541
        prompt_logprobs_dict: dict = {}

        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )
