"""Stage 3 output processing: OutputProcessor.process_outputs single loop,
per-request RequestState, RequestOutputCollector mailbox, OutputProcessorOutput.

Subtract-only companion of vllm/v1/engine/output_processor.py. Subtracted
(per subtraction_plan, all orthogonal subsystems): do_tracing + tracing_enabled
branch; stats accumulation internals (call sites kept as placeholders that early
-return when iteration_stats is None); LoRA fields/LoRARequestStates; the pooling
branch (_new_pooling_output / PoolingRequestOutput / EMPTY_CPU_TENSOR); resumable
streaming-input (StreamingUpdate / input_chunk_queue / apply_streaming_update /
_update_streaming_request_state); routed_experts / kv_transfer_params透传 fields.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from ._types import CompletionOutput, EngineCoreOutput, FinishReason, RequestOutputKind
from .detokenizer import IncrementalDetokenizer
from .logprobs import LogprobsProcessor
from .outputs import RequestOutput
from .parallel_sampling import ParentRequest


# SOURCE: vllm/v1/engine/output_processor.py:L48
class RequestOutputCollector:
    """
    Collects streamed RequestOutputs per individual request,
    for hand-off to the consuming asyncio generate task.

    When streaming deltas, RequestOutputs are merged if the
    producer gets ahead of the consumer.
    """

    # SOURCE: vllm/v1/engine/output_processor.py:L57
    # SUBTRACTED: self._input_stream_task (resumable streaming-input cleanup,
    #   subtraction_plan).
    def __init__(self, output_kind: RequestOutputKind, request_id: str):
        # SOURCE: vllm/v1/engine/output_processor.py:L57
        self.aggregate = output_kind == RequestOutputKind.DELTA
        self.request_id = request_id
        self.output: RequestOutput | Exception | None = None
        self.ready = asyncio.Event()

    # SOURCE: vllm/v1/engine/output_processor.py:L65
    # SUBTRACTED: PoolingRequestOutput merge branch (pooling product line,
    #   subtraction_plan).
    def put(self, output: RequestOutput | Exception) -> None:
        """Non-blocking put operation."""
        # SOURCE: vllm/v1/engine/output_processor.py:L65
        if self.output is None or isinstance(output, Exception):
            self.output = output
            self.ready.set()
        elif isinstance(self.output, RequestOutput) and isinstance(
            output, RequestOutput
        ):
            # This ensures that request outputs with different request indexes
            # (if n > 1) do not override each other.
            self.output.add(output, aggregate=self.aggregate)

    # SOURCE: vllm/v1/engine/output_processor.py:L81
    async def get(self) -> RequestOutput:
        """Get operation blocks on put event."""
        while (output := self.output) is None:
            await self.ready.wait()
        self.output = None
        self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    # SOURCE: vllm/v1/engine/output_processor.py:L91
    def get_nowait(self) -> RequestOutput | None:
        """Non-blocking get operation."""
        output = self.output
        if output is not None:
            self.output = None
            self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    # SUBTRACTED: close / __del__ (_input_stream_task cleanup, output_processor.py
    #   :L98-L106 — resumable streaming-input, subtraction_plan).


# SOURCE: vllm/v1/engine/output_processor.py:L112
@dataclass
class OutputProcessorOutput:
    # SOURCE: vllm/v1/engine/output_processor.py:L112
    request_outputs: list[RequestOutput]
    reqs_to_abort: list[str]


# SUBTRACTED: StreamingUpdate dataclass (output_processor.py:L115-L127) —
#   resumable streaming-input (subtraction_plan).


# SOURCE: vllm/v1/engine/output_processor.py:L132
class RequestState:
    # SOURCE: vllm/v1/engine/output_processor.py:L133
    # SUBTRACTED: lora_request/lora_name, prompt_embeds, max_tokens_param,
    #   top_p/n/temperature (tracing inputs), stats(RequestStateStats),
    #   stream_input/input_chunk_queue/streaming_input — orthogonal
    #   (LoRA/metrics/tracing/streaming-input, subtraction_plan).
    def __init__(
        self,
        request_id: str,
        external_req_id: str,
        parent_req: ParentRequest | None,
        request_index: int,
        output_kind: RequestOutputKind,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        logprobs_processor: LogprobsProcessor | None,
        detokenizer: IncrementalDetokenizer | None,
        queue: RequestOutputCollector | None,
        stream_interval: int,
    ):
        # SOURCE: vllm/v1/engine/output_processor.py:L133
        self.request_id = request_id
        self.external_req_id = external_req_id
        self.parent_req = parent_req
        self.request_index = request_index
        self.output_kind = output_kind
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.logprobs_processor = logprobs_processor
        self.detokenizer = detokenizer
        self.is_prefilling = True
        self.queue = queue
        self.num_cached_tokens = 0

        # Stream Interval
        self.stream_interval = stream_interval
        self.sent_tokens_offset = 0  # Offset of sent tokens

    # SOURCE: vllm/v1/engine/output_processor.py:L210
    # SUBTRACTED: tokenizer/log_stats/max_tokens/top_p/n/temperature/stream_input
    #   wiring + pooling_params branch — orthogonal (subtraction_plan).
    @classmethod
    def from_new_request(
        cls,
        tokenizer,
        request,
        prompt: str | None,
        parent_req: ParentRequest | None,
        request_index: int,
        queue: RequestOutputCollector | None,
        stream_interval: int,
    ) -> "RequestState":
        # SOURCE: vllm/v1/engine/output_processor.py:L210
        sampling_params = request.sampling_params
        assert sampling_params is not None
        output_kind = sampling_params.output_kind
        logprobs_processor = LogprobsProcessor.from_new_request(
            tokenizer=tokenizer,
            request=request,
        )
        detokenizer = IncrementalDetokenizer.from_new_request(
            tokenizer=tokenizer,
            request=request,
        )
        return cls(
            request_id=request.request_id,
            external_req_id=request.external_req_id,
            parent_req=parent_req,
            request_index=request_index,
            output_kind=output_kind,
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            logprobs_processor=logprobs_processor,
            detokenizer=detokenizer,
            queue=queue,
            stream_interval=stream_interval,
        )

    # SOURCE: vllm/v1/engine/output_processor.py:L272
    # SUBTRACTED: pooling_output branch (_new_pooling_output) — pooling product
    #   line (subtraction_plan).
    def make_request_output(
        self,
        new_token_ids: list[int],
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
    ) -> RequestOutput | None:
        # SOURCE: vllm/v1/engine/output_processor.py:L272
        finished = finish_reason is not None
        final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

        if not finished and final_only:
            # Only the final output is required in FINAL_ONLY mode.
            return None

        if self.stream_interval > 1:
            assert self.detokenizer is not None

            # Send output request only when
            # 1. It has finished, or
            # 2. It is the first token, or
            # 3. It has reached the stream interval number of tokens
            if not (
                finished
                or self.sent_tokens_offset == 0
                or self.detokenizer.num_output_tokens() - self.sent_tokens_offset
                >= self.stream_interval
            ):
                return None

            if self.output_kind == RequestOutputKind.DELTA:
                # Send tokens from the offset in DELTA mode, otherwise all
                # tokens are sent.
                new_token_ids = self.detokenizer.output_token_ids[
                    self.sent_tokens_offset :
                ]
                self.sent_tokens_offset = self.detokenizer.num_output_tokens()

        external_req_id = self.external_req_id

        output = self._new_completion_output(new_token_ids, finish_reason, stop_reason)

        if self.parent_req is None:
            outputs = [output]
        else:
            outputs, finished = self.parent_req.get_outputs(self.request_id, output)
            if not outputs:
                return None
            external_req_id = self.parent_req.external_req_id

        return self._new_request_output(
            external_req_id, outputs, finished, kv_transfer_params
        )

    # SOURCE: vllm/v1/engine/output_processor.py:L356
    # SUBTRACTED: prompt_embeds placeholder, PoolingRequestOutput branch, LoRA /
    #   metrics / num_cached_tokens carry — orthogonal (subtraction_plan).
    def _new_request_output(
        self,
        external_req_id: str,
        outputs: list[CompletionOutput],
        finished: bool,
        kv_transfer_params: dict[str, Any] | None = None,
    ) -> RequestOutput:
        # SOURCE: vllm/v1/engine/output_processor.py:L356
        prompt_token_ids = self.prompt_token_ids
        assert self.logprobs_processor is not None
        if self.output_kind == RequestOutputKind.DELTA:
            # Side effect: logprobs processor forgets prompt logprobs
            prompt_logprobs = self.logprobs_processor.pop_prompt_logprobs()
        else:
            prompt_logprobs = self.logprobs_processor.prompt_logprobs

        return RequestOutput(
            request_id=external_req_id,  # request_id is what was provided externally
            prompt=self.prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=prompt_logprobs,
            outputs=outputs,
            finished=finished,
            kv_transfer_params=kv_transfer_params,
        )

    # SOURCE: vllm/v1/engine/output_processor.py:L401
    # SUBTRACTED: routed_experts field on CompletionOutput (MoE orthogonal,
    #   subtraction_plan).
    def _new_completion_output(
        self,
        token_ids: list[int],
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
    ) -> CompletionOutput:
        # SOURCE: vllm/v1/engine/output_processor.py:L401
        assert self.detokenizer is not None
        assert self.logprobs_processor is not None
        finished = finish_reason is not None
        delta = self.output_kind == RequestOutputKind.DELTA

        # Prepare text and token_ids, based on delta mode
        text = self.detokenizer.get_next_output_text(finished, delta)
        if not delta:
            token_ids = self.detokenizer.output_token_ids

        # Prepare logprobs, based on delta mode
        logprobs = self.logprobs_processor.logprobs
        if delta and logprobs:
            logprobs = logprobs[-len(token_ids) :]

        return CompletionOutput(
            index=self.request_index,
            text=text,
            token_ids=token_ids,
            logprobs=logprobs,
            cumulative_logprob=self.logprobs_processor.cumulative_logprob,
            finish_reason=str(finish_reason) if finished else None,
            stop_reason=stop_reason if finished else None,
        )


# SOURCE: vllm/v1/engine/output_processor.py:L438
class OutputProcessor:
    """Process EngineCoreOutputs into RequestOutputs."""

    # SOURCE: vllm/v1/engine/output_processor.py:L441
    # SUBTRACTED: log_stats/lora_states(LoRARequestStates)/tokenizer_field/
    #   tracing_enabled — orthogonal subsystems (subtraction_plan). tokenizer kept
    #   for from_new_request wiring.
    def __init__(
        self,
        tokenizer,
        *,
        stream_interval: int = 1,
    ):
        # SOURCE: vllm/v1/engine/output_processor.py:L441
        self.tokenizer = tokenizer
        self.stream_interval = stream_interval
        self.request_states: dict[str, RequestState] = {}
        self.parent_requests: dict[str, ParentRequest] = {}
        self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)

    # SOURCE: vllm/v1/engine/output_processor.py:L458
    def get_num_unfinished_requests(self):
        return len(self.request_states)

    # SOURCE: vllm/v1/engine/output_processor.py:L461
    def has_unfinished_requests(self) -> bool:
        return len(self.request_states) > 0

    # SOURCE: vllm/v1/engine/output_processor.py:L464
    def propagate_error(self, e: Exception):
        """Propagate error to all generate() tasks."""
        for _, state in self.request_states.items():
            assert state.queue is not None
            state.queue.put(e)

    # SOURCE: vllm/v1/engine/output_processor.py:L533
    # SUBTRACTED: _update_streaming_request_state branch (resumable
    #   streaming-input) + lora wiring (subtraction_plan).
    def add_request(
        self,
        request,
        prompt: str | None,
        parent_req: ParentRequest | None = None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        # SOURCE: vllm/v1/engine/output_processor.py:L533
        request_id = request.request_id
        req_state = RequestState.from_new_request(
            tokenizer=self.tokenizer,
            request=request,
            prompt=prompt,
            parent_req=parent_req,
            request_index=request_index,
            queue=queue,
            stream_interval=self.stream_interval,
        )
        self.request_states[request_id] = req_state
        if parent_req:
            self.parent_requests[parent_req.request_id] = parent_req

        # Track the external_req_id -> [internal_req_id, ...] mapping
        self.external_req_ids[req_state.external_req_id].append(request_id)

    # SUBTRACTED: _update_stats_from_output / _update_stats_from_finished bodies —
    #   metrics subsystem (subtraction_plan: keep call site as placeholder).
    def _update_stats_from_output(self, req_state, engine_core_output, ts, stats):
        # SOURCE: vllm/v1/engine/output_processor.py (_update_stats_from_output 占位)
        if stats is None:
            return

    def _update_stats_from_finished(self, req_state, finish_reason, stats):
        # SOURCE: vllm/v1/engine/output_processor.py (_update_stats_from_finished 占位)
        if stats is None:
            return

    # SOURCE: vllm/v1/engine/output_processor.py:L597
    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float | None = None,
        iteration_stats=None,
    ) -> OutputProcessorOutput:
        """
        Process the EngineCoreOutputs:
        1) Compute stats for logging
        2) Detokenize
        3) Create and handle RequestOutput objects:
            * If there is a queue (for usage with AsyncLLM),
              put the RequestOutput objects into the queue for
              handling by the per-request generate() tasks.
            * If there is no queue (for usage with LLMEngine),
              return a list of RequestOutput objects.

        NOTE FOR DEVELOPERS

        vLLM V1 minimizes the number of python loops over the full
        batch to ensure system overheads are minimized. This is the
        only function that should loop over EngineCoreOutputs.
        """
        request_outputs: list[RequestOutput] = []
        reqs_to_abort: list[str] = []
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                # Ignore output for already-aborted request.
                continue

            # 1) Compute stats for this iteration.
            self._update_stats_from_output(
                req_state, engine_core_output, engine_core_timestamp, iteration_stats
            )

            new_token_ids = engine_core_output.new_token_ids
            finish_reason = engine_core_output.finish_reason
            stop_reason = engine_core_output.stop_reason
            kv_transfer_params = engine_core_output.kv_transfer_params

            if req_state.is_prefilling:
                if engine_core_output.prefill_stats is not None:
                    req_state.num_cached_tokens = (
                        engine_core_output.prefill_stats.num_cached_tokens
                    )
                req_state.is_prefilling = False

            # SUBTRACTED: pooling_output branch — generation path always
            #   detokenizes (pooling product line, subtraction_plan).
            assert req_state.detokenizer is not None
            assert req_state.logprobs_processor is not None
            # 2) Detokenize the token ids into text and perform stop checks.
            stop_string = req_state.detokenizer.update(
                new_token_ids, finish_reason == FinishReason.STOP
            )
            if stop_string:
                finish_reason = FinishReason.STOP
                stop_reason = stop_string

            # 3) Compute sample and prompt logprobs for request, if required.
            req_state.logprobs_processor.update_from_output(engine_core_output)

            # 4) Create and handle RequestOutput objects.
            if request_output := req_state.make_request_output(
                new_token_ids,
                finish_reason,
                stop_reason,
                kv_transfer_params,
            ):
                if req_state.queue is not None:
                    # AsyncLLM: put into queue for handling by generate().
                    req_state.queue.put(request_output)
                else:
                    # LLMEngine: return list of RequestOutputs.
                    request_outputs.append(request_output)

            # Free completed requests.
            if finish_reason is not None:
                # SUBTRACTED: streaming_input branch (resumable input chunk queue,
                #   subtraction_plan) —普通一次性 prompt 直接 finish。
                self._finish_request(req_state)
                if not engine_core_output.finished:
                    # If req not finished in EngineCore, but Detokenizer
                    # detected stop string, abort needed in EngineCore.
                    reqs_to_abort.append(req_id)

                # Track per-request stats
                self._update_stats_from_finished(
                    req_state, finish_reason, iteration_stats
                )
                # SUBTRACTED: tracing_enabled / do_tracing branch (OpenTelemetry,
                #   subtraction_plan).

        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
        )

    # SOURCE: vllm/v1/engine/output_processor.py:L714
    def _finish_request(self, req_state: RequestState) -> None:
        req_id = req_state.request_id
        self.request_states.pop(req_id)

        internal_ids = self.external_req_ids[req_state.external_req_id]
        internal_ids.remove(req_id)
        if not internal_ids:
            del self.external_req_ids[req_state.external_req_id]

        # Remove parent request if applicable.
        parent_req = req_state.parent_req
        if parent_req and not parent_req.child_requests:
            self.parent_requests.pop(parent_req.request_id, None)

    # SUBTRACTED: abort_requests / update_scheduler_stats / do_tracing —
    #   abort path + metrics + OpenTelemetry tracing (subtraction_plan); the
    #   stop-string反向 abort goes through reqs_to_abort + output_handler instead.


# SOURCE: vllm/v1/engine/async_llm.py:L656 output_handler (f3 producer)
# SUBTRACTED: log_stats/IterationStats, logger_ref metrics recording,
#   update_scheduler_stats, propagate_error handling reduced to the essential
#   producer loop — metrics/error-broadcast plumbing (subtraction_plan).
async def output_handler(
    engine_core,
    output_processor: OutputProcessor,
    chunk_size: int,
):
    # SOURCE: vllm/v1/engine/async_llm.py:L656
    while True:
        # 1) Pull EngineCoreOutputs from the EngineCore.
        outputs = await engine_core.get_output_async()
        num_outputs = len(outputs.outputs)

        # Split outputs into chunks of at most chunk_size, so that we don't
        # block the event loop for too long.
        engine_core_outputs = outputs.outputs
        for start in range(0, num_outputs, chunk_size):
            end = start + chunk_size
            outputs_slice = engine_core_outputs[start:end]
            # 2) Process EngineCoreOutputs.
            processed_outputs = output_processor.process_outputs(
                outputs_slice, outputs.timestamp, None
            )
            # NOTE: RequestOutputs are pushed to their queues.
            assert not processed_outputs.request_outputs

            # Allow other asyncio tasks to run between chunks
            if end < num_outputs:
                await asyncio.sleep(0)

            # 3) Abort any reqs that finished due to stop strings.
            if processed_outputs.reqs_to_abort:
                await engine_core.abort_requests_async(
                    processed_outputs.reqs_to_abort
                )


# SOURCE: vllm/v1/engine/async_llm.py:L576 generate() consumer loop (f3 consumer)
# SUBTRACTED: add_request / cancellation+error handling around the loop
#   (request lifecycle covered in ch04, subtraction_plan elide). Only the
#   pull-yield loop body is kept.
async def generate(q: RequestOutputCollector, stream_finished) -> Iterable:
    # SOURCE: vllm/v1/engine/async_llm.py:L576
    results = []
    # The output_handler task pushes items into the queue.
    # This task pulls from the queue and yields to caller.
    finished = False
    while not finished:
        # Note: drain queue without await if possible (avoids
        # task switching under load which helps performance).
        out = q.get_nowait() or await q.get()

        # Note: both OutputProcessor and EngineCore handle their
        # own request cleanup based on finished.
        assert isinstance(out, RequestOutput)
        finished = out.finished
        if out is not stream_finished:
            results.append(out)
    return results
