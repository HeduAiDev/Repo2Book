# Subtract-only companion for v3 ch04 «两个使用面，一套三件套» (API 进程:
# 使用面 + 双登记 + client_index 盖章).
#
# FAITHFUL SUBSET of the real vLLM front-end request path at pin
# v0.27.1 (6e448d0ea). It keeps vLLM's names, structure and control flow;
# it only DELETES branches approved in the dossier subtraction_plan (plus the
# mechanical deletions listed in impl-notes.md) and marks every deletion with
# `# SUBTRACTED:`. Mapping rule: take the real vLLM source, drop every
# SUBTRACTED branch, and you should get (approximately) this file.
#
# Goal line (dossier subtraction_plan._principle): keep the full skeleton of
# «two usage faces + one trio + double registration + client_index stamping»,
# with the EngineCore replaced by an in-process stub (same client-facing
# surface: add_request_async/get_output_async/abort_requests_async and
# add_request/get_output/abort_requests) and the ZMQ/msgpack physical layer
# replaced by in-process queues -- so AsyncLLM (generate/add_request/
# _add_request/output_handler) and LLMEngine (add_request/step) read almost
# verbatim, with both drivers, the queue fork and the double-registration
# reconciliation plainly visible. The IPC physical layer is ch5.
#
# Runs on a CPU host WITHOUT torch/vllm/zmq: the Renderer (ch6), the EngineCore
# scheduler loop (ch9) and the msgspec structs are seams (see impl-notes.md).
# Every def/class carries a `# SOURCE: vllm/...:Lxxx` ref into the pinned tree.

from __future__ import annotations

import asyncio
import copy
import enum
import queue
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import AsyncGenerator, Iterable
from dataclasses import dataclass, field
from typing import Any


# ============================================================================
# Host seams — stdlib stand-ins for vllm.* infrastructure so the front end
# runs without the vllm package. Each mirrors the real interface subset.
# ============================================================================


# SOURCE: vllm/logger.py init_logger — logging seam with the *_once helpers
def init_logger(name: str):
    import logging

    log = logging.getLogger(name)
    if not log.handlers:
        log.addHandler(logging.NullHandler())
    seen: set[str] = set()

    # SOURCE: vllm/logger.py LoggingContext once-messaging (info_once/warning_once)
    class _Once:
        # SOURCE: vllm/logger.py once-messaging wrapper
        def __init__(self, fn):
            self._fn = fn

        def __call__(self, msg, *args):  # SOURCE: vllm/logger.py once-wrapper call
            key = (self._fn.__name__, msg)
            if key not in seen:
                seen.add(key)
                self._fn(msg, *args)

    log.info_once = _Once(log.info)
    log.warning_once = _Once(log.warning)
    return log


logger = init_logger(__name__)


# SOURCE: vllm/envs.py envs — environment flag seam (defaults per pin)
class envs:
    # SOURCE: vllm/envs.py:L149 VLLM_ENABLE_V1_MULTIPROCESSING
    VLLM_ENABLE_V1_MULTIPROCESSING: bool = True
    # SOURCE: vllm/envs.py:L210 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION
    VLLM_DISABLE_REQUEST_ID_RANDOMIZATION: bool = False
    # SOURCE: vllm/envs.py:L160 VLLM_V1_OUTPUT_PROC_CHUNK_SIZE
    VLLM_V1_OUTPUT_PROC_CHUNK_SIZE: int = 128


# SOURCE: vllm/utils/__init__.py:L11 random_uuid
def random_uuid() -> str:
    MASK_64_BITS = (1 << 64) - 1
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars


# SOURCE: vllm/utils/counter.py:L6 Counter — itertools.count wrapper
class Counter:
    # SOURCE: vllm/utils/counter.py:L6 Counter.__init__
    def __init__(self, start: int = 0) -> None:
        super().__init__()
        self.counter = start

    # SOURCE: vllm/utils/counter.py Counter.__next__
    def __next__(self) -> int:
        i = self.counter
        self.counter += 1
        return i

    # SOURCE: vllm/utils/counter.py Counter.iter
    def __iter__(self):
        return self


# SOURCE: vllm/utils/collection_utils.py:L49 as_list
def as_list(maybe_list):
    """Convert iterable to list, unless it's already a list."""
    return maybe_list if isinstance(maybe_list, list) else list(maybe_list)


# SOURCE: vllm/usage/usage_lib.py:L112-L118 UsageContext
class UsageContext(str, enum.Enum):
    LLM_CLASS = "LLM_CLASS"
    OPENAI_API_SERVER = "OPENAI_API_SERVER"
    ENGINE_CONTEXT = "ENGINE_CONTEXT"


# SOURCE: vllm/lora/request.py LoRARequest (seam — LoRA axis is off this
# chapter's path; RequestState only reads lora_name)
class LoRARequest:
    # SOURCE: vllm/lora/request.py LoRARequest.__init__ (field subset)
    def __init__(self, lora_name: str = "", lora_int_id: int = 1):
        self.lora_name = lora_name
        self.lora_int_id = lora_int_id


# SOURCE: vllm/v1/engine/exceptions.py:L6 EngineGenerateError
class EngineGenerateError(Exception):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""
    pass


# SOURCE: vllm/v1/engine/exceptions.py:L12 EngineDeadError
class EngineDeadError(Exception):
    """Raised when the EngineCore dies. Unrecoverable."""

    # SOURCE: vllm/v1/engine/exceptions.py:L15 EngineDeadError.__init__
    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        ENGINE_DEAD_MESSAGE = (  # noqa: N806
            "EngineCore encountered an issue. See stack trace (above) "
            "for the root cause."
        )
        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context


# Config seams — only the fields the two usage faces read (the full assembly
# line from EngineArgs to VllmConfig is the ch03 companion chapter).
# SOURCE: vllm/config/model.py ModelConfig (field seam)
@dataclass
# SOURCE: vllm/config/model.py ModelConfig.runner_type
class ModelConfig:
    runner_type: str = "generate"


# SOURCE: vllm/config/scheduler.py:L153 SchedulerConfig.stream_interval (seam)
@dataclass
# SOURCE: vllm/config/scheduler.py:L26 SchedulerConfig
class SchedulerConfig:
    stream_interval: int = 1


# SOURCE: vllm/config/parallel.py:L129 ParallelConfig.data_parallel_size (seam)
@dataclass
# SOURCE: vllm/config/parallel.py:L118 ParallelConfig
class ParallelConfig:
    data_parallel_size: int = 1


# SOURCE: vllm/config/observability.py ObservabilityConfig (field seam)
@dataclass
# SOURCE: vllm/config/observability.py ObservabilityConfig
class ObservabilityConfig:
    otlp_traces_endpoint: str | None = None


# SOURCE: vllm/config/vllm.py:L331 VllmConfig (field seam — ch03 walks the rest)
@dataclass
# SOURCE: vllm/config/vllm.py:L331 VllmConfig
class VllmConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    observability_config: ObservabilityConfig = field(
        default_factory=ObservabilityConfig
    )


# SOURCE: vllm/v1/executor/abstract.py:L37 Executor (seam — the backend
# selection table is ch03; the faces only need a class to hand downstream)
class Executor:
    # SOURCE: vllm/v1/executor/abstract.py:L47-L92 Executor.get_class
    @staticmethod
    # SOURCE: vllm/v1/executor/abstract.py:L47 Executor.get_class
    def get_class(vllm_config: VllmConfig) -> type["Executor"]:
        return Executor


# SOURCE: vllm/renderers/base.py:L73-L80 BaseRenderer (seam — render internals
# are ch6; the faces only consume renderer.tokenizer)
class BaseRenderer:
    # SOURCE: vllm/renderers/base.py:L73 BaseRenderer.__init__
    def __init__(self, config: VllmConfig | None = None, tokenizer=None) -> None:
        # SUBTRACTED: async tokenizer executor + multimodal processor cache
        #   (vllm/renderers/base.py:L82-L99) — ch6.
        self.tokenizer = tokenizer


# SOURCE: vllm/renderers/registry.py:L82 renderer_from_config (seam)
def renderer_from_config(config: VllmConfig, **kwargs) -> BaseRenderer:
    # SUBTRACTED: tokenizer construction + renderer registry dispatch
    #   (vllm/renderers/registry.py:L83-L88) — ch6. Tokenizer-less companion
    #   (skip_tokenizer_init equivalent): detokenization then runs on the real
    #   no-tokenizer path (detokenizer.py:L57-L59), so the front-end loop is
    #   the one vLLM itself runs in that mode.
    return BaseRenderer(config)


# SOURCE: vllm/inputs.py TokensPrompt — rendered token-ids EngineInput (the
# Renderer output the sync fast path consumes; dict with "type" discriminator)
EngineInput = dict


# ============================================================================
# SamplingParams surface — vllm/sampling_params.py (field subset: exactly what
# the kept front-end code reads; the verify() machinery is ch6).
# ============================================================================


# SOURCE: vllm/sampling_params.py:L182-L188 RequestOutputKind
class RequestOutputKind(enum.Enum):
    # Return entire output so far in every RequestOutput
    CUMULATIVE = 0
    # Return only deltas in each RequestOutput
    DELTA = 1
    # Do not return intermediate RequestOutput
    FINAL_ONLY = 2


# SOURCE: vllm/sampling_params.py:L199 SamplingParams (field subset)
@dataclass
class SamplingParams:
    n: int = 1                                             # L213
    detokenize: bool = True                                # L293
    max_tokens: int | None = None                          # L362
    top_p: float | None = None
    temperature: float | None = None
    num_logprobs: int | None = None
    prompt_logprobs: int | None = None
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE  # L301
    stream_interval: int | None = None                     # L302
    skip_clone: bool = False                               # L307
    # SUBTRACTED: ~80 further SamplingParams fields (stop/seed/
    #   structured-output/... , vllm/sampling_params.py) — not read by this
    #   chapter's kept path; stop-string evaluation lives in the detokenizer
    #   (ch7), logprobs payloads in ch7 (num_logprobs/prompt_logprobs stay:
    #   LogprobsProcessor reads them to decide the None path).

    # SOURCE: vllm/sampling_params.py:L748 SamplingParams.clone
    def clone(self) -> "SamplingParams":
        """If skip_clone is True, uses shallow copy instead of deep copy."""
        # SOURCE: vllm/sampling_params.py:L749 SamplingParams.clone body
        if self.skip_clone:
            return copy.copy(self)
        return copy.deepcopy(self)


# ============================================================================
# Outputs handed to the user — vllm/outputs.py (field subset).
# ============================================================================


# SOURCE: vllm/outputs.py:L21-L51 CompletionOutput
@dataclass
class CompletionOutput:
    """The output data of one completion output of a request."""

    index: int
    text: str
    token_ids: list[int]
    cumulative_logprob: float | None
    logprobs: list | None
    # SUBTRACTED: routed_experts / lora_request fields (vllm/outputs.py:L45,L48)
    #   — MoE metrics and LoRA axis are off this chapter's path.
    finish_reason: str | None = None
    stop_reason: int | str | None = None

    # SOURCE: vllm/outputs.py:L50 CompletionOutput.finished
    def finished(self) -> bool:
        return self.finish_reason is not None


# SOURCE: vllm/outputs.py:L85 RequestOutput
class RequestOutput:
    """The output data of a completion request to the LLM."""

    # SOURCE: vllm/outputs.py:L112-L150 RequestOutput.__init__
    def __init__(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_logprobs,
        outputs: list[CompletionOutput],
        finished: bool,
        metrics=None,
        lora_request: LoRARequest | None = None,
        encoder_prompt: str | None = None,
        encoder_prompt_token_ids: list[int] | None = None,
        num_cached_tokens: int | None = None,
        num_cache_creation_tokens: int | None = None,
        *,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.request_id = request_id
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_logprobs = prompt_logprobs
        self.outputs = outputs
        self.finished = finished
        self.metrics = metrics
        self.lora_request = lora_request
        self.encoder_prompt = encoder_prompt
        self.encoder_prompt_token_ids = encoder_prompt_token_ids
        self.num_cached_tokens = num_cached_tokens
        self.num_cache_creation_tokens = num_cache_creation_tokens
        self.kv_transfer_params = kv_transfer_params
        self.ec_transfer_params = ec_transfer_params

    # SOURCE: vllm/outputs.py:L152-L181 RequestOutput.add
    def add(self, next_output: "RequestOutput", aggregate: bool) -> None:
        """Merge subsequent RequestOutput into this one"""

        self.finished |= next_output.finished
        self.kv_transfer_params = next_output.kv_transfer_params
        self.ec_transfer_params = next_output.ec_transfer_params

        for next_completion in next_output.outputs:
            for i, completion in enumerate(self.outputs):
                if completion.index == next_completion.index:
                    if aggregate:
                        # Merge outputs with same index
                        completion.text += next_completion.text
                        completion.token_ids = list(completion.token_ids)
                        completion.token_ids.extend(next_completion.token_ids)
                        if next_completion.logprobs:
                            assert completion.logprobs is not None
                            completion.logprobs.extend(next_completion.logprobs)
                        completion.cumulative_logprob = (
                            next_completion.cumulative_logprob
                        )
                        completion.finish_reason = next_completion.finish_reason
                        completion.stop_reason = next_completion.stop_reason
                    else:
                        # Replace the output with the new one
                        self.outputs[i] = next_completion
                    break
            else:
                self.outputs.append(next_completion)


# SUBTRACTED: STREAM_FINISHED sentinel (vllm/outputs.py:L201-L208) — streaming
#   inputs only (dossier delete item 3); generate()'s sentinel check goes with it.


# ============================================================================
# Engine wire formats — vllm/v1/engine/__init__.py.
# msgspec.Struct -> dataclass is a mechanical host substitution (no msgspec on
# the host); field order and defaults follow the real structs.
# ============================================================================


# SOURCE: vllm/v1/engine/__init__.py:L31 FINISH_REASON_STRINGS
FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")


# SOURCE: vllm/v1/engine/__init__.py:L43 FinishReason
class FinishReason(enum.IntEnum):
    """
    Reason a request finished - stop, length, abort, error, or repetition.

    Int rather than Str for more compact serialization.

    stop - a stop string was emitted
    length - max_tokens was consumed, or max_model_len was reached
    abort - aborted by client
    error - retryable request-level internal error (e.g., KV load failure).
            Invariant: always converted to 500 Internal Server Error.
    repetition - repetitive token pattern detected (hallucination)

    """

    STOP = 0
    LENGTH = 1
    ABORT = 2
    ERROR = 3
    REPETITION = 4

    # SOURCE: vllm/v1/engine/__init__.py:L64 FinishReason.__str__
    def __str__(self):
        return FINISH_REASON_STRINGS[self.value]


# SOURCE: vllm/v1/engine/__init__.py:L97 EngineCoreRequest
@dataclass
class EngineCoreRequest:
    request_id: str
    prompt_token_ids: list[int] | None
    sampling_params: SamplingParams | None
    arrival_time: float
    lora_request: LoRARequest | None = None
    # SUBTRACTED: mm_features / pooling_params / cache_salt / data_parallel_rank
    #   / prompt_embeds / prompt_is_token_ids (vllm/v1/engine/__init__.py:
    #   L104-L118) — multimodal is ch6, pooling/priority axis is dossier delete
    #   item 7, embeds are the Renderer's business (ch6).

    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    client_index: int = 0

    # SUBTRACTED: current_wave (L124-L127, DP lockstep — item 2 / ch34) and
    #   priority (item 7).
    trace_headers: dict[str, str] | None = None
    resumable: bool = False

    # The user-provided request ID. This field is set internally,
    # copied from the provided request_id that's originally assigned
    # to the request_id field, see InputProcessor.assign_request_id().
    # Used in outputs and to support abort(req_id, internal=False).
    external_req_id: str | None = None

    # SUBTRACTED: reasoning_ended / reasoning_parser_kwargs (L139-L140, item 7)
    #   and abort_immediately (L142-L146, disagg cleanup — item 8 adjacent).

    # SOURCE: vllm/v1/engine/__init__.py:L148-L154 EngineCoreRequest.params
    @property
    # SOURCE: vllm/v1/engine/__init__.py:L148 EngineCoreRequest.params
    def params(self) -> SamplingParams:
        """Return the processed params (sampling or pooling)."""
        if self.sampling_params is not None:
            return self.sampling_params
        # SUBTRACTED: pooling_params fallback (L152-L153, item 7)
        raise AssertionError("sampling_params must be set (pooling subtracted)")


# SOURCE: vllm/v1/engine/__init__.py:L184 EngineCoreOutput
@dataclass
class EngineCoreOutput:
    request_id: str
    new_token_ids: list[int]

    # SUBTRACTED: new_logprobs / new_prompt_logprobs_tensors (L193-L194) —
    #   logprob payloads are ch7; pooling_output (L196) is item 7;
    #   events (L200) is timing telemetry.
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    kv_transfer_params: dict[str, Any] | None = None
    ec_transfer_params: dict[str, Any] | None = None
    # SUBTRACTED: trace_headers (L204, tracing — item 9), prefill_stats (L206)
    #   and routed_experts / num_nans_in_logits (L208-L211) — metrics/MoE axes.

    # SOURCE: vllm/v1/engine/__init__.py:L213-L215 EngineCoreOutput.finished
    @property
    # SOURCE: vllm/v1/engine/__init__.py:L213 EngineCoreOutput.finished
    def finished(self) -> bool:
        return self.finish_reason is not None


# SOURCE: vllm/v1/engine/__init__.py:L230 EngineCoreOutputs
@dataclass
class EngineCoreOutputs:
    # SUBTRACTED: per-batch columnar packing note (L236-L237).

    engine_index: int = 0
    # [num_reqs]
    outputs: list[EngineCoreOutput] = field(default_factory=list)
    # SUBTRACTED: scheduler_stats (L243 — item 5), utility_output (L246 —
    #   item 8), finished_requests (L247 — DPLB bookkeeping, item 2),
    #   wave_complete / start_wave (L249-L254 — DP, item 2).
    timestamp: float = 0.0

    # SOURCE: vllm/v1/engine/__init__.py:L256-L258 EngineCoreOutputs.__post_init__
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


# SOURCE: vllm/v1/engine/__init__.py:L261 EngineCoreRequestType (member subset)
class EngineCoreRequestType(enum.Enum):
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    # SUBTRACTED: START_DP_WAVE (DP, item 2), UTILITY (item 8),
    #   EXECUTOR_FAILED / WAKEUP (EngineCoreProc shutdown sentinels, item 1).


# ============================================================================
# Detokenizer / logprobs — the ch7 depths, kept as the real no-tokenizer
# baseline the front end actually runs on in skip_tokenizer_init mode.
# ============================================================================


# SOURCE: vllm/v1/engine/detokenizer.py:L31 IncrementalDetokenizer
class IncrementalDetokenizer:
    # SOURCE: vllm/v1/engine/detokenizer.py:L32 IncrementalDetokenizer.__init__
    def __init__(self):
        self.token_ids: list[int] = []

    # SOURCE: vllm/v1/engine/detokenizer.py:L35 output_token_ids property
    @property
    # SOURCE: vllm/v1/engine/detokenizer.py:L35 output_token_ids
    def output_token_ids(self) -> list[int]:
        return self.token_ids

    # SOURCE: vllm/v1/engine/detokenizer.py:L39 num_output_tokens
    def num_output_tokens(self) -> int:
        return len(self.token_ids)

    # SOURCE: vllm/v1/engine/detokenizer.py:L42 update
    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        self.token_ids.extend(new_token_ids)
        return None

    # SOURCE: vllm/v1/engine/detokenizer.py:L46 get_next_output_text
    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        return ""

    # SOURCE: vllm/v1/engine/detokenizer.py:L49 from_new_request
    @classmethod
    # SOURCE: vllm/v1/engine/detokenizer.py:L49 IncrementalDetokenizer.from_new_request
    def from_new_request(
        cls,
        tokenizer,
        request: EngineCoreRequest,
    ) -> "IncrementalDetokenizer":
        assert request.sampling_params is not None

        if tokenizer is None:
            # No tokenizer => skipping detokenization.
            return IncrementalDetokenizer()

        # SUBTRACTED: Fast/Slow incremental detokenizer selection
        #   (vllm/v1/engine/detokenizer.py:L61-L66 + BaseIncrementalDetokenizer)
        #   — the tokenizer-backed increment/stop-string algorithm is ch7; the
        #   companion runs tokenizer-less (the real None branch above).
        return IncrementalDetokenizer()


# SOURCE: vllm/v1/engine/logprobs.py:L29 LogprobsProcessor (field subset)
@dataclass
class LogprobsProcessor:
    # Tokenizer for this request,
    # None if detokenization is disabled.
    tokenizer: Any = None

    # Logprobs for this request
    logprobs: list | None = None
    prompt_logprobs: list | None = None
    cumulative_logprob: float | None = None
    num_logprobs: int | None = None
    num_prompt_logprobs: int | None = None

    # SOURCE: vllm/v1/engine/logprobs.py:L42 from_new_request
    @classmethod
    # SOURCE: vllm/v1/engine/logprobs.py:L42 LogprobsProcessor.from_new_request
    def from_new_request(
        cls,
        tokenizer,
        request: EngineCoreRequest,
    ) -> "LogprobsProcessor":
        sampling_params = request.sampling_params
        assert sampling_params is not None
        num_logprobs = sampling_params.num_logprobs
        num_prompt_logprobs = sampling_params.prompt_logprobs
        return cls(
            tokenizer=tokenizer,
            cumulative_logprob=(None if num_logprobs is None else 0.0),
            # SUBTRACTED: create_sample_logprobs / create_prompt_logprobs from
            #   flat_logprobs (vllm/v1/engine/logprobs.py:L55-L64) — ch7; None
            #   when logprobs are not requested (the kept path).
            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
        )

    # SOURCE: vllm/v1/engine/logprobs.py:L189 pop_prompt_logprobs
    def pop_prompt_logprobs(self):
        """Pop and return all request prompt logprobs ..."""
        # SUBTRACTED: aggregation-across-prefill-chunks bookkeeping (L198-L206)
        #   — always None on the kept path; the DELTA side effect is preserved.
        prompt_logprobs = self.prompt_logprobs
        self.prompt_logprobs = None
        return prompt_logprobs

    # SOURCE: vllm/v1/engine/logprobs.py:L348 update_from_output
    def update_from_output(self, output: EngineCoreOutput) -> None:
        # SUBTRACTED: sample/prompt logprob assembly from output.new_logprobs
        #   (L349-L352 + helpers) — ch7; the EngineCoreOutput logprob payload
        #   fields are subtracted with it, so there is nothing to update.
        return None


# ============================================================================
# OutputProcessor & friends — vllm/v1/engine/output_processor.py.
# ============================================================================


# SOURCE: vllm/v1/engine/output_processor.py:L45 RequestOutputCollector
class RequestOutputCollector:
    """
    Collects streamed RequestOutputs per individual request,
    for hand-off to the consuming asyncio generate task.

    When streaming deltas, RequestOutputs are merged if the
    producer gets ahead of the consumer.
    """

    # SOURCE: vllm/v1/engine/output_processor.py:L54 RequestOutputCollector.__init__
    def __init__(self, output_kind: RequestOutputKind, request_id: str):
        self.aggregate = output_kind == RequestOutputKind.DELTA
        self.request_id = request_id
        self.output: RequestOutput | Exception | None = None
        self.ready = asyncio.Event()
        # SUBTRACTED: _input_stream_task (L60) — streaming inputs (item 3).

    # SOURCE: vllm/v1/engine/output_processor.py:L62 RequestOutputCollector.put
    def put(self, output: RequestOutput | Exception) -> None:
        """Non-blocking put operation."""
        if self.output is None or isinstance(output, Exception):
            self.output = output
            self.ready.set()
        elif isinstance(self.output, RequestOutput) and isinstance(
            output, RequestOutput
        ):
            # This ensures that request outputs with different request indexes
            # (if n > 1) do not override each other.
            self.output.add(output, aggregate=self.aggregate)
        # SUBTRACTED: PoolingRequestOutput arm (L73-L76) — pooling (item 7).
        # NOTE: the merge arm above is kept verbatim: DELTA streams with the
        #   producer ahead of the consumer need it at n == 1 too (dropping it
        #   would silently lose deltas, changing observable behavior).

    # SOURCE: vllm/v1/engine/output_processor.py:L78 RequestOutputCollector.get
    async def get(self) -> RequestOutput:
        """Get operation blocks on put event."""
        while (output := self.output) is None:
            await self.ready.wait()
        self.output = None
        self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    # SOURCE: vllm/v1/engine/output_processor.py:L88 RequestOutputCollector.get_nowait
    def get_nowait(self) -> RequestOutput | None:
        """Non-blocking get operation."""
        output = self.output
        if output is not None:
            self.output = None
            self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    # SOURCE: vllm/v1/engine/output_processor.py:L98 RequestOutputCollector.close
    def close(self):
        # SUBTRACTED: streaming-input task cancel (L99-L100) — no-op for
        #   one-shot prompts (item 3).
        pass

    # SUBTRACTED: __del__ (L103-L106) — only guards the streaming task (item 3).


# SOURCE: vllm/v1/engine/output_processor.py:L109 OutputProcessorOutput
@dataclass
# SOURCE: vllm/v1/engine/output_processor.py:L109 OutputProcessorOutput
class OutputProcessorOutput:
    request_outputs: list[RequestOutput]
    reqs_to_abort: list[str]


# SUBTRACTED: StreamingUpdate (vllm/v1/engine/output_processor.py:L115-L126) —
#   streaming inputs (dossier delete item 3).


# SOURCE: vllm/v1/engine/output_processor.py:L129 RequestState
class RequestState:
    # SOURCE: vllm/v1/engine/output_processor.py:L130 RequestState.__init__
    def __init__(
        self,
        request_id: str,
        external_req_id: str,
        request_index: int,
        lora_request: LoRARequest | None,
        output_kind: RequestOutputKind,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        logprobs_processor: LogprobsProcessor | None,
        detokenizer: IncrementalDetokenizer | None,
        max_tokens_param: int | None,
        arrival_time: float,
        queue: RequestOutputCollector | None,
        stream_interval: int,
        top_p: float | None = None,
        n: int | None = None,
        temperature: float | None = None,
    ):
        # SUBTRACTED: parent_req / prompt_embeds / log_stats->RequestStateStats
        #   / routed_experts_chunks / streaming_input & input_chunk_queue
        #   (vllm/v1/engine/output_processor.py:L134,L139-L141,L177-L190) —
        #   n>1 fan-out (item 4), embeds (ch6), stats (item 5), streaming (item 3).
        self.request_id = request_id
        self.external_req_id = external_req_id
        self.request_index = request_index
        self.lora_request = lora_request
        self.lora_name = lora_request.lora_name if lora_request is not None else None
        self.output_kind = output_kind
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.logprobs_processor = logprobs_processor
        self.detokenizer = detokenizer
        self.max_tokens_param = max_tokens_param
        self.top_p = top_p
        self.n = n
        self.temperature = temperature
        self.queue = queue

        # Stream Interval
        self.stream_interval = stream_interval
        # SUBTRACTED: sent_tokens_offset (L184) — consumed by the ch7 slicing.

    # SUBTRACTED: apply_streaming_update (L192-L209) — streaming inputs (item 3).

    # SOURCE: vllm/v1/engine/output_processor.py:L211 RequestState.from_new_request
    @classmethod
    # SOURCE: vllm/v1/engine/output_processor.py:L211 RequestState.from_new_request
    def from_new_request(
        cls,
        tokenizer,
        request: EngineCoreRequest,
        prompt: str | None,
        request_index: int,
        queue: RequestOutputCollector | None,
        stream_interval: int,
    ) -> "RequestState":
        # SUBTRACTED: parent_req parameter threading (item 4) and log_stats
        #   parameter (item 5 — drives RequestStateStats, subtracted above).
        if sampling_params := request.sampling_params:
            if not sampling_params.detokenize:
                tokenizer = None
            output_kind = sampling_params.output_kind
            if sampling_params.stream_interval is not None:
                # clamp to the engine-level stream interval.
                stream_interval = max(sampling_params.stream_interval, stream_interval)
            logprobs_processor = LogprobsProcessor.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            detokenizer = IncrementalDetokenizer.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            max_tokens_param = sampling_params.max_tokens
            top_p = sampling_params.top_p
            n = sampling_params.n
            temperature = sampling_params.temperature
        # SUBTRACTED: pooling else-arm (L242-L250, item 7).
        assert request.external_req_id is not None
        return cls(
            request_id=request.request_id,
            external_req_id=request.external_req_id,
            request_index=request_index,
            lora_request=request.lora_request,
            output_kind=output_kind,
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            logprobs_processor=logprobs_processor,
            detokenizer=detokenizer,
            max_tokens_param=max_tokens_param,
            top_p=top_p,
            n=n,
            temperature=temperature,
            arrival_time=request.arrival_time,
            queue=queue,
            stream_interval=stream_interval,
        )

    # SOURCE: vllm/v1/engine/output_processor.py:L276 RequestState.make_request_output
    def make_request_output(
        self,
        new_token_ids: list[int],
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
    ) -> RequestOutput | None:
        # SUBTRACTED: pooling_output parameter (L279) — pooling (item 7).
        finished = finish_reason is not None
        final_only = self.output_kind == RequestOutputKind.FINAL_ONLY

        if not finished and final_only:
            # Only the final output is required in FINAL_ONLY mode.
            return None

        # SUBTRACTED: stream_interval > 1 send gating + DELTA offset slicing
        #   (vllm/v1/engine/output_processor.py:L292-L313) — ch7.

        external_req_id = self.external_req_id

        # SUBTRACTED: pooling branch (L317-L322, item 7).
        output = self._new_completion_output(new_token_ids, finish_reason, stop_reason)

        # SUBTRACTED: ParentRequest aggregation (L326-L333) — n>1 fan-out
        #   (item 4); a lone child returns directly.
        outputs = [output]

        return self._new_request_output(
            external_req_id,
            outputs,
            finished,
            kv_transfer_params,
            ec_transfer_params,
        )

    # SOURCE: vllm/v1/engine/output_processor.py:L342 RequestState._new_request_output
    def _new_request_output(
        self,
        external_req_id: str,
        outputs: list[CompletionOutput],
        finished: bool,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
    ) -> RequestOutput:
        # SUBTRACTED: prompt_embeds placeholder ids (L350-L353) — embeds (ch6).
        prompt_token_ids = self.prompt_token_ids
        assert prompt_token_ids is not None

        # SUBTRACTED: PoolingRequestOutput arm (L356-L365, item 7).
        if self.output_kind == RequestOutputKind.DELTA:
            # Side effect: logprobs processor forgets prompt logprobs
            prompt_logprobs = self.logprobs_processor.pop_prompt_logprobs()
        else:
            prompt_logprobs = self.logprobs_processor.prompt_logprobs

        return RequestOutput(
            request_id=external_req_id,  # request_id is what was provided externally
            lora_request=self.lora_request,
            prompt=self.prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=prompt_logprobs,
            outputs=outputs,
            finished=finished,
            kv_transfer_params=kv_transfer_params,
            ec_transfer_params=ec_transfer_params,
        )

    # SOURCE: vllm/v1/engine/output_processor.py:L388 RequestState._new_completion_output
    def _new_completion_output(
        self,
        token_ids: list[int],
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
    ) -> CompletionOutput:
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

        # SUBTRACTED: routed_experts concatenation on finish (L409-L412) — MoE.
        return CompletionOutput(
            index=self.request_index,
            text=text,
            token_ids=token_ids,
            logprobs=logprobs,
            cumulative_logprob=self.logprobs_processor.cumulative_logprob,
            finish_reason=str(finish_reason) if finished else None,
            stop_reason=stop_reason if finished else None,
        )


# SOURCE: vllm/v1/engine/output_processor.py:L429 OutputProcessor
class OutputProcessor:
    """Process EngineCoreOutputs into RequestOutputs."""

    # SOURCE: vllm/v1/engine/output_processor.py:L432 OutputProcessor.__init__
    def __init__(
        self,
        tokenizer,
        *,
        log_stats: bool,
        stream_interval: int = 1,
        tracing_enabled: bool = False,
    ):
        self.log_stats = log_stats
        self.tokenizer = tokenizer
        self.stream_interval = stream_interval
        self.request_states: dict[str, RequestState] = {}
        # SUBTRACTED: parent_requests table (L444) — n>1 fan-out (item 4).
        self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)
        # SUBTRACTED: lora_states (L446, LoRA run-state metrics — item 5) and
        #   tracing_enabled (L447 — item 9); the kwarg is kept in the signature
        #   so both faces' construction sites stay verbatim.
        self.tracing_enabled = tracing_enabled

    # SOURCE: vllm/v1/engine/output_processor.py:L449 OutputProcessor.get_num_unfinished_requests
    def get_num_unfinished_requests(self):
        return len(self.request_states)

    # SOURCE: vllm/v1/engine/output_processor.py:L452 OutputProcessor.has_unfinished_requests
    def has_unfinished_requests(self) -> bool:
        return len(self.request_states) > 0

    # SOURCE: vllm/v1/engine/output_processor.py:L455 OutputProcessor.propagate_error
    def propagate_error(self, e: Exception):
        """Propagate error to all generate() tasks."""

        for _, state in self.request_states.items():
            assert state.queue is not None
            state.queue.put(e)

    # SOURCE: vllm/v1/engine/output_processor.py:L462 OutputProcessor.abort_requests
    def abort_requests(self, request_ids: Iterable[str], internal: bool) -> list[str]:
        """Abort a list of requests.

        The request_ids may be either external request IDs (those passed to
        InputProcessor.process_inputs()) or internal request IDs (those randomly
        generated when creating the EngineCoreRequest).

        If an external request ID is provided, and that external request ID
        was used for multiple requests, all requests associated with that external
        request ID are aborted.

        SUBTRACTED: the parallel-sampling parent paragraph (L473-L475) — the
        n>1 fan-out is dossier delete item 4.
        """
        internal_req_ids = []
        for request_id in request_ids:
            if internal:
                # Internal ID - this may be a parent request
                internal_req_ids.append(request_id)

                # Remove internal ID from the external->internal mapping
                if req_state := self.request_states.get(request_id):
                    external_req_id = req_state.external_req_id
                    internal_ids = self.external_req_ids[external_req_id]
                    internal_ids.remove(request_id)
                    if not internal_ids:
                        del self.external_req_ids[external_req_id]
            elif internal_ids := self.external_req_ids.pop(request_id, []):
                # External ID - abort all requests in the external->internal mapping
                internal_req_ids.extend(internal_ids)

        request_ids_to_abort = []
        for request_id in internal_req_ids:
            req_state = self.request_states.pop(request_id, None)
            if req_state is not None:
                # SUBTRACTED: lora_states.request_finished (L498) — item 5.
                request_ids_to_abort.append(request_id)
                # Produce final abort output.
                if req_state.queue is not None and (
                    request_output := req_state.make_request_output(
                        new_token_ids=[],
                        # SUBTRACTED: pooling EMPTY_CPU_TENSOR arm (L504-L508)
                        #   — pooling (item 7); text requests pass None.
                        finish_reason=FinishReason.ABORT,
                        stop_reason=None,
                        kv_transfer_params=None,
                        ec_transfer_params=None,
                    )
                ):
                    req_state.queue.put(request_output)
            # SUBTRACTED: parent-request abort branch (L516-L522) — item 4.
        return request_ids_to_abort

    # SOURCE: vllm/v1/engine/output_processor.py:L525 OutputProcessor.add_request
    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req=None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        # SUBTRACTED: parent_req parameter type (ParentRequest, item 4); the
        #   slot is kept so both faces' call sites stay positional-identical.
        request_id = request.request_id
        # SUBTRACTED: streaming-resume branch (L533-L537) — item 3: a state
        #   already under this internal id only happens for streaming inputs.
        req_state = RequestState.from_new_request(
            tokenizer=self.tokenizer,
            request=request,
            prompt=prompt,
            request_index=request_index,
            queue=queue,
            stream_interval=self.stream_interval,
        )
        # SUBTRACTED: log_stats parameter (item 5 — RequestStateStats subtracted).
        self.request_states[request_id] = req_state
        # SUBTRACTED: parent_requests registration (L550-L551) — item 4.

        # Track the external_req_id -> [internal_req_id, ...] mapping
        self.external_req_ids[req_state.external_req_id].append(request_id)

    # SUBTRACTED: _update_streaming_request_state
    #   (vllm/v1/engine/output_processor.py:L556-L587) — streaming (item 3).

    # SOURCE: vllm/v1/engine/output_processor.py:L589 OutputProcessor.process_outputs
    def process_outputs(
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float | None = None,
        iteration_stats: Any | None = None,
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

        If you need to touch every element of the batch, do it from
        within the loop below.
        """
        # SUBTRACTED: iteration_stats bookkeeping — the parameter is kept so
        #   both drivers' call sites stay verbatim; callers pass None (item 5).

        request_outputs: list[RequestOutput] = []
        reqs_to_abort: list[str] = []
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                # Ignore output for already-aborted request.
                continue

            # SUBTRACTED: 1) stats for this iteration (L626-L629) — item 5.

            new_token_ids = engine_core_output.new_token_ids
            # SUBTRACTED: pooling_output unpack (L632) — item 7.
            finish_reason = engine_core_output.finish_reason
            stop_reason = engine_core_output.stop_reason
            kv_transfer_params = engine_core_output.kv_transfer_params
            ec_transfer_params = engine_core_output.ec_transfer_params
            # SUBTRACTED: routed_experts chunk accumulation + prefill_stats /
            #   is_prefilling carry (L637-L650) — MoE/metrics axes.

            # SUBTRACTED: pooling_output None guard (L652) — pooling (item 7);
            #   text requests always detokenize:
            assert req_state.detokenizer is not None
            assert req_state.logprobs_processor is not None
            # 2) Detokenize the token ids into text and perform stop checks.
            stop_string = req_state.detokenizer.update(
                new_token_ids, finish_reason == FinishReason.STOP
            )
            if stop_string:
                finish_reason = FinishReason.STOP
                stop_reason = stop_string

            # 3) Compute sample and prompt logprobs for request,
            # if required.
            req_state.logprobs_processor.update_from_output(engine_core_output)

            # 4) Create and handle RequestOutput objects.
            if request_output := req_state.make_request_output(
                new_token_ids,
                finish_reason,
                stop_reason,
                kv_transfer_params,
                ec_transfer_params,
            ):
                # SUBTRACTED: streaming_input finished=False override
                #   (L676-L677) — item 3.
                if req_state.queue is not None:
                    # AsyncLLM: put into queue for handling by generate().
                    req_state.queue.put(request_output)
                else:
                    # LLMEngine: return list of RequestOutputs.
                    request_outputs.append(request_output)

            # Free completed requests.
            if finish_reason is not None:
                # SUBTRACTED: streaming input_chunk_queue pop (L688-L693) — item 3.
                self._finish_request(req_state)
                if not engine_core_output.finished:
                    # If req not finished in EngineCore, but Detokenizer
                    # detected stop string, abort needed in EngineCore.
                    reqs_to_abort.append(req_id)

                # SUBTRACTED: finished-stats + tracing (L701-L706) — items 5/9.

        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
        )

    # SOURCE: vllm/v1/engine/output_processor.py:L713 OutputProcessor._finish_request
    def _finish_request(self, req_state: RequestState) -> None:
        req_id = req_state.request_id
        self.request_states.pop(req_id)

        internal_ids = self.external_req_ids[req_state.external_req_id]
        internal_ids.remove(req_id)
        if not internal_ids:
            del self.external_req_ids[req_state.external_req_id]

        # SUBTRACTED: parent-request removal (L722-L725) — item 4.

    # SUBTRACTED: update_scheduler_stats (L727-L728) — item 5.
    # SUBTRACTED: do_tracing / _update_stats_from_output / _update_stats_from_finished
    #   (vllm/v1/engine/output_processor.py:L730-L836) — items 5/9.


# ============================================================================
# EngineCore — vllm/v1/engine/core.py, reduced to the in-process stub the
# dossier sanctions (delete item 1): the scheduling/model loop is ch9; this
# keeps the engine-side halves the chapter needs — the request table (the far
# side of the double registration) and the output IO routing by client_index.
# ============================================================================


# SOURCE: vllm/v1/engine/core.py:L103 EngineCore
class EngineCore:
    """Inner loop of vLLM's Engine.

    (companion: scheduling/model-execution internals are SUBTRACTED — ch9;
    this in-process stub stands in for the background EngineCore process per
    dossier delete item 1.)
    """

    # SOURCE: vllm/v1/engine/core.py:L106 EngineCore.__init__
    def __init__(
        self,
        vllm_config: VllmConfig | None = None,
        executor_class: type | None = None,
        log_stats: bool = False,
        **kwargs,
    ):
        # SUBTRACTED: model executor / KV-cache init / scheduler assembly /
        #   busy loop / input+output IO threads / ZMQ / handshake
        #   (vllm/v1/engine/core.py:L106-L234, L1645-L1810) — ch9/ch5.
        self.vllm_config = vllm_config
        # Engine-side request table: the "separate process" ledger of the
        # double registration (WC2). SUBTRACTED: scheduler admission — ch9.
        self.requests: dict[str, EngineCoreRequest] = {}
        # (request_type, request) messages that crossed the boundary — the
        # in-process stand-in for the engine input queue the socket thread
        # feeds (core.py:L1740-L1741 input_queue.put_nowait).
        self.input_queue: queue.Queue = queue.Queue()
        # One output sink per front-end client, indexed by client_index — the
        # in-process stand-in for the per-client PUSH sockets the output IO
        # thread selects among (core.py:L1761-L1766).
        self.sockets: list = [queue.Queue()]
        # SUBTRACTED: engine_dead is maintained by the deleted monitor thread
        #   (core_client.py:L708-L735); the flag itself stays observable.
        self.engine_dead = False

    # SOURCE: vllm/v1/engine/core.py:L361 EngineCore.get_supported_tasks
    def get_supported_tasks(self) -> tuple[str, ...]:
        # SUBTRACTED: model_executor.supported_tasks probe + pooler logging
        #   (L362-L363) — ch9; a text-generation engine reports "generate".
        return ("generate",)

    # SOURCE: vllm/v1/engine/core.py:L439 EngineCore.add_request
    def add_request(self, request: EngineCoreRequest, request_wave: int = 0):
        """Add request to the scheduler.

        `request_wave`: indicate which wave of requests this is expected to
        belong to in DP case
        """
        # SUBTRACTED: wave plumbing (DP, item 2) — parameter kept verbatim.
        # Validate the request_id type.
        if not isinstance(request.request_id, str):
            raise TypeError(
                f"request_id must be a string, got {type(request.request_id)}"
            )
        # SUBTRACTED: pooling task check (L451-L458) — item 7; scheduler
        #   admission (L459+, ch9) reduces to recording the request.
        self.requests[request.request_id] = request

    # SOURCE: vllm/v1/engine/core.py:L485 EngineCore.abort_requests
    def abort_requests(self, request_ids: list[str]):
        """Abort requests from the scheduler."""

        # TODO: The scheduler doesn't really need to know the
        # specific finish reason, TBD whether we propagate that
        # (i.e. client-aborted vs stop criteria met).
        # SUBTRACTED: scheduler.finish_requests (L491) — ch9; idempotent drop.
        for request_id in request_ids:
            self.requests.pop(request_id, None)

    # SOURCE: vllm/v1/engine/core.py:L751 EngineCore.shutdown
    def shutdown(self):
        # SUBTRACTED: structured-output/executor/scheduler/gc teardown
        #   (L752-L766) — ch9.
        self.requests.clear()

    # SOURCE: vllm/v1/engine/core.py:L969 EngineCore.preprocess_add_request
    def preprocess_add_request(self, request: EngineCoreRequest):
        """Preprocess the request.

        This function could be directly used in input processing thread to allow
        request initialization running in parallel with Model forward
        """
        # SUBTRACTED: mm receiver cache + Request.from_engine_core_request +
        #   structured-output grammar init (L975-L998) — ch6/ch31/ch9; the stub
        #   passes the EngineCoreRequest through unchanged (wave 0 default).
        return request, 0

    # ---- companion seams: the in-process stand-in for the deleted IPC layer
    # (dossier delete item 1). Names anchor to the real wiring they replace. --

    # wiring that launch_core_engines performs via ZMQ addresses (core.py:
    # get_engine_zmq_addresses -> engines PUSH-connect to each client's PULL)
    # SOURCE: vllm/v1/engine/utils.py launch_core_engines — address wiring
    def attach_output_socket(self, client_index: int, sink) -> None:
        while len(self.sockets) <= client_index:
            self.sockets.append(None)
        self.sockets[client_index] = sink

    # input socket thread's decode-and-dispatch (core.py:L1711-L1741)
    # SOURCE: vllm/v1/engine/core.py:L1711-L1741 process_input_sockets dispatch
    def handle_client_message(self, request_type, request) -> None:
        if request_type == EngineCoreRequestType.ADD:
            self.add_request(request)
        elif request_type == EngineCoreRequestType.ABORT:
            self.abort_requests(request)
        # SUBTRACTED: UTILITY dispatch (L1722-L1729) and DP READY handling —
        #   items 2/8.

    # SOURCE: vllm/v1/engine/core.py:L1436-L1444 EngineCore busy-loop step
    #   (step_fn -> output_queue) fused with the output IO thread loop body
    # SOURCE: vllm/v1/engine/core.py:L1778-L1810 process_output_sockets routing
    def emit_step_outputs(self, step_outputs: list[tuple[int, EngineCoreOutputs]]) -> None:
        """Hand the engine one step's worth of outputs.

        In real vLLM the busy loop's step_fn() produces these
        (client_index, EngineCoreOutputs) tuples (ch9) and the output IO
        thread routes each batch to the right front-end. The companion's
        caller (test/explainer) plays the scheduler: pass the tuples a step
        produced; pending input messages are drained first, exactly as the
        busy loop consumes its input queue before each step.
        """
        while True:
            try:
                request_type, request = self.input_queue.get_nowait()
            except queue.Empty:
                break
            self.handle_client_message(request_type, request)
        for output in step_outputs:
            # --- process_output_sockets loop body (core.py:L1778-L1810) ---
            assert not isinstance(output, bytes)
            client_index, outputs = output
            # SUBTRACTED: engine_index stamp (L1786) — single engine (DP/ch34).
            # SUBTRACTED: client_index == -1 DP-coordinator branch (L1788-L1793)
            #   — item 2 / ch34.
            # SUBTRACTED: buffer reuse / MessageTracker zero-copy engineering
            #   (L1795-L1810) — ch5.
            self.sockets[client_index].put_nowait(outputs)
            # Engine-side ledger cleanup: the scheduler frees each request
            # whose output carries a finish_reason (WC2 cost: both ledgers
            # must clean up in sync). SUBTRACTED: the real free path
            # (scheduler.finish_requests -> Request freed) — ch9.
            for engine_core_output in outputs.outputs:
                if engine_core_output.finished:
                    self.requests.pop(engine_core_output.request_id, None)


# ============================================================================
# Protocol face — vllm/engine/protocol.py. OpenAI serving programs against
# this abstraction only; AsyncLLM is its sole implementation in the tree.
# ============================================================================


# SUBTRACTED: StreamingInput (vllm/engine/protocol.py:L29-L38) — streaming
#   inputs (dossier delete item 3).


# SOURCE: vllm/engine/protocol.py:L41 EngineClient
class EngineClient(ABC):
    """Protocol class for Clients to Engine"""

    vllm_config: VllmConfig
    model_config: ModelConfig
    renderer: BaseRenderer
    input_processor: "InputProcessor"

    # SOURCE: vllm/engine/protocol.py:L49-L51 EngineClient.is_running
    @property
    @abstractmethod
    # SOURCE: vllm/engine/protocol.py:L49 EngineClient.is_running
    def is_running(self) -> bool: ...

    # SOURCE: vllm/engine/protocol.py:L53-L55 EngineClient.is_stopped
    @property
    @abstractmethod
    # SOURCE: vllm/engine/protocol.py:L53 EngineClient.is_stopped
    def is_stopped(self) -> bool: ...

    # SOURCE: vllm/engine/protocol.py:L57-L59 EngineClient.errored
    @property
    @abstractmethod
    # SOURCE: vllm/engine/protocol.py:L57 EngineClient.errored
    def errored(self) -> bool: ...

    # SOURCE: vllm/engine/protocol.py:L61-L63 EngineClient.dead_error
    @property
    @abstractmethod
    # SOURCE: vllm/engine/protocol.py:L61 EngineClient.dead_error
    def dead_error(self) -> BaseException: ...

    # SOURCE: vllm/engine/protocol.py:L65-L85 EngineClient.generate
    @abstractmethod
    # SOURCE: vllm/engine/protocol.py:L66 EngineClient.generate
    def generate(
        self,
        prompt: EngineInput,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        trace_headers: dict[str, str] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        """Generate outputs for a request."""
        ...
        # SUBTRACTED: EngineCoreRequest / AsyncGenerator[StreamingInput] union
        #   arms (deprecated direct-pass — item 6; streaming — item 3) and
        #   tokenization_kwargs / priority / data_parallel_rank /
        #   reasoning_ended / reasoning_parser_kwargs (items 6/7).

    # SUBTRACTED: encode (vllm/engine/protocol.py:L87-L100) — pooling (item 7).

    # SOURCE: vllm/engine/protocol.py:L102-L110 EngineClient.abort
    @abstractmethod
    # SOURCE: vllm/engine/protocol.py:L102 EngineClient.abort
    async def abort(self, request_id: str | Iterable[str]) -> None:
        """Abort a request.

        Args:
            request_id: The unique id of the request,
                        or an iterable of such ids.
        """
        ...

    # SUBTRACTED: notify_kv_transfer_request_rejected (L112-L124, disagg) /
    #   is_tracing_enabled / do_log_stats / check_health / profiling / cache
    #   resets / sleep & wake / lora / pause & resume (L126-L215) — items 5/7/8/9.

    # SOURCE: vllm/engine/protocol.py:L217-L220 EngineClient.shutdown
    @abstractmethod
    # SOURCE: vllm/engine/protocol.py:L217 EngineClient.shutdown
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown the engine with optional timeout."""
        ...

    # SUBTRACTED: scale_elastic_ep / collective_rpc / handle_fault / get_status
    #   / weight-transfer surface (vllm/engine/protocol.py:L222-L281) — item 8.

    # SOURCE: vllm/engine/protocol.py:L248-L250 EngineClient.get_supported_tasks
    async def get_supported_tasks(self) -> tuple[str, ...]:
        """Get supported tasks"""
        raise NotImplementedError


# ============================================================================
# Client family — vllm/v1/engine/core_client.py. The ZMQ/msgpack physical
# layer is dossier delete item 1: MP clients keep their names, queues and
# blocking semantics, and hand messages to the in-process EngineCore stub.
# ============================================================================


# SOURCE: vllm/v1/engine/core_client.py:L78 EngineCoreClient
class EngineCoreClient(ABC):
    """
    EngineCoreClient: subclasses handle different methods for pushing
        and pulling from the EngineCore for asyncio / multiprocessing.

    Subclasses:
    * InprocClient: In process EngineCore (for V0-style LLMEngine use)
    * SyncMPClient: ZMQ + background proc EngineCore (for LLM)
    * AsyncMPClient: ZMQ + background proc EngineCore w/ asyncio (for AsyncLLM)
    """

    # SOURCE: vllm/v1/engine/core_client.py:L89-L112 EngineCoreClient.make_client
    @staticmethod
    # SOURCE: vllm/v1/engine/core_client.py:L89 EngineCoreClient.make_client
    def make_client(
        multiprocess_mode: bool,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
    ) -> "EngineCoreClient":
        # SUBTRACTED: @instrument span decorator (L115) — telemetry (item 9).
        # TODO: support this for debugging purposes.
        if asyncio_mode and not multiprocess_mode:
            raise NotImplementedError(
                "Running EngineCore in asyncio without multiprocessing "
                "is not currently supported."
            )

        if multiprocess_mode and asyncio_mode:
            return EngineCoreClient.make_async_mp_client(
                vllm_config, executor_class, log_stats
            )

        if multiprocess_mode and not asyncio_mode:
            return SyncMPClient(vllm_config, executor_class, log_stats)

        return InprocClient(vllm_config, executor_class, log_stats)

    # SOURCE: vllm/v1/engine/core_client.py:L114-L139 EngineCoreClient.make_async_mp_client
    @staticmethod
    # SOURCE: vllm/v1/engine/core_client.py:L116 EngineCoreClient.make_async_mp_client
    def make_async_mp_client(
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncMPClient":
        parallel_config = vllm_config.parallel_config
        client_args = (
            vllm_config,
            executor_class,
            log_stats,
            client_addresses,
            client_count,
            client_index,
        )
        # SUBTRACTED: DP>1 split into DPAsyncMPClient (external LB, one client
        #   per DP rank) / DPLBAsyncMPClient (internal LB)
        #   (vllm/v1/engine/core_client.py:L133-L138) — dossier delete item 2;
        #   DP topology is ch34. Single engine lands on AsyncMPClient.
        return AsyncMPClient(*client_args)

    # SOURCE: vllm/v1/engine/core_client.py:L141-L142 EngineCoreClient.shutdown
    @abstractmethod
    # SOURCE: vllm/v1/engine/core_client.py:L141 EngineCoreClient.shutdown
    def shutdown(self, timeout: float | None = None) -> None: ...

    # SOURCE: vllm/v1/engine/core_client.py:L144-L145 EngineCoreClient.get_output
    def get_output(self) -> EngineCoreOutputs:
        raise NotImplementedError

    # SOURCE: vllm/v1/engine/core_client.py:L147-L148 EngineCoreClient.get_supported_tasks
    def get_supported_tasks(self) -> tuple[str, ...]:
        raise NotImplementedError

    # SOURCE: vllm/v1/engine/core_client.py:L150-L151 EngineCoreClient.add_request
    def add_request(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    # SOURCE: vllm/v1/engine/core_client.py:L194-L195 EngineCoreClient.abort_requests
    def abort_requests(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    # SOURCE: vllm/v1/engine/core_client.py:L234-L235 EngineCoreClient.get_output_async
    async def get_output_async(self) -> EngineCoreOutputs:
        raise NotImplementedError

    # SOURCE: vllm/v1/engine/core_client.py:L237-L238 EngineCoreClient.get_supported_tasks_async
    async def get_supported_tasks_async(self) -> tuple[str, ...]:
        raise NotImplementedError

    # SOURCE: vllm/v1/engine/core_client.py:L240-L241 EngineCoreClient.add_request_async
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        raise NotImplementedError

    # SOURCE: vllm/v1/engine/core_client.py:L268-L269 EngineCoreClient.abort_requests_async
    async def abort_requests_async(self, request_ids: list[str]) -> None:
        raise NotImplementedError

    # SUBTRACTED: profile / cache-reset / sleep / lora / save-sharded-state /
    #   collective_rpc / dp-engines / elastic-EP / fault-tolerance stub surface
    #   (vllm/v1/engine/core_client.py:L153-L303) — items 5/8/9.


# SOURCE: vllm/v1/engine/core_client.py:L306 InprocClient
class InprocClient(EngineCoreClient):
    """
    InprocClient: client for in-process EngineCore. Intended
    for use in LLMEngine for V0-style add_request() and step()
        EngineCore setup in this process (no busy loop).

        * pushes EngineCoreRequest directly into the EngineCore
        * pulls EngineCoreOutputs by stepping the EngineCore
    """

    # SOURCE: vllm/v1/engine/core_client.py:L316-L317 InprocClient.__init__
    def __init__(self, *args, **kwargs):
        self.engine_core = EngineCore(*args, **kwargs)

    # SOURCE: vllm/v1/engine/core_client.py:L319-L322 InprocClient.get_output
    def get_output(self) -> EngineCoreOutputs:
        # SUBTRACTED: engine step_fn()/post_step() execution (L320-L321) — the
        #   busy loop is ch9; the companion's caller plays the engine via
        #   EngineCore.emit_step_outputs, which routes into socket 0.
        outputs = self.engine_core.sockets[0].get()
        if isinstance(outputs, Exception):
            raise outputs
        # SUBTRACTED: EngineCoreOutputsCollector .get(0) unwrap (L322) — the
        #   stub routes EngineCoreOutputs directly.
        return outputs

    # SOURCE: vllm/v1/engine/core_client.py:L324-L325 InprocClient.get_supported_tasks
    def get_supported_tasks(self) -> tuple[str, ...]:
        return self.engine_core.get_supported_tasks()

    # SOURCE: vllm/v1/engine/core_client.py:L327-L329 InprocClient.add_request
    def add_request(self, request: EngineCoreRequest) -> None:
        req, request_wave = self.engine_core.preprocess_add_request(request)
        self.engine_core.add_request(req, request_wave)

    # SOURCE: vllm/v1/engine/core_client.py:L331-L333 InprocClient.abort_requests
    def abort_requests(self, request_ids: list[str]) -> None:
        if len(request_ids) > 0:
            self.engine_core.abort_requests(request_ids)

    # SOURCE: vllm/v1/engine/core_client.py:L335-L336 InprocClient.shutdown
    def shutdown(self, timeout: float | None = None) -> None:
        self.engine_core.shutdown()

    # SUBTRACTED: profile / cache-reset / sleep / lora / save-sharded-state /
    #   collective_rpc bodies (vllm/v1/engine/core_client.py:L338-L399) —
    #   items 5/8/9.

    # SOURCE: vllm/v1/engine/core_client.py:L401-L402 InprocClient.dp_engines_running
    def dp_engines_running(self) -> bool:
        return False


# SOURCE: vllm/v1/engine/core_client.py:L503 MPClient
class MPClient(EngineCoreClient):
    """
    MPClient: base client for multi-proc EngineCore.
        EngineCore runs in a background process busy loop, getting
        new EngineCoreRequests and returning EngineCoreOutputs

        * pushes EngineCoreRequests via input_socket
        * pulls EngineCoreOutputs via output_socket

        * AsyncMPClient subclass for AsyncLLM usage
        * SyncMPClient subclass for LLM usage
    """

    # SOURCE: vllm/v1/engine/core_client.py:L516-L523 MPClient.__init__
    def __init__(
        self,
        asyncio_mode: bool,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
    ):
        self.vllm_config = vllm_config

        # SUBTRACTED: ZMQ context/socket setup + BackgroundResources finalizer +
        #   external-management address binding + engine process launch +
        #   ready handshake + monitor thread + DP engine-rank bookkeeping
        #   (vllm/v1/engine/core_client.py:L526-L680) — dossier delete items
        #   1-2: the in-process EngineCore stub replaces the background
        #   process, and queues replace the wire. client_addresses is kept in
        #   the signature (its consumers are ch34).
        self.engine_core = EngineCore(vllm_config, executor_class, log_stats)
        # SUBTRACTED: engine_dead maintained by the deleted monitor thread
        #   (L708-L735); the flag stays readable so `errored` keeps semantics.
        self.engine_dead = False

    # SOURCE: vllm/v1/engine/core_client.py:L682-L693 MPClient.shutdown
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown engine manager under timeout and clean up resources."""
        # SUBTRACTED: engine-manager + ZMQ background-resource teardown
        #   (L684-L693) — item 1.
        if engine_core := getattr(self, "engine_core", None):
            engine_core.shutdown()

    # SOURCE: vllm/v1/engine/core_client.py:L695-L699 MPClient._format_exception
    def _format_exception(self, e: Exception) -> Exception:
        """If errored, use EngineDeadError so root cause is clear."""
        # SUBTRACTED: resources.engine_dead read (L698) — flag moved up (item 1).
        return EngineDeadError(suppress_context=True) if self.engine_dead else e

    # SOURCE: vllm/v1/engine/core_client.py:L701-L703 MPClient.ensure_alive
    def ensure_alive(self):
        if self.engine_dead:
            raise EngineDeadError()

    # SOURCE: vllm/v1/engine/core_client.py:L705-L706 MPClient.dp_engines_running
    def dp_engines_running(self) -> bool:
        """Returns True if data parallel engines are collectively in a
        running state."""
        # SUBTRACTED: engines_running DP flag (item 2) — single engine: False.
        return False

    # SUBTRACTED: start_engine_core_monitor / _apply_ready_response
    #   (vllm/v1/engine/core_client.py:L708-L777) — item 1 (monitor thread +
    #   ready-response config sync).
    # SUBTRACTED: _process_utility_output (L780-L800) — item 8 (utility RPC).


# SOURCE: vllm/v1/engine/core_client.py:L802 SyncMPClient
class SyncMPClient(MPClient):
    """Synchronous client for multi-proc EngineCore."""

    # SOURCE: vllm/v1/engine/core_client.py:L805-L814 SyncMPClient.__init__ head
    def __init__(
        self, vllm_config: VllmConfig, executor_class: type[Executor], log_stats: bool
    ):
        super().__init__(
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
        )
        # SUBTRACTED: is_dp flag (L816) — DP (item 2).
        self.outputs_queue = queue.Queue[EngineCoreOutputs | Exception]()

        # SUBTRACTED: EngineCoreOutputQueueThread daemon thread + shutdown PAIR
        #   socket + socket-ownership handoff
        #   (vllm/v1/engine/core_client.py:L819-L870) — dossier delete item 1:
        #   the stub engine routes straight into outputs_queue (the in-process
        #   queue replaces the ZMQ PULL socket + decode thread).
        self.engine_core.attach_output_socket(0, self.outputs_queue)

    # SOURCE: vllm/v1/engine/core_client.py:L872-L882 SyncMPClient.get_output
    def get_output(self) -> EngineCoreOutputs:
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        outputs = self.outputs_queue.get()

        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        # SUBTRACTED: wave_complete -> engines_running=False (L880-L881) — DP
        #   (item 2).
        return outputs

    # SOURCE: vllm/v1/engine/core_client.py:L884-L891 SyncMPClient._send_input
    def _send_input(self, request_type: EngineCoreRequestType, request: Any):
        self.ensure_alive()
        # SUBTRACTED: msgpack encode + ZMQ ROUTER send (L887-L891) — item 1;
        #   the (request_type, request) message crosses via the engine's
        #   in-process input queue (mirrors core.py:L1741), to be consumed by
        #   the busy loop (here: EngineCore.emit_step_outputs drains it).
        self.engine_core.input_queue.put((request_type, request))

    # SUBTRACTED: call_utility future machinery
    #   (vllm/v1/engine/core_client.py:L893-L899) — item 8; get_supported_tasks
    #   answers directly from the (now in-process) engine.

    # SOURCE: vllm/v1/engine/core_client.py:L901-L902 SyncMPClient.get_supported_tasks
    def get_supported_tasks(self) -> tuple[str, ...]:
        return self.engine_core.get_supported_tasks()

    # SOURCE: vllm/v1/engine/core_client.py:L904-L907 SyncMPClient.add_request
    def add_request(self, request: EngineCoreRequest) -> None:
        # SUBTRACTED: is_dp -> engines_running=True (L905-L906) — DP (item 2).
        self._send_input(EngineCoreRequestType.ADD, request)

    # SOURCE: vllm/v1/engine/core_client.py:L909-L911 SyncMPClient.abort_requests
    def abort_requests(self, request_ids: list[str]) -> None:
        if request_ids and not self.engine_dead:
            # SUBTRACTED: resources.engine_dead read — flag moved up (item 1).
            self._send_input(EngineCoreRequestType.ABORT, request_ids)

    # SUBTRACTED: profile / cache-reset / sleep / lora / save-sharded-state /
    #   collective_rpc bodies (vllm/v1/engine/core_client.py:L913-L971) —
    #   items 5/8/9.


# SOURCE: vllm/v1/engine/core_client.py:L974 AsyncMPClient
class AsyncMPClient(MPClient):
    """Asyncio-compatible client for multi-proc EngineCore."""

    # SOURCE: vllm/v1/engine/core_client.py:L977-L997 AsyncMPClient.__init__ head
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ):
        super().__init__(
            asyncio_mode=True,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=log_stats,
            client_addresses=client_addresses,
        )

        self.client_count = client_count
        self.client_index = client_index
        self.outputs_queue = asyncio.Queue[EngineCoreOutputs | Exception]()

        # SUBTRACTED: fault-tolerance engine-status cache (L999-L1005) — item 8.
        # SUBTRACTED: eager output-socket task start (L1006-L1014) — the task
        #   itself is item 1.
        # Wire-up (replaces the deleted ZMQ PUSH->PULL wiring, item 1): the
        # stub engine routes into this client's queue slot client_index.
        # NOTE: companion constraint — with the socket task gone, the engine
        #   side must hand outputs over from the event-loop thread (real ZMQ
        #   has no such constraint).
        self.engine_core.attach_output_socket(self.client_index, self.outputs_queue)

    # SOURCE: vllm/v1/engine/core_client.py:L1016 AsyncMPClient._ensure_output_queue_task
    def _ensure_output_queue_task(self):
        # SUBTRACTED: process_outputs_socket asyncio task (L1017-L1091) —
        #   dossier delete item 1: the stub engine routes directly into
        #   outputs_queue; there is no socket to poll. Kept as the no-op the
        #   verbatim add_request_async line still calls.
        return

    # SOURCE: vllm/v1/engine/core_client.py:L1093-L1102 AsyncMPClient.get_output_async
    async def get_output_async(self) -> EngineCoreOutputs:
        self._ensure_output_queue_task()
        # If an exception arises in process_outputs_socket task,
        # it is forwarded to the outputs_queue so we can raise it
        # from this (run_output_handler) task to shut down the server.
        assert self.outputs_queue is not None
        outputs = await self.outputs_queue.get()
        if isinstance(outputs, Exception):
            raise self._format_exception(outputs) from None
        return outputs

    # SOURCE: vllm/v1/engine/core_client.py:L1104-L1114 AsyncMPClient._send_input
    async def _send_input(
        self,
        request_type: EngineCoreRequestType,
        request: Any,
        engine: Any = None,
    ) -> None:
        # SUBTRACTED: engine-identity routing + msgpack encode + ZMQ ROUTER
        #   send (L1110-L1123) — items 1-2; the in-process input queue crosses.
        self.ensure_alive()
        self.engine_core.input_queue.put((request_type, request))

    # SUBTRACTED: call_utility_async / _call_utility_async future machinery
    #   (vllm/v1/engine/core_client.py:L1125-L1140) — item 8.

    # SOURCE: vllm/v1/engine/core_client.py:L1142-L1143 AsyncMPClient.get_supported_tasks_async
    async def get_supported_tasks_async(self) -> tuple[str, ...]:
        return self.engine_core.get_supported_tasks()

    # SOURCE: vllm/v1/engine/core_client.py:L1145-L1148 AsyncMPClient.add_request_async
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        request.client_index = self.client_index
        await self._send_input(EngineCoreRequestType.ADD, request)
        self._ensure_output_queue_task()

    # SOURCE: vllm/v1/engine/core_client.py:L1150-L1152 AsyncMPClient.abort_requests_async
    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if request_ids and not self.engine_dead:
            # SUBTRACTED: resources.engine_dead read — flag moved up (item 1).
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)

    # SUBTRACTED: pause/resume scheduler, profiling, cache resets, sleep/wake,
    #   lora, save-sharded-state, collective_rpc, fault tolerance
    #   (vllm/v1/engine/core_client.py:L1154-L1246) — items 5/8/9.

    # SUBTRACTED: DPAsyncMPClient / DPLBAsyncMPClient
    #   (vllm/v1/engine/core_client.py:L1249-L1872) — dossier delete item 2:
    #   DP load balancing, waves, elastic EP — ch34.


# ============================================================================
# Trio member #1 — vllm/v1/engine/input_processor.py (EngineCoreRequest's
# only birthplace; validation/tokenization internals are ch6).
# ============================================================================


# SOURCE: vllm/v1/engine/input_processor.py:L38 InputProcessor
class InputProcessor:
    # SOURCE: vllm/v1/engine/input_processor.py:L39 InputProcessor.__init__
    def __init__(
        self,
        vllm_config: VllmConfig,
        renderer: BaseRenderer | None = None,
        *,
        mm_registry: Any = None,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        # SUBTRACTED: sub-config unpacking + multimodal budget +
        #   InputPreprocessor assembly (L48-L75) — ch6.
        self.renderer = renderer or renderer_from_config(vllm_config)

        # SUBTRACTED: process_inputs_async thread-pool wrapper (L77-L82) —
        #   dossier delete item 6 (raw-prompt path).

    # SOURCE: vllm/v1/engine/input_processor.py:L84 InputProcessor.tokenizer
    @property
    # SOURCE: vllm/v1/engine/input_processor.py:L84-L86
    def tokenizer(self):
        return self.renderer.tokenizer

    # SUBTRACTED: _validate_params / _validate_lora / _get_mm_identifier /
    #   inject_into_mm_cache (vllm/v1/engine/input_processor.py:L91-L229) —
    #   validation internals are ch6; no raw-prompt/multimodal path remains.

    # SOURCE: vllm/v1/engine/input_processor.py:L231 InputProcessor.assign_request_id
    @staticmethod
    # SOURCE: vllm/v1/engine/input_processor.py:L232 assign_request_id
    def assign_request_id(request: EngineCoreRequest):
        """Replace the externally supplied request ID with an internal request ID
        that adds 8 random characters in order to ensure uniqueness.
        """
        if request.external_req_id is not None:
            raise ValueError(
                "The external_req_id field should not be set on EngineCoreRequests"
                " passed to vLLM; use the request_id field."
            )
        request.external_req_id = request.request_id
        if envs.VLLM_DISABLE_REQUEST_ID_RANDOMIZATION:
            logger.warning_once(
                "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION is set and will be "
                "removed in a future release. Duplicate externally-provided "
                "request IDs may cause failures and/or subtle correctness errors."
            )
        else:
            request.request_id = f"{request.external_req_id}-{random_uuid():.8}"

    # SOURCE: vllm/v1/engine/input_processor.py:L251 InputProcessor.process_inputs
    def process_inputs(
        self,
        request_id: str,
        prompt: EngineInput,
        params: SamplingParams,
        supported_tasks: tuple[str, ...],
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        trace_headers: dict[str, str] | None = None,
        resumable: bool = False,
    ) -> EngineCoreRequest:
        # SUBTRACTED: tokenization_kwargs / priority / data_parallel_rank
        #   parameters (items 6/7) and the resumable streaming caller.
        # SUBTRACTED: _validate_params / _validate_lora call pair (L265-L266)
        #   — validation internals are ch6.
        # SUBTRACTED: data_parallel_rank range check (L268-L276) — item 7.

        if isinstance(prompt, dict) and "type" in prompt:
            # SUBTRACTED: tokenization_kwargs deprecation warning (L279-L284)
            #   — item 6.

            if arrival_time is None:
                arrival_time = prompt.get("arrival_time", time.time())

            processed_inputs: EngineInput = prompt
        else:
            # SUBTRACTED: raw-prompt preprocessing else-arm (L290-L303) —
            #   dossier delete item 6: the Renderer (ch6) renders prompts; the
            #   companion only accepts already-rendered EngineInput dicts (the
            #   sync fast path the OpenAI face always takes).
            raise TypeError(
                "companion accepts rendered EngineInput dicts only "
                "(raw-prompt preprocessing is ch6)"
            )

        # SUBTRACTED: platform validate_request + encoder/decoder split +
        #   prompt-length/vocab validation (L305-L308, L396-L505) — ch6.
        # SUBTRACTED: prompt_embeds / prompt_is_token_ids extraction (L310-L318)
        #   — embeds path is ch6.
        prompt_token_ids = processed_inputs["prompt_token_ids"]

        sampling_params = None
        if isinstance(params, SamplingParams):
            # TODO: can we avoid cloning here in multiproc case?
            sampling_params = params.clone()
            # SUBTRACTED: max_tokens defaulting from max_model_len +
            #   generation-config/tokenizer updates (L326-L337) — ch6.
        # SUBTRACTED: pooling_params clone else-arm (L338-L339) — item 7.
        # SUBTRACTED: multimodal feature assembly (L341-L377) — ch6.

        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            # SUBTRACTED: mm_features / prompt_embeds / prompt_is_token_ids /
            #   cache_salt / priority / data_parallel_rank kwargs — ch6/items 2/7.
            sampling_params=sampling_params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            trace_headers=trace_headers,
            resumable=resumable,
        )


# ============================================================================
# Online usage face — vllm/v1/engine/async_llm.py.
# ============================================================================


# SOURCE: vllm/v1/engine/async_llm.py:L72 AsyncLLM
class AsyncLLM(EngineClient):
    """An asynchronous wrapper for the vLLM engine."""

    # SOURCE: vllm/v1/engine/async_llm.py:L75 AsyncLLM.__init__
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        mm_registry: Any = None,
        log_requests: bool = True,
        start_engine_loop: bool = True,
        stat_loggers: list | None = None,
        aggregate_engine_logging: bool = False,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> None:
        # SUBTRACTED: the Args docstring's stat-logger caveats (L90-L108) and
        #   maybe_register_config_serialize_by_value (L110).
        self.vllm_config = vllm_config
        # SUBTRACTED: _elastic_ep_lock (L113) — item 8.
        self.model_config = vllm_config.model_config
        self.observability_config = vllm_config.observability_config

        tracing_endpoint = self.observability_config.otlp_traces_endpoint
        if tracing_endpoint is not None:
            # SUBTRACTED: init_tracer (L119) — item 9.
            pass

        self.log_requests = log_requests

        # SUBTRACTED: custom stat-logger plugin discovery (L123-L133) — item 5.
        self.log_stats = log_stats

        self.renderer = renderer = renderer_from_config(self.vllm_config)

        # Convert EngineInput --> EngineCoreRequest.
        self.input_processor = InputProcessor(self.vllm_config, renderer)

        # Converts EngineCoreOutputs --> RequestOutput.
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
            tracing_enabled=tracing_endpoint is not None,
        )

        # EngineCore (starts the engine in background process).
        self.engine_core = EngineCoreClient.make_async_mp_client(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

        # SUBTRACTED: StatLoggerManager block (L158-L169) — item 5.

        self._client_count = client_count

        self.output_handler: asyncio.Task | None = None
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass

        # SUBTRACTED: torch profiler setup (L181-L203) — item 9.

    # SOURCE: vllm/v1/engine/async_llm.py:L205 AsyncLLM.from_vllm_config
    @classmethod
    # SOURCE: vllm/v1/engine/async_llm.py:L206 from_vllm_config
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list | None = None,
        enable_log_requests: bool = False,
        aggregate_engine_logging: bool = False,
        disable_log_stats: bool = False,
        client_addresses: dict[str, Any] | None = None,
        client_count: int = 1,
        client_index: int = 0,
    ) -> "AsyncLLM":
        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            start_engine_loop=start_engine_loop,
            stat_loggers=stat_loggers,
            log_requests=enable_log_requests,
            log_stats=not disable_log_stats,
            aggregate_engine_logging=aggregate_engine_logging,
            usage_context=usage_context,
            client_addresses=client_addresses,
            client_count=client_count,
            client_index=client_index,
        )

    # SOURCE: vllm/v1/engine/async_llm.py:L234 AsyncLLM.from_engine_args
    @classmethod
    # SOURCE: vllm/v1/engine/async_llm.py:L235 from_engine_args
    def from_engine_args(
        cls,
        engine_args,
        start_engine_loop: bool = True,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list | None = None,
    ) -> "AsyncLLM":
        """Create an AsyncLLM from the EngineArgs."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        # Create the AsyncLLM.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_requests=engine_args.enable_log_requests,
            log_stats=not engine_args.disable_log_stats,
            start_engine_loop=start_engine_loop,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
        )

    # SOURCE: vllm/v1/engine/async_llm.py:L259 AsyncLLM.__del__
    def __del__(self):
        self.shutdown()

    # SOURCE: vllm/v1/engine/async_llm.py:L262 AsyncLLM.shutdown
    def shutdown(self, timeout: float | None = None) -> None:
        """Shutdown, cleaning up the background proc and IPC."""
        # SUBTRACTED: shutdown_prometheus + renderer.shutdown (L264-L267) —
        #   items 5/6.
        if engine_core := getattr(self, "engine_core", None):
            engine_core.shutdown(timeout=timeout)

        handler = getattr(self, "output_handler", None)
        if handler is not None:
            # SUBTRACTED: cancel_task_threadsafe cross-thread cancel util —
            #   plain cancel suffices on the host seam.
            handler.cancel()

    # SOURCE: vllm/v1/engine/async_llm.py:L276 AsyncLLM.get_supported_tasks
    async def get_supported_tasks(self) -> tuple[str, ...]:
        if not hasattr(self, "_supported_tasks"):
            # Cache the result
            self._supported_tasks = await self.engine_core.get_supported_tasks_async()

        return self._supported_tasks

    # SOURCE: vllm/v1/engine/async_llm.py:L283 AsyncLLM.add_request
    async def add_request(
        self,
        request_id: str,
        prompt: EngineInput,
        params: SamplingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        trace_headers: dict[str, str] | None = None,
        prompt_text: str | None = None,
    ) -> RequestOutputCollector:
        """Add new request to the AsyncLLM."""
        # SUBTRACTED: AsyncGenerator[StreamingInput] and EngineCoreRequest
        #   union arms + tokenization_kwargs / priority / data_parallel_rank /
        #   reasoning_ended / reasoning_parser_kwargs parameters
        #   (L285-L299) — dossier delete items 3/6/7.

        if self.errored:
            raise EngineDeadError()

        # SUBTRACTED: pooling branch + kv_sharing_fast_prefill check
        #   (L306-L317) — item 7.
        # SUBTRACTED: streaming-input AsyncGenerator branch (L319-L334) — item 3.
        # SUBTRACTED: EngineCoreRequest direct-pass deprecated branch
        #   (L336-L350) — item 6.

        # Convert Input --> Request.
        if isinstance(prompt, dict) and "type" in prompt:
            # Rendered EngineInput; no blocking preprocessing needed.
            request = self.input_processor.process_inputs(
                request_id,
                prompt,
                params,
                supported_tasks=await self.get_supported_tasks(),
                arrival_time=arrival_time,
                lora_request=lora_request,
                trace_headers=trace_headers,
            )
        # SUBTRACTED: raw-prompt thread-pool else-arm (L366-L380) — item 6;
        #   only the sync fast path (the one the OpenAI face always takes)
        #   remains.
        # SUBTRACTED: extract_prompt_components text extraction (L381) — the
        #   Renderer-side helper is ch6; prompt_text stays the caller's value.
        # SUBTRACTED: reasoning_ended / reasoning_parser_kwargs passthrough
        #   (L383-L386) — item 7.

        self.input_processor.assign_request_id(request)

        # We start the output_handler on the first call to add_request() so
        # we can call __init__ before the event loop, which enables us
        # to handle startup failure gracefully in the OpenAI server.
        self._run_output_handler()

        # Create a new output collector for the request.
        queue = RequestOutputCollector(params.output_kind, request.request_id)

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        # SUBTRACTED: is_pooling arm (L401) — item 7.
        if params.n == 1:
            await self._add_request(request, prompt_text, None, 0, queue)
            return queue

        # SUBTRACTED: n>1 fan-out via ParentRequest (L405-L418) — item 4;
        #   each child would walk the same double registration (ch7).

    # SOURCE: vllm/v1/engine/async_llm.py:L420 AsyncLLM._add_request
    async def _add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req,  # SUBTRACTED: ParentRequest annotation (item 4)
        index: int,
        queue: RequestOutputCollector,
    ):
        # The parent_req slot is kept (no default, like the real signature) so
        # add_request's call stays positional-identical.
        # Add the request to the OutputProcessor (this process).
        self.output_processor.add_request(request, prompt, parent_req, index, queue)

        # Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)

        if self.log_requests:
            logger.info("Added request %s.", request.request_id)

    # SUBTRACTED: _add_streaming_input_request + handle_inputs +
    #   _validate_streaming_input_sampling_params
    #   (vllm/v1/engine/async_llm.py:L437-L537) — streaming inputs (item 3).

    # SOURCE: vllm/v1/engine/async_llm.py:L544 AsyncLLM.generate
    async def generate(
        self,
        prompt: EngineInput,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: LoRARequest | None = None,
        trace_headers: dict[str, str] | None = None,
    ) -> AsyncGenerator[RequestOutput, None]:
        # SUBTRACTED: union arms + parameters per add_request (items 3/6/7).
        """
        Main function called by the API server to kick off a request
            * 1) Making an AsyncStream corresponding to the Request.
            * 2) Processing the Input.
            * 3) Adding the Request to the Detokenizer.
            * 4) Adding the Request to the EngineCore (separate process).

        A separate output_handler loop runs in a background AsyncIO task,
        pulling outputs from EngineCore and putting them into the
        per-request AsyncStream.

        The caller of generate() iterates the returned AsyncGenerator,
        returning the RequestOutput back to the caller.
        """

        q: RequestOutputCollector | None = None
        try:
            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                trace_headers=trace_headers,
                prompt_text=prompt_text,
            )

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
                # SUBTRACTED: STREAM_FINISHED sentinel check (L605) — streaming
                #   inputs (item 3).
                yield out

        # If the request is disconnected by the client, generate()
        # is cancelled or the generator is garbage collected. So,
        # we abort the request if we end up here.
        except (asyncio.CancelledError, GeneratorExit):
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                logger.info("Request %s aborted.", request_id)
            raise

        # Engine is dead. Do not abort since we shut down.
        except EngineDeadError:
            if self.log_requests:
                logger.info("Request %s failed (engine dead).", request_id)
            raise

        # SUBTRACTED: VLLMClientError arm (L625-L628) — request-validation
        #   taxonomy is ch6; and the InputStreamError arm (L631-L636, item 3).

        # Unexpected error in the generate() task (possibly recoverable).
        except Exception as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            if self.log_requests:
                # SUBTRACTED: exception-print failure guard (L643-L651).
                logger.info("Request %s failed.", request_id)
            raise EngineGenerateError() from e
        finally:
            if q is not None:
                q.close()

    # SOURCE: vllm/v1/engine/async_llm.py:L657 AsyncLLM._run_output_handler
    def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        if self.output_handler is not None:
            return

        # Ensure that the task doesn't have a circular ref back to the AsyncLLM
        # object, or else it won't be garbage collected and cleaned up properly.
        engine_core = self.engine_core
        output_processor = self.output_processor
        # SUBTRACTED: log_stats / _logger_ref / renderer captures (L667-L673) —
        #   the logging block they feed is item 5.
        chunk_size = envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE

        # SOURCE: vllm/v1/engine/async_llm.py:L676 output_handler closure
        async def output_handler():
            try:
                while True:
                    # 1) Pull EngineCoreOutputs from the EngineCore.
                    outputs = await engine_core.get_output_async()
                    num_outputs = len(outputs.outputs)

                    # SUBTRACTED: IterationStats creation (L683-L685) — item 5.

                    # Split outputs into chunks of at most
                    # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                    # event loop for too long.
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

                    # SUBTRACTED: update_scheduler_stats + 4) logging
                    #   (L711-L722) — item 5.
            except Exception as e:
                logger.exception("AsyncLLM output_handler failed.")
                output_processor.propagate_error(e)

        self.output_handler = asyncio.create_task(output_handler())

    # SOURCE: vllm/v1/engine/async_llm.py:L729 AsyncLLM.abort
    async def abort(
        self, request_id: str | Iterable[str], internal: bool = False
    ) -> None:
        """Abort RequestId in OutputProcessor and EngineCore."""

        request_ids = (
            (request_id,) if isinstance(request_id, str) else as_list(request_id)
        )
        all_request_ids = self.output_processor.abort_requests(request_ids, internal)
        await self.engine_core.abort_requests_async(all_request_ids)

        if self.log_requests:
            logger.info("Aborted request(s) %s.", ",".join(request_ids))

    # SUBTRACTED: notify_kv_transfer_request_rejected (L743-L768) and the
    #   RL/ops control surface: pause/resume/is_paused, encode (pooling —
    #   item 7), profiling, cache resets, sleep/wake, lora, collective_rpc,
    #   checkpoints, drain/elastic-EP, handle_fault/get_status, weight
    #   transfer (vllm/v1/engine/async_llm.py:L743-L1083) — items 7/8/9.

    # SOURCE: vllm/v1/engine/async_llm.py:L906 AsyncLLM.tokenizer
    @property
    # SOURCE: vllm/v1/engine/async_llm.py:L906-L908
    def tokenizer(self):
        return self.renderer.tokenizer

    # SUBTRACTED: get_tokenizer (L910-L911) — renderer seam (ch6).

    # SOURCE: vllm/v1/engine/async_llm.py:L1085 AsyncLLM.is_running
    @property
    # SOURCE: vllm/v1/engine/async_llm.py:L1086 is_running
    def is_running(self) -> bool:
        # Is None before the loop is started.
        return self.output_handler is None or not self.output_handler.done()

    # SOURCE: vllm/v1/engine/async_llm.py:L1090 AsyncLLM.is_stopped
    @property
    # SOURCE: vllm/v1/engine/async_llm.py:L1091 is_stopped
    def is_stopped(self) -> bool:
        return self.errored

    # SOURCE: vllm/v1/engine/async_llm.py:L1094 AsyncLLM.errored
    @property
    # SOURCE: vllm/v1/engine/async_llm.py:L1095 errored
    def errored(self) -> bool:
        # SUBTRACTED: resources.engine_dead read — flag moved up (item 1).
        return self.engine_core.engine_dead or not self.is_running

    # SOURCE: vllm/v1/engine/async_llm.py:L1098 AsyncLLM.dead_error
    @property
    # SOURCE: vllm/v1/engine/async_llm.py:L1099 dead_error
    def dead_error(self) -> BaseException:
        return EngineDeadError()


# SOURCE: vllm/engine/async_llm_engine.py:L6 AsyncLLMEngine
AsyncLLMEngine = AsyncLLM  # type: ignore
"""The `AsyncLLMEngine` class is an alias of [vllm.v1.engine.async_llm.AsyncLLM][]."""
# (v0 relic, WC1 old_design in the flesh: the 1032-line AsyncLLMEngine adapter
# of the two-engine era is today a 7-line alias shim — vllm/engine/
# async_llm_engine.py:L1-L7 verbatim.)


# ============================================================================
# Offline usage face — vllm/v1/engine/llm_engine.py.
# ============================================================================


# SOURCE: vllm/v1/engine/llm_engine.py:L48 LLMEngine
class LLMEngine:
    """Legacy LLMEngine for backwards compatibility."""

    # SOURCE: vllm/v1/engine/llm_engine.py:L51 LLMEngine.__init__
    def __init__(
        self,
        vllm_config: VllmConfig,
        executor_class: type[Executor],
        log_stats: bool,
        aggregate_engine_logging: bool = False,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list | None = None,
        mm_registry: Any = None,
        multiprocess_mode: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.observability_config = vllm_config.observability_config

        tracing_endpoint = self.observability_config.otlp_traces_endpoint
        if tracing_endpoint is not None:
            # SUBTRACTED: init_tracer (L68) — item 9.
            pass

        self.log_stats = log_stats

        # SUBTRACTED: external-launcher DP group setup + dp_group init
        #   (L72-L88) — DP (item 2); single engine below.
        # SUBTRACTED: should_execute_dummy_batch flag (L89) — DP dummy batches
        #   (item 2); step()'s consumer branch goes with it.

        self.renderer = renderer = renderer_from_config(self.vllm_config)

        # Convert EngineInput --> EngineCoreRequest.
        self.input_processor = InputProcessor(self.vllm_config, renderer)

        # Converts EngineCoreOutputs --> RequestOutput.
        self.output_processor = OutputProcessor(
            renderer.tokenizer,
            log_stats=self.log_stats,
            stream_interval=self.vllm_config.scheduler_config.stream_interval,
            tracing_enabled=tracing_endpoint is not None,
        )

        # EngineCore (gets EngineCoreRequests and gives EngineCoreOutputs)
        self.engine_core = EngineCoreClient.make_client(
            multiprocess_mode=multiprocess_mode,
            asyncio_mode=False,
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=self.log_stats,
        )

        # SUBTRACTED: StatLoggerManager block (L113-L121) — item 5.
        # SUBTRACTED: v0-compat model_executor exposure + finalizer
        #   (L123-L133) — needs a real in-process executor (ch9).
        # SUBTRACTED: external_launcher_dp cpu_group (L135-L138) — item 2.
        # SUBTRACTED: reset_mm_cache (L140-L141) — multimodal (ch6).

    # SOURCE: vllm/v1/engine/llm_engine.py:L143 LLMEngine.from_vllm_config
    @classmethod
    # SOURCE: vllm/v1/engine/llm_engine.py:L144 from_vllm_config
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list | None = None,
        disable_log_stats: bool = False,
    ) -> "LLMEngine":
        return cls(
            vllm_config=vllm_config,
            executor_class=Executor.get_class(vllm_config),
            log_stats=(not disable_log_stats),
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=envs.VLLM_ENABLE_V1_MULTIPROCESSING,
        )

    # SOURCE: vllm/v1/engine/llm_engine.py:L160 LLMEngine.from_engine_args
    @classmethod
    # SOURCE: vllm/v1/engine/llm_engine.py:L161 from_engine_args
    def from_engine_args(
        cls,
        engine_args,
        usage_context: UsageContext = UsageContext.ENGINE_CONTEXT,
        stat_loggers: list | None = None,
        enable_multiprocessing: bool = False,
    ) -> "LLMEngine":
        """Creates an LLM engine from the engine arguments."""

        # Create the engine configs.
        vllm_config = engine_args.create_engine_config(usage_context)
        executor_class = Executor.get_class(vllm_config)

        if envs.VLLM_ENABLE_V1_MULTIPROCESSING:
            logger.debug("Enabling multiprocessing for LLMEngine.")
            enable_multiprocessing = True

        # Create the LLMEngine.
        return cls(
            vllm_config=vllm_config,
            executor_class=executor_class,
            log_stats=not engine_args.disable_log_stats,
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            multiprocess_mode=enable_multiprocessing,
        )

    # SOURCE: vllm/v1/engine/llm_engine.py:L188 LLMEngine.get_num_unfinished_requests
    def get_num_unfinished_requests(self) -> int:
        return self.output_processor.get_num_unfinished_requests()

    # SOURCE: vllm/v1/engine/llm_engine.py:L191 LLMEngine.has_unfinished_requests
    def has_unfinished_requests(self) -> bool:
        has_unfinished = self.output_processor.has_unfinished_requests()
        # SUBTRACTED: dp_group aggregation branch (L193-L195) — item 2; the
        #   non-DP arm reads the (single) engine's running flag like real.
        return has_unfinished or self.engine_core.dp_engines_running()

    # SUBTRACTED: has_unfinished_requests_dp (L197-L203) — DP (item 2).

    # SOURCE: vllm/v1/engine/llm_engine.py:L205 LLMEngine.get_supported_tasks
    def get_supported_tasks(self) -> tuple[str, ...]:
        if not hasattr(self, "_supported_tasks"):
            # Cache the result
            self._supported_tasks = self.engine_core.get_supported_tasks()

        return self._supported_tasks

    # SOURCE: vllm/v1/engine/llm_engine.py:L212 LLMEngine.abort_request
    def abort_request(self, request_ids: list[str], internal: bool = False) -> None:
        """Remove request_ids from EngineCore and Detokenizer."""

        request_ids = self.output_processor.abort_requests(request_ids, internal)
        self.engine_core.abort_requests(request_ids)

    # SOURCE: vllm/v1/engine/llm_engine.py:L218 LLMEngine.add_request
    def add_request(
        self,
        request_id: str,
        prompt: EngineInput,
        params: SamplingParams,
        arrival_time: float | None = None,
        lora_request: LoRARequest | None = None,
        trace_headers: dict[str, str] | None = None,
        prompt_text: str | None = None,
    ) -> str:
        # SUBTRACTED: EngineCoreRequest union arm + tokenization_kwargs /
        #   priority parameters (L221-L227) — items 6/7.
        # Validate the request_id type.
        if not isinstance(request_id, str):
            raise TypeError(f"request_id must be a string, got {type(request_id)}")

        # Process raw inputs into the request.
        # SUBTRACTED: EngineCoreRequest direct-pass deprecated branch
        #   (L235-L248) — item 6.
        request = self.input_processor.process_inputs(
            request_id,
            prompt,
            params,
            supported_tasks=self.get_supported_tasks(),
            arrival_time=arrival_time,
            lora_request=lora_request,
            trace_headers=trace_headers,
        )
        # SUBTRACTED: extract_prompt_components text extraction (L261) — ch6.

        self.input_processor.assign_request_id(request)

        req_id = request.request_id

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        n = params.n if isinstance(params, SamplingParams) else 1

        if n == 1:
            # Make a new RequestState and queue.
            self.output_processor.add_request(request, prompt_text, None, 0)
            # Add the request to EngineCore.
            self.engine_core.add_request(request)
            return req_id

        # SUBTRACTED: n>1 fan-out via ParentRequest (L279-L293) — item 4.

    # SOURCE: vllm/v1/engine/llm_engine.py:L296 LLMEngine.step
    def step(self) -> list[RequestOutput]:
        # SUBTRACTED: should_execute_dummy_batch branch (L297-L301) — DP
        #   (item 2); non-DP is always False.

        # 1) Get EngineCoreOutput from the EngineCore.
        # SUBTRACTED: record_function_or_nullcontext profiler wrap (L303) —
        #   observability (item 5).
        outputs = self.engine_core.get_output()

        # 2) Process EngineCoreOutputs.
        # SUBTRACTED: profiler wrap (L307).
        # SUBTRACTED: IterationStats creation (L308) — item 5; callers pass None.
        processed_outputs = self.output_processor.process_outputs(
            outputs.outputs,
            engine_core_timestamp=outputs.timestamp,
            iteration_stats=None,
        )
        # SUBTRACTED: update_scheduler_stats (L314) — item 5.

        # 3) Abort any reqs that finished due to stop strings.
        # SUBTRACTED: profiler wrap (L317).
        self.engine_core.abort_requests(processed_outputs.reqs_to_abort)

        # 4) Record stats — SUBTRACTED (L320-L332, item 5).

        return processed_outputs.request_outputs

    # SUBTRACTED: profiling / cache resets / sleep & wake / metrics /
    #   do_log_stats* / lora / collective_rpc / weight version / apply_model /
    #   driver-model cleanup / __del__ DP teardown
    #   (vllm/v1/engine/llm_engine.py:L336-L455) — items 5/8/9/2.

    # SOURCE: vllm/v1/engine/llm_engine.py:L382 LLMEngine.tokenizer
    @property
    # SOURCE: vllm/v1/engine/llm_engine.py:L383 tokenizer
    def tokenizer(self):
        return self.renderer.tokenizer

    # SUBTRACTED: get_tokenizer (L386-L387) — renderer seam (ch6).


# ============================================================================
# Offline face skin — vllm/entrypoints/offline_utils.py (OfflineInferenceMixin).
# ============================================================================


# SOURCE: vllm/entrypoints/offline_utils.py:L49 OfflineInferenceMixin
class OfflineInferenceMixin:
    """Offline inference utils"""

    request_counter: Counter
    renderer: BaseRenderer
    llm_engine: LLMEngine
    model_config: ModelConfig

    # SUBTRACTED: _resolve_mm_lora (L57-...) — multimodal default LoRA (ch6).
    # SUBTRACTED: _params_to_seq / _lora_request_to_seq / _priority_to_seq
    #   sequence normalization (vllm/entrypoints/offline_utils.py:L230-L288) —
    #   the companion takes already-sequenced params; priority passthrough is
    #   dossier delete item 7.

    # SUBTRACTED: _add_completion_requests render pipeline
    #   (vllm/entrypoints/offline_utils.py:L290-L324) — _preprocess_cmpl_one
    #   walks the Renderer (ch6); the companion's prompts arrive pre-rendered.

    # SOURCE: vllm/entrypoints/offline_utils.py:L326 OfflineInferenceMixin._run_completion
    def _run_completion(
        self,
        prompts,
        params,
        output_type: type,
        *,
        use_tqdm: bool = True,
        lora_request=None,
        tokenization_kwargs=None,
        mm_processor_kwargs=None,
    ):
        # SUBTRACTED: priority parameter (item 7) and the render-preprocess
        #   generator plumbing (L340-L348) — ch6.
        self._render_and_add_requests(
            prompts=prompts,
            params=params,
            lora_requests=lora_request,
        )
        return self._run_engine(use_tqdm=use_tqdm, output_type=output_type)

    # SUBTRACTED: _run_chat / chat variants (L351-...) — Renderer chat path (ch6).

    # SOURCE: vllm/entrypoints/offline_utils.py:L523 OfflineInferenceMixin._render_and_add_requests
    def _render_and_add_requests(
        self,
        prompts,
        params,
        *,
        lora_requests=None,
    ) -> list[str]:
        # SUBTRACTED: priorities parameter (item 7).
        added_request_ids: list[str] = []

        try:
            for i, prompt in enumerate(prompts):
                request_id = self._add_request(
                    prompt,
                    params[i],
                    # SUBTRACTED: _resolve_mm_lora lookup (L538-L541) — ch6.
                    lora_request=None if lora_requests is None else lora_requests[i],
                )
                added_request_ids.append(request_id)
        except Exception as e:
            if added_request_ids:
                self.llm_engine.abort_request(added_request_ids, internal=True)
            raise e

        return added_request_ids

    # SOURCE: vllm/entrypoints/offline_utils.py:L552 OfflineInferenceMixin._add_request
    def _add_request(
        self,
        prompt: EngineInput,
        params: SamplingParams,
        lora_request: LoRARequest | None = None,
    ) -> str:
        # SUBTRACTED: priority parameter (item 7).
        if isinstance(params, SamplingParams):
            # We only care about the final output
            params.output_kind = RequestOutputKind.FINAL_ONLY

        request_id = str(next(self.request_counter))

        return self.llm_engine.add_request(
            request_id,
            prompt,
            params,
            lora_request=lora_request,
        )

    # SOURCE: vllm/entrypoints/offline_utils.py:L573 OfflineInferenceMixin._run_engine
    def _run_engine(
        self,
        output_type: type,
        *,
        use_tqdm: bool = True,
    ) -> list:
        # SUBTRACTED: tqdm progress/throughput display
        #   (vllm/entrypoints/offline_utils.py:L579-L588, L600-L619, L621-L622)
        #   — dossier delete item 5; the use_tqdm kwarg is kept for signature
        #   fidelity.

        # Run the engine.
        outputs: list = []
        # SUBTRACTED: token throughput accumulators (L592-L593) — tqdm display.
        while self.llm_engine.has_unfinished_requests():
            step_outputs = self.llm_engine.step()
            for output in step_outputs:
                assert isinstance(output, output_type)
                if output.finished:
                    outputs.append(output)
                    # SUBTRACTED: tqdm update (L600-L619) — item 5.

        # Sort the outputs by request ID.
        # This is necessary because some requests may be finished earlier than
        # its previous requests.
        return sorted(outputs, key=lambda x: int(x.request_id))


# ============================================================================
# Offline user entry — vllm/entrypoints/llm.py.
# ============================================================================


# SOURCE: vllm/entrypoints/llm.py:L67 LLM
class LLM(OfflineInferenceMixin):
    """An LLM for generating texts from given prompts and sampling parameters.

    This class includes a tokenizer, a language model (possibly distributed
    across multiple GPUs), and GPU memory space allocated for intermediate
    states (aka KV cache). Given a set of prompts and sampling parameters,
    this class generates texts from the model, using an intelligent batching
    mechanism and efficient memory management."""

    # SUBTRACTED: BeamSearchOfflineMixin / PoolingOfflineMixin bases
    #   (vllm/entrypoints/llm.py:L67) — beam search and pooling (item 7);
    #   generation-only companion.

    # SOURCE: vllm/entrypoints/llm.py:L295 LLM.__init__ (arg subset)
    def __init__(self, engine_args):
        # SUBTRACTED: the ~100-kwarg -> EngineArgs assembly + pooler/attention/
        #   structured/profiler instance pre-construction + log_non_default_args
        #   (vllm/entrypoints/llm.py:L160-L337) — the whole assembly line is
        #   the ch03 companion; this chapter starts at the engine hookup.

        self.llm_engine = LLMEngine.from_engine_args(
            engine_args=engine_args, usage_context=UsageContext.LLM_CLASS
        )
        self.model_config = self.llm_engine.model_config
        self.engine_class = type(self.llm_engine)

        self.request_counter = Counter()
        # SUBTRACTED: default_sampling_params / supported_tasks surface /
        #   renderer+chat-template warmup (L346-L356) — ch6.

    # SOURCE: vllm/entrypoints/llm.py:L414 LLM.generate
    def generate(
        self,
        prompts,
        sampling_params,
        *,
        use_tqdm: bool = True,
        lora_request=None,
    ) -> list[RequestOutput]:
        # SUBTRACTED: priority / tokenization_kwargs / mm_processor_kwargs
        #   parameters (L421-L423) — items 6/7.
        """Generates the completions for the input prompts.

        The offline face of ch04: batch-add every prompt (FINAL_ONLY +
        auto-increment ids), then drive the engine with the bare
        while-step() loop and restore the input order at the end.
        """

        runner_type = self.model_config.runner_type
        if runner_type != "generate":
            raise ValueError(
                "LLM.generate() is only supported for generative models. "
                "Try passing `--runner generate` to use the model as a "
                "generative model."
            )

        # SUBTRACTED: get_default_sampling_params fallback (L465-L466) —
        #   model-derived defaults are ch6; pass sampling_params explicitly.

        return self._run_completion(
            prompts=prompts,
            params=sampling_params,
            output_type=RequestOutput,
            use_tqdm=use_tqdm,
            lora_request=lora_request,
        )

    # SUBTRACTED: enqueue/wait_for_completion/chat/tokenize/classify/...
    #   (vllm/entrypoints/llm.py:L479+) — ch6 variants and pooling (item 7).
