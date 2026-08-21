# Subtract-only companion for v3 ch07 «上行：从 token 到文字» (Part II: API
# 进程上行泳道放大——引擎回程的一串 token id 在前端进程内被变回文字、按请求
# 拆流、流到各自的消费者手上；客户端断连时反向 abort 把请求从引擎里抠掉).
#
# FAITHFUL SUBSET of the real vLLM front-end uplink at pin v0.27.1
# (6e448d0ea). It keeps vLLM's names, structure and control flow; it only
# DELETES branches approved in the dossier subtraction_plan (every deletion
# marked `# SUBTRACTED:` with its source span) plus the documented HOST SEAMs
# (external machinery this chapter treats as a black box). Mapping rule: take
# the real vLLM source, drop every SUBTRACTED branch, and you should get
# (approximately) this file.
#
# Fully real (the chapter's product):
# - RequestOutputCollector: the per-request SINGLE-SLOT mailbox (asyncio.Event
#   + producer-side merge via RequestOutput.add, aggregate=DELTA) — pointedly
#   NOT an asyncio.Queue (WC1);
# - OutputProcessor.process_outputs: the ONE loop over the batch (NOTE FOR
#   DEVELOPERS verbatim): demux by internal id -> incremental detokenize +
#   stop strings -> make_request_output -> queue.put / list.append fork ->
#   finish cleanup + reqs_to_abort;
# - make_request_output's three gates: FINAL_ONLY zero-construction,
#   stream_interval throttle (DELTA from sent_tokens_offset), n>1 parent
#   aggregation (ParentRequest.get_outputs);
# - IncrementalDetokenizer hierarchy: null shell + three-way factory, Base
#   update (stop-token skip / min_tokens window / check_stop_strings), Fast
#   (Rust DecodeStream, native prefill, invalid-prefix recovery) and Slow
#   (prefix/read double-offset window, byte-fallback UTF-8 boundary) paths,
#   get_next_output_text delta slicing with stop-buffer holdback;
# - AsyncLLM uplink face: add_request collector birth + n>1 fan-out (idx_
#   prefixed child ids), _run_output_handler chunked pull (chunk=128 +
#   asyncio.sleep(0)), generate() consumption loop with the disconnect ->
#   abort(internal=True) ladder, abort's two hops;
# - AsyncMPClient output face: the EngineCoreOutputQueueTask (recv ->
#   validate_alive -> msgpack decode -> outputs_queue; exceptions and the
#   dead sentinel ride the same queue) + get_output_async.
#
# Runs on a CPU host WITHOUT the vllm package. Every def/class carries a
# `# SOURCE: vllm/...:Lxxx` ref into the pinned tree (line numbers re-verified
# against v0.27.1, not copied from v2's v0.21.0 assets). HOST SEAMs stand in
# for: the tokenizers backend wrapper class, the Slow-path TokenizerLike, the
# config family, msgspec (wire-compatible shim in _msgspec_seam.py, real
# msgpack bytes), the PULL output socket & engine input sink (ch05's product)
# and the input renderer (ch06's product).

from __future__ import annotations

import asyncio
import enum
import logging
import os
import sys
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import (
    Iterable,
    Mapping,
    MutableSequence,
)
from collections.abc import Sequence as GenericSequence
from copy import copy
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
import torch
import tokenizers
import tokenizers.decoders
from packaging import version
from tokenizers import Tokenizer

import _msgspec_seam
from _msgspec_seam import seam_msgspec

# The pinned vLLM does `import msgspec` / `from msgspec import msgpack`;
# both names below are the HOST SEAM namespace (see _msgspec_seam.py).
msgspec = seam_msgspec

try:
    # Real import path of the pin (vllm/v1/engine/detokenizer.py:L10).
    from transformers import TokenizersBackend
except ImportError:
    # HOST SEAM: host transformers (4.57) predates TokenizersBackend; the
    # class only needs to be nameable for isinstance + carry `._tokenizer`
    # (the raw Rust tokenizers.Tokenizer) — exactly the surface the kept
    # FastIncrementalDetokenizer touches (detokenizer.py:L61, L178).
    class TokenizersBackend:  # SOURCE: vllm/v1/engine/detokenizer.py:L10 import 位（HOST SEAM 替身）
        def __init__(self, tokenizer: Tokenizer, *args, **kwargs):  # SOURCE: vllm/v1/engine/detokenizer.py:L178 tokenizer._tokenizer 触达面
            self._tokenizer = tokenizer


# ============================================================================
# §0 Host seams — stdlib stand-ins so the module runs without the vllm
# package. Each mirrors the real interface subset the kept code touches.
# ============================================================================


# SOURCE: vllm/logger.py init_logger — logging seam with the *_once helpers
_ONCE_SEEN: set = set()  # HOST SEAM: shared once-dedup registry (test-resettable)


# SOURCE: vllm/logger.py init_logger — logging seam with the *_once helpers
def init_logger(name: str):
    log = logging.getLogger(name)
    if not log.handlers:
        log.addHandler(logging.NullHandler())
    seen = _ONCE_SEEN

    # SOURCE: vllm/logger.py once-messaging wrapper (info_once/warning_once)
    class _Once:  # HOST SEAM
        # SOURCE: vllm/logger.py once-messaging wrapper (_Once.__init__ — host seam)
        def __init__(self, fn):
            self._fn = fn

        # SOURCE: vllm/logger.py once-wrapper call
        def __call__(self, msg, *args):  # SOURCE: vllm/logger.py once-wrapper call
            key = (self._fn.__name__, msg)
            if key not in seen:
                seen.add(key)
                self._fn(msg, *args)

    log.info_once = _Once(log.info)
    log.warning_once = _Once(log.warning)
    log.debug_once = _Once(log.debug)
    return log


logger = init_logger(__name__)


# SOURCE: vllm/envs.py:L160 + L1371-1372 VLLM_V1_OUTPUT_PROC_CHUNK_SIZE — env flag seam
class envs:  # HOST SEAM (real envs.py reads the env var at get_envs() time)
    # SOURCE: vllm/envs.py:L1371-1372 lambda: int(os.getenv(..., "128"))
    VLLM_V1_OUTPUT_PROC_CHUNK_SIZE = int(
        os.getenv("VLLM_V1_OUTPUT_PROC_CHUNK_SIZE", "128")
    )
    # SOURCE: vllm/envs.py VLLM_DISABLE_REQUEST_ID_RANDOMIZATION（ch06 域旗标）
    VLLM_DISABLE_REQUEST_ID_RANDOMIZATION = False


# SOURCE: vllm/exceptions.py:L9-L42 VLLMError family — exception seams
class VLLMError(Exception):  # HOST SEAM (verbatim shape of vllm/exceptions.py:L9)
    pass


# SOURCE: vllm/exceptions.py:L19 VLLMClientError
class VLLMClientError(VLLMError):  # HOST SEAM (vllm/exceptions.py:L19)
    pass


# SOURCE: vllm/exceptions.py:L23 VLMServerError
class VLLMServerError(VLLMError):  # HOST SEAM (vllm/exceptions.py:L23)
    pass


# SOURCE: vllm/v1/engine/exceptions.py:L6-L9 EngineGenerateError — verbatim
class EngineGenerateError(VLLMServerError):
    """Raised when a AsyncLLM.generate() fails. Recoverable."""

    pass


# SOURCE: vllm/v1/engine/exceptions.py:L12-L21 EngineDeadError — verbatim
class EngineDeadError(VLLMServerError):
    """Raised when the EngineCore dies. Unrecoverable."""

    def __init__(self, *args, suppress_context: bool = False, **kwargs):  # SOURCE: vllm/v1/engine/exceptions.py:L15-L18
        ENGINE_DEAD_MESSAGE = "EngineCore encountered an issue. See stack trace (above) for the root cause."  # noqa: E501

        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)
        # Make stack trace clearer when using with LLMEngine by
        # silencing irrelevant ZMQError.
        self.__suppress_context__ = suppress_context


# SOURCE: vllm/utils/__init__.py:L11-L12 random_uuid — verbatim
def random_uuid() -> str:  # SOURCE: vllm/utils/__init__.py:L11-L12
    MASK_64_BITS = (1 << 64) - 1
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars


# SOURCE: vllm/utils/__init__.py:L15-L36 length_from_prompt_token_ids_or_embeds — verbatim
def length_from_prompt_token_ids_or_embeds(  # SOURCE: vllm/utils/__init__.py:L15-L18
    prompt_token_ids: list[int] | torch.Tensor | None,
    prompt_embeds: torch.Tensor | None,
) -> int:
    """Calculate the request length (in number of tokens) give either
    prompt_token_ids or prompt_embeds.
    """
    prompt_token_len = None if prompt_token_ids is None else len(prompt_token_ids)
    prompt_embeds_len = None if prompt_embeds is None else len(prompt_embeds)

    if prompt_token_len is None:
        if prompt_embeds_len is None:
            raise ValueError("Neither prompt_token_ids nor prompt_embeds were defined.")
        return prompt_embeds_len
    else:
        if prompt_embeds_len is not None and prompt_embeds_len != prompt_token_len:
            raise ValueError(
                "Prompt token ids and prompt embeds had different lengths"
                f" prompt_token_ids={prompt_token_len}"
                f" prompt_embeds={prompt_embeds_len}"
            )
        return prompt_token_len


# SOURCE: vllm/utils/collection_utils.py:L49-L51 as_list — verbatim
def as_list(maybe_list: Iterable) -> list:  # SOURCE: vllm/utils/collection_utils.py:L49-L51
    """Convert iterable to list, unless it's already a list."""
    return maybe_list if isinstance(maybe_list, list) else list(maybe_list)


# ============================================================================
# §1 Wire structs — vllm/v1/engine/__init__.py (decoded EngineCoreOutputs ride
# the outputs_queue; the struct fields are the schema contract, kept whole).
# ============================================================================


# SOURCE: vllm/v1/engine/__init__.py:L29-L31 FINISH_REASON_STRINGS — verbatim
FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")


# SOURCE: vllm/v1/engine/__init__.py:L43-L62 FinishReason — verbatim
class FinishReason(enum.IntEnum):  # SOURCE: vllm/v1/engine/__init__.py:L43-L56
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

    def __str__(self):  # SOURCE: vllm/v1/engine/__init__.py:L64-L65
        return FINISH_REASON_STRINGS[self.value]


# SOURCE: vllm/v1/engine/__init__.py:L261-L274 EngineCoreRequestType — verbatim
class EngineCoreRequestType(enum.Enum):  # SOURCE: vllm/v1/engine/__init__.py:L261-L265
    """
    Request types defined as hex byte strings, so it can be sent over sockets
    without separate encoding step.
    """

    ADD = b"\x00"
    ABORT = b"\x01"
    START_DP_WAVE = b"\x02"
    UTILITY = b"\x03"
    # Sentinel used within EngineCoreProc.
    EXECUTOR_FAILED = b"\x04"
    # Sentinel used to wake up input_queue.get() during shutdown.
    WAKEUP = b"\x05"


# SOURCE: vllm/v1/engine/__init__.py:L97-L146 EngineCoreRequest — 字段全保留（线格式
# schema 契约）；mm/lora/pooling 等 ch6/ch8 域字段类型放宽为 Any（HOST SEAM 注解）
class EngineCoreRequest(  # SOURCE: vllm/v1/engine/__init__.py:L97-L102
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: Any | None
    sampling_params: Any | None
    pooling_params: Any | None
    arrival_time: float
    lora_request: Any | None
    cache_salt: str | None
    data_parallel_rank: int | None
    prompt_embeds: torch.Tensor | None = None

    # SOURCE: vllm/v1/engine/__init__.py:L114-L118 mixed-mode 注释逐字
    # Per-position mask for mixed-mode inputs (e.g chat completion with
    # prompt_embeds content parts). `True` means the position is a real
    # token ID; `False` means the position uses a pre-computed entry from
    # `prompt_embeds`. `None` for pure-tokens and pure-embeds requests.
    prompt_is_token_ids: list[bool] | None = None

    # SOURCE: vllm/v1/engine/__init__.py:L120-L122 client_index 注释逐字
    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    client_index: int = 0

    # SOURCE: vllm/v1/engine/__init__.py:L124-L127 current_wave 注释逐字
    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    current_wave: int = 0
    priority: int = 0

    trace_headers: Mapping[str, str] | None = None
    resumable: bool = False

    # SOURCE: vllm/v1/engine/__init__.py:L133-L137 external_req_id 注释逐字
    # The user-provided request ID. This field is set internally,
    # copied from the provided request_id that's originally assigned
    # to the request_id field, see InputProcessor.assign_request_id().
    # Used in outputs and to support abort(req_id, internal=False).
    external_req_id: str | None = None

    reasoning_ended: bool | None = None
    reasoning_parser_kwargs: dict[str, Any] | None = None

    # SOURCE: vllm/v1/engine/__init__.py:L142-L146 abort_immediately 注释逐字
    # If True, the request should be added to the scheduler's waiting queue
    # and immediately aborted, so connector-side cleanup runs via the standard
    # request_finished hook. Used to free P-side prefill blocks when a
    # KV-transfer request is rejected on the D node before engine admission.
    abort_immediately: bool = False

    @property
    def params(self):  # SOURCE: vllm/v1/engine/__init__.py:L148-L154
        """Return the processed params (sampling or pooling)."""
        if self.sampling_params is not None:
            return self.sampling_params
        assert self.pooling_params is not None
        return self.pooling_params


# SOURCE: vllm/v1/engine/__init__.py:L184-L215 EngineCoreOutput — 上行消费最小字段集
# 全保留（logprobs/pooling/trace/experts 字段归 ch8/ch2/ch36 章，注解放宽为 Any）
class EngineCoreOutput(  # SOURCE: vllm/v1/engine/__init__.py:L184-L188
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    new_token_ids: list[int]

    new_logprobs: Any | None = None  # LogprobsLists — ch8 域
    new_prompt_logprobs_tensors: Any | None = None  # LogprobsTensors — ch8 域

    pooling_output: torch.Tensor | None = None  # pooling 产品线（delete 项 2，字段为 schema 保留）

    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None
    events: Any | None = None  # list[EngineCoreEvent] — tracing 域
    kv_transfer_params: dict[str, Any] | None = None
    ec_transfer_params: dict[str, Any] | None = None

    trace_headers: Mapping[str, str] | None = None

    prefill_stats: Any | None = None  # PrefillStats — metrics 域

    routed_experts: np.ndarray | None = None
    # SOURCE: vllm/v1/engine/__init__.py:L209-L211 num_nans_in_logits 注释逐字
    # The number of NaNs in logits.
    # A value greater than 0 indicates that the output is corrupted.
    num_nans_in_logits: int = 0

    @property
    def finished(self) -> bool:  # SOURCE: vllm/v1/engine/__init__.py:L213-L215
        return self.finish_reason is not None


# SOURCE: vllm/v1/engine/__init__.py:L230-L258 EngineCoreOutputs — 按步聚合载体
class EngineCoreOutputs(  # SOURCE: vllm/v1/engine/__init__.py:L230-L234
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    # SOURCE: vllm/v1/engine/__init__.py:L236-L237 NOTE(Nick) 注释逐字
    # NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout

    engine_index: int = 0

    # [num_reqs]
    outputs: list[EngineCoreOutput] = []
    scheduler_stats: Any | None = None  # SchedulerStats — metrics 域
    timestamp: float = 0.0

    utility_output: Any | None = None  # UtilityOutput — 控制面回执（ch5 域）
    finished_requests: set[str] | None = None

    # SOURCE: vllm/v1/engine/__init__.py:L249-L254 DP wave 注释逐字
    # In DP case, used to signal that the current wave of requests
    # has finished and the engines are paused.
    wave_complete: int | None = None
    # In DP case, used to signal that a request was received for an
    # "old" wave, so the next wave needs to be started in other engines.
    start_wave: int | None = None

    def __post_init__(self):  # SOURCE: vllm/v1/engine/__init__.py:L256-L258
        if self.timestamp == 0.0:
            self.timestamp = time.monotonic()


# ============================================================================
# §2 vllm/sampling_params.py — the three-state contract + params seam
# ============================================================================


# SOURCE: vllm/sampling_params.py:L182-L188 RequestOutputKind — verbatim
class RequestOutputKind(enum.Enum):  # SOURCE: vllm/sampling_params.py:L182
    # Return entire output so far in every RequestOutput
    CUMULATIVE = 0
    # Return only deltas in each RequestOutput
    DELTA = 1
    # Do not return intermediate RequestOutput
    FINAL_ONLY = 2


# SOURCE: vllm/sampling_params.py SamplingParams — HOST SEAM 字段面（参数域归 ch06；
# 本章触达的字段 + 真实默认值）
class SamplingParams:  # HOST SEAM
    def __init__(  # SOURCE: vllm/sampling_params.py 字段声明区（ch06 域 seam）
        self,
        n: int = 1,
        stop: Any = (),
        min_tokens: int = 0,
        include_stop_str_in_output: bool = False,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        detokenize: bool = True,
        output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE,
        stream_interval: int | None = None,
        seed: int | None = None,
        max_tokens: int = 16,
        top_p: float = 1.0,
        temperature: float = 1.0,
    ):
        self.n = n
        self.stop = stop
        self.min_tokens = min_tokens
        self.include_stop_str_in_output = include_stop_str_in_output
        self.skip_special_tokens = skip_special_tokens
        self.spaces_between_special_tokens = spaces_between_special_tokens
        self.detokenize = detokenize
        self.output_kind = output_kind
        self.stream_interval = stream_interval
        self.seed = seed
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.temperature = temperature


# SOURCE: vllm/pooling_params.py PoolingParams — HOST SEAM（pooling 产品线不入，字段面
# 仅为 EngineCoreRequest schema 与 params property 的 isinstance 面）
class PoolingParams:  # SOURCE: vllm/pooling_params.py PoolingParams（HOST SEAM 字段面）
    def __init__(self, output_kind: RequestOutputKind = RequestOutputKind.FINAL_ONLY):  # SOURCE: vllm/pooling_params.py output_kind 字段
        self.output_kind = output_kind


# ============================================================================
# §3 vllm/outputs.py — the public output objects
# ============================================================================


SampleLogprobs: TypeAlias = Any  # vllm.logprobs.SampleLogprobs — ch8 域
LoRARequest: TypeAlias = Any  # vllm.lora.request.LoRARequest — delete 项 4 域


# SOURCE: vllm/outputs.py:L21-L48 CompletionOutput — 字段全保留
@dataclass
class CompletionOutput:  # SOURCE: vllm/outputs.py:L21-L38
    """The output data of one completion output of a request.

    Args:
        index: The index of the output in the request.
        text: The generated output text.
        token_ids: The token IDs of the generated output text.
        cumulative_logprob: The cumulative log probability of the generated
            output text.
        logprobs: The log probabilities of the top probability words at each
            position if the logprobs are requested.
        finish_reason: The reason why the sequence is finished.
        stop_reason: The stop string or token id that caused the completion
            to stop, None if the completion finished for some other reason
            including encountering the EOS token.
        lora_request: The LoRA request that was used to generate the output.
    """

    index: int
    text: str
    token_ids: GenericSequence[int]
    # 机械调整：cumulative_logprob/logprobs 加默认 None（真码无默认）——logprobs 准备段
    # 随 delete 项 1 删除后调用点不再传这两参，ch8 回填时恢复传参即可。
    cumulative_logprob: float | None = None
    logprobs: SampleLogprobs | None = None
    routed_experts: np.ndarray | None = None  # [seq_len,layer_num,topk]
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    lora_request: LoRARequest | None = None

    def finished(self) -> bool:  # SOURCE: vllm/outputs.py:L50-L51
        return self.finish_reason is not None

    def __repr__(self) -> str:  # SOURCE: vllm/outputs.py:L53-L63
        return (
            f"CompletionOutput(index={self.index}, "
            f"text={self.text!r}, "
            f"token_ids={self.token_ids}, "
            f"routed_experts={self.routed_experts}, "
            f"cumulative_logprob={self.cumulative_logprob}, "
            f"logprobs={self.logprobs}, "
            f"finish_reason={self.finish_reason}, "
            f"stop_reason={self.stop_reason})"
        )


# SOURCE: vllm/outputs.py:L85-L110 RequestOutput — docstring 逐字（Pooling 类不引入）
class RequestOutput:  # SOURCE: vllm/outputs.py:L85
    """The output data of a completion request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
                For encoder/decoder models, this is the
                decoder input prompt.
        prompt_token_ids: The token IDs of the prompt.
                          For encoder/decoder models, this is the
                          decoder input prompt token ids.
        prompt_logprobs: The log probabilities to return per prompt token.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
        metrics: Metrics associated with the request.
        lora_request: The LoRA request that was used to generate the output.
        encoder_prompt: The encoder prompt string of the request.
                        None if decoder-only.
        encoder_prompt_token_ids: The token IDs of the encoder prompt.
                                  None if decoder-only.
        num_cached_tokens: The number of tokens with prefix cache hit.
        num_cache_creation_tokens: Prompt tokens currently counted as local
            prefix-cache writes for this request.
        kv_transfer_params: The params for remote K/V transfer.
        ec_transfer_params: The params for remote encoder-cache transfer.
    """

    def __init__(  # SOURCE: vllm/outputs.py:L112-L132（签名逐字，kwargs 前向兼容保留）
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_logprobs: SampleLogprobs | None,
        outputs: list[CompletionOutput],
        finished: bool,
        metrics: Any | None = None,
        lora_request: LoRARequest | None = None,
        encoder_prompt: str | None = None,
        encoder_prompt_token_ids: list[int] | None = None,
        num_cached_tokens: int | None = None,
        num_cache_creation_tokens: int | None = None,
        *,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
        # Forward compatibility, code that uses args added in new release can
        # still run with older versions of vLLM without breaking.
        **kwargs: Any,
    ) -> None:
        if kwargs:
            logger.warning_once(
                "RequestOutput: Ignoring extra arguments: %s", str(kwargs)
            )
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

    def add(self, next_output: "RequestOutput", aggregate: bool) -> None:  # SOURCE: vllm/outputs.py:L152-L153
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
                        if not isinstance(completion.token_ids, MutableSequence):
                            completion.token_ids = list(completion.token_ids)
                        completion.token_ids.extend(next_completion.token_ids)
                        if next_completion.logprobs:
                            assert completion.logprobs is not None
                            completion.logprobs.extend(next_completion.logprobs)  # type: ignore[arg-type]
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

    def __repr__(self) -> str:  # SOURCE: vllm/outputs.py:L183-L197
        return (
            f"RequestOutput(request_id={self.request_id}, "
            f"prompt={self.prompt!r}, "
            f"prompt_token_ids={self.prompt_token_ids}, "
            f"encoder_prompt={self.encoder_prompt!r}, "
            f"encoder_prompt_token_ids={self.encoder_prompt_token_ids}, "
            f"prompt_logprobs={self.prompt_logprobs}, "
            f"outputs={self.outputs}, "
            f"finished={self.finished}, "
            f"metrics={self.metrics}, "
            f"lora_request={self.lora_request}, "
            f"num_cached_tokens={self.num_cached_tokens}, "
            f"num_cache_creation_tokens={self.num_cache_creation_tokens})"
        )

    # SUBTRACTED: STREAM_FINISHED 哨兵（L200-L208）——流式输入专用（delete 项 3）


# ============================================================================
# §4 vllm/tokenizers/detokenizer_utils.py — Slow-path helpers
# ============================================================================


# SOURCE: vllm/tokenizers/detokenizer_utils.py:L11-L14 _replace_none_with_empty — verbatim
def _replace_none_with_empty(tokens: list):  # SOURCE: vllm/tokenizers/detokenizer_utils.py:L11-L14
    for i, token in enumerate(tokens):
        if token is None:
            tokens[i] = ""


# SUBTRACTED: _convert_tokens_to_string_with_added_encoders（L17-L54）与
#   _get_leading_space_marker/_restore_leading_spaces/convert_ids_list_to_tokens
#   （L62-L170）——added-vocab 切段与 logprobs 域 marker 三件套（delete 项 8）

# SOURCE: vllm/tokenizers/detokenizer_utils.py:L57-L59 — verbatim（注释逐字）
# 5 is an arbitrary value that should work for all
# tokenizers (bigger = more conservative).
INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET = 5


# SOURCE: vllm/tokenizers/detokenizer_utils.py:L119-L140 convert_prompt_ids_to_tokens — verbatim
def convert_prompt_ids_to_tokens(  # SOURCE: vllm/tokenizers/detokenizer_utils.py:L119-L123
    tokenizer: Any,
    prompt_ids: list[int],
    skip_special_tokens: bool = False,
) -> tuple[list[str], int, int]:
    """Converts the prompt ids to tokens and returns the tokens and offsets
    for incremental detokenization.

    Note that not all tokens are converted to strings. Only the tokens that
    are necessary for incremental detokenization are converted to strings.
    """
    # We do not need to convert the whole prompt to tokens.
    # Offset a little more in case we have special tokens.
    new_tokens = tokenizer.convert_ids_to_tokens(
        prompt_ids[-INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET - 2 :],
        skip_special_tokens=skip_special_tokens,
    )
    read_offset = len(new_tokens)
    prefix_offset = max(read_offset - INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET, 0)
    # This is required to guard against out-of-vocab prompt token ids
    _replace_none_with_empty(new_tokens)  # type: ignore[arg-type]
    return new_tokens, prefix_offset, read_offset


# SOURCE: vllm/tokenizers/detokenizer_utils.py:L173-L268 detokenize_incrementally —
# 主线逐字（docstring/控制流/UTF-8 边界判定全真）；else 分支随 delete 项 8 删
def detokenize_incrementally(  # SOURCE: vllm/tokenizers/detokenizer_utils.py:L176-L184
    tokenizer: Any,
    all_input_ids: list[int],
    prev_tokens: list[str] | None,
    prefix_offset: int,
    read_offset: int,
    skip_special_tokens: bool = False,
    spaces_between_special_tokens: bool = True,
) -> tuple[list[str], str, int, int]:
    """Detokenizes the input ids incrementally and returns the new tokens
    and the new text.

    If `prev_tokens` is None, this function will convert the input ids to
    tokens and return the tokens and the new text. Otherwise, it will return the
    new tokens and the new text.

    This function will also return the new prefix offset and the new read
    offset to be used in the next iteration.

    The offsets are necessary to defeat cleanup algorithms in the decode which
    decide to add a space or not depending on the surrounding ids.

    Args:
        tokenizer: The tokenizer to use.
        all_input_ids: The input ids. The last id is the new token id.
        prev_tokens: The previous tokens. If None, this function will convert
            the input ids to tokens and return the tokens and the new text.
        prefix_offset: The prefix offset.
        read_offset: The read offset.
        skip_special_tokens: Whether to skip special tokens.
        spaces_between_special_tokens: Whether to add spaces between special
            tokens.
    """
    new_token_id = all_input_ids[-1]
    # This is the first iteration for this sequence
    is_first_iter = prev_tokens is None
    if is_first_iter:
        (prev_tokens, prefix_offset, read_offset) = convert_prompt_ids_to_tokens(
            tokenizer, all_input_ids[:-1], skip_special_tokens=skip_special_tokens
        )
    assert prev_tokens is not None

    # If the new token id is out of bounds, return an empty string.
    if 0 <= new_token_id < len(tokenizer):
        # Put new_token_id in a list so skip_special_tokens is respected
        new_tokens = tokenizer.convert_ids_to_tokens(
            [new_token_id], skip_special_tokens=skip_special_tokens
        )
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
        else:
            # This is required to guard against out-of-vocab prompt token ids
            # (for example when using dummy weights)
            _replace_none_with_empty(new_tokens)  # type: ignore[arg-type]
    else:
        new_tokens = [""]
    output_tokens = prev_tokens + new_tokens

    # If this is the first iteration, return all tokens.
    if is_first_iter:
        new_tokens = output_tokens

    # The prefix text is necessary only to defeat cleanup algorithms in
    # the decode which decide to add a space or not depending on the
    # surrounding ids.
    # SUBTRACTED: else 分支（L246-L258）——非 fast 且带 added vocab 的 tokenizer 走
    #   _convert_tokens_to_string_with_added_encoders 逐 added-token 切段，逻辑同构
    #   （delete 项 8）——退化为统一 convert_tokens_to_string
    prefix_text = tokenizer.convert_tokens_to_string(
        output_tokens[prefix_offset:read_offset]
    )
    new_text = tokenizer.convert_tokens_to_string(output_tokens[prefix_offset:])

    if len(new_text) <= len(prefix_text) or new_text.endswith("�"):
        # utf-8 char at the end means it's a potential unfinished byte sequence
        # from byte fallback tokenization.
        # If it's in the middle, it's probably a real invalid id generated
        # by the model
        return new_tokens, "", prefix_offset, read_offset

    new_text = new_text[len(prefix_text) :]
    return new_tokens, new_text, read_offset, len(output_tokens)


# ============================================================================
# §5 vllm/v1/engine/detokenizer.py — the incremental detokenizer hierarchy
# ============================================================================

# TokenizerLike: vllm.tokenizers.TokenizerLike 的鸭子面（registry 域）——HOST SEAM 别名
TokenizerLike: TypeAlias = Any

# SOURCE: vllm/v1/engine/detokenizer.py:L23-L25 — verbatim（注释逐字）
# Only tokenizers >= 0.22.0 supports DecodeStream with native prefill
# (ids parameter) used for FastIncrementalDetokenizer.
USE_FAST_DETOKENIZER = version.parse(tokenizers.__version__) >= version.parse("0.22.0")

# SOURCE: vllm/v1/engine/detokenizer.py:L27-L28 — verbatim（注释逐字）
# Error string from https://github.com/huggingface/tokenizers/blob/909fdde2a4ffedd9295206f705eb612be2a91b12/tokenizers/src/tokenizer/mod.rs#L1042
INVALID_PREFIX_ERR_MSG = "Invalid prefix encountered"


# SOURCE: vllm/v1/engine/detokenizer.py:L31-L66 IncrementalDetokenizer — verbatim
class IncrementalDetokenizer:  # SOURCE: vllm/v1/engine/detokenizer.py:L31
    def __init__(self):  # SOURCE: vllm/v1/engine/detokenizer.py:L32-L33
        self.token_ids: list[int] = []

    @property
    def output_token_ids(self) -> list[int]:  # SOURCE: vllm/v1/engine/detokenizer.py:L35-L37
        return self.token_ids

    def num_output_tokens(self) -> int:  # SOURCE: vllm/v1/engine/detokenizer.py:L39-L40
        return len(self.token_ids)

    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:  # SOURCE: vllm/v1/engine/detokenizer.py:L42-L44
        self.token_ids.extend(new_token_ids)
        return None

    def get_next_output_text(self, finished: bool, delta: bool) -> str:  # SOURCE: vllm/v1/engine/detokenizer.py:L46-L47
        return ""

    @classmethod
    def from_new_request(  # SOURCE: vllm/v1/engine/detokenizer.py:L49-L54
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
    ) -> "IncrementalDetokenizer":
        assert request.sampling_params is not None

        if tokenizer is None:
            # No tokenizer => skipping detokenization.
            return IncrementalDetokenizer()

        if USE_FAST_DETOKENIZER and isinstance(tokenizer, TokenizersBackend):
            # Fast tokenizer => use tokenizers library DecodeStream.
            return FastIncrementalDetokenizer(tokenizer, request)

        # Fall back to slow python-based incremental detokenization.
        return SlowIncrementalDetokenizer(tokenizer, request)


# SOURCE: vllm/v1/engine/detokenizer.py:L69 BaseIncrementalDetokenizer — 抽象基类
class BaseIncrementalDetokenizer(IncrementalDetokenizer, ABC):  # SOURCE: vllm/v1/engine/detokenizer.py:L69
    """与具体 tokenizer 无关的公共逻辑承载者（stop/min_tokens/holdback 解析、
    update 主流程、get_next_output_text 的 delta/全量取文本与尾字符扣留）。"""

    def __init__(self, request: EngineCoreRequest):  # SOURCE: vllm/v1/engine/detokenizer.py:L70-L71
        super().__init__()

        # Stop strings
        params = request.sampling_params
        assert params is not None
        if params.stop is None:
            self.stop = []
        elif isinstance(params.stop, str):
            self.stop = [params.stop]
        else:
            self.stop = params.stop
        self.min_tokens = params.min_tokens
        self.include_stop_str_in_output = params.include_stop_str_in_output

        # Number of chars to hold back when stop strings are to be excluded
        # from streamed output.
        if self.stop and not self.include_stop_str_in_output:
            self.stop_buffer_length = max(len(s) for s in self.stop) - 1
        else:
            self.stop_buffer_length = 0
        self._last_output_text_offset: int = 0

        # Generation data
        self.output_text = ""

    # SOURCE: vllm/v1/engine/detokenizer.py:L96-L103 update — verbatim（docstring 逐字）
    def update(self, new_token_ids: list[int], stop_terminated: bool) -> str | None:
        """
        Update RequestState for the request_id by:
            1) Detokenize the new token ids incrementally.
            2) Evaluate stop criteria.

        Return matched stop string or None.
        """
        if not new_token_ids:
            # Skip detokenization if no new token ids.
            return None

        if stop_terminated and not self.include_stop_str_in_output:
            # If stop-terminated, exclude last token from detokenization
            # based on include_stop_str_in_output parameter.
            skipped_stop_token_id = new_token_ids[-1]
            new_token_ids = new_token_ids[:-1]
        else:
            skipped_stop_token_id = None

        # 1) Detokenize the new token ids incrementally.
        stop_check_offset = len(self.output_text)
        for new_token_id in new_token_ids:
            self.token_ids.append(new_token_id)
            self.output_text += self.decode_next(new_token_id)
            # Support min_tokens, see https://github.com/vllm-project/vllm/pull/22014
            if self.min_tokens and self.num_output_tokens() <= self.min_tokens:
                stop_check_offset = len(self.output_text)

        if skipped_stop_token_id is not None:
            # Cleanup after skipping detokenization.
            self.token_ids.append(skipped_stop_token_id)

        # 2) Evaluate stop strings.
        stop_string = None
        if self.stop and self.num_output_tokens() > self.min_tokens:
            stop = check_stop_strings(
                output_text=self.output_text,
                new_char_count=len(self.output_text) - stop_check_offset,
                stop=self.stop,
                include_in_output=self.include_stop_str_in_output,
            )
            if stop is not None:
                stop_string, truncate_to = stop
                if truncate_to != -1:
                    self.output_text = self.output_text[:truncate_to]

        return stop_string

    # SOURCE: vllm/v1/engine/detokenizer.py:L145-L147 decode_next — abstract
    @abstractmethod
    def decode_next(self, next_token_id: int) -> str:  # SOURCE: vllm/v1/engine/detokenizer.py:L145-L147
        raise NotImplementedError

    # SOURCE: vllm/v1/engine/detokenizer.py:L149-L165 get_next_output_text — verbatim
    def get_next_output_text(self, finished: bool, delta: bool) -> str:
        """If delta is True, only new text since the last call to
        this method is returned"""

        # We return the full output text if the sequence is finished.
        buffer_length = 0 if finished else self.stop_buffer_length
        if not delta:
            if not buffer_length:
                return self.output_text
            return self.output_text[:-buffer_length]

        length = len(self.output_text) - buffer_length
        last_offset = self._last_output_text_offset
        if last_offset < length:
            self._last_output_text_offset = length
            return self.output_text[last_offset:length]
        return ""


# SOURCE: vllm/v1/engine/detokenizer.py:L168-L248 FastIncrementalDetokenizer —
# 快路径（DecodeStream native prefill + _protected_step 容错逐字；空格抑制分支删）
class FastIncrementalDetokenizer(BaseIncrementalDetokenizer):  # SOURCE: vllm/v1/engine/detokenizer.py:L168
    def __init__(self, tokenizer: TokenizersBackend, request: EngineCoreRequest):  # SOURCE: vllm/v1/engine/detokenizer.py:L169-L170
        super().__init__(request)

        sampling_params = request.sampling_params
        assert sampling_params is not None

        self.request_id = request.request_id
        self.skip_special_tokens = sampling_params.skip_special_tokens

        self.tokenizer: Tokenizer = tokenizer._tokenizer

        # Use native prefill to prime the decode stream with prompt tokens.
        # Look up DecodeStream on the module so backend patches (e.g. the
        # fastokens shim that replaces ``tokenizers.decoders.DecodeStream``)
        # are honored regardless of import order.
        self.stream = tokenizers.decoders.DecodeStream(
            ids=request.prompt_token_ids,
            skip_special_tokens=self.skip_special_tokens,
        )

        # SUBTRACTED: spaces_between_special_tokens=False 的 added_token_ids 预计算
        #   （L189-L209，delete 项 7）——skip 或 spaces 任一为真即短路，可选优化不入

    # SOURCE: vllm/v1/engine/detokenizer.py:L211-L222 decode_next（删 L214-L220）
    def decode_next(self, next_token_id: int) -> str:
        token = self._protected_step(next_token_id)

        # SUBTRACTED: 相邻特殊 token 间空格抑制段（L214-L220，delete 项 7）

        return token or ""

    # SOURCE: vllm/v1/engine/detokenizer.py:L224-L248 _protected_step — verbatim
    def _protected_step(self, next_token_id: int) -> str | None:
        try:
            token = self.stream.step(self.tokenizer, next_token_id)
        except (OverflowError, TypeError):
            # Handle rare observed overflow, still to be diagnosed.
            # See https://github.com/vllm-project/vllm/issues/21951.
            logger.exception("Encountered invalid token id: %r", next_token_id)
            token = None
        except Exception as e:
            if not str(e).startswith(INVALID_PREFIX_ERR_MSG):
                raise e
            # Recover from edge case where tokenizer can produce non-monotonic,
            # invalid UTF-8 output, which breaks the internal state of
            # tokenizers' DecodeStream.
            # See https://github.com/vllm-project/vllm/issues/17448.
            logger.warning(
                "Encountered invalid prefix detokenization error"
                " for request %s, resetting decode stream.",
                self.request_id,
            )
            self.stream = tokenizers.decoders.DecodeStream(
                skip_special_tokens=self.skip_special_tokens
            )
            token = self.stream.step(self.tokenizer, next_token_id)
        return token


# SOURCE: vllm/v1/engine/detokenizer.py:L251-L307 SlowIncrementalDetokenizer —
# 慢路径（prompt_embeds 兜底随 delete 项 9 删）
class SlowIncrementalDetokenizer(BaseIncrementalDetokenizer):  # SOURCE: vllm/v1/engine/detokenizer.py:L251
    def __init__(self, tokenizer: TokenizerLike, request: EngineCoreRequest):  # SOURCE: vllm/v1/engine/detokenizer.py:L252-L253
        super().__init__(request)

        self.tokenizer = tokenizer
        params = request.sampling_params
        assert params is not None

        self.prompt_len = length_from_prompt_token_ids_or_embeds(
            request.prompt_token_ids, request.prompt_embeds
        )

        # Metadata for incremental detokenization.
        self.tokens, self.prefix_offset, self.read_offset = (
            convert_prompt_ids_to_tokens(
                tokenizer=tokenizer,
                prompt_ids=request.prompt_token_ids,
                skip_special_tokens=params.skip_special_tokens,
            )
        )
        # SUBTRACTED: prompt_embeds 兜底分支（L272-L276 else——源码自注 'cannot be
        #   detokenized, in general'，本就产占位空转；delete 项 9）
        self.token_ids.extend(request.prompt_token_ids)

        self.skip_special_tokens = params.skip_special_tokens
        self.spaces_between_special_tokens = params.spaces_between_special_tokens

    @property
    def output_token_ids(self) -> list[int]:  # SOURCE: vllm/v1/engine/detokenizer.py:L283-L287
        if self.prompt_len:
            return self.token_ids[self.prompt_len :]
        return self.token_ids

    def num_output_tokens(self) -> int:  # SOURCE: vllm/v1/engine/detokenizer.py:L289-L290
        return len(self.token_ids) - self.prompt_len

    # SOURCE: vllm/v1/engine/detokenizer.py:L292-L307 decode_next — verbatim
    def decode_next(self, next_token_id: int) -> str:
        new_tokens, decoded_text, prefix_offset, read_offset = detokenize_incrementally(
            tokenizer=self.tokenizer,
            all_input_ids=self.token_ids,
            prev_tokens=self.tokens,
            prefix_offset=self.prefix_offset,
            read_offset=self.read_offset,
            skip_special_tokens=self.skip_special_tokens,
            spaces_between_special_tokens=self.spaces_between_special_tokens,
        )

        self.tokens.extend(new_tokens)
        self.prefix_offset = prefix_offset
        self.read_offset = read_offset

        return decoded_text


# SOURCE: vllm/v1/engine/detokenizer.py:L310-L362 check_stop_strings — verbatim
def check_stop_strings(  # SOURCE: vllm/v1/engine/detokenizer.py:L310-L315
    output_text: str,
    new_char_count: int,
    stop: list[str],
    include_in_output: bool,
) -> tuple[str, int] | None:
    """Check if any stop strings are matched and truncate sequence
    output text accordingly.

    Returns tuple (stop_string, offset) if matched or else None.

    Where stop_string is the matched stop string and offset is the
    length to which output_text should be truncated, or -1 for no
    truncation.

    When several stop strings match within the newly generated text (for
    example when speculative decoding appends multiple tokens in a single
    step), the stop string that completes earliest in the text is selected,
    so the result matches appending one token at a time. Ties are broken by
    stop-list order.
    """
    if not new_char_count or not stop:
        return None

    best_stop_str: str | None = None
    best_stop_index = 0
    best_end = sys.maxsize
    for stop_str in stop:
        stop_string_len = len(stop_str)
        # Avoid searching already-searched text.
        stop_index = output_text.find(stop_str, 1 - new_char_count - stop_string_len)
        if stop_index == -1:
            continue

        # Prefer the stop string that completes earliest in the text.
        end = stop_index + stop_string_len
        if end < best_end:
            best_stop_str = stop_str
            best_stop_index = stop_index
            best_end = end

    if best_stop_str is None:
        return None

    if include_in_output:
        # Truncate to end of stop string.
        if best_end >= len(output_text):
            # No truncation required.
            return best_stop_str, -1
        return best_stop_str, best_end

    # Truncate the output text to the beginning of the stop string.
    return best_stop_str, best_stop_index


# ============================================================================
# §6 vllm/v1/engine/parallel_sampling.py — n>1 parent aggregation
# ============================================================================


# SOURCE: vllm/v1/engine/parallel_sampling.py:L13-L34 ParentRequest — 注释/注解逐字
class ParentRequest:  # SOURCE: vllm/v1/engine/parallel_sampling.py:L13-L18
    """Info, state & processing for parallel sampling request.

    Store parent request ID and sampling params.
    Facilitate generating child request sampling params.
    """

    request_id: str
    external_req_id: str
    sampling_params: SamplingParams

    # To track the completion of child requests
    child_requests: set[str]

    # To aggregate child completions when not streaming
    output_aggregator: list[CompletionOutput]

    # To find the max number of generated tokens across all children
    max_num_generation_tokens: int

    # To efficiently obtain child sampling params
    cached_child_sampling_params: SamplingParams | None

    def __init__(self, request: EngineCoreRequest) -> None:  # SOURCE: vllm/v1/engine/parallel_sampling.py:L36-L37
        assert request.external_req_id is not None
        sampling_params = request.params
        self.request_id = request.request_id
        self.external_req_id = request.external_req_id
        self.sampling_params = sampling_params

        self.child_requests = set()
        self.output_aggregator = (
            [cast(CompletionOutput, None)] * sampling_params.n
            if (sampling_params.output_kind == RequestOutputKind.FINAL_ONLY)
            else []
        )
        self.max_num_generation_tokens = 0
        self.cached_child_sampling_params = None

    # SOURCE: vllm/v1/engine/parallel_sampling.py:L52-L81 _get_child_sampling_params — verbatim
    def _get_child_sampling_params(
        self,
        index: int,
    ) -> SamplingParams:
        """Efficiently obtain child `sampling_params`

        If `sampling_params.seed` is not `None` then
        each child request requires a unique clone of
        parent `sampling_params` with a unique seed.

        Args:
          index: index within `n` child requests

        Returns:
          Child `sampling_params` instance.
        """
        seed = self.sampling_params.seed
        if self.cached_child_sampling_params:
            # Reuse child sampling_params data structure
            return self.cached_child_sampling_params
        # Build child sampling_params
        child_sampling_params = copy(self.sampling_params)
        child_sampling_params.n = 1
        if seed is None:
            # Cache child sampling_params for later reuse
            self.cached_child_sampling_params = child_sampling_params
        else:
            # Each child gets a clone with a unique seed
            child_sampling_params.seed = seed + index
        return child_sampling_params

    # SOURCE: vllm/v1/engine/parallel_sampling.py:L83-L94 get_child_info — verbatim
    def get_child_info(self, index: int) -> tuple[str, SamplingParams]:
        """Get child request ID and sampling params.

        Args:
          index: index within `n` child requests.

        Returns:
          (request ID, sampling_params) tuple
        """
        child_req_id = f"{index}_{self.request_id}"
        self.child_requests.add(child_req_id)
        return child_req_id, self._get_child_sampling_params(index)

    @property
    def n(self) -> int:  # SOURCE: vllm/v1/engine/parallel_sampling.py:L96-L98
        return self.sampling_params.n

    # SOURCE: vllm/v1/engine/parallel_sampling.py:L100-L126 get_outputs — verbatim
    def get_outputs(
        self,
        child_request_id: str,
        completion_output: CompletionOutput,
    ) -> tuple[list[CompletionOutput], bool]:
        already_finished_and_returned: bool = False
        if completion_output.finished():
            if child_request_id in self.child_requests:
                self.child_requests.remove(child_request_id)
            else:
                # child request ID is not available in child_requests
                # which means the request had finished in previous
                # batch step and returned to the client earlier
                already_finished_and_returned = True

        if self.sampling_params.output_kind != RequestOutputKind.FINAL_ONLY:
            # If streaming, just return the current output
            #
            # DO NOT output finished and already returned child request to client again
            outputs = [] if already_finished_and_returned else [completion_output]
        else:
            # If not streaming, aggregate the n final outputs.
            self.output_aggregator[completion_output.index] = completion_output
            outputs = [] if self.child_requests else self.output_aggregator

        finished = not self.child_requests
        return outputs, finished

    # SUBTRACTED: observe_num_generation_tokens / observe_finished_request
    #   （L128-L150）——stats 观测域（delete 项 5）


# ============================================================================
# §7 vllm/v1/engine/output_processor.py — the uplink unpacker
# ============================================================================

# SUBTRACTED: EMPTY_CPU_TENSOR（L41-L42）——pooling abort 分支占位张量（delete 项 2）


# SOURCE: vllm/v1/engine/output_processor.py:L45-L52 RequestOutputCollector — verbatim docstring
class RequestOutputCollector:  # SOURCE: vllm/v1/engine/output_processor.py:L45
    """
    Collects streamed RequestOutputs per individual request,
    for hand-off to the consuming asyncio generate task.

    When streaming deltas, RequestOutputs are merged if the
    producer gets ahead of the consumer.
    """

    def __init__(self, output_kind: RequestOutputKind, request_id: str):  # SOURCE: vllm/v1/engine/output_processor.py:L54-L58
        self.aggregate = output_kind == RequestOutputKind.DELTA
        self.request_id = request_id
        # 删 PoolingRequestOutput 联合（delete 项 2）
        self.output: RequestOutput | Exception | None = None
        self.ready = asyncio.Event()

        # SUBTRACTED: _input_stream_task（L60）——流式输入任务句柄（delete 项 3）

    # SOURCE: vllm/v1/engine/output_processor.py:L62-L63 put — 主干逐字（pooling 分支删）
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
        # SUBTRACTED: PoolingRequestOutput 直接替换分支（L73-L76，delete 项 2）

    # SOURCE: vllm/v1/engine/output_processor.py:L78-L86 get — verbatim
    async def get(self) -> RequestOutput:
        """Get operation blocks on put event."""
        while (output := self.output) is None:
            await self.ready.wait()
        self.output = None
        self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    # SOURCE: vllm/v1/engine/output_processor.py:L88-L96 get_nowait — verbatim
    def get_nowait(self) -> RequestOutput | None:
        """Non-blocking get operation."""
        output = self.output
        if output is not None:
            self.output = None
            self.ready.clear()
        if isinstance(output, Exception):
            raise output
        return output

    # SUBTRACTED: close/__del__（L98-L106）——流式输入任务清理（delete 项 3）


# SOURCE: vllm/v1/engine/output_processor.py:L109-L112 OutputProcessorOutput — verbatim（删 Pooling 联合）
@dataclass
class OutputProcessorOutput:  # SOURCE: vllm/v1/engine/output_processor.py:L109-L112（删 Pooling 联合）
    request_outputs: list[RequestOutput]
    reqs_to_abort: list[str]

    # SUBTRACTED: StreamingUpdate（L115-L127）——流式输入更新载荷（delete 项 3）


# SOURCE: vllm/v1/engine/output_processor.py:L129 RequestState — per-request 上行状态
class RequestState:  # SOURCE: vllm/v1/engine/output_processor.py:L129
    def __init__(  # SOURCE: vllm/v1/engine/output_processor.py:L130-L152（删 lora_request/stream_input 形参）
        self,
        request_id: str,
        external_req_id: str,
        parent_req: ParentRequest | None,
        request_index: int,
        # SUBTRACTED: lora_request（delete 项 4）
        output_kind: RequestOutputKind,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_embeds: torch.Tensor | None,
        logprobs_processor: Any | None,  # LogprobsProcessor 类型面（ch8 域；恒 None）
        detokenizer: IncrementalDetokenizer | None,
        max_tokens_param: int | None,
        arrival_time: float,
        queue: RequestOutputCollector | None,
        log_stats: bool,
        stream_interval: int,
        top_p: float | None = None,
        n: int | None = None,
        temperature: float | None = None,
        # SUBTRACTED: stream_input（delete 项 3）
    ):
        self.request_id = request_id
        self.external_req_id = external_req_id
        self.parent_req = parent_req
        self.request_index = request_index
        # SUBTRACTED: lora_request/lora_name（L157-L158，delete 项 4）
        self.output_kind = output_kind
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.prompt_embeds = prompt_embeds
        self.prompt_len = length_from_prompt_token_ids_or_embeds(
            self.prompt_token_ids, self.prompt_embeds
        )
        self.logprobs_processor = logprobs_processor
        self.detokenizer = detokenizer
        self.max_tokens_param = max_tokens_param
        self.top_p = top_p
        self.n = n
        self.temperature = temperature
        self.is_prefilling = True
        self.queue = queue
        self.num_cached_tokens = 0
        self.num_cache_creation_tokens = 0

        # SUBTRACTED: stats 字段（L177）——观测性域（delete 项 5）
        # SUBTRACTED: routed_experts_chunks（L179-L180）——MoE 专家路由记录（delete 项 6）

        # Stream Interval
        # SOURCE: vllm/v1/engine/output_processor.py:L182-L184
        self.stream_interval = stream_interval
        self.sent_tokens_offset = 0  # Offset of sent tokens

        # SUBTRACTED: streaming_input/input_chunk_queue（L186-L190）——流式输入（delete 项 3）

    # SUBTRACTED: apply_streaming_update（L192-L209）——流式输入状态推进（delete 项 3）

    @classmethod
    def from_new_request(  # SOURCE: vllm/v1/engine/output_processor.py:L211-L221
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        request_index: int,
        queue: RequestOutputCollector | None,
        log_stats: bool,
        stream_interval: int,
    ) -> "RequestState":
        # SOURCE: vllm/v1/engine/output_processor.py:L223-L241（detokenize=False→None 与 clamp 逐字）
        if sampling_params := request.sampling_params:
            if not sampling_params.detokenize:
                tokenizer = None
            output_kind = sampling_params.output_kind
            if sampling_params.stream_interval is not None:
                # clamp to the engine-level stream interval.
                stream_interval = max(sampling_params.stream_interval, stream_interval)
            # SUBTRACTED: LogprobsProcessor.from_new_request（L230-L233，delete 项 1）——ch8 域，恒 None
            logprobs_processor = None
            detokenizer = IncrementalDetokenizer.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            max_tokens_param = sampling_params.max_tokens
            top_p = sampling_params.top_p
            n = sampling_params.n
            temperature = sampling_params.temperature
        # SUBTRACTED: pooling else 分支（L242-L250）——池化产品线（delete 项 2）

        assert request.external_req_id is not None
        # SOURCE: vllm/v1/engine/output_processor.py:L253-L274（删 lora_request/stream_input 实参）
        return cls(
            request_id=request.request_id,
            external_req_id=request.external_req_id,
            parent_req=parent_req,
            request_index=request_index,
            output_kind=output_kind,
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=request.prompt_embeds,
            logprobs_processor=logprobs_processor,
            detokenizer=detokenizer,
            max_tokens_param=max_tokens_param,
            top_p=top_p,
            n=n,
            temperature=temperature,
            arrival_time=request.arrival_time,
            queue=queue,
            log_stats=log_stats,
            stream_interval=stream_interval,
        )

    # SOURCE: vllm/v1/engine/output_processor.py:L276-L340 make_request_output — 三道闸
    def make_request_output(
        self,
        new_token_ids: list[int],
        # SUBTRACTED: pooling_output 形参（delete 项 2 全链）
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        # SUBTRACTED: kv/ec_transfer_params 形参（delete 项 6）
    ) -> RequestOutput | None:
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

        # SUBTRACTED: pooling 分支（L317-L322）——池化产品线（delete 项 2）

        output = self._new_completion_output(new_token_ids, finish_reason, stop_reason)

        if self.parent_req is None:
            outputs = [output]
        else:
            outputs, finished = self.parent_req.get_outputs(self.request_id, output)
            if not outputs:
                return None
            external_req_id = self.parent_req.external_req_id

        # SUBTRACTED: kv/ec_transfer_params 透传实参（L338-L339，delete 项 6）
        return self._new_request_output(external_req_id, outputs, finished)

    # SOURCE: vllm/v1/engine/output_processor.py:L342-L349 _new_request_output — 外部 id 写回
    def _new_request_output(
        self,
        external_req_id: str,
        outputs: list[CompletionOutput],
        finished: bool,
        # SUBTRACTED: kv/ec_transfer_params 形参（L347-L348，delete 项 6）
    ) -> RequestOutput:
        # If prompt embeds were used, put placeholder prompt token ids
        prompt_token_ids = self.prompt_token_ids
        if prompt_token_ids is None and self.prompt_embeds is not None:
            prompt_token_ids = [0] * len(self.prompt_embeds)
        assert prompt_token_ids is not None

        # SUBTRACTED: PoolingOutput 分支（L356-L365）——池化产品线（delete 项 2）
        # SUBTRACTED: prompt_logprobs 分支（L366-L371，delete 项 1）——ch8 域，恒 None
        prompt_logprobs = None

        # SOURCE: vllm/v1/engine/output_processor.py:L373-L386（删 lora/kv/ec/metrics 实参）
        return RequestOutput(
            request_id=external_req_id,  # request_id is what was provided externally
            prompt=self.prompt,
            prompt_token_ids=prompt_token_ids,
            prompt_logprobs=prompt_logprobs,
            outputs=cast(list[CompletionOutput], outputs),
            finished=finished,
            num_cached_tokens=self.num_cached_tokens,
            num_cache_creation_tokens=self.num_cache_creation_tokens,
        )

    # SOURCE: vllm/v1/engine/output_processor.py:L388-L393 _new_completion_output — DELTA 产出
    def _new_completion_output(
        self,
        token_ids: list[int],
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
    ) -> CompletionOutput:
        assert self.detokenizer is not None
        # SUBTRACTED: assert logprobs_processor（L395）——恒 None（delete 项 1）
        finished = finish_reason is not None
        delta = self.output_kind == RequestOutputKind.DELTA

        # Prepare text and token_ids, based on delta mode
        text = self.detokenizer.get_next_output_text(finished, delta)
        if not delta:
            token_ids = self.detokenizer.output_token_ids

        # SUBTRACTED: logprobs 准备段（L404-L407）与 logprobs/cumulative_logprobs 传参
        #   （L419-L420）——ch8 域（delete 项 1）
        # SUBTRACTED: routed_experts 拼接（L409-L412）——MoE 正交特性（delete 项 6）

        return CompletionOutput(
            index=self.request_index,
            text=text,
            token_ids=token_ids,
            finish_reason=str(finish_reason) if finished else None,
            stop_reason=stop_reason if finished else None,
        )

    # SUBTRACTED: _new_pooling_output（L425-L426）——池化产品线（delete 项 2）


# SOURCE: vllm/v1/engine/output_processor.py:L429-L430 OutputProcessor — 上行解包总入口
class OutputProcessor:  # SOURCE: vllm/v1/engine/output_processor.py:L429
    """Process EngineCoreOutputs into RequestOutputs."""

    def __init__(  # SOURCE: vllm/v1/engine/output_processor.py:L432-L438（删 tracing 之外无改动）
        self,
        tokenizer: TokenizerLike | None,
        *,
        log_stats: bool,
        stream_interval: int = 1,
        tracing_enabled: bool = False,
    ):
        self.log_stats = log_stats
        self.tokenizer = tokenizer
        self.stream_interval = stream_interval
        self.request_states: dict[str, RequestState] = {}
        self.parent_requests: dict[str, ParentRequest] = {}
        self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)
        # SUBTRACTED: lora_states（L446）——LoRA 状态（delete 项 4）
        self.tracing_enabled = tracing_enabled

    # SOURCE: vllm/v1/engine/output_processor.py:L449-L450 — verbatim
    def get_num_unfinished_requests(self):
        return len(self.request_states)

    # SOURCE: vllm/v1/engine/output_processor.py:L452-L453 — verbatim
    def has_unfinished_requests(self) -> bool:
        return len(self.request_states) > 0

    # SOURCE: vllm/v1/engine/output_processor.py:L455-L460 propagate_error — verbatim
    def propagate_error(self, e: Exception):
        """Propagate error to all generate() tasks."""

        for _, state in self.request_states.items():
            assert state.queue is not None
            state.queue.put(e)

    # SOURCE: vllm/v1/engine/output_processor.py:L462-L476 abort_requests docstring — verbatim
    def abort_requests(self, request_ids: Iterable[str], internal: bool) -> list[str]:
        """Abort a list of requests.

        The request_ids may be either external request IDs (those passed to
        InputProcessor.process_inputs()) or internal request IDs (those randomly
        generated when creating the EngineCoreRequest).

        If an external request ID is provided, and that external request ID
        was used for multiple requests, all requests associated with that external
        request ID are aborted.

        In the case of parallel sampling, a request ID may be used to identify
        a parent request, in which case the associated child requests are aborted
        also.
        """
        # SOURCE: vllm/v1/engine/output_processor.py:L477-L492 双轨展开 — verbatim
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

        # SOURCE: vllm/v1/engine/output_processor.py:L494-L523 终态解阻塞与父联动（删 lora/pooling）
        request_ids_to_abort = []
        for request_id in internal_req_ids:
            req_state = self.request_states.pop(request_id, None)
            if req_state is not None:
                # SUBTRACTED: lora_states.request_finished（L498）——LoRA 观测（delete 项 4）
                request_ids_to_abort.append(request_id)
                # Produce final abort output.
                if req_state.queue is not None and (
                    request_output := req_state.make_request_output(
                        new_token_ids=[],
                        # SUBTRACTED: EMPTY_CPU_TENSOR 三元（L504-L508）——pooling 分支占位（delete 项 2）
                        finish_reason=FinishReason.ABORT,
                        stop_reason=None,
                        # SUBTRACTED: kv/ec_transfer_params 实参（L511-L512，delete 项 6）
                    )
                ):
                    req_state.queue.put(request_output)
            elif parent := self.parent_requests.get(request_id):
                # Abort children prior to removing the parent.
                if parent.child_requests:
                    child_reqs = list(parent.child_requests)
                    child_reqs = self.abort_requests(child_reqs, internal=True)
                    request_ids_to_abort.extend(child_reqs)
                self.parent_requests.pop(request_id, None)
        return request_ids_to_abort

    # SOURCE: vllm/v1/engine/output_processor.py:L525-L532 add_request — 本进程登记
    def add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None = None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        # SUBTRACTED: 流式更新早分支（L533-L537 → _update_streaming_request_state）——
        #   流式输入（delete 项 3）；内部 id 唯一后该查永不命中
        request_id = request.request_id
        req_state = RequestState.from_new_request(
            tokenizer=self.tokenizer,
            request=request,
            prompt=prompt,
            parent_req=parent_req,
            request_index=request_index,
            queue=queue,
            log_stats=self.log_stats,
            stream_interval=self.stream_interval,
        )
        self.request_states[request_id] = req_state
        if parent_req:
            self.parent_requests[parent_req.request_id] = parent_req

        # Track the external_req_id -> [internal_req_id, ...] mapping
        # SOURCE: vllm/v1/engine/output_processor.py:L553-L554
        self.external_req_ids[req_state.external_req_id].append(request_id)

    # SUBTRACTED: _update_streaming_request_state（L556-L587）——流式输入（delete 项 3）

    # SOURCE: vllm/v1/engine/output_processor.py:L589-L594 process_outputs — 唯一单循环
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

        # SOURCE: vllm/v1/engine/output_processor.py:L617-L618
        request_outputs: list[RequestOutput] = []
        reqs_to_abort: list[str] = []
        # SOURCE: vllm/v1/engine/output_processor.py:L619-L624 demux — verbatim
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                # Ignore output for already-aborted request.
                continue

            # 1) Compute stats for this iteration.
            # SOURCE: vllm/v1/engine/output_processor.py:L626-L629（调用点保留，早返回体见下）
            self._update_stats_from_output(
                req_state, engine_core_output, engine_core_timestamp, iteration_stats
            )

            # SOURCE: vllm/v1/engine/output_processor.py:L631-L634（删 pooling/kv/ec 读取）
            new_token_ids = engine_core_output.new_token_ids
            # SUBTRACTED: pooling_output 读取（L632）——pooling 全链（delete 项 2）
            finish_reason = engine_core_output.finish_reason
            stop_reason = engine_core_output.stop_reason
            # SUBTRACTED: kv/ec_transfer_params 读取（L635-L636）——透传字段（delete 项 6）
            # SUBTRACTED: routed_experts 累积（L637-L640）——MoE 记录（delete 项 6）

            # SOURCE: vllm/v1/engine/output_processor.py:L642-L650 prefill 翻转 — verbatim
            if req_state.is_prefilling:
                if engine_core_output.prefill_stats is not None:
                    req_state.num_cached_tokens = (
                        engine_core_output.prefill_stats.num_cached_tokens
                    )
                    req_state.num_cache_creation_tokens = (
                        engine_core_output.prefill_stats.num_cache_creation_tokens
                    )
                req_state.is_prefilling = False

            # SUBTRACTED: pooling 分支守卫（L652 `if pooling_output is None:`）——delete 项 2
            assert req_state.detokenizer is not None
            # 2) Detokenize the token ids into text and perform stop checks.
            # SOURCE: vllm/v1/engine/output_processor.py:L655-L661 — verbatim
            stop_string = req_state.detokenizer.update(
                new_token_ids, finish_reason == FinishReason.STOP
            )
            if stop_string:
                finish_reason = FinishReason.STOP
                stop_reason = stop_string

            # 3) Compute sample and prompt logprobs for request,
            # if required.
            # SUBTRACTED: logprobs_processor.update_from_output（L663-L665）——ch8 域（delete 项 1）

            # 4) Create and handle RequestOutput objects.
            # SOURCE: vllm/v1/engine/output_processor.py:L668-L675（删 pooling/kv/ec 实参）
            if request_output := req_state.make_request_output(
                new_token_ids,
                finish_reason,
                stop_reason,
            ):
                # SUBTRACTED: streaming_input 分支（L676-L677）——流式输入（delete 项 3）

                if req_state.queue is not None:
                    # AsyncLLM: put into queue for handling by generate().
                    req_state.queue.put(request_output)
                else:
                    # LLMEngine: return list of RequestOutputs.
                    request_outputs.append(request_output)

            # Free completed requests.
            # SOURCE: vllm/v1/engine/output_processor.py:L686-L699（删流式分支与 stats 内部）
            if finish_reason is not None:
                # SUBTRACTED: streaming_input 分支（L688-L693）——流式输入（delete 项 3）
                self._finish_request(req_state)
                if not engine_core_output.finished:
                    # If req not finished in EngineCore, but Detokenizer
                    # detected stop string, abort needed in EngineCore.
                    reqs_to_abort.append(req_id)

                # Track per-request stats
                # SOURCE: vllm/v1/engine/output_processor.py:L701-L704（调用点保留，早返回体见下）
                self._update_stats_from_finished(
                    req_state, finish_reason, iteration_stats
                )
                # SUBTRACTED: tracing_enabled → do_tracing（L705-L706）——tracing（delete 项 5）

        return OutputProcessorOutput(
            request_outputs=request_outputs,
            reqs_to_abort=reqs_to_abort,
        )

    # SOURCE: vllm/v1/engine/output_processor.py:L713-L725 _finish_request — verbatim
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

    # SUBTRACTED: update_scheduler_stats（L727-L728）——观测性（delete 项 5）
    # SUBTRACTED: do_tracing（L730-L790）——tracing（delete 项 5）

    # SOURCE: vllm/v1/engine/output_processor.py:L792-L798 _update_stats_from_output —
    # 调用点保留（delete 项 5：内部删除，iteration_stats 恒 None 走早返回）
    def _update_stats_from_output(  # SOURCE: vllm/v1/engine/output_processor.py:L792-L798
        self,
        req_state: RequestState,
        engine_core_output: EngineCoreOutput,
        engine_core_timestamp: float | None,
        iteration_stats: Any | None,
    ):
        if iteration_stats is None:
            return
        # SUBTRACTED: stats 内部（L802-L811）——观测性域（delete 项 5，恒不可达）

    # SOURCE: vllm/v1/engine/output_processor.py:L813-L818 _update_stats_from_finished —
    # 调用点保留（delete 项 5：内部删除，iteration_stats 恒 None 走早返回）
    def _update_stats_from_finished(  # SOURCE: vllm/v1/engine/output_processor.py:L813-L818
        self,
        req_state: RequestState,
        finish_reason: FinishReason | None,
        iteration_stats: Any | None,
    ):
        if iteration_stats is None:
            return
        # SUBTRACTED: stats 内部（L822-L836）——观测性域（delete 项 5，恒不可达）


# ============================================================================
# §8 vllm/v1/engine/core_client.py（AsyncMPClient 上行面）+ ch05 域 socket seam
# ============================================================================

# SOURCE: vllm/v1/engine/core.py:L1011 EngineCoreProc.ENGINE_CORE_DEAD — 死讯哨兵
ENGINE_CORE_DEAD = b"ENGINE_CORE_DEAD"  # HOST SEAM 常量（EngineCoreProc = ch05 域）


class SeamFrame:  # HOST SEAM — zmq.Frame 替身（被触面只有 .buffer）
    """One received frame; `.buffer` mirrors zmq.Frame's payload view."""

    # SOURCE: vllm/v1/engine/core_client.py:L490-L493 validate_alive 消费的帧面
    def __init__(self, payload: bytes):  # HOST SEAM
        self.buffer = payload


class SeamOutputSocket:  # HOST SEAM — ch05 产品的 PULL socket 替身（测试注入已编码帧）
    """Async recv_multipart face of the real PULL socket; tests feed whole
    frame lists (msgpack bytes produced exactly like the engine side)."""

    def __init__(self):  # SOURCE: vllm/v1/engine/core_client.py:L1030 output_socket 的 seam 面
        self._frames: asyncio.Queue = asyncio.Queue()
        self._exc: BaseException | None = None

    def feed(self, frames):  # SOURCE: vllm/v1/engine/core_client.py:L1040 recv 对端（测试注入）
        self._frames.put_nowait(frames)

    def feed_exception(self, exc: BaseException):  # SOURCE: vllm/v1/engine/core_client.py:L1084-L1085 异常入队语义（测试注入）
        self._exc = exc
        self._frames.put_nowait(None)

    async def recv_multipart(self, copy=False):  # SOURCE: vllm/v1/engine/core_client.py:L1040 调用面
        frames = await self._frames.get()
        if frames is None:
            raise self._exc
        return frames


# SOURCE: vllm/v1/engine/core_client.py:L406-L501 BackgroundResources — HOST SEAM 子集
class BackgroundResources:  # HOST SEAM（ch05 域资源面：仅上行触达的字段）
    def __init__(self, output_socket):  # SOURCE: vllm/v1/engine/core_client.py:L406-L429 __init__（seam 子集）
        self.output_socket = output_socket
        self.output_queue_task: asyncio.Task | None = None
        self.engine_dead = False

    # SOURCE: vllm/v1/engine/core_client.py:L490-L493 validate_alive — verbatim
    def validate_alive(self, frames):
        if len(frames) == 1 and (frames[0].buffer == ENGINE_CORE_DEAD):
            self.engine_dead = True
            raise EngineDeadError()


# SOURCE: vllm/v1/serial_utils.py:L313-L348 MsgpackDecoder — HOST SEAM 单面子
class MsgpackDecoder:  # HOST SEAM（ch05 全量实现的子集：本章线载体无张量/aux 帧）
    """Decoder turning the first frame into a typed struct — mirrors the real
    decode() (bufs[0] decodes; aux buffers are ch05's tensor domain, unused
    on this chapter's payloads)."""

    def __init__(self, t=None):  # SOURCE: vllm/v1/serial_utils.py:L323-L338 构造面（seam 单面）
        self._decoder = msgspec.msgpack.Decoder(t=t)

    def decode(self, bufs):  # SOURCE: vllm/v1/serial_utils.py:L340-L348
        if isinstance(bufs, (bytes, bytearray, memoryview)):
            return self._decoder.decode(bufs)
        return self._decoder.decode(bufs[0])


# SOURCE: vllm/v1/engine/core_client.py:L974-L989 AsyncMPClient — 上行面（装配=ch05 域 seam）
class AsyncMPClient:  # SOURCE: vllm/v1/engine/core_client.py:L974-L979
    """Async client for the EngineCore in a separate process."""

    def __init__(self, output_socket, client_index: int = 0):  # HOST SEAM ctor（真实经 MPClient.__init__ L503+ 起 ZMQ/encoder——ch05 产品域）
        self.client_index = client_index
        # SOURCE: vllm/v1/engine/core_client.py:L997
        self.outputs_queue: asyncio.Queue[EngineCoreOutputs | Exception] = (
            asyncio.Queue()
        )
        self.resources = BackgroundResources(output_socket)  # HOST SEAM（ch05 域子集）
        self.decoder = MsgpackDecoder(EngineCoreOutputs)  # HOST SEAM（单面 decoder）
        self.sent: list[tuple[EngineCoreRequestType, Any]] = []  # HOST SEAM：记录 (帧标签, 载荷) 代替 ZMQ 发送
        try:
            # SOURCE: vllm/v1/engine/core_client.py:L1006-L1010（注释逐字）
            # If we are running in an asyncio event loop, start the queue task.
            # Otherwise, it will be started lazily. If it is not started here,
            # we could miss EXECUTOR_FAILED messages from engine core if they
            # occur prior to any requests being sent.
            asyncio.get_running_loop()
            self._ensure_output_queue_task()
        except RuntimeError:
            pass

    # SOURCE: vllm/v1/engine/core_client.py:L1016-L1031 — 闭包捕获段逐字（删项 11 的旁支除外）
    def _ensure_output_queue_task(self):
        resources = self.resources
        if resources.output_queue_task is not None:
            return

        # Perform IO in separate task to parallelize as much as possible.
        # Avoid task having direct reference back to the client.
        decoder = self.decoder
        # SUBTRACTED: utility_results/output_handler 回调查找/_self_ref weakref（L1024-L1035）
        outputs_queue = self.outputs_queue
        output_socket = resources.output_socket
        assert output_socket is not None

        # SOURCE: vllm/v1/engine/core_client.py:L1037-L1087 — 主干（删项 11）
        async def process_outputs_socket():
            try:
                while True:
                    frames = await output_socket.recv_multipart(copy=False)
                    resources.validate_alive(frames)
                    outputs: EngineCoreOutputs = decoder.decode(frames)
                    # SUBTRACTED: utility/EEP/FT 分支与 output_handler 回调（L1043-L1080）——
                    #   控制面回执与弹性通知（delete 项 11，ch5/ch34/ch39 域）
                    if outputs.outputs or outputs.scheduler_stats:
                        outputs_queue.put_nowait(outputs)
            except Exception as e:
                outputs_queue.put_nowait(e)
            except asyncio.CancelledError:
                outputs_queue.put_nowait(EngineDeadError())

        resources.output_queue_task = asyncio.create_task(
            # SOURCE: vllm/v1/engine/core_client.py:L1089-L1091（name 参数逐字）
            process_outputs_socket(),
            name="EngineCoreOutputQueueTask",
        )

    # SOURCE: vllm/v1/engine/core_client.py:L1093-L1102 get_output_async — verbatim
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

    # SOURCE: vllm/v1/engine/core_client.py:L695-L699 _format_exception — verbatim
    def _format_exception(self, e: Exception) -> Exception:
        """If errored, use EngineDeadError so root cause is clear."""
        return (
            EngineDeadError(suppress_context=True) if self.resources.engine_dead else e
        )

    # SOURCE: vllm/v1/engine/core_client.py:L1104-L1114 _send_input — HOST SEAM 记录面
    async def _send_input(self, request_type: EngineCoreRequestType, request: Any):
        # HOST SEAM：真实拼 (type, *encode(request)) 帧并 send_multipart 返回其
        # awaitable（ch05 帧/编码域）；本章止于出发之前——记录帧标签即可断言
        # ADD/ABORT 行为（async 以贴合调用点的 await）
        self.sent.append((request_type, request))

    # SOURCE: vllm/v1/engine/core_client.py:L1145-L1148 add_request_async — verbatim
    async def add_request_async(self, request: EngineCoreRequest) -> None:
        request.client_index = self.client_index
        await self._send_input(EngineCoreRequestType.ADD, request)
        self._ensure_output_queue_task()

    # SOURCE: vllm/v1/engine/core_client.py:L1150-L1152 abort_requests_async — verbatim
    async def abort_requests_async(self, request_ids: list[str]) -> None:
        if request_ids and not self.resources.engine_dead:
            await self._send_input(EngineCoreRequestType.ABORT, request_ids)


# ============================================================================
# §9 vllm/v1/engine/async_llm.py — the AsyncLLM uplink face
# ============================================================================


# SOURCE: vllm/v1/engine/input_processor.py:L231-L249 assign_request_id — verbatim
class InputProcessor:  # HOST SEAM（ch06 产品域；上行登记只触到 assign_request_id）
    @staticmethod
    def assign_request_id(request: EngineCoreRequest):  # SOURCE: vllm/v1/engine/input_processor.py:L231-L235
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

    # SUBTRACTED: process_inputs/_validate_* 面（input_processor.py 其余）——ch06 产品域


# SUBTRACTED: InputStreamError（vllm/v1/engine/async_llm.py:L60-L69）——流式输入错误包装（delete 项 3）


# SOURCE: vllm/v1/engine/async_llm.py:L72-L73 AsyncLLM — 上行面（装配线=HOST SEAM）
class AsyncLLM:  # SOURCE: vllm/v1/engine/async_llm.py:L72-L73
    """An asynchronous wrapper for the vLLM engine."""

    def __init__(  # HOST SEAM 装配（真实 L75-L203 的 renderer/input_processor 构造、
        # profiler/logger_manager/engine 启动是 ch03/ch06/ch09 域；保留上行侧触达的
        # 字段 + L141-L146 + L173-L179 逐字）
        self,
        engine_core: AsyncMPClient,
        tokenizer: TokenizerLike | None = None,
        stream_interval: int = 1,
        log_stats: bool = False,
    ) -> None:
        self.input_processor = InputProcessor()
        # Converts EngineCoreOutputs --> RequestOutput.
        # SOURCE: vllm/v1/engine/async_llm.py:L140-L146（tokenizer 来自 renderer——ch06 域 seam）
        self.output_processor = OutputProcessor(
            tokenizer,
            log_stats=log_stats,
            stream_interval=stream_interval,
            tracing_enabled=False,
        )

        # EngineCore (starts the engine in background process).
        # SOURCE: vllm/v1/engine/async_llm.py:L148-L149
        self.engine_core = engine_core

        # SOURCE: vllm/v1/engine/async_llm.py:L173
        self.output_handler: asyncio.Task | None = None
        # SOURCE: vllm/v1/engine/async_llm.py:L174-L179（eager 启动逐字）
        try:
            # Start output handler eagerly if we are in the asyncio eventloop.
            asyncio.get_running_loop()
            self._run_output_handler()
        except RuntimeError:
            pass

    # SUBTRACTED: from_vllm_config/from_engine_args/__del__/shutdown/get_supported_tasks
    #   等装配与生命周期面（L205-L281）——ch03/ch04/ch05 域，上行主线不触达

    # SOURCE: vllm/v1/engine/async_llm.py:L283-L300 add_request — 签名逐字
    async def add_request(
        self,
        request_id: str,
        prompt: EngineCoreRequest | Any,
        params: SamplingParams | PoolingParams,
        arrival_time: float | None = None,
        lora_request: Any | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        prompt_text: str | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ) -> RequestOutputCollector:
        """Add new request to the AsyncLLM."""

        # SOURCE: vllm/v1/engine/async_llm.py:L303-L304
        if self.errored:
            raise EngineDeadError()

        # SUBTRACTED: is_pooling 判定与 kv_sharing_fast_prefill 校验（L306-L317）——
        #   pooling（delete 项 2）与 ch06/ch08 域校验
        # SUBTRACTED: 流式输入分支（L319-L334）——resumable 输入（delete 项 3）

        # Convert Input --> Request.
        # SOURCE: vllm/v1/engine/async_llm.py:L337-L350（EngineCoreRequest 直入路径逐字）
        if isinstance(prompt, EngineCoreRequest):
            logger.warning_once(
                "Passing EngineCoreRequest to AsyncLLM.generate() and .add_requests() "
                "is deprecated and will be removed in v0.18. You should instead pass "
                "the outputs of Renderer.render_cmpl() or Renderer.render_chat()."
            )

            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "AsyncLLM.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute. The "
                    "latter will be used, and the former will be ignored."
                )
        # SUBTRACTED: dict/raw prompt 渲染分流（L352-L381）——ch06 产品域
        #   （renderer.render/process_inputs_async/extract_prompt_components）；
        #   本章自 EngineCoreRequest 起步（ch6 的出发侧已完成渲染）

        # SOURCE: vllm/v1/engine/async_llm.py:L383-L386 — verbatim
        if reasoning_ended is not None:
            request.reasoning_ended = reasoning_ended
        if reasoning_parser_kwargs is not None:
            request.reasoning_parser_kwargs = reasoning_parser_kwargs

        # SOURCE: vllm/v1/engine/async_llm.py:L388
        self.input_processor.assign_request_id(request)

        # We start the output_handler on the first call to add_request() so
        # we can call __init__ before the event loop, which enables us
        # to handle startup failure gracefully in the OpenAI server.
        # SOURCE: vllm/v1/engine/async_llm.py:L390-L393（lazy 启动逐字）
        self._run_output_handler()

        # Create a new output collector for the request.
        # SOURCE: vllm/v1/engine/async_llm.py:L395-L396
        queue = RequestOutputCollector(params.output_kind, request.request_id)

        # Use cloned params that may have been updated in process_inputs()
        # SOURCE: vllm/v1/engine/async_llm.py:L398-L399
        params = request.params

        # SOURCE: vllm/v1/engine/async_llm.py:L401（删 is_pooling 联合——delete 项 2）
        if params.n == 1:
            await self._add_request(request, prompt_text, None, 0, queue)
            return queue

        parent_params = params
        assert isinstance(parent_params, SamplingParams)

        # Fan out child requests (for n>1).
        # SOURCE: vllm/v1/engine/async_llm.py:L408-L418 — verbatim
        parent_request = ParentRequest(request)
        for idx in range(parent_params.n):
            request_id, child_params = parent_request.get_child_info(idx)
            child_request = request if idx == parent_params.n - 1 else copy(request)
            child_request.request_id = request_id
            child_request.sampling_params = child_params
            await self._add_request(
                child_request, prompt_text, parent_request, idx, queue
            )
        return queue

    # SOURCE: vllm/v1/engine/async_llm.py:L420-L435 _add_request — 双登记（日志行删）
    async def _add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: ParentRequest | None,
        index: int,
        queue: RequestOutputCollector,
    ):
        # Add the request to OutputProcessor (this process).
        self.output_processor.add_request(request, prompt, parent_req, index, queue)

        # Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)

        # SUBTRACTED: log_requests 日志行（L434-L435）——纯日志（delete 项 12）

    # SUBTRACTED: _add_streaming_input_request/_validate_streaming_input_sampling_params
    #   （L437-L538）——流式输入（delete 项 3）

    # SOURCE: vllm/v1/engine/async_llm.py:L544-L575 generate — docstring 逐字
    async def generate(
        self,
        prompt: EngineCoreRequest | Any,
        sampling_params: SamplingParams,
        request_id: str,
        *,
        prompt_text: str | None = None,
        lora_request: Any | None = None,
        tokenization_kwargs: dict[str, Any] | None = None,
        trace_headers: Mapping[str, str] | None = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: dict[str, Any] | None = None,
    ):
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

        # SOURCE: vllm/v1/engine/async_llm.py:L577-L591（透传面逐字）
        q: RequestOutputCollector | None = None
        try:
            q = await self.add_request(
                request_id,
                prompt,
                sampling_params,
                lora_request=lora_request,
                tokenization_kwargs=tokenization_kwargs,
                trace_headers=trace_headers,
                priority=priority,
                data_parallel_rank=data_parallel_rank,
                prompt_text=prompt_text,
                reasoning_ended=reasoning_ended,
                reasoning_parser_kwargs=reasoning_parser_kwargs,
            )

            # The output_handler task pushes items into the queue.
            # This task pulls from the queue and yields to caller.
            # SOURCE: vllm/v1/engine/async_llm.py:L593-L599（注释逐字）
            finished = False
            while not finished:
                # Note: drain queue without await if possible (avoids
                # task switching under load which helps performance).
                out = q.get_nowait() or await q.get()

                # Note: both OutputProcessor and EngineCore handle their
                # own request cleanup based on finished.
                assert isinstance(out, RequestOutput)
                finished = out.finished
                # SUBTRACTED: STREAM_FINISHED 判别（L605）——流式输入哨兵（delete 项 3）
                yield out

        # If the request is disconnected by the client, generate()
        # is cancelled or the generator is garbage collected. So,
        # we abort the request if we end up here.
        # SOURCE: vllm/v1/engine/async_llm.py:L608-L616（注释逐字；日志行删）
        except (asyncio.CancelledError, GeneratorExit):
            if q is not None:
                await self.abort(q.request_id, internal=True)
            # SUBTRACTED: log_requests 日志行（L614-L615）——纯日志（delete 项 12）
            raise

        # Engine is dead. Do not abort since we shut down.
        # SOURCE: vllm/v1/engine/async_llm.py:L618-L622（注释逐字；日志行删）
        except EngineDeadError:
            raise

        # Request validation error.
        # SOURCE: vllm/v1/engine/async_llm.py:L624-L628（注释逐字；日志行删）
        except VLLMClientError:
            raise

        # SUBTRACTED: InputStreamError 分支（L630-L636）——流式输入（delete 项 3）

        # Unexpected error in the generate() task (possibly recoverable).
        # SOURCE: vllm/v1/engine/async_llm.py:L638-L652（日志构造压缩——delete 项 12）
        except Exception as e:
            if q is not None:
                await self.abort(q.request_id, internal=True)
            raise EngineGenerateError() from e
        # SUBTRACTED: finally q.close()（L653-L655）——close 随流式输入清理删除（delete 项 3）

    # SOURCE: vllm/v1/engine/async_llm.py:L657-L661 _run_output_handler — 逐字
    def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        if self.output_handler is not None:
            return

        # Ensure that the task doesn't have a circular ref back to the AsyncLLM
        # object, or else it won't be garbage collected and cleaned up properly.
        # SOURCE: vllm/v1/engine/async_llm.py:L663-L666（闭包捕获段逐字）
        engine_core = self.engine_core
        output_processor = self.output_processor
        # SUBTRACTED: log_stats/logger_ref/_logger_ref/logger_manager/renderer
        #   （L667-L673）——观测与弹性 EP 期 logger 重建（delete 项 5/10）
        # SOURCE: vllm/v1/engine/async_llm.py:L674
        chunk_size = envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE

        # SOURCE: vllm/v1/engine/async_llm.py:L676-L727 — 拉批分块循环
        async def output_handler():
            try:
                while True:
                    # 1) Pull EngineCoreOutputs from the EngineCore.
                    # SOURCE: vllm/v1/engine/async_llm.py:L679-L681
                    outputs = await engine_core.get_output_async()
                    num_outputs = len(outputs.outputs)

                    # SUBTRACTED: IterationStats 构造（L683-L685）——stats 观测域
                    #   （delete 项 5）；早返回体保留于 process_outputs 调用点
                    iteration_stats = None

                    # Split outputs into chunks of at most
                    # VLLM_V1_OUTPUT_PROC_CHUNK_SIZE, so that we don't block the
                    # event loop for too long.
                    # SOURCE: vllm/v1/engine/async_llm.py:L687-L693（注释逐字）
                    engine_core_outputs = outputs.outputs
                    for start in range(0, num_outputs, chunk_size):
                        end = start + chunk_size
                        outputs_slice = engine_core_outputs[start:end]
                        # 2) Process EngineCoreOutputs.
                        # SOURCE: vllm/v1/engine/async_llm.py:L694-L697
                        processed_outputs = output_processor.process_outputs(
                            outputs_slice, outputs.timestamp, iteration_stats
                        )
                        # NOTE: RequestOutputs are pushed to their queues.
                        assert not processed_outputs.request_outputs

                        # Allow other asyncio tasks to run between chunks
                        # SOURCE: vllm/v1/engine/async_llm.py:L701-L703
                        if end < num_outputs:
                            await asyncio.sleep(0)

                        # 3) Abort any reqs that finished due to stop strings.
                        # SOURCE: vllm/v1/engine/async_llm.py:L705-L709
                        if processed_outputs.reqs_to_abort:
                            await engine_core.abort_requests_async(
                                processed_outputs.reqs_to_abort
                            )

                    # SUBTRACTED: update_scheduler_stats 与第 4 步 Logging
                    #   （L711-L722）——观测性（delete 项 5）
            except Exception as e:
                # SOURCE: vllm/v1/engine/async_llm.py:L723-L725
                logger.exception("AsyncLLM output_handler failed.")
                output_processor.propagate_error(e)

        # SOURCE: vllm/v1/engine/async_llm.py:L727
        self.output_handler = asyncio.create_task(output_handler())

    # SOURCE: vllm/v1/engine/async_llm.py:L729-L738 abort — 两跳（日志行删）
    async def abort(
        self, request_id: str | Iterable[str], internal: bool = False
    ) -> None:
        """Abort RequestId in OutputProcessor and EngineCore."""

        # SOURCE: vllm/v1/engine/async_llm.py:L734-L736
        request_ids = (
            (request_id,) if isinstance(request_id, str) else as_list(request_id)
        )
        all_request_ids = self.output_processor.abort_requests(request_ids, internal)
        await self.engine_core.abort_requests_async(all_request_ids)

        # SUBTRACTED: log_requests 日志行（L740-L741）——纯日志（delete 项 12）

    # SUBTRACTED: encode/notify_kv_transfer_request_rejected/pause_generation/
    #   resume_generation/check_health/profile/reset_*/sleep/wake_up/weight-*
    #   面板（L743-L1046）——服务面/权重面/弹性域，上行主线不触达（ch38/ch34/ch39 域）

    # SOURCE: vllm/v1/engine/async_llm.py:L1085-L1088 is_running — verbatim
    @property
    def is_running(self) -> bool:  # SOURCE: vllm/v1/engine/async_llm.py:L1085-L1088
        # Is None before the loop is started.
        return self.output_handler is None or not self.output_handler.done()

    # SOURCE: vllm/v1/engine/async_llm.py:L1094-L1096 errored — verbatim
    @property
    def errored(self) -> bool:  # SOURCE: vllm/v1/engine/async_llm.py:L1094-L1096
        return self.engine_core.resources.engine_dead or not self.is_running

    # SOURCE: vllm/v1/engine/async_llm.py:L1098-L1100 dead_error — verbatim
    @property
    def dead_error(self) -> BaseException:  # SOURCE: vllm/v1/engine/async_llm.py:L1098-L1100
        return EngineDeadError()
