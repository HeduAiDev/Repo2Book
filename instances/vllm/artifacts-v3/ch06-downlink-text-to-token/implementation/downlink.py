# Subtract-only companion for v3 ch06 «下行：从文本到 token» (Part II: API
# 进程下行泳道放大——一段话在前端进程内被切成 token、组装成
# EngineCoreRequest、盖双轨 id 后出港过线).
#
# FAITHFUL SUBSET of the real vLLM front-end downlink at pin v0.27.1
# (6e448d0ea). It keeps vLLM's names, structure and control flow; it only
# DELETES branches approved in the dossier subtraction_plan (every deletion
# marked `# SUBTRACTED:` with its source span) plus the documented HOST SEAMs
# (external machinery this chapter treats as a black box). Mapping rule: take
# the real vLLM source, drop every SUBTRACTED branch, and you should get
# (approximately) this file.
#
# Fully real (the chapter's product):
# - BaseRenderer dual thread pools (renderer_num_workers tokenize pool +
#   single-worker mm pool, #38418) and the four-step pipeline render ->
#   tokenize -> extras -> process_for_engine (chat/completion isomorphic,
#   async = gather);
# - OnlineRenderer.render_chat gatekeeping (tool_choice availability, chat
#   template trust) -> preprocess_chat -> render_chat_async;
# - AsyncLLM.add_request input-form dispatch: rendered EngineInput (dict with
#   'type') -> sync fast path; raw prompt -> await process_inputs_async on
#   the renderer pool ("must not block the event loop", PR #49608);
# - InputProcessor.process_inputs: validation chain -> params.clone()
#   completion (max_tokens default, eos injection, bad words) -> mm flatten
#   (argsort_mm_positions -> list[MultiModalFeatureSpec]) -> EngineCoreRequest
#   construction (NO text field, #11963) -> assign_request_id dual-track id
#   (PR #27987);
# - _add_request double registration (local OutputProcessor BEFORE the
#   cross-process engine core) and the add_request_async client_index stamp.
#
# Runs on a CPU host WITHOUT the vllm package. Every def/class carries a
# `# SOURCE: vllm/...:Lxxx` ref into the pinned tree (line numbers re-verified
# against v0.27.1 on 2026-08-17, not copied from v2's v0.21.0 assets).
# HOST SEAMs stand in for: the HF tokenizer & chat-template engine, the mm
# processor internals (per-modality processing = black box per dossier scope),
# the config family, msgspec (wire-compatible shim in _msgspec_seam.py, real
# msgpack bytes), the engine-core client (ch05's product) and the
# OutputProcessor (ch04/ch07's product).

from __future__ import annotations

import asyncio
import contextlib
import copy as copy_mod
import enum
import hashlib
import logging
import os
import threading
import time
import uuid
from collections import defaultdict
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable, Mapping, Sequence
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cached_property, partial
from io import BytesIO
from types import SimpleNamespace
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    TypeAlias,
    TypeVar,
    Union,
    get_args,
    overload,
)

import torch
from typing_extensions import NotRequired, TypedDict, assert_never

import _msgspec_seam
from _msgspec_seam import seam_msgspec

# The pinned vLLM does `import msgspec` / `from msgspec import msgpack`;
# both names below are the HOST SEAM namespace (see _msgspec_seam.py).
msgspec = seam_msgspec
msgpack_ext = msgspec.msgpack

if TYPE_CHECKING:
    from _typeshed import SupportsRichComparison  # noqa: F401 (typing parity)


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


# SOURCE: vllm/envs.py:L210 VLLM_DISABLE_REQUEST_ID_RANDOMIZATION — env flag seam
class envs:  # HOST SEAM
    VLLM_DISABLE_REQUEST_ID_RANDOMIZATION = False


# SOURCE: vllm/exceptions.py:L9-L42 VLLMError family
class VLLMError(Exception):  # HOST SEAM (verbatim shape of vllm/exceptions.py:L9)
    pass


# SOURCE: vllm/exceptions.py:L19 VLLMClientError
class VLLMClientError(VLLMError):  # HOST SEAM (vllm/exceptions.py:L19)
    pass


# SOURCE: vllm/exceptions.py:L23 VLLMServerError
class VLLMServerError(VLLMError):  # HOST SEAM (vllm/exceptions.py:L23)
    pass


# SOURCE: vllm/exceptions.py:L27-L42 VLLMValidationError(message, parameter=, value=)
class VLLMValidationError(VLLMClientError):
    """vLLM-specific validation error for request validation failures."""

    # SOURCE: vllm/exceptions.py:L36-L42 keyword-only parameter/value attrs
    def __init__(self, message, *, parameter=None, value=None):  # HOST SEAM init
        super().__init__(message)
        self.parameter = parameter
        self.value = value


# SOURCE: vllm/v1/engine/exceptions.py:L12-L18 EngineDeadError
class EngineDeadError(VLLMServerError):
    """Raised when the EngineCore dies. Unrecoverable."""

    # SOURCE: vllm/v1/engine/exceptions.py:L15-L18 canned message
    def __init__(self, *args, suppress_context: bool = False, **kwargs):
        ENGINE_DEAD_MESSAGE = (
            "EngineCore encountered an issue. See stack trace (above) for "
            "the root cause."
        )
        super().__init__(ENGINE_DEAD_MESSAGE, *args, **kwargs)


# ── config family seams (the EngineArgs -> VllmConfig assembly line is the
#    ch03 companion; only the fields this chapter's kept code reads) ─────────


# SOURCE: vllm/config/model.py ModelConfig hf_config (attribute seam)
class HFConfig:  # HOST SEAM of the transformers config object
    # SOURCE: vllm/config/model.py ModelConfig.hf_config attribute surface
    def __init__(self, **attrs):  # HOST SEAM
        self.model_type = attrs.pop("model_type", "seam-model")
        self.__dict__.update(attrs)


# SOURCE: vllm/config/model.py:L355 renderer_num_workers (field seam)
@dataclass
# SOURCE: vllm/config/model.py ModelConfig — fields the downlink reads (seam)
class ModelConfig:
    model: str = "seam-model"  # HOST SEAM of ModelConfig.model
    max_model_len: int = 256
    # SOURCE: vllm/config/model.py runner_type ("generate" for LLMs)
    runner_type: str = "generate"
    # SOURCE: vllm/config/model.py ModelConfig.is_encoder_decoder
    is_encoder_decoder: bool = False
    # SOURCE: vllm/config/model.py ModelConfig.hf_config
    hf_config: HFConfig = field(default_factory=HFConfig)
    # SOURCE: vllm/config/model.py ModelConfig.encoder_config (dict or None)
    encoder_config: dict | None = None
    # SOURCE: vllm/config/model.py ModelConfig.enable_prompt_embeds
    enable_prompt_embeds: bool = False
    # SOURCE: vllm/config/model.py ModelConfig.multimodal_config
    multimodal_config: "MultimodalConfig | None" = None
    # SOURCE: vllm/config/model.py:L355-L364 renderer_num_workers
    renderer_num_workers: int = 1
    # SOURCE: vllm/config/model.py ModelConfig vocab (get_vocab_size seam)
    vocab_size: int = 100
    # SOURCE: vllm/config/model.py ModelConfig.try_get_generation_config (seam)
    generation_config_fields: dict = field(default_factory=dict)

    # SOURCE: vllm/config/model.py ModelConfig.get_vocab_size
    def get_vocab_size(self) -> int:  # HOST SEAM (reads the seam field)
        return self.vocab_size

    # SOURCE: vllm/config/model.py ModelConfig.try_get_generation_config
    def try_get_generation_config(self) -> dict:  # HOST SEAM
        return dict(self.generation_config_fields)

    # SOURCE: vllm/config/model.py ModelConfig.get_multimodal_config
    def get_multimodal_config(self):  # HOST SEAM
        if self.multimodal_config is None:
            raise ValueError("The config has no multimodal config")
        return self.multimodal_config


# SOURCE: vllm/config/multimodal.py MultimodalConfig — seam knobs marked seam_*
@dataclass
# SOURCE: vllm/config/multimodal.py MultimodalConfig (field seam)
class MultimodalConfig:
    # SOURCE: vllm/config/multimodal.py mm_processor_cache_gb (0 disables)
    mm_processor_cache_gb: float = 4
    # SOURCE: vllm/config/multimodal.py media_io_kwargs
    media_io_kwargs: dict | None = None
    # SOURCE: vllm/config/multimodal.py limit_per_prompt
    limit_per_prompt: dict | None = None
    # SOURCE: vllm/config/multimodal.py mm_ipc_gpu_memory_gb (deleted consumer)
    mm_ipc_gpu_memory_gb: float = 0
    encoder_cache_size: int = 1024  # HOST SEAM knob (real: encoder_budget.py)
    seam_marker_ids: dict | None = None  # HOST SEAM: marker token per modality
    seam_placeholder_ids: dict | None = None  # HOST SEAM: placeholder token id
    seam_tokens_per_item: dict | None = None  # HOST SEAM: placeholder length


# SOURCE: vllm/config/parallel.py ParallelConfig — fields the downlink reads
@dataclass
# SOURCE: vllm/config/parallel.py ParallelConfig (field seam)
class ParallelConfig:
    # SOURCE: vllm/config/parallel.py data_parallel_size
    data_parallel_size: int = 1
    # SOURCE: vllm/config/parallel.py data_parallel_size_local
    data_parallel_size_local: int = 1
    # SOURCE: vllm/config/parallel.py local_engines_only
    local_engines_only: bool = False
    # SOURCE: vllm/config/parallel.py _api_process_rank
    _api_process_rank: int = 0
    # SOURCE: vllm/config/parallel.py _api_process_count
    _api_process_count: int = 1


# SOURCE: vllm/config/cache.py CacheConfig.enable_prefix_caching
@dataclass
# SOURCE: vllm/config/cache.py CacheConfig (field seam)
class CacheConfig:
    enable_prefix_caching: bool = True


# SOURCE: vllm/config/lora.py LoRAConfig.enable_tower_connector_lora
@dataclass
# SOURCE: vllm/config/lora.py LoRAConfig (field seam)
class LoRAConfig:
    enable_tower_connector_lora: bool = False


# SOURCE: vllm/config/scheduler.py SchedulerConfig (field seam — unread here)
@dataclass
# SOURCE: vllm/config/scheduler.py SchedulerConfig
class SchedulerConfig:
    stream_interval: int = 1


# SOURCE: vllm/config/observability.py ObservabilityConfig (field seam)
@dataclass
# SOURCE: vllm/config/observability.py ObservabilityConfig
class ObservabilityConfig:
    otlp_traces_endpoint: str | None = None


# SOURCE: vllm/config/vllm.py:L331 VllmConfig — field seam (ch03 walks the rest)
@dataclass
# SOURCE: vllm/config/vllm.py:L331 VllmConfig
class VllmConfig:
    model_config: ModelConfig = field(default_factory=ModelConfig)
    # SOURCE: vllm/config/vllm.py cache_config
    cache_config: CacheConfig = field(default_factory=CacheConfig)
    # SOURCE: vllm/config/vllm.py parallel_config
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    # SOURCE: vllm/config/vllm.py scheduler_config
    scheduler_config: SchedulerConfig = field(default_factory=SchedulerConfig)
    # SOURCE: vllm/config/vllm.py lora_config
    lora_config: LoRAConfig | None = None
    # SOURCE: vllm/config/vllm.py speculative_config
    speculative_config: Any | None = None
    # SOURCE: vllm/config/vllm.py structured_outputs_config
    structured_outputs_config: Any | None = None
    # SOURCE: vllm/config/vllm.py observability_config
    observability_config: ObservabilityConfig = field(default_factory=ObservabilityConfig)
    # SOURCE: vllm/config/vllm.py use_v2_model_runner
    use_v2_model_runner: bool = False
    # SOURCE: vllm/config/vllm.py reasoning_config (deleted consumer)
    reasoning_config: Any | None = None


# SOURCE: vllm/lora/request.py LoRARequest (field seam — lora_name is read)
@dataclass
# SOURCE: vllm/lora/request.py LoRARequest
class LoRARequest:
    # SOURCE: vllm/lora/request.py LoRARequest.lora_name
    lora_name: str = ""


# SOURCE: vllm/platforms/interface.py:L1123-L1129 Platform.validate_request (default no-op)
class _Platform:  # HOST SEAM of vllm/platforms current_platform
    # SOURCE: vllm/platforms/interface.py:L1123-L1129 validate_request
    @classmethod
    # SOURCE: vllm/platforms/interface.py:L1123-L1129 validate_request
    def validate_request(cls, processed_inputs, params) -> None:
        """Raises if this request is unsupported on this platform"""
        return None


# SOURCE: vllm/platforms/__init__.py current_platform
current_platform = _Platform()  # HOST SEAM


# SOURCE: vllm/utils/mistral.py:L31-L38 is_mistral_tool_parser marker protocol
class MistralTokenizer:  # HOST SEAM marker base (vllm/utils/mistral.py mt)
    IS_MISTRAL_TOKENIZER = True


# SOURCE: vllm/utils/mistral.py:L19-L28 is_mistral_tokenizer
def is_mistral_tokenizer(obj) -> bool:
    """Return true if the tokenizer is a MistralTokenizer instance."""
    cls = type(obj)
    # Check for special class attribute, this avoids importing the class to
    # do an isinstance() check.  If the attribute is True, do an isinstance
    # check to be sure we have the correct type.
    return bool(
        getattr(cls, "IS_MISTRAL_TOKENIZER", False)
        and isinstance(obj, MistralTokenizer)
    )


# SOURCE: vllm/utils/mistral.py:L31-L38 is_mistral_tool_parser
def is_mistral_tool_parser(cls: type | None) -> bool:
    """Return true if *cls* carries the ``IS_MISTRAL_TOOL_PARSER`` marker."""
    return bool(getattr(cls, "IS_MISTRAL_TOOL_PARSER", False))


# SOURCE: vllm/parser/registry.py ParserManager.get_parser (seam — the tool/
# reasoning parser machinery is the entrypoints parser domain; the common
# case of this chapter has no parser configured)
class ParserManager:  # HOST SEAM
    # SOURCE: vllm/parser/registry.py ParserManager.get_parser
    @staticmethod
    # SOURCE: vllm/parser/registry.py ParserManager.get_parser
    def get_parser(tool_parser_name=None, reasoning_parser_name=None,
                   enable_auto_tools=False, model_name=None, is_harmony=False):
        return None  # HOST SEAM: no parser configured in the common case


# SOURCE: vllm/entrypoints/openai/engine/protocol.py:L70 ErrorResponse (seam)
class ErrorResponse:  # HOST SEAM
    # SOURCE: vllm/entrypoints/openai/engine/protocol.py ErrorResponse fields
    def __init__(self, message: str, err_type: str = "BadRequestError",
                 status_code: int = 400, param: str | None = None):
        self.message = message
        self.err_type = err_type
        self.status_code = status_code
        self.param = param


# SOURCE: vllm/entrypoints/serve/utils/error_response.py:L16-L21 create_error_response
def create_error_response(message, err_type: str = "BadRequestError",
                          status_code: int = 400,
                          param: str | None = None) -> ErrorResponse:
    return ErrorResponse(str(message), err_type, status_code, param)  # HOST SEAM


# SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:L207 ChatCompletionNamedToolChoiceParam
class ChatCompletionNamedToolChoiceParam:  # HOST SEAM of the pydantic model
    # SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py named-tool shape
    def __init__(self, function_name: str):  # HOST SEAM
        self.function = SimpleNamespace(name=function_name)


# SOURCE: vllm/entrypoints/chat_utils.py:L406 ChatTemplateContentFormatOption
ChatTemplateContentFormatOption = Literal["auto", "string", "openai"]
# SOURCE: vllm/entrypoints/chat_utils.py:L376 ConversationMessage (TypedDict)
ConversationMessage: TypeAlias = dict  # HOST SEAM of the TypedDict
# SOURCE: vllm/entrypoints/openai/protocol.py ChatCompletionMessageParam
ChatCompletionMessageParam: TypeAlias = dict  # HOST SEAM of the TypedDict

# SOURCE: vllm/multimodal/media/connector.py merge_media_io_kwargs (seam)
def merge_media_io_kwargs(defaults, overrides):  # HOST SEAM
    return (defaults or {}) | (overrides or {})


# ── params family: RequestOutputKind / SamplingParams / PoolingParams ──────


# SOURCE: vllm/sampling_params.py:L182-L188 RequestOutputKind
class RequestOutputKind(enum.Enum):
    # Return entire output so far in every RequestOutput
    CUMULATIVE = 0
    # Return only deltas in each RequestOutput
    DELTA = 1
    # Do not return intermediate RequestOutput
    FINAL_ONLY = 2


# SOURCE: vllm/sampling_params.py:L199 SamplingParams — field subset: exactly
# what the downlink reads/writes (the ~80 other fields and the 7 verify()
# validators are outside this chapter's mechanisms)
@dataclass
class SamplingParams:
    n: int = 1                                             # L213
    max_tokens: int | None = None                          # L362
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE  # L301
    ignore_eos: bool = False                               # L259
    stop: list[str] | None = None                          # L242
    stop_token_ids: list[int] | None = None                # L255
    bad_words: list[str] | None = None                     # L342
    prompt_logprobs: int | None = None                     # L295
    skip_clone: bool = False                               # L307
    # SUBTRACTED: ~80 further SamplingParams fields (vllm/sampling_params.py
    #   L199-L370 — stop strings/seed/structured-output/logit-bias/...) — not
    #   read by this chapter's kept path (stop-string evaluation is the
    #   detokenizer's, ch7; sampling decisions live engine-side, ch11+).

    _eos_token_id: int | None = None                                # L316
    _all_stop_token_ids: set[int] = field(default_factory=set)      # L318
    _bad_words_token_ids: list[list[int]] | None = None             # L344

    # SOURCE: vllm/sampling_params.py:L497-L507 __post_init__ normalization +
    #   eos_token_id-added-by-engine note (kept subset)
    # SOURCE: vllm/sampling_params.py:L497-L507 __post_init__ normalization +
    def __post_init__(self) -> None:
        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            self.stop = [self.stop]
        if self.stop_token_ids is None:
            self.stop_token_ids = []
        if self.bad_words is None:
            self.bad_words = []
        # eos_token_id is added to this by the engine
        self._all_stop_token_ids.update(self.stop_token_ids)

    # SOURCE: vllm/sampling_params.py:L748-L753 SamplingParams.clone
    def clone(self) -> "SamplingParams":
        """If skip_clone is True, uses shallow copy instead of deep copy."""
        if self.skip_clone:
            return copy_mod.copy(self)

        return copy_mod.deepcopy(self)

    # SOURCE: vllm/sampling_params.py:L646-L674 update_from_generation_config
    def update_from_generation_config(
        self,
        generation_config: dict[str, Any],
        eos_token_id: int | None = None,
    ) -> None:
        """Update if there are non-default values from generation_config"""
        if not self.ignore_eos:
            self._eos_token_id = eos_token_id

        if eos_token_id is not None:
            # Add the eos token id into the sampling_params to support
            # min_tokens processing.
            self._all_stop_token_ids.add(eos_token_id)

        # Update eos_token_id for generation
        if (eos_ids := generation_config.get("eos_token_id")) is not None:
            # it can be either int or list of int
            eos_ids = {eos_ids} if isinstance(eos_ids, int) else set(eos_ids)
            if eos_token_id is not None:
                # We don't need to include the primary eos_token_id in
                # stop_token_ids since it's handled separately for stopping
                # purposes.
                eos_ids.discard(eos_token_id)
            if eos_ids:
                self._all_stop_token_ids.update(eos_ids)
                if not self.ignore_eos:
                    assert self.stop_token_ids is not None
                    eos_ids.update(self.stop_token_ids)
                    self.stop_token_ids = list(eos_ids)

    # SOURCE: vllm/sampling_params.py:L676-L715 update_from_tokenizer
    def update_from_tokenizer(self, tokenizer) -> None:
        if not self.bad_words:
            return
        self._bad_words_token_ids = []
        for bad_word in self.bad_words:
            # To prohibit words both at the beginning
            # and in the middle of text
            # (related to add_prefix_space tokenizer parameter)
            for add_prefix_space in [False, True]:
                prefix = " " if add_prefix_space else ""
                prompt = prefix + bad_word.lstrip()
                prompt_token_ids = tokenizer.encode(
                    text=prompt, add_special_tokens=False
                )

                # If no space at the beginning
                # or if prefix space produces a new word token
                if (not add_prefix_space) or (
                    add_prefix_space
                    and prompt_token_ids[0] != self._bad_words_token_ids[-1][0]
                    and len(prompt_token_ids) == len(self._bad_words_token_ids[-1])
                ):
                    self._bad_words_token_ids.append(prompt_token_ids)

        invalid_token_ids = [
            token_id
            for bad_words_token_ids in self._bad_words_token_ids
            for token_id in bad_words_token_ids
            if token_id < 0 or token_id > tokenizer.max_token_id
        ]
        if len(invalid_token_ids) > 0:
            raise VLLMValidationError(
                f"The model vocabulary size is {tokenizer.max_token_id + 1},"
                f" but the following tokens"
                f" were specified as bad: {invalid_token_ids}."
                f" All token id values should be integers satisfying:"
                f" 0 <= token_id <= {tokenizer.max_token_id}.",
                parameter="bad_words",
                value=self.bad_words,
            )

    # SOURCE: vllm/sampling_params.py:L755-L770 verify (seam — the 7
    # _validate_* calls are param-domain validation, not this chapter's)
    # SOURCE: vllm/sampling_params.py:L755-L770 verify (seam — the 7
    def verify(self, model_config, speculative_config,
               structured_outputs_config, tokenizer) -> None:  # HOST SEAM
        return None

    # SOURCE: vllm/sampling_params.py:L725-L727 eos_token_id property
    @property
    # SOURCE: vllm/sampling_params.py:L725-L727 eos_token_id property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    # SOURCE: vllm/sampling_params.py:L729-L731 all_stop_token_ids property
    @property
    # SOURCE: vllm/sampling_params.py:L729-L731 all_stop_token_ids property
    def all_stop_token_ids(self) -> set:
        return self._all_stop_token_ids

    # SOURCE: vllm/sampling_params.py:L733-L736 bad_words_token_ids property
    @property
    # SOURCE: vllm/sampling_params.py:L733-L736 bad_words_token_ids property
    def bad_words_token_ids(self) -> list[list[int]] | None:
        # For internal use only. Backward compatibility not guaranteed
        return self._bad_words_token_ids


# SOURCE: vllm/pooling_params.py:L38 PoolingParams (field seam)
@dataclass
class PoolingParams:
    task: str | None = None
    # SUBTRACTED: the per-task extra fields (pooling_params.py "embed": ... map
    #   et al.) — pooling internals are not this chapter's mechanisms.
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE

    # SOURCE: vllm/pooling_params.py:L86-L88 PoolingParams.clone
    def clone(self) -> "PoolingParams":
        """Returns a deep copy of the PoolingParams instance."""
        return copy_mod.deepcopy(self)

    # SOURCE: vllm/pooling_params.py:L90 PoolingParams.verify (seam)
    def verify(self, model_config) -> None:  # HOST SEAM
        return None


# SOURCE: vllm/engine/protocol.py:L30-L38 StreamingInput
@dataclass
# SOURCE: vllm/engine/protocol.py:L30-L38 StreamingInput
class StreamingInput:
    """Input data for a streaming generation request.

    This is used with generate() to support multi-turn streaming sessions
    where inputs are provided via an async generator.
    """

    prompt: "EngineInput"
    sampling_params: "SamplingParams | None" = None


# ── front-end neighbours (ch04/ch05/ch07 products) as recording seams ──────


# SOURCE: vllm/v1/engine/output_processor.py:L45-L60 RequestOutputCollector
class RequestOutputCollector:  # HOST SEAM (put/add machinery is ch07's)
    """
    Collects streamed RequestOutputs per individual request,
    for hand-off to the consuming asyncio generate task.
    """

    # SOURCE: vllm/v1/engine/output_processor.py:L54-L60 __init__
    def __init__(self, output_kind: RequestOutputKind, request_id: str):
        self.aggregate = output_kind == RequestOutputKind.DELTA
        self.request_id = request_id
        self.output: Any = None
        self.ready = asyncio.Event()

        self._input_stream_task: asyncio.Task | None = None
    # SUBTRACTED: put()/add-merge machinery (output_processor.py:L62-L99) —
    #   the upstream loop is ch07's product; the downlink only creates the
    #   collector keyed by the INTERNAL request id.


# SOURCE: vllm/v1/engine/parallel_sampling.py ParentRequest (seam — n>1 fan-out
# is deleted per the subtraction plan and is ch07's product)
class ParentRequest:  # HOST SEAM
    # SOURCE: vllm/v1/engine/parallel_sampling.py ParentRequest.__init__
    def __init__(self, request):  # HOST SEAM
        self.request_id = request.request_id


# SOURCE: vllm/v1/engine/output_processor.py:L429+ OutputProcessor — seam that
# records the local-process half of the double registration (ch04's product;
# external_req_ids demux bookkeeping is ch07's)
class OutputProcessor:  # HOST SEAM
    # SOURCE: vllm/v1/engine/output_processor.py OutputProcessor.__init__ (seam)
    def __init__(self, events: list | None = None, **kwargs):
        self.events = events if events is not None else []
        self.request_states: dict[str, Any] = {}
        self.external_req_ids: defaultdict[str, list[str]] = defaultdict(list)

    # SOURCE: vllm/v1/engine/output_processor.py:L525-L532 add_request signature
    def add_request(
        self,
        request: "EngineCoreRequest",
        prompt: str | None,
        parent_req: ParentRequest | None = None,
        request_index: int = 0,
        queue: RequestOutputCollector | None = None,
    ) -> None:
        request_id = request.request_id
        # SOURCE: vllm/v1/engine/output_processor.py external→internal map
        self.external_req_ids[request.external_req_id or request_id].append(
            request_id
        )
        self.request_states[request_id] = SimpleNamespace(
            request=request, prompt=prompt, index=request_index, queue=queue,
        )
        self.events.append(("output_processor.add_request", request_id))
    # SUBTRACTED: RequestState construction internals / streaming-input update
    #   path (output_processor.py:L533-L560) — ch04/ch07 章域.


# SOURCE: vllm/v1/engine/__init__.py:L261-L274 EngineCoreRequestType
class EngineCoreRequestType(enum.Enum):
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
    # Sentinel to wake up input_queue.get() during shutdown.
    WAKEUP = b"\x05"


# SOURCE: vllm/v1/engine/core_client.py:L974+ AsyncMPClient — seam of the ch05
# product: the ZMQ ROUTER/DEALER wire, handshake and output loop all live in
# the ch05 companion; this seam keeps the observable downlink surface
# (client_index stamp + ADD frame + get_supported_tasks RPC).
class AsyncMPClient:  # HOST SEAM
    # SOURCE: vllm/v1/engine/core_client.py AsyncMPClient.__init__ (seam)
    def __init__(self, events: list | None = None, client_index: int = 0,
                 supported_tasks: tuple = ("generate",)):
        self.events = events if events is not None else []
        self.client_index = client_index
        self.supported_tasks = supported_tasks
        # SOURCE: vllm/v1/engine/core_client.py BackgroundResources.engine_dead
        self.resources = SimpleNamespace(engine_dead=False)

    # SOURCE: vllm/v1/engine/core_client.py:L1142-L1143 get_supported_tasks_async
    async def get_supported_tasks_async(self) -> tuple:
        # HOST SEAM: the real call is a UTILITY RPC over the ch05 wire.
        return self.supported_tasks

    # SOURCE: vllm/v1/engine/core_client.py:L1145-L1148 add_request_async — the
    # real three lines, verbatim (client_index stamp + ADD frame)
    # SOURCE: vllm/v1/engine/core_client.py:L1145-L1148 add_request_async — the
    async def add_request_async(self, request: "EngineCoreRequest") -> None:
        request.client_index = self.client_index
        await self._send_input(EngineCoreRequestType.ADD, request)
        self._ensure_output_queue_task()

    # SOURCE: vllm/v1/engine/core_client.py:L1103-L1113 _send_input/_send_input_message
    async def _send_input(self, request_type, request) -> None:
        # HOST SEAM: records the frame instead of hitting the ZMQ wire —
        # chapter boundary per scope (帧序/编码已由 ch05 讲透, 本页止于 _send_input).
        self.events.append(("send_input", request_type, request))

    # SOURCE: vllm/v1/engine/core_client.py AsyncMPClient._ensure_output_queue_task
    def _ensure_output_queue_task(self) -> None:  # HOST SEAM
        self.events.append(("output_queue_task",))


# ── torch thread helper (real) ──────────────────────────────────────────────


# SOURCE: vllm/utils/torch_utils.py:L153-L177 set_default_torch_num_threads
@contextlib.contextmanager
# SOURCE: vllm/utils/torch_utils.py:L153-L177 set_default_torch_num_threads
def set_default_torch_num_threads(num_threads: int | None = None):
    """
    Sets the default number of threads for PyTorch to the given value.

    `None` means using the value of the environment variable `OMP_NUM_THREADS`
    (or `1` if that is not available).
    """
    if num_threads is None:
        num_threads = 1

        try:
            num_threads = int(os.environ["OMP_NUM_THREADS"])
        except KeyError:
            logger.debug_once(
                "OMP_NUM_THREADS is not set; defaulting Torch threads to %d.",
                num_threads,
            )
        except ValueError:
            logger.warning_once(
                "OMP_NUM_THREADS is invalid; defaulting Torch threads to %d.",
                num_threads,
            )

    old_num_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)

    try:
        yield
    finally:
        torch.set_num_threads(old_num_threads)


# ── small real utils the kept code calls ────────────────────────────────────


# SOURCE: vllm/utils/__init__.py:L8 MASK_64_BITS
MASK_64_BITS = (1 << 64) - 1


# SOURCE: vllm/utils/__init__.py:L11-L12 random_uuid
def random_uuid() -> str:
    return f"{uuid.uuid4().int & MASK_64_BITS:016x}"  # 16 hex chars


# SOURCE: vllm/utils/__init__.py:L15-L36 length_from_prompt_token_ids_or_embeds
def length_from_prompt_token_ids_or_embeds(
    prompt_token_ids: "list[int] | torch.Tensor | None",
    prompt_embeds: "torch.Tensor | None",
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


# SOURCE: vllm/utils/async_utils.py:L28-L45 make_async
def make_async(
    func: Callable,
    executor: Executor | None = None,
) -> Callable[..., Awaitable]:
    """
    Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """

    # SOURCE: vllm/utils/async_utils.py:L40-L44 _async_wrapper
    def _async_wrapper(*args, **kwargs):
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=executor, func=p_func)

    return _async_wrapper


# SOURCE: vllm/utils/counter.py:L21-L45 AtomicCounter
class AtomicCounter:
    """An atomic, thread-safe counter"""

    # SOURCE: vllm/utils/counter.py:L24-L29 AtomicCounter.__init__
    def __init__(self, initial: int = 0) -> None:
        """Initialize a new atomic counter to given initial value"""
        super().__init__()

        self._value = initial
        self._lock = threading.Lock()

    # SOURCE: vllm/utils/counter.py:L31-L33 AtomicCounter.value
    @property
    # SOURCE: vllm/utils/counter.py:L31-L33 AtomicCounter.value
    def value(self) -> int:
        return self._value

    # SOURCE: vllm/utils/counter.py:L34-L38 AtomicCounter.inc
    def inc(self, num: int = 1) -> int:
        """Atomically increment the counter by num and return the new value"""
        with self._lock:
            self._value += num
            return self._value

    # SOURCE: vllm/utils/counter.py:L39-L43 AtomicCounter.dec
    def dec(self, num: int = 1) -> int:
        """Atomically decrement the counter by num and return the new value"""
        with self._lock:
            self._value -= num
            return self._value


# SOURCE: vllm/utils/jsontree.py:L40-L45 json_iter_leaves
def json_iter_leaves(value) -> Iterable:
    """Iterate through each leaf in a nested JSON structure."""
    if isinstance(value, dict):
        for v in value.values():
            yield from json_iter_leaves(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from json_iter_leaves(v)
    else:
        yield value


# SOURCE: vllm/utils/collection_utils.py:L54-L64 is_list_of
def is_list_of(value: object, typ, *, check: str = "first") -> bool:
    if not isinstance(value, list):
        return False

    if check == "first":
        return len(value) == 0 or isinstance(value[0], typ)
    return all(isinstance(v, typ) for v in value)
    # SUBTRACTED: the TypeIs narrowing + heterogeneous-tuple branch
    #   (collection_utils.py:L54-L96) — typing surface only.


# SOURCE: vllm/tasks.py:L7-L43 task literals
GenerationTask = Literal["generate", "transcription", "realtime"]
# SOURCE: vllm/tasks.py:L8 GENERATION_TASKS
GENERATION_TASKS: tuple = get_args(GenerationTask)
# SOURCE: vllm/tasks.py:L10-L18 POOLING_TASKS
PoolingTask = Literal[
    "embed", "classify", "token_embed", "token_classify", "plugin",
    "embed&token_classify",
]
# SOURCE: vllm/tasks.py:L18
POOLING_TASKS: tuple = get_args(PoolingTask)
# SOURCE: vllm/tasks.py:L40-L43 FrontendTask / SupportedTask
FrontendTask = Literal["render"]
SupportedTask = Literal[GenerationTask, PoolingTask, FrontendTask]

# ============================================================================
# §1 Prompt schemas — vllm/inputs/llm.py (real)
# ============================================================================

# SOURCE: vllm/inputs/llm.py:L45 MultiModalDataDict (ModalityData union seam'd to Any)
ModalityData: TypeAlias = Any  # HOST SEAM of the per-modality data unions
# SOURCE: vllm/inputs/llm.py:L45
MultiModalDataDict: TypeAlias = Mapping[str, ModalityData]
"""
A dictionary containing an entry for each modality type to input.
"""
# SOURCE: vllm/inputs/llm.py:L53
MultiModalUUIDDict: TypeAlias = Mapping[str, Sequence[str | None] | str]
"""
A dictionary containing user-provided UUIDs for items in each modality.
"""


# SOURCE: vllm/inputs/llm.py:L64-L96 _PromptOptions
class _PromptOptions(TypedDict):
    """
    Additional options available to all
    [`SingletonPrompt`][vllm.inputs.llm.SingletonPrompt] types.
    """

    multi_modal_data: NotRequired[MultiModalDataDict | None]
    """
    Optional multi-modal data to pass to the model,
    if the model supports it.
    """

    mm_processor_kwargs: NotRequired[dict[str, Any] | None]
    """
    Optional multi-modal processor kwargs to be forwarded to the
    multimodal input mapper & processor.
    """

    multi_modal_uuids: NotRequired[MultiModalUUIDDict]
    """
    Optional user-specified UUIDs for multimodal items, mapped by modality.
    """

    cache_salt: NotRequired[str]
    """
    Optional cache salt to be used for prefix caching.
    """


# SOURCE: vllm/inputs/llm.py:L99-L103 TextPrompt
class TextPrompt(_PromptOptions):
    """Schema for a text prompt."""

    prompt: str
    """The input text to be tokenized before passing to the model."""


# SOURCE: vllm/inputs/llm.py:L106-L122 TokensPrompt
class TokensPrompt(_PromptOptions):
    """Schema for a tokenized prompt."""

    prompt_token_ids: list[int]
    """A list of token IDs to pass to the model."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token IDs, if available."""

    # SUBTRACTED: token_type_ids / prompt_token_offsets fields
    #   (vllm/inputs/llm.py:L115-L122) — cross-encoder offsets feature
    #   (delete item 3) and cross-encoder input, off the downlink spine.


# SOURCE: vllm/inputs/llm.py:L125-L143 EmbedsPrompt
class EmbedsPrompt(_PromptOptions):
    """Schema for a prompt provided via token embeddings."""

    prompt_embeds: "torch.Tensor"
    """The embeddings of the prompt."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token embeddings, if available."""

    prompt_token_ids: NotRequired[list[int]]
    """Token IDs for mixed-mode inputs (chat completion with
    `prompt_embeds` content parts)."""

    prompt_is_token_ids: NotRequired[list[bool]]
    """Per-position mask, `True` uses the real token ID, `False` uses
    the corresponding entry from `prompt_embeds`."""


# SOURCE: vllm/inputs/llm.py:L146-L158 DecoderOnlyPrompt
DecoderOnlyPrompt: TypeAlias = (
    str | TextPrompt | list[int] | TokensPrompt | EmbedsPrompt
)
# SOURCE: vllm/inputs/llm.py:L162
EncoderPrompt: TypeAlias = str | TextPrompt | list[int] | TokensPrompt
# SOURCE: vllm/inputs/llm.py:L172
DecoderPrompt: TypeAlias = str | TextPrompt | list[int] | TokensPrompt


# SOURCE: vllm/inputs/llm.py:L185-L201 ExplicitEncoderDecoderPrompt
class ExplicitEncoderDecoderPrompt(TypedDict):
    """
    Schema of a pair of encoder and decoder singleton prompts.
    """

    encoder_prompt: EncoderPrompt
    """The prompt for the encoder part of the model."""

    decoder_prompt: DecoderPrompt | None
    """
    The prompt for the decoder part of the model.

    Passing `None` will cause the decoder prompt to be inferred automatically.
    """


# SOURCE: vllm/inputs/llm.py:L204-L210 EncoderDecoderPrompt
EncoderDecoderPrompt: TypeAlias = EncoderPrompt | ExplicitEncoderDecoderPrompt
# SOURCE: vllm/inputs/llm.py:L213-L218 SingletonPrompt
SingletonPrompt: TypeAlias = DecoderOnlyPrompt | EncoderPrompt | DecoderPrompt
# SOURCE: vllm/inputs/llm.py:L221-L226 PromptType
PromptType: TypeAlias = DecoderOnlyPrompt | EncoderDecoderPrompt


# ============================================================================
# §2 Renderer-internal prompt types — vllm/renderers/inputs/* (real)
# ============================================================================


# SOURCE: vllm/renderers/inputs/preprocess.py:L30-L56 prompt_to_seq (+overloads)
@overload
# SOURCE: vllm/renderers/inputs/preprocess.py:L30-L56 prompt_to_seq (+overloads)
def prompt_to_seq(
    prompt_or_prompts: "SingletonPrompt | bytes | Sequence[SingletonPrompt | bytes]",
) -> "Sequence[SingletonPrompt]": ...


# SOURCE: vllm/renderers/inputs/preprocess.py:L35-L39 prompt_to_seq overload 2
@overload
# SOURCE: vllm/renderers/inputs/preprocess.py:L35-L39 prompt_to_seq overload 2
def prompt_to_seq(  # type: ignore[misc]
    prompt_or_prompts: "ExplicitEncoderDecoderPrompt | Sequence[ExplicitEncoderDecoderPrompt]",
) -> "Sequence[ExplicitEncoderDecoderPrompt]": ...


# SOURCE: vllm/renderers/inputs/preprocess.py:L42-L45 prompt_to_seq overload 3
@overload
# SOURCE: vllm/renderers/inputs/preprocess.py:L42-L45 prompt_to_seq overload 3
def prompt_to_seq(  # type: ignore[misc]
    prompt_or_prompts: "PromptType | Sequence[PromptType]",
) -> "Sequence[PromptType]": ...


# SOURCE: vllm/renderers/inputs/preprocess.py:L48-L56 prompt_to_seq body
def prompt_to_seq(
    prompt_or_prompts: "PromptType | bytes | Sequence[PromptType | bytes]",
) -> "Sequence[PromptType]":
    if isinstance(prompt_or_prompts, (dict, str, bytes)) or (
        len(prompt_or_prompts) > 0 and is_list_of(prompt_or_prompts, int)
    ):
        return [prompt_or_prompts]  # type: ignore[list-item]

    return prompt_or_prompts  # type: ignore[return-value]


# SOURCE: vllm/renderers/inputs/preprocess.py:L59-L68 conversation_to_seq
def conversation_to_seq(
    conversation_or_conversations: "list[ChatCompletionMessageParam] | Sequence[list[ChatCompletionMessageParam]]",
) -> "Sequence[list[ChatCompletionMessageParam]]":
    if len(conversation_or_conversations) > 0 and is_list_of(
        conversation_or_conversations, dict
    ):
        return [conversation_or_conversations]  # type: ignore[list-item]

    return conversation_or_conversations  # type: ignore[return-value]


# SOURCE: vllm/renderers/inputs/preprocess.py:L71-L81 DecoderOnlyDictPrompt
DecoderOnlyDictPrompt: TypeAlias = TextPrompt | TokensPrompt | EmbedsPrompt
# SOURCE: vllm/renderers/inputs/preprocess.py:L78
EncoderDictPrompt: TypeAlias = TextPrompt | TokensPrompt
# SOURCE: vllm/renderers/inputs/preprocess.py:L85
DecoderDictPrompt: TypeAlias = TextPrompt | TokensPrompt


# SOURCE: vllm/renderers/inputs/preprocess.py:L92-L100 EncoderDecoderDictPrompt
class EncoderDecoderDictPrompt(TypedDict):
    encoder_prompt: EncoderDictPrompt
    decoder_prompt: DecoderDictPrompt | None


# SOURCE: vllm/renderers/inputs/preprocess.py:L103-L109 SingletonDictPrompt
SingletonDictPrompt: TypeAlias = (
    DecoderOnlyDictPrompt | EncoderDictPrompt | DecoderDictPrompt
)
# SOURCE: vllm/renderers/inputs/preprocess.py:L112-L116 DictPrompt
DictPrompt: TypeAlias = DecoderOnlyDictPrompt | EncoderDecoderDictPrompt


# SOURCE: vllm/renderers/inputs/preprocess.py:L119-L129 _validate_prompt_dict
def _validate_prompt_dict(prompt: Mapping[str, object]) -> None:
    """Reject malformed dict prompts before renderer tokenization."""
    if (
        "prompt" not in prompt
        or "prompt_token_ids" in prompt
        or "prompt_embeds" in prompt
    ):
        return

    if not isinstance(prompt["prompt"], str):
        raise TypeError("Prompt text should be a string")


# SOURCE: vllm/renderers/inputs/preprocess.py:L132-L160 parse_dec_only_prompt
def parse_dec_only_prompt(prompt: "PromptType | object") -> "DecoderOnlyDictPrompt":
    """
    Parse a prompt for a decoder-only model and normalize it to a dictionary.
    """
    if isinstance(prompt, str):
        return TextPrompt(prompt=prompt)

    if isinstance(prompt, list):
        if not is_list_of(prompt, int):
            raise TypeError("Token prompt should be a list of integers")

        return TokensPrompt(prompt_token_ids=prompt)

    if isinstance(prompt, dict):
        if "encoder_prompt" in prompt:
            raise TypeError("Cannot pass encoder-decoder prompt to decoder-only models")

        _validate_prompt_dict(prompt)

        if (
            "prompt" in prompt
            or "prompt_token_ids" in prompt
            or "prompt_embeds" in prompt
        ):
            return prompt  # type: ignore[return-value]

        raise TypeError("Prompt dictionary must contain text, tokens, or embeddings")

    raise TypeError("Prompt should be a string, list of tokens, or dictionary")


# SOURCE: vllm/renderers/inputs/preprocess.py:L163-L184 _parse_enc_prompt
def _parse_enc_prompt(prompt: "PromptType | object") -> "EncoderDictPrompt":
    if isinstance(prompt, str):
        return TextPrompt(prompt=prompt)

    if isinstance(prompt, list):
        if not is_list_of(prompt, int):
            raise TypeError("Token prompt should be a list of integers")

        return TokensPrompt(prompt_token_ids=prompt)

    if isinstance(prompt, dict):
        _validate_prompt_dict(prompt)

        if "prompt_embeds" in prompt:
            raise TypeError("Cannot pass embeddings prompt to encoder-decoder models")

        if "prompt" in prompt or "prompt_token_ids" in prompt:
            return prompt  # type: ignore[return-value]

        raise TypeError("Prompt dictionary must contain text or tokens")

    raise TypeError("Prompt should be a string, list of tokens, or dictionary")


# SOURCE: vllm/renderers/inputs/preprocess.py:L187-L215 _parse_dec_prompt
def _parse_dec_prompt(prompt: "PromptType | object") -> "DecoderDictPrompt":
    if isinstance(prompt, str):
        return TextPrompt(prompt=prompt)

    if isinstance(prompt, list):
        if not is_list_of(prompt, int):
            raise TypeError("Token prompt should be a list of integers")

        return TokensPrompt(prompt_token_ids=prompt)

    if isinstance(prompt, dict):
        _validate_prompt_dict(prompt)

        if "prompt_embeds" in prompt:
            raise TypeError("Cannot pass embeddings prompt to encoder-decoder models")

        if (
            "multi_modal_data" in prompt
            or "mm_processor_kwargs" in prompt
            or "multi_modal_uuids" in prompt
        ):
            raise TypeError("Cannot pass multi-modal inputs to decoder prompt")

        if "prompt" in prompt or "prompt_token_ids" in prompt:
            return prompt  # type: ignore[return-value]

        raise TypeError("Prompt dictionary must contain text or tokens")

    raise TypeError("Prompt should be a string, list of tokens, or dictionary")


# SOURCE: vllm/renderers/inputs/preprocess.py:L218-L232 parse_enc_dec_prompt
def parse_enc_dec_prompt(prompt: "PromptType | object") -> "EncoderDecoderDictPrompt":
    """
    Parse a prompt for an encoder-decoder model and normalize it to a dictionary.
    """
    if isinstance(prompt, dict) and "encoder_prompt" in prompt:
        enc_prompt = prompt["encoder_prompt"]  # type: ignore[typeddict-item]
        dec_prompt = prompt["decoder_prompt"]  # type: ignore[typeddict-item]
    else:
        enc_prompt = prompt
        dec_prompt = None

    return EncoderDecoderDictPrompt(
        encoder_prompt=_parse_enc_prompt(enc_prompt),
        decoder_prompt=None if dec_prompt is None else _parse_dec_prompt(dec_prompt),
    )


# SOURCE: vllm/renderers/inputs/preprocess.py:L235-L239 parse_model_prompt
def parse_model_prompt(model_config: "ModelConfig", prompt: object):
    if model_config.is_encoder_decoder:
        return parse_enc_dec_prompt(prompt)

    return parse_dec_only_prompt(prompt)


# SOURCE: vllm/renderers/inputs/preprocess.py:L242-L245 PromptComponents
class PromptComponents:
    # SOURCE: vllm/renderers/inputs/preprocess.py:L242-L245 NamedTuple fields
    def __init__(self, text=None, token_ids=None, embeds=None):  # seam of the NamedTuple
        self.text = text
        self.token_ids = token_ids
        self.embeds = embeds

    def __iter__(self):  # SOURCE: vllm/renderers/inputs/preprocess.py NamedTuple unpack
        return iter((self.text, self.token_ids, self.embeds))

    def __eq__(self, other):  # SOURCE: NamedTuple equality (seam)
        return (self.text, self.token_ids, self.embeds) == (
            other.text, other.token_ids, other.embeds
        )


# SOURCE: vllm/renderers/inputs/preprocess.py:L248-L253 extract_target_prompt
def extract_target_prompt(model_config: "ModelConfig", prompt: object):
    return (
        parse_enc_dec_prompt(prompt)["encoder_prompt"]
        if model_config.is_encoder_decoder
        else parse_dec_only_prompt(prompt)
    )


# SOURCE: vllm/renderers/inputs/preprocess.py:L256-L266 extract_prompt_components
def extract_prompt_components(
    model_config: "ModelConfig",
    prompt: "PromptType | EngineInput",
) -> PromptComponents:
    target_prompt = extract_target_prompt(model_config, prompt)

    return PromptComponents(
        text=target_prompt.get("prompt"),
        token_ids=target_prompt.get("prompt_token_ids"),
        embeds=target_prompt.get("prompt_embeds"),
    )


# SOURCE: vllm/renderers/inputs/preprocess.py:L269-L278 extract_prompt_len
def extract_prompt_len(
    model_config: "ModelConfig",
    prompt: "PromptType | EngineInput",
):
    target_prompt = extract_target_prompt(model_config, prompt)

    return length_from_prompt_token_ids_or_embeds(
        target_prompt.get("prompt_token_ids"),
        target_prompt.get("prompt_embeds"),
    )


# SOURCE: vllm/renderers/inputs/tokenize.py:L11-L57 TokPrompt aliases
DecoderOnlyTokPrompt: TypeAlias = TokensPrompt | EmbedsPrompt
EncoderTokPrompt: TypeAlias = TokensPrompt
DecoderTokPrompt: TypeAlias = TokensPrompt


# SOURCE: vllm/renderers/inputs/tokenize.py:L32-L41 EncoderDecoderTokPrompt
class EncoderDecoderTokPrompt(TypedDict):
    encoder_prompt: EncoderTokPrompt
    decoder_prompt: DecoderTokPrompt | None


# SOURCE: vllm/renderers/inputs/tokenize.py:L44-L50 SingletonTokPrompt
SingletonTokPrompt: TypeAlias = (
    DecoderOnlyTokPrompt | EncoderTokPrompt | DecoderTokPrompt
)
# SOURCE: vllm/renderers/inputs/tokenize.py:L53-L57 TokPrompt
TokPrompt: TypeAlias = DecoderOnlyTokPrompt | EncoderDecoderTokPrompt


# ============================================================================
# §3 EngineInput family — vllm/inputs/engine.py (real, whole)
# ============================================================================


# SOURCE: vllm/inputs/engine.py:L18-L28 _InputOptions
class _InputOptions(TypedDict):
    """
    Additional options available to all
    [`SingletonInput`][vllm.inputs.engine.SingletonInput] types.
    """

    arrival_time: NotRequired[float]
    """The time when the input was received (before rendering)."""

    cache_salt: NotRequired[str]
    """Optional cache salt to be used for prefix caching."""


# SOURCE: vllm/inputs/engine.py:L31-L50 TokensInput
class TokensInput(_InputOptions):
    """Represents token-based input to the engine."""

    type: Literal["token"]
    """The type of input."""

    prompt_token_ids: list[int]
    """The token IDs of the prompt."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token IDs, if available."""

    # SUBTRACTED: prompt_token_offsets field (engine.py:L43-L45) — offsets
    #   feature, delete item 3.
    # SUBTRACTED: assistant_tokens_mask field (engine.py:L47-L50) — chat
    #   template {% generation %} feature, off the downlink spine.


# SOURCE: vllm/inputs/engine.py:L53-L70 tokens_input
def tokens_input(
    prompt_token_ids: list[int],
    *,
    prompt: str | None = None,
    cache_salt: str | None = None,
) -> TokensInput:
    """
    Construct [`TokensInput`][vllm.inputs.engine.TokensInput]
    from optional values.
    """
    inputs = TokensInput(type="token", prompt_token_ids=prompt_token_ids)

    if prompt is not None:
        inputs["prompt"] = prompt
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt

    return inputs


# SOURCE: vllm/inputs/engine.py:L73-L96 EmbedsInput
class EmbedsInput(_InputOptions):
    """Represents embeddings-based input to the engine."""

    type: Literal["embeds"]
    """The type of input."""

    prompt_embeds: "torch.Tensor"
    """The embeddings of the prompt."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token IDs, if available."""

    prompt_token_ids: NotRequired[list[int]]
    """Token IDs of the rendered prompt. Only set for mixed-mode inputs
    (chat completion with `prompt_embeds` content parts)."""

    is_token_ids: NotRequired[list[bool]]
    """Per-position mask for mixed-mode inputs."""


# SOURCE: vllm/inputs/engine.py:L99-L122 embeds_input
def embeds_input(
    prompt_embeds: "torch.Tensor",
    *,
    prompt: str | None = None,
    cache_salt: str | None = None,
    prompt_token_ids: list[int] | None = None,
    is_token_ids: list[bool] | None = None,
) -> EmbedsInput:
    """
    Construct [`EmbedsInput`][vllm.inputs.engine.EmbedsInput]
    from optional values.
    """
    inputs = EmbedsInput(type="embeds", prompt_embeds=prompt_embeds)

    if prompt is not None:
        inputs["prompt"] = prompt
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt
    if prompt_token_ids is not None:
        inputs["prompt_token_ids"] = prompt_token_ids
    if is_token_ids is not None:
        inputs["is_token_ids"] = is_token_ids

    return inputs


# SOURCE: vllm/inputs/engine.py:L125-L134 MultiModalHashes / MultiModalPlaceholders
MultiModalHashes: TypeAlias = Mapping[str, list[str]]
"""
A dictionary containing per-item hashes for each modality.
"""
MultiModalPlaceholders: TypeAlias = Mapping[str, Sequence["PlaceholderRange"]]
"""
A dictionary containing per-item placeholder ranges for each modality.
"""


# SOURCE: vllm/inputs/engine.py:L137-L164 MultiModalInput
class MultiModalInput(_InputOptions):
    """Represents multi-modal input to the engine."""

    type: Literal["multimodal"]
    """The type of input."""

    prompt_token_ids: list[int]
    """The processed token IDs which includes placeholder tokens."""

    prompt: NotRequired[str]
    """The prompt text corresponding to the token IDs, if available."""

    mm_kwargs: "MultiModalKwargsOptionalItems"
    """Keyword arguments to be directly passed to the model after batching."""

    mm_hashes: MultiModalHashes
    """The hashes of the multi-modal data."""

    mm_placeholders: MultiModalPlaceholders
    """
    For each modality, information about the placeholder tokens in
    `prompt_token_ids`.
    """
    # SUBTRACTED: assistant_tokens_mask field (engine.py:L161-L164) — off the
    #   downlink spine.


# SOURCE: vllm/inputs/engine.py:L167-L189 mm_input
def mm_input(
    prompt_token_ids: list[int],
    mm_kwargs: "MultiModalKwargsOptionalItems",
    mm_hashes: MultiModalHashes,
    mm_placeholders: MultiModalPlaceholders,
    *,
    prompt: str | None = None,
    cache_salt: str | None = None,
) -> MultiModalInput:
    inputs = MultiModalInput(
        type="multimodal",
        prompt_token_ids=prompt_token_ids,
        mm_kwargs=mm_kwargs,
        mm_hashes=mm_hashes,
        mm_placeholders=mm_placeholders,
    )

    if prompt is not None:
        inputs["prompt"] = prompt
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt

    return inputs


# SOURCE: vllm/inputs/engine.py:L192-L231 MultiModalEncDecInput + mm_enc_dec_input
class MultiModalEncDecInput(MultiModalInput):
    """
    Represents multi-modal input to the engine for encoder-decoder models.
    """

    encoder_prompt_token_ids: list[int]
    """The processed token IDs of the encoder prompt."""

    encoder_prompt: NotRequired[str]
    """The prompt text corresponding to the encoder token IDs, if available."""


# SOURCE: vllm/inputs/engine.py:L209-L231 mm_enc_dec_input
def mm_enc_dec_input(
    encoder_inputs: MultiModalInput,
    decoder_prompt_token_ids: list[int],
    *,
    decoder_prompt: str | None = None,
) -> MultiModalEncDecInput:
    inputs = MultiModalEncDecInput(
        type="multimodal",
        prompt_token_ids=decoder_prompt_token_ids,
        encoder_prompt_token_ids=encoder_inputs["prompt_token_ids"],
        mm_kwargs=encoder_inputs["mm_kwargs"],
        mm_hashes=encoder_inputs["mm_hashes"],
        mm_placeholders=encoder_inputs["mm_placeholders"],
    )

    if decoder_prompt is not None:
        inputs["prompt"] = decoder_prompt
    if "prompt" in encoder_inputs:
        inputs["encoder_prompt"] = encoder_inputs["prompt"]
    if "cache_salt" in encoder_inputs:
        inputs["cache_salt"] = encoder_inputs["cache_salt"]

    return inputs


# SOURCE: vllm/inputs/engine.py:L234-L238 DecoderOnlyEngineInput
DecoderOnlyEngineInput: TypeAlias = TokensInput | EmbedsInput | MultiModalInput
# SOURCE: vllm/inputs/engine.py:L241-L245 EncoderInput
EncoderInput: TypeAlias = TokensInput | MultiModalEncDecInput
# SOURCE: vllm/inputs/engine.py:L248-L252 DecoderEngineInput
DecoderEngineInput: TypeAlias = TokensInput | MultiModalInput


# SOURCE: vllm/inputs/engine.py:L255-L270 EncoderDecoderInput
class EncoderDecoderInput(TypedDict):
    """
    A rendered [`EncoderDecoderPrompt`][vllm.inputs.llm.EncoderDecoderPrompt]
    which can be passed to `LLMEngine.add_request` or `AsyncLLM.add_request`.
    """

    type: Literal["enc_dec"]

    encoder_prompt: EncoderInput
    """The inputs for the encoder portion."""

    decoder_prompt: DecoderEngineInput
    """The inputs for the decoder portion."""

    arrival_time: NotRequired[float]
    """The time when the input was received (before rendering)."""


# SOURCE: vllm/inputs/engine.py:L273-L277 SingletonInput
SingletonInput: TypeAlias = DecoderOnlyEngineInput | MultiModalEncDecInput
# SOURCE: vllm/inputs/engine.py:L280-L284 EngineInput
EngineInput: TypeAlias = DecoderOnlyEngineInput | EncoderDecoderInput


# SOURCE: vllm/inputs/engine.py:L287-L302 _validate_enc_input
def _validate_enc_input(enc_input: SingletonInput) -> EncoderInput:
    if enc_input["type"] == "embeds":
        raise VLLMValidationError(
            "Embedding inputs are not supported for encoder-decoder models"
        )

    if (
        enc_input["type"] == "multimodal"
        and "encoder_prompt_token_ids" not in enc_input
    ):
        raise RuntimeError(
            "You should register an encoder-decoder multi-modal processor "
            "for encoder-decoder models."
        )

    return enc_input  # type: ignore[return-value]


# SOURCE: vllm/inputs/engine.py:L305-L311 _validate_dec_input
def _validate_dec_input(dec_input: SingletonInput) -> DecoderEngineInput:
    if dec_input["type"] == "embeds":
        raise VLLMValidationError(
            "Embedding inputs are not supported for encoder-decoder models"
        )

    return dec_input


# SOURCE: vllm/inputs/engine.py:L314-L328 _prepare_decoder_input_ids_for_generation
def _prepare_decoder_input_ids_for_generation(
    decoder_input_ids: list[int],
    decoder_start_token_id: int,
) -> list[int]:
    """
    Prepare `decoder_input_ids` for generation with encoder-decoder models,
    according to `GenerationMixin._prepare_decoder_input_ids_for_generation()`.
    """
    if len(decoder_input_ids) == 0 or decoder_input_ids[0] != decoder_start_token_id:
        decoder_input_ids = [decoder_start_token_id] + decoder_input_ids

    return decoder_input_ids


# SOURCE: vllm/inputs/engine.py:L331-L378 build_enc_dec_input
def build_enc_dec_input(
    encoder_input: SingletonInput,
    decoder_input: SingletonInput | None,
    decoder_start_token_id: int,
    skip_decoder_start_token: bool = False,
) -> EncoderDecoderInput:
    enc_input = _validate_enc_input(encoder_input)

    if decoder_input is None:
        dec_input: DecoderEngineInput = enc_input
    else:
        dec_input = _validate_dec_input(decoder_input)

    enc_input_new: EncoderInput
    dec_input_new: DecoderEngineInput

    if enc_input["type"] == "multimodal":
        enc_input_new = tokens_input(
            enc_input["encoder_prompt_token_ids"],
            prompt=enc_input.get("encoder_prompt"),
        )
        dec_input_new = mm_input(
            prompt_token_ids=dec_input["prompt_token_ids"],
            prompt=dec_input.get("prompt"),
            mm_kwargs=enc_input["mm_kwargs"],
            mm_hashes=enc_input["mm_hashes"],
            mm_placeholders=enc_input["mm_placeholders"],
        )
    elif enc_input["type"] == "token":
        enc_input_new = tokens_input(prompt_token_ids=[])
        dec_input_new = dec_input
    else:
        assert_never(enc_input)

    if not skip_decoder_start_token:
        dec_input_new["prompt_token_ids"] = _prepare_decoder_input_ids_for_generation(
            dec_input_new["prompt_token_ids"],
            decoder_start_token_id,
        )

    if cache_salt := enc_input.get("cache_salt"):
        dec_input_new["cache_salt"] = cache_salt

    return EncoderDecoderInput(
        type="enc_dec",
        encoder_prompt=enc_input_new,
        decoder_prompt=dec_input_new,
    )


# SOURCE: vllm/inputs/engine.py:L381-L387 split_enc_dec_input
def split_enc_dec_input(
    inputs: EngineInput,
) -> tuple[SingletonInput | None, SingletonInput]:
    if inputs["type"] == "enc_dec":
        return inputs["encoder_prompt"], inputs["decoder_prompt"]

    return None, inputs


# ============================================================================
# §4 Renderer params — vllm/renderers/params.py (real, whole)
# ============================================================================


# SOURCE: vllm/renderers/params.py:L25 _S TypeVar
_S = TypeVar("_S", list[int], "torch.Tensor")


# SOURCE: vllm/renderers/params.py:L28-L40 merge_kwargs
def merge_kwargs(
    defaults: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
    /,
    *,
    unset_values: tuple[object, ...] = (None, "auto"),
) -> dict[str, Any]:
    if defaults is None:
        defaults = {}
    if overrides is None:
        overrides = {}

    return defaults | {k: v for k, v in overrides.items() if v not in unset_values}


# SOURCE: vllm/renderers/params.py:L43-L68 recursively_merge_kwargs
def recursively_merge_kwargs(
    defaults: dict[str, Any] | None,
    overrides: dict[str, Any] | None,
    /,
    *,
    unset_values: tuple[object, ...] = (None, "auto"),
) -> dict[str, Any]:
    if defaults is None:
        defaults = {}
    if overrides is None:
        overrides = {}

    merged = dict(defaults)

    for k, v in overrides.items():
        if v in unset_values:
            continue

        if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
            merged[k] = recursively_merge_kwargs(
                merged[k], v, unset_values=unset_values
            )
        else:
            merged[k] = v

    return merged


# SOURCE: vllm/renderers/params.py:L71-L137 ChatParams
@dataclass(frozen=True)
class ChatParams:
    """Configuration to control how to parse chat messages."""

    chat_template: str | None = None
    """The chat template to apply."""

    chat_template_content_format: "ChatTemplateContentFormatOption" = "auto"
    """The format of the chat template."""

    chat_template_kwargs: dict[str, Any] = field(default_factory=dict)
    """The kwargs to pass to the chat template."""

    media_io_kwargs: dict[str, dict[str, Any]] | None = None
    """Per-modality kwargs for media I/O (loading/decoding images, videos, etc.)."""

    mm_processor_kwargs: dict[str, Any] | None = None
    """The kwargs to pass to the multi-modal processor."""

    return_assistant_tokens_mask: bool = False
    """Request a per-token assistant mask from apply_chat_template."""

    tool_choice: Any | None = None
    """Request-level tool choice for renderers that need API metadata."""

    response_format: Any | None = None
    """Request-level response format for renderers that need API metadata."""

    # SOURCE: vllm/renderers/params.py:L99-L130 ChatParams.with_defaults
    def with_defaults(
        self,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        default_media_io_kwargs: dict[str, dict[str, Any]] | None = None,
        default_mm_processor_kwargs: dict[str, Any] | None = None,
    ):
        if (
            not default_chat_template_kwargs
            and not default_media_io_kwargs
            and not default_mm_processor_kwargs
        ):
            return self

        return ChatParams(
            chat_template=self.chat_template,
            chat_template_content_format=self.chat_template_content_format,
            chat_template_kwargs=merge_kwargs(
                default_chat_template_kwargs,
                self.chat_template_kwargs,
            ),
            media_io_kwargs=merge_media_io_kwargs(
                default_media_io_kwargs,
                self.media_io_kwargs,
            ),
            mm_processor_kwargs=recursively_merge_kwargs(
                default_mm_processor_kwargs,
                self.mm_processor_kwargs,
            ),
            return_assistant_tokens_mask=self.return_assistant_tokens_mask,
            tool_choice=self.tool_choice,
            response_format=self.response_format,
        )

    # SOURCE: vllm/renderers/params.py:L132-L137 get_apply_chat_template_kwargs
    def get_apply_chat_template_kwargs(self) -> dict[str, Any]:
        """The arguments to pass to `tokenizer.apply_chat_template`."""
        return merge_kwargs(
            self.chat_template_kwargs,
            dict(chat_template=self.chat_template, return_dict=False),
        )


# SOURCE: vllm/renderers/params.py:L140-L199 TokenizeParams fields
@dataclass(frozen=True)
class TokenizeParams:
    """Configuration to control how prompts are tokenized."""

    max_total_tokens: int | None
    """
    Maximum allowed number of input + output tokens.

    Usually, this refers to the model's context length.
    """

    max_output_tokens: int = 0
    """Maximum requested number of output tokens."""

    pad_prompt_tokens: int | None = None
    """
    Number of tokens to pad to:
    - `None` means no padding.
    - `-1` maps to `max_input_tokens`.
    """

    truncate_prompt_tokens: int | None = None
    """
    Number of tokens to keep:
    - `None` means no truncation.
    - `-1` maps to `max_input_tokens`.
    """

    truncation_side: Literal["left", "right"] | None = None
    """
    Which side to truncate from when ``truncate_prompt_tokens`` is active.
    """

    do_lower_case: bool = False
    """Whether to normalize text to lower case before tokenization."""

    add_special_tokens: bool = True
    """Whether to add special tokens."""

    return_token_offsets: bool = False
    """If true, request char-level (start, end) offsets per token."""

    needs_detokenization: bool = False
    """
    Whether the tokenized prompt needs to contain the original text.
    """

    max_total_tokens_param: str = "max_total_tokens"
    """Override this to edit the message for validation errors."""

    max_output_tokens_param: str = "max_output_tokens"
    """Override this to edit the message for validation errors."""

    truncate_prompt_tokens_param: str = "truncate_prompt_tokens"
    """Override this to edit the message for validation errors."""

    # SOURCE: vllm/renderers/params.py:L204-L210 max_input_tokens property
    @property
    # SOURCE: vllm/renderers/params.py:L204-L210 max_input_tokens property
    def max_input_tokens(self) -> int | None:
        """Maximum allowed number of input tokens."""
        if self.max_total_tokens is None:
            return None

        return self.max_total_tokens - self.max_output_tokens

    # SOURCE: vllm/renderers/params.py:L212-L251 __post_init__
    def __post_init__(self) -> None:
        max_total_tokens = self.max_total_tokens
        max_output_tokens = self.max_output_tokens
        max_input_tokens = self.max_input_tokens
        truncate_prompt_tokens = self.truncate_prompt_tokens

        if self.truncation_side not in (None, "left", "right"):
            raise VLLMValidationError(
                "`truncation_side` must be either 'left' or 'right'.",
                parameter="truncation_side",
                value=self.truncation_side,
            )

        if (
            max_output_tokens is not None
            and max_total_tokens is not None
            and max_output_tokens > max_total_tokens
        ):
            raise VLLMValidationError(
                f"{self.max_output_tokens_param}={max_output_tokens} "
                f"cannot be greater than "
                f"{self.max_total_tokens_param}={max_total_tokens=}. "
                f"Please request fewer output tokens.",
                parameter=self.max_output_tokens_param,
                value=max_output_tokens,
            )

        if (
            max_input_tokens is not None
            and truncate_prompt_tokens is not None
            and truncate_prompt_tokens > max_input_tokens
        ):
            raise VLLMValidationError(
                f"{self.truncate_prompt_tokens_param}={truncate_prompt_tokens} "
                f"cannot be greater than {self.max_total_tokens_param} - "
                f"{self.max_output_tokens_param} = {max_input_tokens}. "
                f"Please request a smaller truncation size.",
                parameter=self.truncate_prompt_tokens_param,
                value=truncate_prompt_tokens,
            )

    # SOURCE: vllm/renderers/params.py:L253-L313 with_kwargs
    def with_kwargs(self, **tokenization_kwargs: Any):
        max_length = tokenization_kwargs.pop("max_length", self.max_input_tokens)
        pad_prompt_tokens = tokenization_kwargs.pop(
            "pad_prompt_tokens", self.pad_prompt_tokens
        )
        truncate_prompt_tokens = tokenization_kwargs.pop(
            "truncate_prompt_tokens", self.truncate_prompt_tokens
        )
        truncation_side = tokenization_kwargs.pop(
            "truncation_side", self.truncation_side
        )
        do_lower_case = tokenization_kwargs.pop("do_lower_case", self.do_lower_case)
        add_special_tokens = tokenization_kwargs.pop(
            "add_special_tokens", self.add_special_tokens
        )
        needs_detokenization = tokenization_kwargs.pop(
            "needs_detokenization", self.needs_detokenization
        )

        # https://huggingface.co/docs/transformers/en/pad_truncation
        if padding := tokenization_kwargs.pop("padding", None):
            if padding == "max_length":
                pad_prompt_tokens = max_length
            elif padding in (False, "do_not_pad"):
                pad_prompt_tokens = None
            else:
                # To emit the below warning
                tokenization_kwargs["padding"] = padding

        if truncation := tokenization_kwargs.pop("truncation", None):
            if truncation in (True, "longest_first"):
                truncate_prompt_tokens = max_length
            elif truncation in (False, "do_not_truncate"):
                truncate_prompt_tokens = None
            else:
                # To emit the below warning
                tokenization_kwargs["truncation"] = truncation

        if tokenization_kwargs:
            logger.warning(
                "The following tokenization arguments are not supported "
                "by vLLM Renderer and will be ignored: %s",
                tokenization_kwargs,
            )

        max_total_tokens = self.max_total_tokens

        return TokenizeParams(
            max_total_tokens=max_total_tokens,
            max_output_tokens=(
                0
                if max_total_tokens is None or max_length is None
                else max_total_tokens - max_length
            ),
            pad_prompt_tokens=pad_prompt_tokens,
            truncate_prompt_tokens=truncate_prompt_tokens,
            truncation_side=truncation_side,
            do_lower_case=do_lower_case,
            add_special_tokens=add_special_tokens,
            needs_detokenization=needs_detokenization,
        )

    # SOURCE: vllm/renderers/params.py:L315-L340 get_encode_kwargs
    def get_encode_kwargs(self) -> dict[str, Any]:
        """The arguments to pass to `tokenizer.encode`."""
        max_length = self.truncate_prompt_tokens
        if max_length is not None and max_length < 0:
            max_length = self.max_input_tokens
        elif max_length is None and self.max_input_tokens is not None:
            # This prevents tokenization from taking up more resources than necessary
            # while still failing `self._token_len_check` as expected by users
            max_length = self.max_input_tokens + 1

        # Explicit truncation-side overrides require the full token sequence
        # so we can slice from the requested side in _token_truncation.
        # Disable tokenizer-level truncation because its default side may
        # differ from the requested side.
        if self.truncation_side is not None and self.truncate_prompt_tokens is not None:
            return dict(
                truncation=False,
                add_special_tokens=self.add_special_tokens,
            )

        return dict(
            truncation=max_length is not None,
            max_length=max_length,
            add_special_tokens=self.add_special_tokens,
        )

    # SOURCE: vllm/renderers/params.py:L342-L370 _text_len_check
    def _text_len_check(self, tokenizer, text: str) -> str:
        """Apply length checks to prompt text if necessary."""
        max_input_tokens = self.max_input_tokens
        if max_input_tokens is None or tokenizer is None:
            return text

        max_input_chars = max_input_tokens * tokenizer.max_chars_per_token

        if self.truncate_prompt_tokens is None:
            if len(text) > max_input_chars:
                raise VLLMValidationError(
                    f"This model's maximum context length is "
                    f"{self.max_total_tokens} tokens. However, you requested "
                    f"{self.max_output_tokens} output tokens and your prompt "
                    f"contains {len(text)} characters (more than "
                    f"{max_input_chars} characters, which is the upper bound "
                    f"for {max_input_tokens} input tokens). "
                    f"Please reduce the length of the input prompt or the "
                    f"number of requested output tokens.",
                    parameter="input_text",
                    value=len(text),
                )
        elif self.truncation_side is not None and len(text) > max_input_chars:
            if self.truncation_side == "left":
                text = text[-max_input_chars:]
            else:
                text = text[:max_input_chars]

        return text

    # SOURCE: vllm/renderers/params.py:L372-L374 _text_lowercase
    def _text_lowercase(self, tokenizer, text: str) -> str:
        """Apply lowercase to prompt text if necessary."""
        return text.lower() if self.do_lower_case else text

    # SOURCE: vllm/renderers/params.py:L376-L384 _validate_text
    def _validate_text(self, tokenizer, text: str) -> str:
        """Apply all validators to prompt text."""
        for validator in (
            self._text_len_check,
            self._text_lowercase,
        ):
            text = validator(tokenizer, text)

        return text

    # SOURCE: vllm/renderers/params.py:L386-L399 apply_pre_tokenization
    def apply_pre_tokenization(
        self,
        tokenizer,
        prompt: TextPrompt,
    ) -> TextPrompt:
        """
        Ensure that the prompt meets the requirements set out by this config.
        If that is not possible, raise a `VLLMValidationError`.

        This method is run before tokenization occurs.
        """
        prompt["prompt"] = self._validate_text(tokenizer, prompt["prompt"])

        return prompt

    # SOURCE: vllm/renderers/params.py:L401-L415 _token_padding
    def _token_padding(self, tokenizer, tokens: _S) -> _S:
        """Apply padding to prompt tokens if necessary."""
        pad_length = self.pad_prompt_tokens
        if pad_length is not None and pad_length < 0:
            pad_length = self.max_input_tokens

        if pad_length is None or pad_length <= len(tokens):
            return tokens

        if tokenizer is None:
            raise ValueError("Cannot pad tokens when `skip_tokenizer_init=True`")
        if not isinstance(tokens, list):
            raise ValueError("Cannot pad tokens for embedding inputs")

        return tokens + [tokenizer.pad_token_id] * (pad_length - len(tokens))

    # SOURCE: vllm/renderers/params.py:L417-L434 _token_truncation
    def _token_truncation(self, tokenizer, tokens: _S) -> _S:
        """Apply truncation to prompt tokens if necessary."""
        max_length = self.truncate_prompt_tokens
        if max_length is not None and max_length < 0:
            max_length = self.max_input_tokens

        if max_length is None or max_length >= len(tokens):
            return tokens
        if max_length == 0:
            return tokens[:0]

        side = self.truncation_side or (
            tokenizer.truncation_side if tokenizer is not None else None
        )
        if side == "left":
            return tokens[-max_length:]

        return tokens[:max_length]

    # SOURCE: vllm/renderers/params.py:L436-L461 _token_len_check
    def _token_len_check(self, tokenizer, tokens: _S) -> _S:
        """Apply length checks to prompt tokens if necessary."""
        max_input_tokens = self.max_input_tokens
        if max_input_tokens is None:
            return tokens

        if len(tokens) > max_input_tokens:
            token_count = len(tokens)
            # The tokenizer may have truncated the prompt to
            # max_input_tokens + 1 (see get_encode_kwargs), so the
            # actual prompt length could be larger.
            qualifier = "at least " if token_count == max_input_tokens + 1 else ""
            total = token_count + self.max_output_tokens
            raise VLLMValidationError(
                f"This model's maximum context length is "
                f"{self.max_total_tokens} tokens. However, you requested "
                f"{self.max_output_tokens} output tokens and your prompt "
                f"contains {qualifier}{token_count} input tokens, "
                f"for a total of {qualifier}{total} tokens. "
                f"Please reduce the length of the input prompt or the "
                f"number of requested output tokens.",
                parameter="input_tokens",
                value=token_count,
            )

        return tokens

    # SOURCE: vllm/renderers/params.py:L463-L472 _validate_tokens
    def _validate_tokens(self, tokenizer, tokens: _S) -> _S:
        """Apply all validators to a token sequence."""
        for validator in (
            self._token_padding,
            self._token_truncation,
            self._token_len_check,
        ):
            tokens = validator(tokenizer, tokens)

        return tokens

    # SOURCE: vllm/renderers/params.py:L474-L496 apply_post_tokenization
    def apply_post_tokenization(
        self,
        tokenizer,
        prompt: "TokensPrompt | EmbedsPrompt",
    ) -> "TokensPrompt | EmbedsPrompt":
        """
        Ensure that the prompt meets the requirements set out by this config.
        If that is not possible, raise a `VLLMValidationError`.

        This method is run after tokenization occurs.
        """
        if "prompt_token_ids" in prompt:
            prompt["prompt_token_ids"] = self._validate_tokens(
                tokenizer,
                prompt["prompt_token_ids"],
            )
        if "prompt_embeds" in prompt:
            prompt["prompt_embeds"] = self._validate_tokens(
                tokenizer,
                prompt["prompt_embeds"],
            )

        return prompt


# ============================================================================
# §5 Multimodal structures — vllm/multimodal/inputs.py + utils.py (real)
# ============================================================================


# SOURCE: vllm/multimodal/inputs.py:L221-L229 NestedTensors
NestedTensors: TypeAlias = Union[
    list["NestedTensors"],
    list["torch.Tensor"],
    "torch.Tensor",
    tuple["torch.Tensor", ...],
]
"""
Uses a list instead of a tensor if the dimensions of each element do not match.
"""


# SOURCE: vllm/multimodal/inputs.py:L232-L283 nested_tensors_equal
def nested_tensors_equal(
    a: NestedTensors,
    b: NestedTensors,
    check_dtype: bool = True,
) -> bool:
    """
    Equality check between
    [`NestedTensors`][vllm.multimodal.inputs.NestedTensors] objects.

    If `check_dtype` is `True`, the tensors must have the same dtype.
    """
    check_dtype_func = (
        lambda a, b, check_dtype: a.dtype == b.dtype if check_dtype else True
    )
    if isinstance(a, torch.Tensor):
        return (
            isinstance(b, torch.Tensor)
            and torch.equal(a, b)
            and check_dtype_func(a, b, check_dtype)
        )
    elif isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and torch.equal(b, a)
            and check_dtype_func(b, a, check_dtype)
        )

    if isinstance(a, list):
        return (
            isinstance(b, list)
            and len(a) == len(b)
            and all(nested_tensors_equal(a_, b_, check_dtype) for a_, b_ in zip(a, b))
        )
    if isinstance(b, list):
        return (
            isinstance(a, list)
            and len(b) == len(a)
            and all(nested_tensors_equal(b_, a_, check_dtype) for b_, a_ in zip(b, a))
        )

    if isinstance(a, tuple):
        return (
            isinstance(b, tuple)
            and len(a) == len(b)
            and all(nested_tensors_equal(a_, b_, check_dtype) for a_, b_ in zip(a, b))
        )
    if isinstance(b, tuple):
        return (
            isinstance(a, tuple)
            and len(b) == len(a)
            and all(nested_tensors_equal(b_, a_, check_dtype) for b_, a_ in zip(b, a))
        )

    return False


# SOURCE: vllm/multimodal/inputs.py:L121-L219 PlaceholderRange
@dataclass(frozen=True)
class PlaceholderRange:
    """
    Placeholder location information for multi-modal data.

    Example:

    Prompt: `AAAA BBBB What is in these images?`

    Images A and B will have:

    ```
    A: PlaceholderRange(offset=0, length=4)
    B: PlaceholderRange(offset=5, length=4)
    ```
    """

    offset: int
    """The start index of the placeholder in the prompt."""

    length: int
    """The length of the placeholder."""

    is_embed: "torch.Tensor | None" = None
    """
    A boolean mask of shape `(length,)` indicating which positions
    between `offset` and `offset + length` to assign embeddings to.
    """

    # SOURCE: vllm/multimodal/inputs.py:L150-L153 embeds_cumsum cached_property
    @cached_property
    # SOURCE: vllm/multimodal/inputs.py:L150-L153 embeds_cumsum cached_property
    def embeds_cumsum(self) -> list[int] | None:
        # python list so python indexing avoids torch C++ overhead/conversions/deallocs
        return None if self.is_embed is None else self.is_embed.cumsum(dim=0).tolist()

    # SOURCE: vllm/multimodal/inputs.py:L155-L159 get_num_embeds
    def get_num_embeds(self) -> int:
        if self.embeds_cumsum is None:
            return self.length

        return self.embeds_cumsum[-1] if self.embeds_cumsum else 0

    # SOURCE: vllm/multimodal/inputs.py:L161-L180 get_embeds_indices_in_range
    def get_embeds_indices_in_range(
        self, start_idx: int, end_idx: int
    ) -> tuple[int, int]:
        """
        Returns the starting and ending indices of the embeddings of encoder outputs
        in the range of [start_idx, end_idx) in the placeholders.
        """
        if self.embeds_cumsum is None:
            return start_idx, end_idx

        embeds_start_idx = self.embeds_cumsum[start_idx - 1] if start_idx > 0 else 0
        embeds_end_idx = self.embeds_cumsum[end_idx - 1] if end_idx > 0 else 0

        return embeds_start_idx, embeds_end_idx

    # SOURCE: vllm/multimodal/inputs.py:L182-L205 extract_embeds_range
    def extract_embeds_range(self) -> list[tuple[int, int]]:
        """Extract the start and end indices of the embedded region in prompt."""
        if self.is_embed is None:
            return [(self.offset, self.offset + self.length - 1)]

        mask_i = self.is_embed.int()
        starts = torch.nonzero(
            torch.diff(mask_i, prepend=mask_i.new_zeros(1)) == 1
        ).flatten()
        ends = torch.nonzero(
            torch.diff(mask_i, append=mask_i.new_zeros(1)) == -1
        ).flatten()
        ranges = torch.stack((starts, ends), dim=1) + self.offset
        return [tuple(x) for x in ranges.tolist()]

    # SOURCE: vllm/multimodal/inputs.py:L207-L218 __eq__
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False
        if not (self.offset, self.length) == (other.offset, other.length):
            return False

        if self.is_embed is None:
            return other.is_embed is None
        if other.is_embed is None:
            return self.is_embed is None

        return nested_tensors_equal(self.is_embed, other.is_embed)


# SOURCE: vllm/multimodal/inputs.py:L322-L352 MultiModalFeatureSpec
@dataclass
class MultiModalFeatureSpec:
    """
    Represents a single multimodal input with its processed data and metadata.

    Used to track multimodal data through processing and caching.
    A request containing multiple multimodal items will have one
    `MultiModalFeatureSpec` per item.
    """

    data: "MultiModalKwargsItem | None"
    """
    Represents multimodal data for this feature.

    Can be `None` if the item is cached, to skip IPC between API server
    and engine core processes.
    """

    modality: str
    """The input modality, e.g., `"image"`, `"audio"`, `"video"`."""

    identifier: str
    """The hash for caching encoder outputs (with LoRA prefix if applicable)."""

    mm_position: PlaceholderRange
    """
    The location of the `modality` tokens corresponding to this item
    in the prompt, e.g., `PlaceholderRange(offset=2, length=336)`.
    """

    mm_hash: str | None = None
    """The hash for caching processor outputs (without LoRA prefix)."""

    # SOURCE: vllm/multimodal/inputs.py:L354-L365 gather_kwargs
    @staticmethod
    # SOURCE: vllm/multimodal/inputs.py:L354-L365 gather_kwargs
    def gather_kwargs(features: list["MultiModalFeatureSpec"], keys: set[str]):
        kwargs = defaultdict[str, list[NestedTensors]](list)

        for f in features:
            item = f.data
            if item is not None:
                for k in keys:
                    if k in item:
                        kwargs[k].append(item[k].data)

        return dict(kwargs)


# SOURCE: vllm/multimodal/inputs.py MultiModalKwargsItem (opaque item seam)
MultiModalKwargsItem: TypeAlias = Mapping[str, Any]  # HOST SEAM of the item type
# SOURCE: vllm/multimodal/inputs.py MultiModalKwargsOptionalItems
MultiModalKwargsOptionalItems: TypeAlias = Mapping[
    str, Sequence["MultiModalKwargsItem | None"]
]


# SOURCE: vllm/multimodal/utils.py:L145-L165 argsort_mm_positions
def argsort_mm_positions(
    mm_positions: MultiModalPlaceholders,
) -> list[tuple[str, int]]:
    """
    Given a `MultiModalPlaceholders`, output a sequence of keys to
    sort the dictionary by `offset` (starting index in the input sequence)
    in ascending order.

    Returns:
        A list of `(modality, idx)`, which can be used to access an item
        by `mm_positions[modality][idx]`.
    """
    flat_items = (
        (modality, idx, item)
        for modality, items in mm_positions.items()
        for idx, item in enumerate(items)
    )

    sorted_flat_items = sorted(flat_items, key=lambda x: x[2].offset)

    return [(modality, idx) for modality, idx, _ in sorted_flat_items]


# ============================================================================
# §6 mm-processor black box — HOST SEAMs (the per-modality HF processing is
# deliberately out of scope: the chapter only opens up to the SHAPE of
# mm_features and the flatten in InputProcessor). The seam honours the real
# interface the kept code touches: info.parse_mm_data, apply(inputs) ->
# MultiModalInput, and the processor cache's get_and_update_item contract
# (hit -> (None, prompt_updates), miss -> store metadata and return the item).
# ============================================================================


# SOURCE: vllm/multimodal/cache.py:L395-L403 MultiModalProcessorSenderCache init
class MultiModalProcessorCacheItemMetadata:  # HOST SEAM (cache.py item metadata)
    # SOURCE: vllm/multimodal/cache.py MultiModalProcessorCacheItemMetadata(*mm_item)
    def __init__(self, mm_kwargs_item=None, prompt_updates=None):  # HOST SEAM
        self.mm_kwargs_item = mm_kwargs_item
        self.prompt_updates = prompt_updates


# SOURCE: vllm/multimodal/cache.py:L346-L422 BaseMultiModalProcessorCache /
# MultiModalProcessorSenderCache — seam honouring get_and_update_item's
# hit-returns-(None, prompt_updates) contract (cache.py:L410-L422, verbatim shape)
class BaseMultiModalProcessorCache:  # HOST SEAM
    # SOURCE: vllm/multimodal/cache.py:L395-L403 __init__ (LRU seam: plain dict)
    def __init__(self):  # HOST SEAM
        self._cache: dict = {}
        self.hits = 0

    # SOURCE: vllm/multimodal/cache.py:L405-L407 is_cached_item
    def is_cached_item(self, mm_hash: str) -> bool:
        return mm_hash in self._cache

    # SOURCE: vllm/multimodal/cache.py:L410-L422 get_and_update_item —
    # hit -> (None, prompt_updates); miss -> store metadata, return the item
    # SOURCE: vllm/multimodal/cache.py:L410-L422 get_and_update_item —
    def get_and_update_item(self, mm_item, mm_hash: str):
        if (cached_item := self._cache.get(mm_hash)) is not None:
            self.hits += 1
            return None, cached_item.prompt_updates

        assert mm_item is not None, f"Expected a cached item for {mm_hash=}"

        self._cache[mm_hash] = MultiModalProcessorCacheItemMetadata(*mm_item)

        return mm_item

    # SOURCE: vllm/multimodal/cache.py BaseCache.make_stats(delta=True)
    def make_stats(self, delta: bool = False):  # HOST SEAM
        return SimpleNamespace(total=len(self._cache), hits=self.hits)

    # SOURCE: vllm/multimodal/cache.py:L428-L429 clear_cache
    def clear_cache(self) -> None:
        self._cache.clear()

    # SOURCE: vllm/multimodal/cache.py BaseMultiModalProcessorCache.close
    def close(self) -> None:  # HOST SEAM (LRU resource release)
        return None


# SOURCE: vllm/v1/metrics/stats.py:L146-L158 MultiModalCacheStats
class MultiModalCacheStats:  # HOST SEAM (record is verbatim)
    # SOURCE: vllm/v1/metrics/stats.py BaseCacheStats fields
    def __init__(self):  # HOST SEAM
        self.requests = 0
        self.queries = 0
        self.hits = 0
        self.reset = False

    # SOURCE: vllm/v1/metrics/stats.py:L154-L158 MultiModalCacheStats.record
    def record(self, num_queries: int, num_hits: int) -> None:
        """Aggregate request information into the stats."""
        self.requests += 1
        self.queries += num_queries
        self.hits += num_hits


# SOURCE: vllm/multimodal/registry.py MultiModalTimingRegistry (seam)
class MultiModalTimingRegistry:  # HOST SEAM
    # SOURCE: vllm/multimodal/registry.py MultiModalTimingRegistry.__init__
    def __init__(self, observability_config=None):  # HOST SEAM
        self._observability_config = observability_config

    # SOURCE: vllm/multimodal/processing/processor.py TimingContext pass-through
    def get(self, mm_req_id: str):  # HOST SEAM: TimingContext stand-in
        return SimpleNamespace(
            record=lambda *a, **kw: contextlib.nullcontext()
        )


# SOURCE: vllm/multimodal/parse.py:L733 MultiModalUUIDItems
MultiModalUUIDItems: TypeAlias = dict[str, Sequence[str | None]]


# SOURCE: vllm/multimodal/parse.py:L740-L746 parse_mm_uuids
def parse_mm_uuids(mm_uuids: MultiModalUUIDDict | None) -> MultiModalUUIDItems:
    if mm_uuids is None:
        return {}

    return {
        modality: [uuids] if isinstance(uuids, str) else uuids
        for modality, uuids in mm_uuids.items()
    }


# SOURCE: vllm/multimodal/parse.py MultiModalDataItems (dict-of-list seam)
MultiModalDataItems: TypeAlias = dict  # HOST SEAM of the typed items container


# SOURCE: vllm/multimodal/processing/inputs.py:L14-L24 ProcessorInputs
@dataclass
# SOURCE: vllm/multimodal/processing/inputs.py:L14-L24 ProcessorInputs
class ProcessorInputs:
    """
    Represents the keyword arguments to
    [`vllm.multimodal.processing.BaseMultiModalProcessor.apply`][].
    """

    prompt: "str | list[int]"
    mm_data_items: MultiModalDataItems
    mm_uuid_items: MultiModalUUIDItems | None = None
    hf_processor_mm_kwargs: Mapping[str, object] = field(default_factory=dict)
    tokenization_kwargs: Mapping[str, object] = field(default_factory=dict)
    # SUBTRACTED: get_mm_hashes (processing/inputs.py:L26-L70) — hashing lives
    #   in MultiModalHasher; the seam processor hashes deterministically.


# SOURCE: vllm/multimodal/parse.py:L707-L730 MultiModalProcessorInfo.parse_mm_data
class MultiModalProcessorInfo:  # HOST SEAM
    # SOURCE: vllm/multimodal/parse.py MultiModalProcessorInfo ctor (seam)
    def __init__(self, model_config, tokenizer=None):  # HOST SEAM
        self._model_config = model_config
        # SOURCE: vllm/multimodal/processing/processor.py info.default_tok_params
        self.default_tok_params = TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            do_lower_case=False,
            add_special_tokens=True,
        )
        self.skip_prompt_length_check = False  # HOST SEAM default
        self.allowed_mm_limits = {"image": 1, "audio": 1, "video": 1}

    # SOURCE: vllm/multimodal/parse.py:L718-L730 parse_mm_data — normalize the
    # per-modality entries to lists, rejecting unknown modalities
    # SOURCE: vllm/multimodal/parse.py:L718-L730 parse_mm_data — normalize the
    def parse_mm_data(self, mm_data: MultiModalDataDict) -> MultiModalDataItems:
        subparsers = {"audio", "image", "video"}

        mm_items: dict = {}
        for k, v in mm_data.items():
            if k not in subparsers:
                raise ValueError(f"Unsupported modality: {k}")

            mm_items[k] = [v] if isinstance(v, (str, bytes)) else list(v)

        return mm_items


# SOURCE: vllm/multimodal/processing/processor.py:L1762+ BaseMultiModalProcessor.apply —
# HOST SEAM: the HF-processor internals (per-modality encoders, prompt updates
# machinery) are the black box; this seam preserves the observable contract:
# placeholder expansion into prompt_token_ids + per-item kwargs/hashes/
# placeholders + the processor-cache hit -> payload-None behavior.
class BaseMultiModalProcessor:  # HOST SEAM
    # SOURCE: vllm/multimodal/registry.py:L211+ create_processor(...) wiring
    def __init__(self, model_config, tokenizer=None, cache=None):  # HOST SEAM
        mm_config = model_config.multimodal_config or MultimodalConfig()
        self._tokenizer = tokenizer
        self.info = MultiModalProcessorInfo(model_config, tokenizer)
        self.cache = cache
        self._marker_ids = dict(
            mm_config.seam_marker_ids or {"image": 31, "audio": 32}
        )
        self._placeholder_ids = dict(
            mm_config.seam_placeholder_ids or {"image": 40, "audio": 45}
        )
        self._tokens_per_item = dict(
            mm_config.seam_tokens_per_item or {"image": 2, "audio": 3}
        )

    # SOURCE: vllm/multimodal/processing/processor.py apply steps 1-3 (seam)
    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx=None,
    ) -> MultiModalInput:
        """
        Process multi-modal inputs to be used in vLLM.

        The main steps are:

        1. Apply HF Processor on prompt text and multi-modal data together,
           outputting token IDs and processed tensors.
        2. Find and update sequences in the token IDs with placeholder tokens.
           The number of placeholder tokens equals the feature size of the
           multi-modal data outputted by the multi-modal encoder.
        3. Extract information about the placeholder tokens from the
           processed token IDs.
        """
        prompt = inputs.prompt
        if isinstance(prompt, str):
            ids = self._tokenizer(prompt, add_special_tokens=False)["input_ids"]
        else:
            ids = list(prompt)

        mm_kwargs_items: dict[str, list] = {}
        mm_hashes: dict[str, list[str]] = {}
        mm_placeholders: dict[str, list[PlaceholderRange]] = {}

        markers = self._marker_ids
        counters: dict[str, int] = {}
        out_ids: list[int] = []
        for tid in ids:
            modality = None
            for m, marker_id in markers.items():
                items = inputs.mm_data_items.get(m, [])
                if tid == marker_id and counters.get(m, 0) < len(items):
                    modality = m
                    break
            if modality is None:
                out_ids.append(tid)
                continue

            idx = counters.get(modality, 0)
            counters[modality] = idx + 1
            data_item = inputs.mm_data_items[modality][idx]
            length = int(self._tokens_per_item.get(modality, 1))
            base_mm_hash = self._hash_item(data_item)

            payload = data_item
            # SOURCE: vllm/multimodal/cache.py:L410-L416 hit -> (None, updates)
            if self.cache is not None:
                result = self.cache.get_and_update_item(
                    (data_item, length), base_mm_hash
                )
                if isinstance(result, tuple):
                    # cache hit: first element None -> tensor stays front-end
                    payload = result[0]
                else:
                    payload = result[0]

            offset = len(out_ids)
            out_ids.extend([self._placeholder_ids[modality]] * length)
            mm_kwargs_items.setdefault(modality, []).append(payload)
            mm_hashes.setdefault(modality, []).append(base_mm_hash)
            mm_placeholders.setdefault(modality, []).append(
                PlaceholderRange(offset=offset, length=length)
            )

        return mm_input(out_ids, mm_kwargs_items, mm_hashes, mm_placeholders)

    # SOURCE: vllm/multimodal/hashing.py MultiModalHasher (seam: sha1 of repr)
    def _hash_item(self, data_item) -> str:  # HOST SEAM
        return hashlib.sha1(repr(data_item).encode()).hexdigest()[:16]


# SOURCE: vllm/multimodal/processing EncDecMultiModalProcessor (marker seam)
class EncDecMultiModalProcessor(BaseMultiModalProcessor):  # HOST SEAM marker
    # SOURCE: vllm/multimodal/processing skip_decoder_start_token flag
    skip_decoder_start_token = False


# SOURCE: vllm/multimodal/encoder_budget.py MultiModalBudget (seam — the
# budget arithmetic is the mm chapter's product; only the two read fields)
class MultiModalBudget:  # HOST SEAM
    # SOURCE: vllm/multimodal/encoder_budget.py MultiModalBudget ctor
    def __init__(self, vllm_config, mm_registry):  # HOST SEAM
        mm_config = vllm_config.model_config.multimodal_config
        self.encoder_cache_size = (
            1024 if mm_config is None else int(mm_config.encoder_cache_size)
        )
        self.processor = SimpleNamespace(
            info=SimpleNamespace(skip_prompt_length_check=False)
        )

    # SOURCE: vllm/multimodal/encoder_budget.py MultiModalBudget.reset_cache
    def reset_cache(self) -> None:  # HOST SEAM
        return None


# SOURCE: vllm/multimodal/registry.py:L103/L211/L294 registry entry points (seam)
class _MultiModalRegistrySeam:  # HOST SEAM of the global mm registry
    # SOURCE: vllm/multimodal/registry.py:L103 supports_multimodal_inputs
    def supports_multimodal_inputs(self, model_config) -> bool:
        return model_config.multimodal_config is not None

    # SOURCE: vllm/multimodal/registry.py:L294 processor_cache_from_config
    def processor_cache_from_config(self, config):
        mm_config = config.model_config.multimodal_config
        if mm_config is None or mm_config.mm_processor_cache_gb == 0:
            return None
        return BaseMultiModalProcessorCache()

    # SOURCE: vllm/multimodal/registry.py processor_only_cache_from_config
    def processor_only_cache_from_config(self, config):  # HOST SEAM
        return None  # readonly processor is deleted (delete item 5)

    # SOURCE: vllm/multimodal/registry.py:L211 create_processor
    def create_processor(self, model_config, tokenizer=None, cache=None):
        with set_default_torch_num_threads():
            return BaseMultiModalProcessor(model_config, tokenizer=tokenizer, cache=cache)


# SOURCE: vllm/multimodal/__init__.py MULTIMODAL_REGISTRY
MULTIMODAL_REGISTRY = _MultiModalRegistrySeam()  # HOST SEAM
MultiModalRegistry = _MultiModalRegistrySeam  # HOST SEAM (type name)


# SOURCE: vllm/renderers/embed_utils.py:L21-L81 safe_load_prompt_embeds (seam:
# base64 decode + torch.load + shape/hidden-size/dtype checks reduced — the
# client-side payload contract is the embed_utils chapter's domain)
# SOURCE: vllm/renderers/embed_utils.py:L21-L81 safe_load_prompt_embeds (seam:
def safe_load_prompt_embeds(model_config, embed: bytes) -> "torch.Tensor":  # HOST SEAM
    if not model_config.enable_prompt_embeds:
        raise VLLMValidationError(
            "You must set `--enable-prompt-embeds` to input `prompt_embeds`.",
            parameter="prompt_embeds",
        )

    tensor = torch.load(
        BytesIO(embed),
        weights_only=True,
        map_location=torch.device("cpu"),
    )
    if not isinstance(tensor, torch.Tensor):
        raise VLLMValidationError(
            "`prompt_embeds` payload did not deserialize to a torch.Tensor.",
            parameter="prompt_embeds",
        )
    if tensor.dim() > 2:
        tensor = tensor.squeeze(0)
    if tensor.dim() != 2:
        raise VLLMValidationError(
            "`prompt_embeds` must be a 2D tensor of shape "
            f"(num_tokens, hidden_size); got shape {tuple(tensor.shape)}.",
            parameter="prompt_embeds",
        )
    return tensor


# ============================================================================
# §7 BaseRenderer — vllm/renderers/base.py (subtract-only per dossier)
# ============================================================================


# SOURCE: vllm/renderers/base.py:L72 BaseRenderer
class BaseRenderer:
    """Renderer base: owns the tokenizer, the mm processor and the two thread
    pools; drives the four-step pipeline (render -> tokenize -> extras ->
    process_for_engine). The concrete chat-template step (render_messages) is
    provided by the tokenizer-specific subclass (HfRenderer et al.) — the
    template engine is out of this chapter's scope."""

    # SOURCE: vllm/renderers/base.py:L73-L153 BaseRenderer.__init__
    def __init__(self, config: "VllmConfig", tokenizer) -> None:
        self.config = config
        self.model_config = config.model_config
        self.api_process_rank = config.parallel_config._api_process_rank

        self.tokenizer = tokenizer

        # Thread pool executor for blocking tokenizer operations.  The
        # multimodal processor receives a deep-copied tokenizer (see #36557)
        # so it is safe to run tokenization and MM preprocessing concurrently.
        pool_workers = config.model_config.renderer_num_workers
        self._executor = ThreadPoolExecutor(max_workers=pool_workers)

        # Separate single-worker executor so tokenization never queues behind
        # MM preprocessing; must stay single-worker per #38418 (P0/P1 order).
        self._mm_executor: Executor = ThreadPoolExecutor(max_workers=1)

        # Offload tokenization to the thread pool. The sync
        # ``_tokenize_prompt`` already encapsulates the unified ``__call__``
        # path and char-offset extraction, so the async variant is just it
        # offloaded (mirrors ``_process_multimodal_async`` below).
        self._tokenize_prompt_async = make_async(
            self._tokenize_prompt, executor=self._executor
        )
        self._async_tokenizer_decode = make_async(self._decode, executor=self._executor)

        self.mm_processor: "BaseMultiModalProcessor | None" = None
        self._readonly_mm_processor: "BaseMultiModalProcessor | None" = None
        self._mm_cache_stats: MultiModalCacheStats | None = None
        self._clear_mm_cache_async = make_async(
            self.clear_mm_cache, executor=self._mm_executor
        )
        self._process_multimodal_async = make_async(
            self._process_multimodal, executor=self._mm_executor
        )
        self._safe_load_prompt_embeds_async = make_async(
            safe_load_prompt_embeds, executor=self._executor
        )
        # SUBTRACTED: maybe_init_mm_gpu_ipc_pool install (base.py:L114-L122) —
        #   frontend GPU-side mm decoding pool, input-domain detail (delete item 5).
        if MULTIMODAL_REGISTRY.supports_multimodal_inputs(config.model_config):
            mm_processor_cache = MULTIMODAL_REGISTRY.processor_cache_from_config(config)

            with set_default_torch_num_threads():
                self.mm_processor = MULTIMODAL_REGISTRY.create_processor(
                    config.model_config,
                    tokenizer=self.tokenizer,
                    cache=mm_processor_cache,
                )

            if mm_processor_cache:
                self._mm_cache_stats = MultiModalCacheStats()

            # SUBTRACTED: readonly mm processor creation (base.py:L136-L146) —
            #   tokenize-endpoint cache isolation (delete item 5).

            # This is used to generate internal request ID for MM processing
            # It has no relation to the request ID for engine core
            self._mm_req_counter = AtomicCounter()
            self._mm_timing_registry = MultiModalTimingRegistry(
                config.observability_config
            )

    # SOURCE: vllm/renderers/base.py:L155-L160 get_tokenizer
    def get_tokenizer(self):
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise ValueError("Tokenizer not available when `skip_tokenizer_init=True`")

        return tokenizer

    # SOURCE: vllm/renderers/base.py:L162-L163 _decode
    def _decode(self, *args, **kwargs):
        return self.get_tokenizer().decode(*args, **kwargs)

    # SOURCE: vllm/renderers/base.py:L165-L169 get_mm_processor
    def get_mm_processor(self) -> "BaseMultiModalProcessor":
        if self.mm_processor is None:
            raise ValueError("Multi-modal processor not available for text-only models")

        return self.mm_processor

    # SOURCE: vllm/renderers/base.py:L171-L176 mm_processor_cache property
    @property
    # SOURCE: vllm/renderers/base.py:L171-L176 mm_processor_cache property
    def mm_processor_cache(self) -> "BaseMultiModalProcessorCache | None":
        if self.mm_processor is None:
            return None

        return self.mm_processor.cache

    # SOURCE: vllm/renderers/base.py:L178-L185 stat_mm_cache
    def stat_mm_cache(self) -> MultiModalCacheStats | None:
        mm_cache_stats = self._mm_cache_stats
        if mm_cache_stats is None:
            return None

        self._mm_cache_stats = MultiModalCacheStats()

        return mm_cache_stats

    # SOURCE: vllm/renderers/base.py:L187-L193 update_mm_cache_stats
    def update_mm_cache_stats(self) -> None:
        mm_processor_cache = self.mm_processor_cache
        mm_cache_stats = self._mm_cache_stats

        if mm_processor_cache and mm_cache_stats:
            delta = mm_processor_cache.make_stats(delta=True)
            mm_cache_stats.record(delta.total, delta.hits)

    # SOURCE: vllm/renderers/base.py:L195-L201 clear_mm_cache
    def clear_mm_cache(self) -> None:
        mm_processor_cache = self.mm_processor_cache
        if mm_processor_cache is not None:
            mm_processor_cache.clear_cache()

        if self._mm_cache_stats is not None:
            self._mm_cache_stats.reset = True

    # SOURCE: vllm/renderers/base.py:L203-L212 _clear_processor_cache
    @staticmethod
    # SOURCE: vllm/renderers/base.py:L203-L212 _clear_processor_cache
    def _clear_processor_cache(
        processor: "BaseMultiModalProcessor | None",
    ) -> None:
        if processor is None:
            return

        processor_cache = processor.cache
        if processor_cache is not None:
            processor_cache.clear_cache()

    # SUBTRACTED: warmup + _warmup_mm_processor (base.py:L214-L283) — startup
    #   pre-compilation (Jinja2 template compile + dummy-input mm warmup),
    #   not on the request path (delete item 6). The OnlineRenderer.warmup
    #   caller went with it (mechanical consequence, see impl-notes).

    # SOURCE: vllm/renderers/base.py:L285-L288 clear_mm_cache_async
    async def clear_mm_cache_async(self) -> None:
        """Serialize clear_mm_cache through the multimodal executor to avoid
        races with concurrent process_inputs on the mm_processor_cache."""
        await self._clear_mm_cache_async()

    # SOURCE: vllm/renderers/base.py:L290-L301 shutdown
    def shutdown(self) -> None:
        mm_processor_cache = self.mm_processor_cache
        if mm_processor_cache is not None:
            mm_processor_cache.close()

        if executor := getattr(self, "_executor", None):
            executor.shutdown(wait=False)

        if (
            mm_executor := getattr(self, "_mm_executor", None)
        ) is not None and mm_executor is not executor:
            mm_executor.shutdown(wait=False)

    # SOURCE: vllm/renderers/base.py:L303-L310 get_bos_token_id
    def get_bos_token_id(self) -> int | None:
        if self.tokenizer is None:
            logger.warning_once(
                "Using None for BOS token id because tokenizer is not initialized"
            )
            return None

        return self.tokenizer.bos_token_id

    # SOURCE: vllm/renderers/base.py:L312-L319 get_eos_token_id
    def get_eos_token_id(self) -> int | None:
        if self.tokenizer is None:
            logger.warning_once(
                "Using None for EOS token id because tokenizer is not initialized"
            )
            return None

        return self.tokenizer.eos_token_id

    # SOURCE: vllm/renderers/base.py:L321-L340 get_dec_start_token_id
    def get_dec_start_token_id(self) -> int:
        """
        Obtain the decoder start token id employed by an encoder/decoder model,
        raising an error if it is not available.
        """
        dec_start_token_id = getattr(
            self.model_config.hf_config, "decoder_start_token_id", None
        )

        if dec_start_token_id is None:
            logger.warning_once(
                "Falling back on <BOS> for decoder start token id "
                "because decoder start token id is not available."
            )
            dec_start_token_id = self.get_bos_token_id()

        if dec_start_token_id is None:
            raise RuntimeError("Cannot find decoder start token id or <BOS>")

        return dec_start_token_id

    # SOURCE: vllm/renderers/base.py:L342-L355 default_cmpl_tok_params
    @cached_property
    # SOURCE: vllm/renderers/base.py:L342-L355 default_cmpl_tok_params
    def default_cmpl_tok_params(self) -> TokenizeParams:
        mm_processor = self.mm_processor
        if mm_processor is not None:
            return mm_processor.info.default_tok_params

        model_config = self.model_config
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=True,
        )

    # SOURCE: vllm/renderers/base.py:L357-L370 default_chat_tok_params
    @cached_property
    # SOURCE: vllm/renderers/base.py:L357-L370 default_chat_tok_params
    def default_chat_tok_params(self) -> TokenizeParams:
        mm_processor = self.mm_processor
        if mm_processor is not None:
            return mm_processor.info.default_tok_params

        model_config = self.model_config
        encoder_config = model_config.encoder_config or {}

        return TokenizeParams(
            max_total_tokens=model_config.max_model_len,
            do_lower_case=encoder_config.get("do_lower_case", False),
            add_special_tokens=False,
        )

    # Step 1: Convert raw inputs to prompts
    # SOURCE: vllm/renderers/base.py:L373-L381 render_prompt
    def render_prompt(
        self,
        prompt: "DictPrompt | bytes",
    ) -> "DictPrompt":
        if isinstance(prompt, bytes):
            embeds = safe_load_prompt_embeds(self.model_config, prompt)
            prompt = EmbedsPrompt(prompt_embeds=embeds)

        return prompt

    # SOURCE: vllm/renderers/base.py:L383-L390 render_prompts
    def render_prompts(
        self,
        prompts: "Sequence[DictPrompt | bytes]",
    ) -> "list[DictPrompt]":
        if len(prompts) == 0:
            raise ValueError("You must pass at least one prompt")

        return [self.render_prompt(prompt) for prompt in prompts]

    # SOURCE: vllm/renderers/base.py:L392-L402 _render_prompt_async
    async def _render_prompt_async(
        self,
        prompt: "DictPrompt | bytes",
    ) -> "DictPrompt":
        if isinstance(prompt, bytes):
            embeds = await self._safe_load_prompt_embeds_async(
                self.model_config, prompt
            )
            return EmbedsPrompt(prompt_embeds=embeds)

        return prompt

    # SOURCE: vllm/renderers/base.py:L404-L413 render_prompts_async
    async def render_prompts_async(
        self,
        prompts: "Sequence[DictPrompt | bytes]",
    ) -> "list[DictPrompt]":
        if len(prompts) == 0:
            raise ValueError("You must pass at least one prompt")

        return await asyncio.gather(
            *(self._render_prompt_async(prompt) for prompt in prompts)
        )

    # SOURCE: vllm/renderers/base.py:L415-L421 render_messages (abstract step 1)
    def render_messages(
        self,
        messages: "list[ChatCompletionMessageParam]",
        params: ChatParams,
    ) -> "tuple[list[ConversationMessage], DictPrompt]":
        raise NotImplementedError(
            "render_messages is implemented by the tokenizer-specific "
            "renderer subclass (HfRenderer et al.)"
        )

    # SOURCE: vllm/renderers/base.py:L423-L428 render_messages_async
    async def render_messages_async(
        self,
        messages: "list[ChatCompletionMessageParam]",
        params: ChatParams,
    ) -> "tuple[list[ConversationMessage], DictPrompt]":
        return self.render_messages(messages, params)

    # Step 2: Tokenize prompts if necessary
    # SUBTRACTED: _can_produce_offsets / _wants_offsets (base.py:L431-L449) —
    #   return_token_offsets 可选特性 (delete item 3).

    # SOURCE: vllm/renderers/base.py:L451-L470 _build_tokens_prompt
    @staticmethod
    # SOURCE: vllm/renderers/base.py:L451-L470 _build_tokens_prompt
    def _build_tokens_prompt(
        token_ids: Sequence[int],
        prompt: "TextPrompt",
    ) -> "TokensPrompt":
        """Build a TokensPrompt from already-extracted token ids.

        ``offset_mapping`` is the per-token ``(start, end)`` sequence from
        a BatchEncoding; pass it only when offsets were requested, and it
        is attached as ``prompt_token_offsets``.
        """
        # SUBTRACTED: offset_mapping 形参分支 (base.py:L464-L469) — offsets
        #   feature (delete item 3).
        return TokensPrompt(prompt_token_ids=list(token_ids), **prompt)

    # SOURCE: vllm/renderers/base.py:L472-L487 _tokenize_prompt
    def _tokenize_prompt(
        self,
        prompt: TextPrompt,
        params: TokenizeParams,
    ) -> TokensPrompt:
        tokenizer = self.get_tokenizer()
        # SUBTRACTED: want_offsets / return_offsets_mapping 分支
        #   (base.py:L478-L481, delete item 3).
        kwargs = params.get_encode_kwargs()
        encoding = tokenizer(prompt["prompt"], **kwargs)
        return self._build_tokens_prompt(
            encoding["input_ids"],
            prompt,
        )

    # SOURCE: vllm/renderers/base.py:L489-L493 _detokenize_prompt
    def _detokenize_prompt(self, prompt: TokensPrompt) -> TokensPrompt:
        tokenizer = self.get_tokenizer()
        prompt["prompt"] = tokenizer.decode(prompt["prompt_token_ids"])

        return prompt

    # SOURCE: vllm/renderers/base.py:L495-L500 _detokenize_prompt_async
    async def _detokenize_prompt_async(self, prompt: TokensPrompt) -> TokensPrompt:
        prompt["prompt"] = await self._async_tokenizer_decode(
            prompt["prompt_token_ids"]
        )

        return prompt

    # SOURCE: vllm/renderers/base.py:L502-L507 _tokenize_singleton_prompt overload
    @overload
    # SOURCE: vllm/renderers/base.py:L502-L507 _tokenize_singleton_prompt overload
    def _tokenize_singleton_prompt(
        self,
        prompt: "TextPrompt | TokensPrompt",
        params: TokenizeParams,
    ) -> TokensPrompt: ...

    # SOURCE: vllm/renderers/base.py:L509-L514 _tokenize_singleton_prompt overload 2
    @overload
    # SOURCE: vllm/renderers/base.py:L509-L514 _tokenize_singleton_prompt overload 2
    def _tokenize_singleton_prompt(  # type: ignore[misc]
        self,
        prompt: EmbedsPrompt,
        params: TokenizeParams,
    ) -> EmbedsPrompt: ...

    # SOURCE: vllm/renderers/base.py:L516-L536 _tokenize_singleton_prompt body
    def _tokenize_singleton_prompt(
        self,
        prompt: "SingletonDictPrompt",
        params: TokenizeParams,
    ) -> "SingletonTokPrompt":
        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            if not isinstance(prompt.get("prompt"), str):
                raise TypeError(
                    "Expected prompt['prompt'] to be a string before tokenization; "
                    "use 'prompt_token_ids' for token ID inputs"
                )
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]
            prompt = self._tokenize_prompt(prompt, params)

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            prompt = self._detokenize_prompt(prompt)  # type: ignore[arg-type]

        return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]

    # SOURCE: vllm/renderers/base.py:L538-L543 _tokenize_singleton_prompt_async overload
    @overload
    # SOURCE: vllm/renderers/base.py:L538-L543 _tokenize_singleton_prompt_async overload
    async def _tokenize_singleton_prompt_async(
        self,
        prompt: "TextPrompt | TokensPrompt",
        params: TokenizeParams,
    ) -> TokensPrompt: ...

    # SOURCE: vllm/renderers/base.py:L545-L550 overload 2
    @overload
    # SOURCE: vllm/renderers/base.py:L545-L550 overload 2
    async def _tokenize_singleton_prompt_async(  # type: ignore[misc]
        self,
        prompt: EmbedsPrompt,
        params: TokenizeParams,
    ) -> EmbedsPrompt: ...

    # SOURCE: vllm/renderers/base.py:L552-L572 _tokenize_singleton_prompt_async body
    async def _tokenize_singleton_prompt_async(
        self,
        prompt: "SingletonDictPrompt",
        params: TokenizeParams,
    ) -> "SingletonTokPrompt":
        if "prompt_token_ids" not in prompt and "prompt_embeds" not in prompt:
            if not isinstance(prompt.get("prompt"), str):
                raise TypeError(
                    "Expected prompt['prompt'] to be a string before tokenization; "
                    "use 'prompt_token_ids' for token ID inputs"
                )
            prompt = params.apply_pre_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]
            prompt = await self._tokenize_prompt_async(prompt, params)

        if params.needs_detokenization and "prompt" not in prompt:
            if "prompt_token_ids" not in prompt:
                raise RuntimeError("Cannot run detokenization on embeddings")

            prompt = await self._detokenize_prompt_async(prompt)  # type: ignore[arg-type]

        return params.apply_post_tokenization(self.tokenizer, prompt)  # type: ignore[arg-type]

    # SUBTRACTED: _tokenize_enc_dec_prompt / _tokenize_enc_dec_prompt_async
    #   (base.py:L574-L612) — enc-dec 特化 (delete item 7).

    # SOURCE: vllm/renderers/base.py:L614-L622 tokenize_prompt
    def tokenize_prompt(
        self,
        prompt: "DictPrompt",
        params: TokenizeParams,
    ) -> "TokPrompt":
        # SUBTRACTED: "encoder_prompt" -> _tokenize_enc_dec_prompt 分支
        #   (base.py:L619-L620, delete item 7).
        return self._tokenize_singleton_prompt(prompt, params)

    # SOURCE: vllm/renderers/base.py:L624-L629 tokenize_prompts
    def tokenize_prompts(
        self,
        prompts: "Sequence[DictPrompt]",
        params: TokenizeParams,
    ) -> "list[TokPrompt]":
        return [self.tokenize_prompt(prompt, params) for prompt in prompts]

    # SOURCE: vllm/renderers/base.py:L631-L639 tokenize_prompt_async
    async def tokenize_prompt_async(
        self,
        prompt: "DictPrompt",
        params: TokenizeParams,
    ) -> "TokPrompt":
        # SUBTRACTED: "encoder_prompt" 分支 (base.py:L636-L637, delete item 7).
        return await self._tokenize_singleton_prompt_async(prompt, params)

    # SOURCE: vllm/renderers/base.py:L641-L648 tokenize_prompts_async
    async def tokenize_prompts_async(
        self,
        prompts: "Sequence[DictPrompt]",
        params: TokenizeParams,
    ) -> "list[TokPrompt]":
        return await asyncio.gather(
            *(self.tokenize_prompt_async(prompt, params) for prompt in prompts)
        )

    # Step 3: Add extra keys to the prompts
    # SOURCE: vllm/renderers/base.py:L650-L661 _apply_prompt_extras
    def _apply_prompt_extras(
        self,
        prompts: "Sequence[TokPrompt]",
        prompt_extras: dict[str, Any] | None,
    ):
        if not prompt_extras:
            return

        for prompt in prompts:
            target_prompt = extract_target_prompt(self.model_config, prompt)
            target_prompt.update(prompt_extras)  # type: ignore[union-attr]

    # Step 4: Convert to engine inputs
    # SUBTRACTED: _validate_mm_uuids / _process_mm_uuids (base.py:L664-L726) —
    #   mm uuid 机制 (delete item 4).

    # TODO: Remove str and tokenization_kwargs after deprecating InputPreprocessor
    # SOURCE: vllm/renderers/base.py:L729-L767 _process_multimodal
    def _process_multimodal(
        self,
        prompt: "list[int] | str",
        mm_data: MultiModalDataDict,
        mm_uuids: MultiModalUUIDDict | None,
        mm_processor_kwargs: Mapping[str, object] | None,
        tokenization_kwargs: dict[str, Any] | None,
        *,
        skip_mm_cache: bool = False,
    ) -> MultiModalInput:
        # SUBTRACTED: skip_mm_cache -> readonly mm processor 分流
        #   (base.py:L739-L742, delete item 5).
        mm_processor = self.get_mm_processor()

        mm_req_id = f"renderer{self.api_process_rank}-mm-{self._mm_req_counter.inc(1)}"

        mm_data_items = mm_processor.info.parse_mm_data(mm_data)
        mm_uuid_items = parse_mm_uuids(mm_uuids)
        # SUBTRACTED: _process_mm_uuids 调用行 (base.py:L749-L751, delete item 4).

        mm_processor_inputs = ProcessorInputs(
            prompt,
            mm_data_items,
            mm_uuid_items,
            hf_processor_mm_kwargs=mm_processor_kwargs or {},
            tokenization_kwargs=tokenization_kwargs or {},
        )
        mm_timing_ctx = self._mm_timing_registry.get(mm_req_id)

        with set_default_torch_num_threads():
            mm_inputs = mm_processor.apply(mm_processor_inputs, mm_timing_ctx)

        self.update_mm_cache_stats()

        return mm_inputs

    # SOURCE: vllm/renderers/base.py:L769-L803 _process_tokens
    def _process_tokens(
        self,
        prompt: TokensPrompt,
        *,
        skip_mm_cache: bool = False,
    ) -> "TokensInput | MultiModalInput":
        """Process token inputs, with multimodal preprocessing offloaded
        to the shared thread pool in the async variant.
        """
        prompt_token_ids = prompt["prompt_token_ids"]

        engine_input: "TokensInput | MultiModalInput"
        if multi_modal_data := prompt.get("multi_modal_data"):
            engine_input = self._process_multimodal(
                prompt_token_ids,
                multi_modal_data,
                mm_processor_kwargs=prompt.get("mm_processor_kwargs"),
                tokenization_kwargs=None,  # Tokenization already done in Step 2
                mm_uuids=prompt.get("multi_modal_uuids"),
                skip_mm_cache=skip_mm_cache,
            )
        else:
            engine_input = tokens_input(prompt_token_ids)

        if prompt_text := prompt.get("prompt"):
            engine_input["prompt"] = prompt_text
        if cache_salt := prompt.get("cache_salt"):
            engine_input["cache_salt"] = cache_salt
        # SUBTRACTED: prompt_token_offsets 拷贝分支 (base.py:L797-L801) —
        #   offsets feature (delete item 3).

        return engine_input

    # SOURCE: vllm/renderers/base.py:L805-L833 _process_embeds
    def _process_embeds(self, prompt: EmbedsPrompt) -> EmbedsInput:
        if not self.model_config.enable_prompt_embeds:
            raise ValueError(
                "You must set `--enable-prompt-embeds` to input `prompt_embeds`."
            )

        prompt_embeds = prompt["prompt_embeds"]

        # prompt_embeds must be (seq_len, hidden_size), but if the user
        # passes in a batch of size 1, i.e. (1, seq_len, hidden_size),
        # we can unambiguously process the intent by squeezing the batch
        # dimension.
        if prompt_embeds.ndim == 3:
            prompt_embeds = prompt_embeds.squeeze(dim=0)

        if prompt_embeds.ndim != 2:
            raise ValueError("prompt_embeds must be of shape (seq_len, hidden_size).")

        # Tensors must be on CPU for serialization between processes
        # in the MsgpackEncoder. Casting to CPU here ensures that there is no
        # hidden device transfer in the critical path of generation.
        prompt_embeds = prompt_embeds.cpu()

        return embeds_input(
            prompt_embeds=prompt_embeds,
            cache_salt=prompt.get("cache_salt"),
            prompt_token_ids=prompt.get("prompt_token_ids"),
            is_token_ids=prompt.get("prompt_is_token_ids"),
        )

    # SOURCE: vllm/renderers/base.py:L835-L866 _process_tokens_async
    async def _process_tokens_async(
        self,
        prompt: TokensPrompt,
        *,
        skip_mm_cache: bool = False,
    ) -> "TokensInput | MultiModalInput":
        prompt_token_ids = prompt["prompt_token_ids"]

        engine_input: "TokensInput | MultiModalInput"
        if multi_modal_data := prompt.get("multi_modal_data"):
            engine_input = await self._process_multimodal_async(
                prompt_token_ids,
                multi_modal_data,
                mm_processor_kwargs=prompt.get("mm_processor_kwargs"),
                tokenization_kwargs=None,
                mm_uuids=prompt.get("multi_modal_uuids"),
                skip_mm_cache=skip_mm_cache,
            )
        else:
            engine_input = tokens_input(prompt_token_ids)

        if prompt_text := prompt.get("prompt"):
            engine_input["prompt"] = prompt_text
        if cache_salt := prompt.get("cache_salt"):
            engine_input["cache_salt"] = cache_salt
        # SUBTRACTED: prompt_token_offsets 拷贝分支 (base.py:L860-L864, delete item 3).

        return engine_input

    # SOURCE: vllm/renderers/base.py:L868-L877 _process_singleton
    def _process_singleton(
        self,
        prompt: "SingletonTokPrompt",
        *,
        skip_mm_cache: bool = False,
    ) -> SingletonInput:
        if "prompt_embeds" in prompt:
            return self._process_embeds(prompt)  # type: ignore[arg-type]

        return self._process_tokens(prompt, skip_mm_cache=skip_mm_cache)  # type: ignore[arg-type]

    # SOURCE: vllm/renderers/base.py:L879-L888 _process_singleton_async
    async def _process_singleton_async(
        self,
        prompt: "SingletonTokPrompt",
        *,
        skip_mm_cache: bool = False,
    ) -> SingletonInput:
        if "prompt_embeds" in prompt:
            return self._process_embeds(prompt)  # type: ignore[arg-type]

        return await self._process_tokens_async(prompt, skip_mm_cache=skip_mm_cache)  # type: ignore[arg-type]

    # SUBTRACTED: _process_enc_dec / _process_enc_dec_async (base.py:L890-L943)
    #   — enc-dec 特化, build_enc_dec_input 调用位随删 (delete item 7);
    #   InputProcessor 侧 split_enc_dec_input 保留防守.

    # SOURCE: vllm/renderers/base.py:L945-L960 process_for_engine
    def process_for_engine(
        self,
        prompt: "TokPrompt",
        arrival_time: float,
        *,
        skip_mm_cache: bool = False,
    ) -> EngineInput:
        engine_input: EngineInput
        # SUBTRACTED: "encoder_prompt" -> _process_enc_dec 分支
        #   (base.py:L953-L954, delete item 7).
        engine_input = self._process_singleton(prompt, skip_mm_cache=skip_mm_cache)

        engine_input["arrival_time"] = arrival_time

        return engine_input

    # SOURCE: vllm/renderers/base.py:L962-L982 process_for_engine_async
    async def process_for_engine_async(
        self,
        prompt: "TokPrompt",
        arrival_time: float,
        *,
        skip_mm_cache: bool = False,
    ) -> EngineInput:
        engine_input: EngineInput
        # SUBTRACTED: "encoder_prompt" 分支 (base.py:L970-L974, delete item 7).
        engine_input = await self._process_singleton_async(
            prompt, skip_mm_cache=skip_mm_cache
        )

        engine_input["arrival_time"] = arrival_time

        return engine_input

    # Top-level methods
    # SOURCE: vllm/renderers/base.py:L985-L1006 render_cmpl
    def render_cmpl(
        self,
        prompts: "Sequence[DictPrompt | bytes]",
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
        skip_mm_cache: bool = False,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_cmpl_tok_params

        dict_prompts = self.render_prompts(prompts)
        tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        return [
            self.process_for_engine(prompt, arrival_time, skip_mm_cache=skip_mm_cache)
            for prompt in tok_prompts
        ]

    # SOURCE: vllm/renderers/base.py:L1008-L1033 render_cmpl_async
    async def render_cmpl_async(
        self,
        prompts: "Sequence[DictPrompt | bytes]",
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
        skip_mm_cache: bool = False,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_cmpl_tok_params

        dict_prompts = await self.render_prompts_async(prompts)
        tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        return await asyncio.gather(
            *(
                self.process_for_engine_async(
                    p, arrival_time, skip_mm_cache=skip_mm_cache
                )
                for p in tok_prompts
            )
        )

    # SOURCE: vllm/renderers/base.py:L1035-L1069 render_chat
    def render_chat(
        self,
        conversations: "Sequence[list[ChatCompletionMessageParam]]",
        chat_params: ChatParams,
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
        skip_mm_cache: bool = False,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_chat_tok_params

        rendered = [
            self.render_messages(conversation, chat_params)
            for conversation in conversations
        ]

        out_conversations = list()
        dict_prompts = list()
        for conv, prompt in rendered:
            out_conversations.append(conv)
            dict_prompts.append(prompt)

        tok_prompts = self.tokenize_prompts(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        eng_prompts = [
            self.process_for_engine(prompt, arrival_time, skip_mm_cache=skip_mm_cache)
            for prompt in tok_prompts
        ]

        return out_conversations, eng_prompts

    # SOURCE: vllm/renderers/base.py:L1071-L1109 render_chat_async
    async def render_chat_async(
        self,
        conversations: "Sequence[list[ChatCompletionMessageParam]]",
        chat_params: ChatParams,
        tok_params: TokenizeParams | None = None,
        *,
        prompt_extras: dict[str, Any] | None = None,
        skip_mm_cache: bool = False,
    ):
        arrival_time = time.time()

        if tok_params is None:
            tok_params = self.default_chat_tok_params

        rendered = [
            self.render_messages_async(conversation, chat_params)
            for conversation in conversations
        ]

        out_conversations = list()
        dict_prompts = list()
        for conv, prompt in await asyncio.gather(*rendered):
            out_conversations.append(conv)
            dict_prompts.append(prompt)

        tok_prompts = await self.tokenize_prompts_async(dict_prompts, tok_params)

        self._apply_prompt_extras(tok_prompts, prompt_extras)

        eng_prompts = await asyncio.gather(
            *(
                self.process_for_engine_async(
                    p, arrival_time, skip_mm_cache=skip_mm_cache
                )
                for p in tok_prompts
            )
        )

        return out_conversations, eng_prompts


# ============================================================================
# §8 OnlineRenderer — vllm/renderers/online_renderer.py (subtract-only)
# ============================================================================


# SUBTRACTED: _reused_prompt_token_ids (online_renderer.py:L50-L60) —
#   decode-side token reuse (kv_transfer_params 转发 prefill 侧 id),
#   P/D 分离特性 (delete item 2), ch36 领域.


# SOURCE: vllm/renderers/online_renderer.py:L63 OnlineRenderer
class OnlineRenderer:
    """OpenAI 层的渲染门面：chat/completion 请求的参数校验（tool_choice
    可用性、chat 模板信任），委托 BaseRenderer.render_chat_async /
    render_cmpl_async 产 EngineInput。"""

    # SOURCE: vllm/renderers/online_renderer.py:L64-L105 OnlineRenderer.__init__
    def __init__(
        self,
        model_config: "ModelConfig",
        renderer: BaseRenderer,
        *,
        request_logger=None,
        chat_template: str | None,
        chat_template_content_format: "ChatTemplateContentFormatOption",
        trust_request_chat_template: bool = False,
        enable_auto_tools: bool = False,
        exclude_tools_when_tool_choice_none: bool = False,
        tool_parser: str | None = None,
        reasoning_parser: str | None = None,
        default_chat_template_kwargs: dict[str, Any] | None = None,
        log_error_stack: bool = False,
    ) -> None:
        self.model_config = model_config
        self.renderer = renderer
        self.request_logger = request_logger

        self.enable_auto_tools = enable_auto_tools
        self.exclude_tools_when_tool_choice_none = exclude_tools_when_tool_choice_none
        self.use_harmony = model_config.hf_config.model_type == "gpt_oss"
        self.parser = ParserManager.get_parser(
            tool_parser_name=tool_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=enable_auto_tools,
            model_name=model_config.model,
            is_harmony=self.use_harmony,
        )

        self.chat_template = chat_template
        self.chat_template_content_format: "ChatTemplateContentFormatOption" = (
            chat_template_content_format
        )
        self.default_chat_template_kwargs: dict[str, Any] = (
            default_chat_template_kwargs or {}
        )
        self.trust_request_chat_template = trust_request_chat_template

        self.log_error_stack = log_error_stack
        self.supports_browsing = False
        self.supports_code_interpreter = False

    # SUBTRACTED: OnlineRenderer.warmup (online_renderer.py:L108-L115) — the
    #   caller of the deleted BaseRenderer.warmup (delete item 6); it left
    #   with its callee (mechanical consequence, see impl-notes).

    # SOURCE: vllm/renderers/online_renderer.py:L117-L218 OnlineRenderer.render_chat
    async def render_chat(
        self,
        request,
        *,
        skip_mm_cache: bool = False,
    ) -> "tuple[list[ConversationMessage], list[EngineInput]] | ErrorResponse":
        """Core preprocessing logic for chat requests (no model/engine check).

        Called directly by render_chat_request and delegated to by
        OpenAIServingChat.render_chat_request after its engine-aware checks.
        """
        tokenizer = self.renderer.tokenizer

        tool_parser = self.parser.tool_parser_cls if self.parser is not None else None

        # SUBTRACTED: mistral 序列化分支 (online_renderer.py:L138-L143,
        #   maybe_serialize_tool_calls / truncate_tool_call_ids /
        #   validate_request_params) — mistral 特化 (delete item 1).

        # Check if tool parsing is unavailable (common condition)
        tool_parsing_unavailable = (
            tool_parser is None
            and not is_mistral_tokenizer(tokenizer)
            and not self.use_harmony
        )

        # Validate tool_choice when tool parsing is required but unavailable
        if tool_parsing_unavailable and request.tool_choice not in (
            None,
            "none",
        ):
            if request.tool_choice == "auto" and not self.enable_auto_tools:
                # for hf tokenizers, "auto" tools requires
                # --enable-auto-tool-choice and --tool-call-parser
                return self.create_error_response(
                    '"auto" tool choice requires '
                    "--enable-auto-tool-choice and --tool-call-parser to be set"
                )
            elif request.tool_choice != "auto":
                # "required" or named tool requires tool parser
                if isinstance(request.tool_choice, ChatCompletionNamedToolChoiceParam):
                    tool_choice_desc = f'function "{request.tool_choice.function.name}"'
                else:
                    tool_choice_desc = f'"{request.tool_choice}"'
                return self.create_error_response(
                    f"tool_choice={tool_choice_desc} requires "
                    "--tool-call-parser to be set"
                )

        if request.tools is None or (
            request.tool_choice == "none" and self.exclude_tools_when_tool_choice_none
        ):
            tool_dicts = None
        else:
            tool_dicts = [tool.model_dump() for tool in request.tools]

        # SUBTRACTED: GPT-OSS harmony 分支 (online_renderer.py:L202-L216) —
        #   harmony 特化 (delete item 1); common case only.
        # Common case.
        error_check_ret = self.validate_chat_template(
            request_chat_template=request.chat_template,
            chat_template_kwargs=request.chat_template_kwargs,
            trust_request_chat_template=self.trust_request_chat_template,
        )
        if error_check_ret is not None:
            return error_check_ret

        conversation, engine_inputs = await self.preprocess_chat(
            request,
            request.messages,
            default_template=self.chat_template,
            default_template_content_format=self.chat_template_content_format,
            default_template_kwargs=self.default_chat_template_kwargs,
            tool_dicts=tool_dicts,
            parser=self.parser,
            skip_mm_cache=skip_mm_cache,
        )

        return conversation, engine_inputs

    # SUBTRACTED: _make_request_with_harmony (online_renderer.py:L220-L267) —
    #   orphaned harmony helper (delete item 1); only the deleted harmony
    #   branch called it.

    # SOURCE: vllm/renderers/online_renderer.py:L269-L299 OnlineRenderer.render_completion
    async def render_completion(
        self,
        request,
        *,
        skip_mm_cache: bool = False,
    ) -> "list[EngineInput] | ErrorResponse":
        """Core preprocessing logic for completion requests (no model/engine check).

        Called directly by render_completion_request and delegated to by
        OpenAIServingCompletion.render_completion_request after its engine-aware checks.
        """
        # Return error for unsupported features.
        if request.suffix is not None:
            return self.create_error_response("suffix is not currently supported")

        if request.echo and request.prompt_embeds is not None:
            return self.create_error_response("Echo is unsupported with prompt embeds.")

        if request.prompt_logprobs is not None and request.prompt_embeds is not None:
            return self.create_error_response(
                "prompt_logprobs is not compatible with prompt embeds."
            )

        engine_inputs = await self.preprocess_completion(
            request,
            prompt_input=request.prompt,
            prompt_embeds=request.prompt_embeds,
            skip_mm_cache=skip_mm_cache,
        )

        return engine_inputs

    # SOURCE: vllm/renderers/online_renderer.py:L301-L308 create_error_response
    def create_error_response(
        self,
        message: str | Exception,
        err_type: str = "BadRequestError",
        status_code: int = 400,
        param: str | None = None,
    ) -> ErrorResponse:
        return create_error_response(message, err_type, status_code, param)

    # SOURCE: vllm/renderers/online_renderer.py:L310-L329 validate_chat_template
    def validate_chat_template(
        self,
        request_chat_template: str | None,
        chat_template_kwargs: dict[str, Any] | None,
        trust_request_chat_template: bool,
    ) -> "ErrorResponse | None":
        """Copied from GenerateBaseServing._validate_chat_template."""
        if not trust_request_chat_template and (
            request_chat_template is not None
            or (
                chat_template_kwargs
                and chat_template_kwargs.get("chat_template") is not None
            )
        ):
            return self.create_error_response(
                "Chat template is passed with request, but "
                "--trust-request-chat-template is not set. "
                "Refused request with untrusted chat template."
            )
        return None

    # SOURCE: vllm/renderers/online_renderer.py:L331-L345 preprocess_completion
    async def preprocess_completion(
        self,
        request,
        prompt_input: "str | list[str] | list[int] | list[list[int]] | None",
        prompt_embeds: "bytes | list[bytes] | None",
        *,
        skip_mm_cache: bool = False,
    ) -> "list[EngineInput]":
        """Copied from GenerateBaseServing._preprocess_completion."""
        prompts = list()
        if prompt_embeds is not None:  # embeds take higher priority
            prompts.extend(prompt_to_seq(prompt_embeds))
        if prompt_input is not None:
            prompts.extend(prompt_to_seq(prompt_input))
        return await self.preprocess_cmpl(request, prompts, skip_mm_cache=skip_mm_cache)

    # SOURCE: vllm/renderers/online_renderer.py:L347-L377 preprocess_cmpl
    async def preprocess_cmpl(
        self,
        request,
        prompts: "Sequence[PromptType | bytes]",
        *,
        skip_mm_cache: bool = False,
    ) -> "list[EngineInput]":
        """Copied from GenerateBaseServing._preprocess_cmpl."""
        renderer = self.renderer
        model_config = self.model_config

        parsed_prompts = [
            (
                prompt
                if isinstance(prompt, bytes)
                else parse_model_prompt(model_config, prompt)
            )
            for prompt in prompts
        ]
        tok_params = request.build_tok_params(model_config)

        return await renderer.render_cmpl_async(
            parsed_prompts,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
            skip_mm_cache=skip_mm_cache,
        )

    # SOURCE: vllm/renderers/online_renderer.py:L379-L477 preprocess_chat
    async def preprocess_chat(
        self,
        request,
        messages: list,
        default_template: str | None,
        default_template_content_format: "ChatTemplateContentFormatOption",
        default_template_kwargs: dict[str, Any] | None,
        tool_dicts: "list[dict[str, Any]] | None" = None,
        parser=None,
        *,
        skip_mm_cache: bool = False,
    ) -> "tuple[list[ConversationMessage], list[EngineInput]]":
        """Copied from GenerateBaseServing._preprocess_chat."""
        renderer = self.renderer
        mm_config = self.model_config.multimodal_config

        default_template_kwargs = merge_kwargs(
            default_template_kwargs,
            dict(
                tools=tool_dicts,
                tokenize=(
                    is_mistral_tokenizer(renderer.tokenizer)
                    or self.model_config.enable_prompt_embeds
                ),
            ),
        )

        tok_params = request.build_tok_params(self.model_config)
        chat_params = request.build_chat_params(
            default_template, default_template_content_format
        ).with_defaults(
            default_template_kwargs,
            default_media_io_kwargs=(mm_config.media_io_kwargs if mm_config else None),
            default_mm_processor_kwargs=getattr(request, "mm_processor_kwargs", None),
        )

        # SUBTRACTED: decode-side token reuse 分支
        #   (online_renderer.py:L415-L424, delete item 2) — P/D 分离, ch36.
        (conversation,), (engine_input,) = await renderer.render_chat_async(
            [messages],
            chat_params,
            tok_params,
            prompt_extras={
                k: v
                for k in ("mm_processor_kwargs", "cache_salt")
                if (v := getattr(request, k, None)) is not None
            },
            skip_mm_cache=skip_mm_cache,
        )

        # tool parsing is done only if a tool_parser has been set and if
        # tool_choice is not "none" (if tool_choice is "none" but a tool_parser
        # is set, we want to prevent parsing a tool_call hallucinated by the LLM
        #
        # Exception: Mistral grammar-capable tokenizers always call
        # adjust_request — even for tool_choice="none" — so that the grammar
        # factory can prevent special-token leakage.
        if parser is not None:
            tokenizer = renderer.get_tokenizer()
            tool_parser = parser.tool_parser_cls
            tool_choice = getattr(request, "tool_choice", "none")
            is_mistral_grammar_eligible = (
                tool_parser is not None
                and is_mistral_tool_parser(tool_parser)
                and is_mistral_tokenizer(tokenizer)
                and tokenizer.supports_grammar
            )
            should_adjust_request = (
                parser.reasoning_parser_cls is not None
                or tool_choice != "none"
                or is_mistral_grammar_eligible
            )
            if should_adjust_request:
                request = parser(
                    tokenizer,
                    request.tools,
                    model_config=self.model_config,
                    chat_template_kwargs=chat_params.chat_template_kwargs,
                ).adjust_request(
                    request=request,
                )

        return conversation, [engine_input]


# ============================================================================
# §9 InputPreprocessor — vllm/inputs/preprocess.py (real; the deprecated
# raw-prompt fallback path)
# ============================================================================


# SOURCE: vllm/inputs/preprocess.py:L48 InputPreprocessor
class InputPreprocessor:
    # SOURCE: vllm/inputs/preprocess.py:L49-L59 InputPreprocessor.__init__
    def __init__(
        self,
        vllm_config: "VllmConfig",
        renderer: "BaseRenderer | None" = None,
        mm_registry: "MultiModalRegistry" = MULTIMODAL_REGISTRY,
    ) -> None:
        super().__init__()

        self.model_config = vllm_config.model_config
        self.renderer = renderer or renderer_from_config(vllm_config)
        self.mm_registry = mm_registry

    # SOURCE: vllm/inputs/preprocess.py:L61-L63 tokenizer property
    @property
    # SOURCE: vllm/inputs/preprocess.py:L61-L63 tokenizer property
    def tokenizer(self):
        return self.renderer.tokenizer

    # SOURCE: vllm/inputs/preprocess.py:L65-L66 get_tokenizer
    def get_tokenizer(self):
        return self.renderer.get_tokenizer()

    # SOURCE: vllm/inputs/preprocess.py:L68-L88 _tokenize_prompt
    def _tokenize_prompt(
        self,
        prompt: str,
        tokenization_kwargs: "dict[str, Any] | None" = None,
    ) -> list[int]:
        """
        Apply the model's tokenizer to a text prompt, returning the
        corresponding token IDs.
        """
        renderer = self.renderer

        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )

        tok_prompt = renderer._tokenize_singleton_prompt(
            TextPrompt(prompt=prompt),
            tok_params,
        )

        return tok_prompt["prompt_token_ids"]

    # SOURCE: vllm/inputs/preprocess.py:L90-L109 _process_multimodal
    def _process_multimodal(
        self,
        prompt: "str | list[int]",
        mm_data: MultiModalDataDict,
        mm_processor_kwargs: "Mapping[str, object] | None" = None,
        tokenization_kwargs: "dict[str, Any] | None" = None,
        *,
        mm_uuids: MultiModalUUIDDict | None = None,
    ) -> MultiModalInput:
        """
        Apply the model's multi-modal processor to a multi-modal prompt,
        returning the corresponding token IDs and metadata.
        """
        return self.renderer._process_multimodal(
            prompt,
            mm_data,
            mm_uuids=mm_uuids,
            mm_processor_kwargs=mm_processor_kwargs,
            tokenization_kwargs=tokenization_kwargs,
        )

    # SOURCE: vllm/inputs/preprocess.py:L111-L115 _process_embeds
    def _process_embeds(
        self,
        parsed_content: EmbedsPrompt,
    ) -> EmbedsInput:
        return self.renderer._process_embeds(parsed_content)

    # SOURCE: vllm/inputs/preprocess.py:L117-L131 _truncate_inputs
    def _truncate_inputs(
        self, inputs: list[int], tokenization_kwargs: "dict[str, Any] | None" = None
    ) -> list[int]:
        renderer = self.renderer

        tok_params = renderer.default_cmpl_tok_params.with_kwargs(
            **(tokenization_kwargs or {})
        )

        tok_prompt = renderer._tokenize_singleton_prompt(
            TokensPrompt(prompt_token_ids=inputs),
            tok_params,
        )

        return tok_prompt["prompt_token_ids"]

    # SOURCE: vllm/inputs/preprocess.py:L133-L159 _process_tokens
    def _process_tokens(
        self,
        parsed_content: TokensPrompt,
        tokenization_kwargs: "dict[str, Any] | None" = None,
    ) -> "TokensInput | MultiModalInput":
        prompt_token_ids = self._truncate_inputs(
            parsed_content["prompt_token_ids"], tokenization_kwargs
        )

        inputs: "TokensInput | MultiModalInput"
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            inputs = self._process_multimodal(
                prompt_token_ids,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs"),
                tokenization_kwargs=tokenization_kwargs,
                mm_uuids=parsed_content.get("multi_modal_uuids"),
            )
        else:
            inputs = tokens_input(prompt_token_ids)

        if prompt_text := parsed_content.get("prompt"):
            inputs["prompt"] = prompt_text
        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    # SOURCE: vllm/inputs/preprocess.py:L161-L188 _process_text
    def _process_text(
        self,
        parsed_content: TextPrompt,
        tokenization_kwargs: "dict[str, Any] | None" = None,
    ) -> "TokensInput | MultiModalInput":
        prompt_text = parsed_content["prompt"]

        inputs: "TokensInput | MultiModalInput"
        if multi_modal_data := parsed_content.get("multi_modal_data"):
            inputs = self._process_multimodal(
                prompt_text,
                multi_modal_data,
                parsed_content.get("mm_processor_kwargs") or {},
                tokenization_kwargs=tokenization_kwargs,
            )
        else:
            prompt_token_ids = self._tokenize_prompt(
                prompt_text,
                tokenization_kwargs=tokenization_kwargs,
            )
            inputs = tokens_input(prompt_token_ids)

        inputs["prompt"] = prompt_text

        if cache_salt := parsed_content.get("cache_salt"):
            inputs["cache_salt"] = cache_salt

        return inputs

    # SOURCE: vllm/inputs/preprocess.py:L190-L195 _prompt_to_llm_inputs overload
    @overload
    # SOURCE: vllm/inputs/preprocess.py:L190-L195 _prompt_to_llm_inputs overload
    def _prompt_to_llm_inputs(
        self,
        prompt: "EncoderDictPrompt",
        tokenization_kwargs: "dict[str, Any] | None" = None,
    ) -> EncoderInput: ...

    # SOURCE: vllm/inputs/preprocess.py:L197-L202 overload 2
    @overload
    # SOURCE: vllm/inputs/preprocess.py:L197-L202 overload 2
    def _prompt_to_llm_inputs(  # type: ignore[misc]
        self,
        prompt: "DecoderDictPrompt",
        tokenization_kwargs: "dict[str, Any] | None" = None,
    ) -> "DecoderOnlyEngineInput": ...

    # SOURCE: vllm/inputs/preprocess.py:L204-L209 overload 3
    @overload
    # SOURCE: vllm/inputs/preprocess.py:L204-L209 overload 3
    def _prompt_to_llm_inputs(  # type: ignore[misc]
        self,
        prompt: "DecoderOnlyDictPrompt",
        tokenization_kwargs: "dict[str, Any] | None" = None,
    ) -> "DecoderOnlyEngineInput": ...

    # SOURCE: vllm/inputs/preprocess.py:L211-L228 _prompt_to_llm_inputs body
    def _prompt_to_llm_inputs(
        self,
        prompt: "SingletonDictPrompt",
        tokenization_kwargs: "dict[str, Any] | None" = None,
    ) -> SingletonInput:
        if "prompt_embeds" in prompt:
            return self._process_embeds(prompt)  # type: ignore[arg-type]

        if "prompt_token_ids" in prompt:
            return self._process_tokens(prompt)  # type: ignore[arg-type]

        if "prompt" in prompt:
            return self._process_text(
                prompt,  # type: ignore[arg-type]
                tokenization_kwargs=tokenization_kwargs,
            )

        assert_never(prompt)  # type: ignore[arg-type]

    # SOURCE: vllm/inputs/preprocess.py:L230-L262 _process_encoder_decoder_prompt
    def _process_encoder_decoder_prompt(
        self,
        prompt: EncoderDecoderDictPrompt,
        tokenization_kwargs: "dict[str, Any] | None" = None,
    ) -> EncoderDecoderInput:
        encoder_prompt = prompt["encoder_prompt"]
        decoder_prompt = prompt["decoder_prompt"]

        skip_decoder_start_token = False
        if self.renderer.mm_processor is not None:
            if isinstance(self.renderer.mm_processor, EncDecMultiModalProcessor):
                skip_decoder_start_token = (
                    self.renderer.mm_processor.skip_decoder_start_token
                )

        return build_enc_dec_input(
            encoder_input=self._prompt_to_llm_inputs(
                encoder_prompt,
                tokenization_kwargs=tokenization_kwargs,
            ),
            decoder_input=(
                None
                if decoder_prompt is None
                else self._prompt_to_llm_inputs(
                    decoder_prompt,
                    tokenization_kwargs=tokenization_kwargs,
                )
            ),
            decoder_start_token_id=self.renderer.get_dec_start_token_id(),
            skip_decoder_start_token=skip_decoder_start_token,
        )

    # SOURCE: vllm/inputs/preprocess.py:L264-L272 _process_decoder_only_prompt
    def _process_decoder_only_prompt(
        self,
        prompt: "DecoderOnlyDictPrompt",
        tokenization_kwargs: "dict[str, Any] | None" = None,
    ) -> "DecoderOnlyEngineInput":
        return self._prompt_to_llm_inputs(
            prompt,
            tokenization_kwargs=tokenization_kwargs,
        )

    # SOURCE: vllm/inputs/preprocess.py:L274-L291 preprocess
    def preprocess(
        self,
        prompt: PromptType,
        tokenization_kwargs: "dict[str, Any] | None" = None,
    ) -> EngineInput:
        """Preprocess the input prompt."""
        if self.model_config.is_encoder_decoder:
            # Encoder-decoder model requires special mapping of
            # input prompts to encoder & decoder.
            return self._process_encoder_decoder_prompt(
                parse_enc_dec_prompt(prompt),
                tokenization_kwargs,
            )

        return self._process_decoder_only_prompt(
            parse_dec_only_prompt(prompt),
            tokenization_kwargs=tokenization_kwargs,
        )


# SOURCE: vllm/renderers/registry.py:L82-L88 renderer_from_config (seam — the
# tokenizer construction / registry dispatch is the renderer-registry domain;
# tests inject a renderer, the fallback builds the default hf renderer)
def renderer_from_config(config: "VllmConfig", **kwargs) -> BaseRenderer:  # HOST SEAM
    # SOURCE: vllm/renderers/registry.py:L87 RENDERER_REGISTRY.load_renderer
    return BaseRenderer(config, tokenizer=getattr(config, "_seam_tokenizer", None))


# ============================================================================
# §10 EngineCoreRequest — vllm/v1/engine/__init__.py (subtract-only: 3 fields)
# ============================================================================


# SOURCE: vllm/v1/engine/__init__.py:L97-L154 EngineCoreRequest
class EngineCoreRequest(
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    prompt_token_ids: "list[int] | None"
    mm_features: "list[MultiModalFeatureSpec] | None"
    sampling_params: "SamplingParams | None"
    pooling_params: "PoolingParams | None"
    arrival_time: float
    lora_request: "LoRARequest | None"
    cache_salt: "str | None"
    data_parallel_rank: "int | None"
    prompt_embeds: "torch.Tensor | None" = None

    # Per-position mask for mixed-mode inputs (e.g chat completion with
    # prompt_embeds content parts). `True` means the position is a real
    # token ID; `False` means the position uses a pre-computed entry from
    # `prompt_embeds`. `None` for pure-tokens and pure-embeds requests.
    prompt_is_token_ids: "list[bool] | None" = None

    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    client_index: int = 0

    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    current_wave: int = 0
    priority: int = 0

    trace_headers: "Mapping[str, str] | None" = None
    resumable: bool = False

    # The user-provided request ID. This field is set internally,
    # copied from the provided request_id that's originally assigned
    # to the request_id field, see InputProcessor.assign_request_id().
    # Used in outputs and to support abort(req_id, internal=False).
    external_req_id: "str | None" = None

    # SUBTRACTED: reasoning_ended / reasoning_parser_kwargs fields
    #   (vllm/v1/engine/__init__.py:L139-L140) — reasoning parser params,
    #   not used on this chapter's or ch7's main line (delete item 17).
    # SUBTRACTED: abort_immediately field (vllm/v1/engine/__init__.py:L142-L146)
    #   — KV disagg rejection receipt (ch36 domain, delete item 17).

    # SOURCE: vllm/v1/engine/__init__.py:L148-L154 params property
    @property
    # SOURCE: vllm/v1/engine/__init__.py:L148-L154 params property
    def params(self) -> "SamplingParams | PoolingParams":
        """Return the processed params (sampling or pooling)."""
        if self.sampling_params is not None:
            return self.sampling_params
        assert self.pooling_params is not None
        return self.pooling_params


# ============================================================================
# §11 InputProcessor — vllm/v1/engine/input_processor.py (subtract-only)
# ============================================================================


# SOURCE: vllm/v1/engine/input_processor.py:L38 InputProcessor
class InputProcessor:
    # SOURCE: vllm/v1/engine/input_processor.py:L39-L82 InputProcessor.__init__
    def __init__(
        self,
        vllm_config: "VllmConfig",
        renderer: "BaseRenderer | None" = None,
        *,
        mm_registry: "MultiModalRegistry" = MULTIMODAL_REGISTRY,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.structured_outputs_config = vllm_config.structured_outputs_config
        self.observability_config = vllm_config.observability_config
        self.use_v2_model_runner = vllm_config.use_v2_model_runner

        self.generation_config_fields = model_config.try_get_generation_config()

        self.renderer = renderer or renderer_from_config(vllm_config)

        self.supports_mm_inputs = mm_registry.supports_multimodal_inputs(model_config)
        self.mm_encoder_cache_size = 0
        self.skip_prompt_length_check = False
        if self.supports_mm_inputs:
            mm_budget = MultiModalBudget(vllm_config, mm_registry)
            self.mm_encoder_cache_size = mm_budget.encoder_cache_size
            self.skip_prompt_length_check = (
                mm_budget.processor.info.skip_prompt_length_check
            )
            mm_budget.reset_cache()  # Not used anymore

        self.input_preprocessor = InputPreprocessor(
            vllm_config,
            renderer=renderer,
            mm_registry=mm_registry,
        )

        # Raw-prompt preprocessing (tokenization and multimodal processing)
        # is blocking, so async callers should run it on the renderer's
        # thread pool to keep their event loop responsive.
        self.process_inputs_async = make_async(
            self.process_inputs, executor=self.renderer._executor
        )

    # SOURCE: vllm/v1/engine/input_processor.py:L84-L86 tokenizer property
    @property
    # SOURCE: vllm/v1/engine/input_processor.py:L84-L86 tokenizer property
    def tokenizer(self):
        return self.renderer.tokenizer

    # SOURCE: vllm/v1/engine/input_processor.py:L88-L89 get_tokenizer
    def get_tokenizer(self):
        return self.renderer.get_tokenizer()

    # SOURCE: vllm/v1/engine/input_processor.py:L91-L153 _validate_params
    def _validate_params(
        self,
        params: "SamplingParams | PoolingParams",
        supported_tasks: tuple,
    ) -> None:
        """Raise `ValueError` if SamplingParams or PoolingParams is not valid."""
        if isinstance(params, SamplingParams):
            supported_generation_tasks = [
                task for task in supported_tasks if task in GENERATION_TASKS
            ]
            if not supported_generation_tasks:
                raise VLLMValidationError("This model does not support generation")

            params.verify(
                self.model_config,
                self.speculative_config,
                self.structured_outputs_config,
                self.tokenizer,
            )

            # SUBTRACTED: thinking_token_budget / reasoning_config 校验分支
            #   与 use_v2_model_runner 子分支 (input_processor.py:L111-L126) —
            #   reasoning 特性校验 (delete item 9).
        elif isinstance(params, PoolingParams):
            supported_pooling_tasks = [
                task for task in supported_tasks if task in POOLING_TASKS
            ]
            if not supported_pooling_tasks:
                raise VLLMValidationError("This model does not support pooling")

            # SUBTRACTED: PoolingParams task 默认补全 + task 支持校验
            #   (input_processor.py:L134-L146) — pooling task 自动选择次要旁支
            #   (delete item 10); 保留「不支持 pooling 报错 + params.verify」.

            params.verify(self.model_config)
        else:
            raise TypeError(
                f"params must be either SamplingParams or PoolingParams, "
                f"but got {type(params).__name__}"
            )

    # SOURCE: vllm/v1/engine/input_processor.py:L155-L172 _validate_lora
    def _validate_lora(self, lora_request: "LoRARequest | None") -> None:
        if lora_request is None:
            return

        # LoRA request passed in while LoRA is not enabled
        if not self.lora_config:
            raise VLLMValidationError(
                f"Got lora_request {lora_request} but LoRA is not enabled!"
            )

        if self.tokenizer is not None:
            # SUBTRACTED: 冗长 deprecation 警告文案 (input_processor.py:L166-L172)
            #   — 保留分支, 一行提示 (delete item 11).
            logger.warning_once(
                "vLLM has deprecated support for different tokenizers for "
                "different LoRAs."
            )

    # SOURCE: vllm/v1/engine/input_processor.py:L174-L190 _get_mm_identifier
    def _get_mm_identifier(
        self,
        mm_hash: str,
        lora_request: "LoRARequest | None",
    ) -> str:
        """
        When enable_tower_connector_lora is True, multi-modal embeddings
        vary depending on the LoRA request. Therefore, the mm_hash must be
        generated based on the LoRA request to prevent incorrect cache hits.
        """
        if (
            lora_request is None
            or self.lora_config is None
            or not self.lora_config.enable_tower_connector_lora
        ):
            return mm_hash
        return f"{lora_request.lora_name}:{mm_hash}"

    # SUBTRACTED: inject_into_mm_cache (input_processor.py:L192-L229) —
    #   外部预处理张量注入 processor cache 的旁路 (delete item 8).

    # SOURCE: vllm/v1/engine/input_processor.py:L231-L249 assign_request_id
    @staticmethod
    # SOURCE: vllm/v1/engine/input_processor.py:L231-L249 assign_request_id
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
            # SUBTRACTED: 冗长警告文案 (input_processor.py:L243-L247) — 保留
            #   分支, 一行提示 (delete item 11).
            logger.warning_once(
                "VLLM_DISABLE_REQUEST_ID_RANDOMIZATION is set and will be "
                "removed in a future release."
            )
        else:
            request.request_id = f"{request.external_req_id}-{random_uuid():.8}"

    # SOURCE: vllm/v1/engine/input_processor.py:L251-L394 process_inputs
    def process_inputs(
        self,
        request_id: str,
        prompt: "PromptType | EngineInput",
        params: "SamplingParams | PoolingParams",
        supported_tasks: tuple,
        arrival_time: float | None = None,
        lora_request: "LoRARequest | None" = None,
        tokenization_kwargs: "dict[str, Any] | None" = None,
        trace_headers: "Mapping[str, str] | None" = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        resumable: bool = False,
    ) -> EngineCoreRequest:
        self._validate_params(params, supported_tasks)
        self._validate_lora(lora_request)

        parallel_config = self.vllm_config.parallel_config
        dp_size = parallel_config.data_parallel_size
        dp_local_size = parallel_config.data_parallel_size_local
        num_ranks = dp_local_size if parallel_config.local_engines_only else dp_size
        if data_parallel_rank is not None and not (0 <= data_parallel_rank < num_ranks):
            raise VLLMValidationError(
                f"data_parallel_rank {data_parallel_rank} "
                f"is out of range [0, {num_ranks})."
            )

        if isinstance(prompt, dict) and "type" in prompt:
            if tokenization_kwargs:
                # SUBTRACTED: 冗长 deprecation 文案 (input_processor.py:L280-L284)
                #   — 保留分支, 一行提示 (delete item 11).
                logger.warning_once(
                    "Passing tokenization_kwargs to InputProcessor is deprecated "
                    "and will be removed in v0.18."
                )

            if arrival_time is None:
                arrival_time = prompt.get("arrival_time", time.time())  # type: ignore[union-attr]

            processed_inputs: EngineInput = prompt  # type: ignore[assignment]
        else:
            # SUBTRACTED: 冗长 deprecation 文案 (input_processor.py:L291-L295)
            #   — 保留分支, 一行提示 (delete item 11).
            logger.warning_once(
                "Passing raw prompts to InputProcessor is deprecated "
                "and will be removed in v0.18."
            )

            if arrival_time is None:
                arrival_time = time.time()

            processed_inputs = self.input_preprocessor.preprocess(
                prompt,
                tokenization_kwargs=tokenization_kwargs,
            )

        current_platform.validate_request(processed_inputs, params)

        encoder_inputs, decoder_inputs = split_enc_dec_input(processed_inputs)
        self._validate_model_inputs(encoder_inputs, decoder_inputs)

        # Mypy can be conservative for TypedDict unions; normalize access.
        if decoder_inputs["type"] == "embeds":
            prompt_embeds = decoder_inputs["prompt_embeds"]
            prompt_token_ids = decoder_inputs.get("prompt_token_ids")
            prompt_is_token_ids = decoder_inputs.get("is_token_ids")
        else:
            prompt_token_ids = decoder_inputs["prompt_token_ids"]
            prompt_embeds = None
            prompt_is_token_ids = None

        sampling_params = None
        pooling_params = None
        if isinstance(params, SamplingParams):
            # TODO: can we avoid cloning here in multiproc case?
            sampling_params = params.clone()
            # If unset max tokens, then generate up to the max_model_len.
            if sampling_params.max_tokens is None:
                seq_len = length_from_prompt_token_ids_or_embeds(
                    prompt_token_ids, prompt_embeds
                )
                sampling_params.max_tokens = self.model_config.max_model_len - seq_len

            sampling_params.update_from_generation_config(
                self.generation_config_fields,
                self.renderer.get_eos_token_id(),
            )
            if self.tokenizer is not None:
                sampling_params.update_from_tokenizer(self.tokenizer)
        else:
            pooling_params = params.clone()

        # Multimodal related.
        mm_features: "list[MultiModalFeatureSpec] | None" = None

        if decoder_inputs["type"] == "multimodal":
            decoder_mm_inputs = decoder_inputs["mm_kwargs"]
            decoder_mm_positions = decoder_inputs["mm_placeholders"]
            decoder_mm_hashes = decoder_inputs["mm_hashes"]

            if not all(
                isinstance(leaf, str) for leaf in json_iter_leaves(decoder_mm_hashes)
            ):
                # SUBTRACTED: 冗长错误文案 (input_processor.py:L352-L356) —
                #   保留校验语义 (delete item 12 邻域).
                raise ValueError(
                    f"mm_hashes must contain only strings, got: {decoder_mm_hashes}."
                )

            # Merge and flatten multimodal placeholders, hashes and inputs
            # from dictionaries to lists, and sort them by each item's position
            # in the input sequence.
            sorted_mm_idxs = argsort_mm_positions(decoder_mm_positions)

            mm_features = []
            for modality, idx in sorted_mm_idxs:
                base_mm_hash = decoder_mm_hashes[modality][idx]
                mm_features.append(
                    MultiModalFeatureSpec(
                        data=decoder_mm_inputs[modality][idx],
                        modality=modality,
                        identifier=self._get_mm_identifier(
                            base_mm_hash,
                            lora_request,
                        ),
                        mm_position=decoder_mm_positions[modality][idx],
                        mm_hash=base_mm_hash,
                    )
                )

        return EngineCoreRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            prompt_embeds=prompt_embeds,
            prompt_is_token_ids=prompt_is_token_ids,
            mm_features=mm_features,
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            arrival_time=arrival_time,
            lora_request=lora_request,
            cache_salt=decoder_inputs.get("cache_salt"),
            priority=priority,
            data_parallel_rank=data_parallel_rank,
            trace_headers=trace_headers,
            resumable=resumable,
        )

    # SOURCE: vllm/v1/engine/input_processor.py:L396-L441 _validate_prompt_len
    def _validate_prompt_len(
        self,
        prompt_len: int,
        prompt_type: str,
    ):
        if self.skip_prompt_length_check:
            return

        if prompt_len == 0 and prompt_type == "decoder":
            raise VLLMValidationError(f"The {prompt_type} prompt cannot be empty")

        model_config = self.model_config
        max_prompt_len = (
            model_config.max_model_len
            if prompt_type == "decoder"
            else self.mm_encoder_cache_size
        )
        if prompt_len > max_prompt_len:
            # SUBTRACTED: suggestion 文案构造分支 (input_processor.py:L414-L425)
            #   — 错误信息措辞 (delete item 12).
            raise VLLMValidationError(
                f"The {prompt_type} prompt (length {prompt_len}) is "
                f"longer than the maximum model length of {max_prompt_len}."
            )
        elif prompt_len == max_prompt_len and model_config.runner_type == "generate":
            # SUBTRACTED: suggestion 文案构造 (input_processor.py:L432-L436, item 12).
            raise VLLMValidationError(
                f"The {prompt_type} prompt (length {prompt_len}) plus the number of "
                f"requested output tokens (at least 1) is longer than the maximum "
                f"model length of {max_prompt_len}."
            )

    # SOURCE: vllm/v1/engine/input_processor.py:L443-L495 _validate_model_input
    def _validate_model_input(
        self,
        prompt_input: SingletonInput,
        prompt_type: str,
    ) -> None:
        model_config = self.model_config
        tokenizer = self.tokenizer

        prompt_ids = (
            None
            if prompt_input["type"] == "embeds"
            else prompt_input["prompt_token_ids"]
        )
        prompt_embeds = (
            prompt_input["prompt_embeds"] if prompt_input["type"] == "embeds" else None
        )

        prompt_len = length_from_prompt_token_ids_or_embeds(prompt_ids, prompt_embeds)
        self._validate_prompt_len(prompt_len, prompt_type)

        if prompt_input["type"] == "multimodal":
            decoder_mm_positions = prompt_input["mm_placeholders"]
            for modality, mm_positions in decoder_mm_positions.items():
                for mm_position in mm_positions:
                    num_embeds = mm_position.get_num_embeds()
                    if num_embeds > self.mm_encoder_cache_size:
                        raise VLLMValidationError(
                            f"The {prompt_type} prompt contains a(n) {modality} item "
                            f"with {num_embeds} embedding tokens, which exceeds the "
                            f"pre-allocated encoder cache size "
                            f"{self.mm_encoder_cache_size}. Please reduce the input "
                            f"size or increase the encoder cache size "
                            f"by setting --limit-mm-per-prompt at startup."
                        )

        if prompt_ids and tokenizer is not None:
            max_input_id = max(prompt_ids, default=0)

            # SUBTRACTED: Qwen3 vocab 长注释 (input_processor.py:L481-L487) —
            #   注释不影响执行 (delete item 13); 判定逻辑保留:
            # tokenizer.max_token_id is the tokenizer's vocab size while
            # model_config.get_vocab_size() is the model's vocab size — take
            # the max of the two to determine if a token id is truly OOV.
            model_vocab_size = model_config.get_vocab_size()
            if max_input_id > max(tokenizer.max_token_id, model_vocab_size - 1):
                raise VLLMValidationError(
                    f"Token id {max_input_id} is out of vocabulary"
                )

    # SOURCE: vllm/v1/engine/input_processor.py:L497-L505 _validate_model_inputs
    def _validate_model_inputs(
        self,
        encoder_input: "SingletonInput | None",
        decoder_input: SingletonInput,
    ):
        if encoder_input is not None:
            self._validate_model_input(encoder_input, prompt_type="encoder")

        self._validate_model_input(decoder_input, prompt_type="decoder")


# ============================================================================
# §12 AsyncLLM — vllm/v1/engine/async_llm.py (subtract-only: the downlink
# surface; __init__ is a documented HOST SEAM — metrics/tracing/profiler and
# the output_handler loop are ch04/ch07 products)
# ============================================================================


# SOURCE: vllm/v1/engine/async_llm.py:L72 AsyncLLM
class AsyncLLM:
    """An asynchronous wrapper for the vLLM engine."""

    # SOURCE: vllm/v1/engine/async_llm.py:L75-L203 AsyncLLM.__init__ — HOST
    # SEAM: only the downlink wiring (renderer -> InputProcessor ->
    # OutputProcessor -> EngineCore client); the real constructor's metrics
    # managers, tracing, profiler and eager output-handler start are the
    # ch04 domain. Wiring order preserved from L135-L156.
    def __init__(
        self,
        vllm_config: "VllmConfig",
        renderer: "BaseRenderer | None" = None,
        *,
        log_requests: bool = False,
        client_index: int = 0,
        engine_core: "AsyncMPClient | None" = None,
        supported_tasks: tuple = ("generate",),
    ) -> None:  # HOST SEAM
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config

        self.log_requests = log_requests
        self.events: list = []  # HOST SEAM: shared recording bus

        # SOURCE: vllm/v1/engine/async_llm.py:L135-L138 renderer -> InputProcessor
        self.renderer = renderer or renderer_from_config(vllm_config)
        # Convert EngineInput --> EngineCoreRequest.
        self.input_processor = InputProcessor(self.vllm_config, self.renderer)

        # SOURCE: vllm/v1/engine/async_llm.py:L141-L146 OutputProcessor wiring
        # Converts EngineCoreOutputs --> RequestOutput.
        self.output_processor = OutputProcessor(events=self.events)

        # SOURCE: vllm/v1/engine/async_llm.py:L149-L156 EngineCoreClient wiring
        # EngineCore (starts the engine in background process).
        self.engine_core = engine_core or AsyncMPClient(
            events=self.events, client_index=client_index,
            supported_tasks=supported_tasks,
        )

        self.output_handler = None

    # SOURCE: vllm/v1/engine/async_llm.py:L276-L281 get_supported_tasks
    async def get_supported_tasks(self) -> tuple:
        if not hasattr(self, "_supported_tasks"):
            # Cache the result
            self._supported_tasks = await self.engine_core.get_supported_tasks_async()

        return self._supported_tasks

    # SOURCE: vllm/v1/engine/async_llm.py:L283-L300 add_request signature
    async def add_request(
        self,
        request_id: str,
        prompt: "EngineCoreRequest | PromptType | EngineInput | AsyncGenerator[StreamingInput, None]",
        params: "SamplingParams | PoolingParams",
        arrival_time: float | None = None,
        lora_request: "LoRARequest | None" = None,
        tokenization_kwargs: "dict[str, Any] | None" = None,
        trace_headers: "Mapping[str, str] | None" = None,
        priority: int = 0,
        data_parallel_rank: int | None = None,
        prompt_text: str | None = None,
        reasoning_ended: bool | None = None,
        reasoning_parser_kwargs: "dict[str, Any] | None" = None,
    ) -> RequestOutputCollector:
        """Add new request to the AsyncLLM."""

        if self.errored:
            raise EngineDeadError()

        is_pooling = isinstance(params, PoolingParams)

        # SUBTRACTED: kv_sharing_fast_prefill + prompt_logprobs 冲突校验
        #   (async_llm.py:L308-L317) — KV 共享 fast-prefill 配置校验
        #   (delete item 15).

        # SUBTRACTED: 流式输入分支 (async_llm.py:L319-L334,
        #   AsyncGenerator prompt -> _add_streaming_input_request) — 流式输入
        #   与单发下行主线互斥 (delete item 14).

        # Convert Input --> Request.
        if isinstance(prompt, EngineCoreRequest):
            # SUBTRACTED: 冗长 deprecation 文案 (async_llm.py:L338-L350) —
            #   保留两分支, 一行提示 (delete item 11).
            logger.warning_once(
                "Passing EngineCoreRequest to AsyncLLM.generate() and .add_requests() "
                "is deprecated and will be removed in v0.18."
            )

            request = prompt
            if request_id != request.request_id:
                logger.warning_once(
                    "AsyncLLM.add_request() was passed a request_id parameter that "
                    "does not match the EngineCoreRequest.request_id attribute."
                )
        else:
            if isinstance(prompt, dict) and "type" in prompt:
                # Rendered EngineInput; no blocking preprocessing needed.
                request = self.input_processor.process_inputs(
                    request_id,
                    prompt,
                    params,
                    supported_tasks=await self.get_supported_tasks(),
                    arrival_time=arrival_time,
                    lora_request=lora_request,
                    tokenization_kwargs=tokenization_kwargs,
                    trace_headers=trace_headers,
                    priority=priority,
                    data_parallel_rank=data_parallel_rank,
                )
            else:
                # Raw prompts require tokenization and possibly multimodal
                # processing, which must not block the event loop.
                request = await self.input_processor.process_inputs_async(
                    request_id,
                    prompt,
                    params,
                    supported_tasks=await self.get_supported_tasks(),
                    arrival_time=arrival_time,
                    lora_request=lora_request,
                    tokenization_kwargs=tokenization_kwargs,
                    trace_headers=trace_headers,
                    priority=priority,
                    data_parallel_rank=data_parallel_rank,
                )
            prompt_text, _, _ = extract_prompt_components(self.model_config, prompt)

        # SUBTRACTED: reasoning_ended / reasoning_parser_kwargs 注入
        #   (async_llm.py:L383-L386) — reasoning 解析器参数 (delete item 15).

        self.input_processor.assign_request_id(request)

        # We start the output_handler on the first call to add_request() so
        # we can call __init__ before the event loop, which enables us
        # to handle startup failure gracefully in the OpenAI server.
        self._run_output_handler()

        # Create a new output collector for the request.
        queue = RequestOutputCollector(params.output_kind, request.request_id)

        # Use cloned params that may have been updated in process_inputs()
        params = request.params

        if is_pooling or params.n == 1:
            await self._add_request(request, prompt_text, None, 0, queue)
            return queue

        # SUBTRACTED: n>1 fan-out (async_llm.py:L405-L418) — 扇出与输出聚合
        #   归 ch7 (delete item 16); 单请求下行完整正确.

    # SOURCE: vllm/v1/engine/async_llm.py:L420-L435 _add_request
    async def _add_request(
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: "ParentRequest | None",
        index: int,
        queue: RequestOutputCollector,
    ):
        # Add the request to OutputProcessor (this process).
        self.output_processor.add_request(request, prompt, parent_req, index, queue)

        # Add the EngineCoreRequest to EngineCore (separate process).
        await self.engine_core.add_request_async(request)

        if self.log_requests:
            logger.info("Added request %s.", request.request_id)

    # SUBTRACTED: _add_streaming_input_request /
    #   _validate_streaming_input_sampling_params (async_llm.py:L437-L538) —
    #   流式输入 (delete item 14), 后文领域.

    # SOURCE: vllm/v1/engine/async_llm.py:L657-L661 _run_output_handler guard
    # HOST SEAM: the real method spawns the output-handling task (the
    # upstream loop is ch07's product); this seam only records that the
    # first add_request started it, preserving the one-shot guard.
    # SOURCE: vllm/v1/engine/async_llm.py:L657-L661 _run_output_handler guard
    def _run_output_handler(self):
        """Background loop: pulls from EngineCore and pushes to AsyncStreams."""

        if self.output_handler is not None:
            return

        self.output_handler = SimpleNamespace(done=lambda: False)

    # SOURCE: vllm/v1/engine/async_llm.py:L1085-L1087 is_running
    @property
    # SOURCE: vllm/v1/engine/async_llm.py:L1085-L1087 is_running
    def is_running(self) -> bool:
        # Is None before the loop is started.
        return self.output_handler is None or not self.output_handler.done()

    # SOURCE: vllm/v1/engine/async_llm.py:L1089-L1090 is_stopped
    @property
    # SOURCE: vllm/v1/engine/async_llm.py:L1089-L1090 is_stopped
    def is_stopped(self) -> bool:
        return self.errored

    # SOURCE: vllm/v1/engine/async_llm.py:L1092-L1095 errored
    @property
    # SOURCE: vllm/v1/engine/async_llm.py:L1092-L1095 errored
    def errored(self) -> bool:
        return self.engine_core.resources.engine_dead or not self.is_running

    # SOURCE: vllm/v1/engine/async_llm.py:L1097-L1098 dead_error
    @property
    # SOURCE: vllm/v1/engine/async_llm.py:L1097-L1098 dead_error
    def dead_error(self) -> BaseException:
        return EngineDeadError()

