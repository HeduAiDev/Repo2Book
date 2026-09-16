# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/sampling_params.py —— 忠实承载（消费面核心）：本章站 5
# 「协议定型 to_sampling_params」的直接落点——SamplingParams.from_optional
# 的字段面、RequestOutputKind 三态枚举（WC2）、StructuredOutputsParams、
# __post_init__ 校验链逐字；verify/_validate_* 族（引擎侧校验，ch3/ch32
# 域）与 BeamSearchParams（delete[0] beam 全家）删。
"""Sampling parameters for text generation."""

import copy
import math
from dataclasses import field
from enum import Enum, IntEnum
from functools import cached_property
from typing import Annotated, Any

import msgspec
from pydantic import BeforeValidator
from pydantic.dataclasses import dataclass

import vllm.envs as envs
from vllm.exceptions import VLLMValidationError
from vllm.logger import init_logger
from vllm.v1.serial_utils import PydanticMsgspecMixin

logger = init_logger(__name__)

_SAMPLING_EPS = 1e-5
_MAX_TEMP = 1e-2

MAX_LOGPROB_TOKEN_IDS = 128
"""Upper bound on `SamplingParams.logprob_token_ids` list length. Must match
the per-request row width allocated by the sampler's `LogprobTokenIdsState`."""


# SOURCE: vllm/sampling_params.py:L35-L49 —— validate_thinking_token_budget 逐字
def validate_thinking_token_budget(value: int | float | bool | None) -> int | None:
    """Validate ``thinking_token_budget``; return ``None`` if unset."""
    if value is None:
        return None
    if isinstance(value, (bool, float)) or not isinstance(value, int):
        raise VLLMValidationError(
            "`thinking_token_budget` must be a non-negative integer "
            "or -1 for unlimited.",
            parameter="thinking_token_budget",
            value=value,
        )
    if value == -1:
        return None
    if value < 0:
        raise VLLMValidationError(
            "`thinking_token_budget` must be a non-negative integer "
            "or -1 for unlimited.",
            parameter="thinking_token_budget",
            value=value,
        )
    return value


# SOURCE: vllm/sampling_params.py:L52-L55
ThinkingTokenBudget = Annotated[
    int | None,
    BeforeValidator(validate_thinking_token_budget),
]


# SOURCE: vllm/sampling_params.py:L64-L67 —— SamplingType 逐字
class SamplingType(IntEnum):
    GREEDY = 0
    RANDOM = 1
    RANDOM_SEED = 2


# SOURCE: vllm/sampling_params.py:L71-L142 —— StructuredOutputsParams 逐字
@dataclass
class StructuredOutputsParams:
    # One of these fields will be used to build a logit processor.
    json: str | dict | None = None
    regex: str | None = None
    choice: list[str] | None = None
    grammar: str | None = None
    json_object: bool | None = None
    # These are other options that can be set.
    disable_any_whitespace: bool = False
    disable_additional_properties: bool = False
    whitespace_pattern: str | None = None
    structural_tag: str | None = None

    # SOURCE: vllm/sampling_params.py:L85-L88
    _backend: str | None = field(default=None, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""
    # SOURCE: vllm/sampling_params.py:L87-L88
    _backend_was_auto: bool = field(default=False, init=False)
    """CAUTION: Should only be set by Processor._validate_structured_output"""

    # SOURCE: vllm/sampling_params.py:L90-L111
    def __post_init__(self):
        """Validate that some fields are mutually exclusive."""
        count = sum(
            [
                self.json is not None,
                self.regex is not None,
                self.choice is not None,
                self.grammar is not None,
                self.json_object is not None,
                self.structural_tag is not None,
            ]
        )
        if count > 1:
            raise VLLMValidationError(
                "You can only use one kind of structured outputs constraint "
                f"but multiple are specified: {self.__dict__}"
            )
        if count < 1:
            raise VLLMValidationError(
                "You must use one kind of structured outputs constraint "
                f"but none are specified: {self.__dict__}"
            )

    # SOURCE: vllm/sampling_params.py:L113-L127
    def all_constraints_none(self) -> bool:
        """
        Returns True if all structured-output constraint fields are None.
        """
        return all(
            getattr(self, f) is None
            for f in (
                "json",
                "regex",
                "choice",
                "grammar",
                "json_object",
                "structural_tag",
            )
        )

    # SOURCE: vllm/sampling_params.py:L129-L142
    def all_non_structural_tag_constraints_none(self) -> bool:
        """
        Returns True if all structured-output constraint fields are None.
        """
        return all(
            getattr(self, f) is None
            for f in (
                "json",
                "regex",
                "choice",
                "grammar",
                "json_object",
            )
        )


# SOURCE: vllm/sampling_params.py:L145-L179 —— RepetitionDetectionParams 逐字
@dataclass
class RepetitionDetectionParams:
    """Parameters for detecting repetitive N-gram patterns in output tokens."""

    max_pattern_size: int = 0
    """Maximum size of N-gram pattern to detect for sequence repetition.
    Set to 0 to disable. Must be used together with min_count."""

    min_pattern_size: int = 0
    """Minimum size of N-gram pattern to check for sequence repetition.
    If set to 0, it defaults to 1.
    Must be <= max_pattern_size."""

    min_count: int = 0
    """Minimum number of times an N-gram pattern must repeat to trigger
    detection. Must be >= 2. Example: 3 for detecting a phrase repeated
    3 times. Must be used together with max_pattern_size."""

    # SOURCE: vllm/sampling_params.py:L163-L179
    def __post_init__(self):
        if (
            self.max_pattern_size < 0
            or self.min_pattern_size < 0
            or self.min_pattern_size > self.max_pattern_size
        ):
            raise VLLMValidationError(
                "max_pattern_size, min_pattern_size must be >=0, "
                "with min_pattern_size <= max_pattern_size. "
                "Set both to 0 to disable repetitive pattern detection."
            )
        if self.max_pattern_size > 0 and self.min_count < 2:
            raise VLLMValidationError(
                "min_count must be >= 2 to detect repetitive patterns "
                "in engine output. If you do not wish to detect repetitive "
                "patterns, set max_pattern_size to 0."
            )


# SOURCE: vllm/sampling_params.py:L182-L188 —— RequestOutputKind 逐字
# （WC2 三态契约：使用面在入口声明消费方式；ch7 站 13 裁剪侧已立，
# 本章 to_sampling_params 是 chat 面的声明处）
class RequestOutputKind(Enum):
    # Return entire output so far in every RequestOutput
    CUMULATIVE = 0
    # Return only deltas in each RequestOutput
    DELTA = 1
    # Do not return intermediate RequestOutput
    FINAL_ONLY = 2


# SUBTRACTED: vllm/sampling_params.py:L191-L196 _is_non_tekken_mistral /
# _get_llg_tokenizer——Mistral tokenizer 专用支路（delete[5]）。


# SOURCE: vllm/sampling_params.py:L199-L358 —— SamplingParams 字段面逐字
# （docstring 压缩为逐字段单行注释，字段与默认值不变）
class SamplingParams(
    PydanticMsgspecMixin,
    msgspec.Struct,
    omit_defaults=True,  # type: ignore[call-arg]
    # required for @cached_property.
    dict=True,
):  # type: ignore[call-arg]
    """Sampling parameters for text generation.

    Overall, we follow the sampling parameters from the OpenAI text completion
    API (https://platform.openai.com/docs/api-reference/completions/create).
    """

    # SOURCE: vllm/sampling_params.py:L213 —— n
    n: int = 1
    # SOURCE: vllm/sampling_params.py:L224 —— presence_penalty
    presence_penalty: float = 0.0
    # SOURCE: vllm/sampling_params.py:L228 —— frequency_penalty
    frequency_penalty: float = 0.0
    # SOURCE: vllm/sampling_params.py:L232 —— repetition_penalty
    repetition_penalty: float = 1.0
    # SOURCE: vllm/sampling_params.py:L236 —— temperature
    temperature: float = 1.0
    # SOURCE: vllm/sampling_params.py:L240 —— top_p
    top_p: float = 1.0
    # SOURCE: vllm/sampling_params.py:L243 —— top_k
    top_k: int = 0
    # SOURCE: vllm/sampling_params.py:L246 —— min_p
    min_p: float = 0.0
    # SOURCE: vllm/sampling_params.py:L250 —— seed
    seed: int | None = None
    # SOURCE: vllm/sampling_params.py:L252 —— stop
    stop: str | list[str] | None = None
    # SOURCE: vllm/sampling_params.py:L255 —— stop_token_ids
    stop_token_ids: list[int] | None = None
    # SOURCE: vllm/sampling_params.py:L259 —— ignore_eos
    ignore_eos: bool = False
    # SOURCE: vllm/sampling_params.py:L262 —— max_tokens
    max_tokens: int | None = 16
    # SOURCE: vllm/sampling_params.py:L264 —— min_tokens
    min_tokens: int = 0
    # SOURCE: vllm/sampling_params.py:L267 —— logprobs
    logprobs: int | None = None
    # SOURCE: vllm/sampling_params.py:L275 —— prompt_logprobs
    prompt_logprobs: int | None = None
    # SOURCE: vllm/sampling_params.py:L278 —— logprob_token_ids
    logprob_token_ids: list[int] | None = None
    # SOURCE: vllm/sampling_params.py:L284 —— flat_logprobs
    flat_logprobs: bool = False
    # SOURCE: vllm/sampling_params.py:L293 —— detokenize
    detokenize: bool = True
    # SOURCE: vllm/sampling_params.py:L295 —— skip_special_tokens
    skip_special_tokens: bool = True
    # SOURCE: vllm/sampling_params.py:L297 —— spaces_between_special_tokens
    spaces_between_special_tokens: bool = True
    # SOURCE: vllm/sampling_params.py:L299 —— include_stop_str_in_output
    include_stop_str_in_output: bool = False
    # SOURCE: vllm/sampling_params.py:L301 —— output_kind（三态）
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE
    # SOURCE: vllm/sampling_params.py:L302 —— stream_interval
    stream_interval: int | None = None
    # SOURCE: vllm/sampling_params.py:L307 —— skip_clone
    skip_clone: bool = False

    # The below fields are not supposed to be used as an input.
    # They are set in post_init.
    # SOURCE: vllm/sampling_params.py:L316-L318
    output_text_buffer_length: int = 0
    _eos_token_id: int | None = None
    _all_stop_token_ids: set[int] = msgspec.field(default_factory=set)

    # Fields used to construct logits processors
    # SOURCE: vllm/sampling_params.py:L321-L322
    structured_outputs: StructuredOutputsParams | None = None
    # SOURCE: vllm/sampling_params.py:L323-L325
    logit_bias: dict[int, float] | None = None
    # SOURCE: vllm/sampling_params.py:L326-L328
    allowed_token_ids: list[int] | None = None
    # SOURCE: vllm/sampling_params.py:L329-L332
    extra_args: dict[str, Any] | None = None
    # SOURCE: vllm/sampling_params.py:L333-L339
    routed_experts_prompt_start: int = 0

    # Fields used for bad words
    # SOURCE: vllm/sampling_params.py:L341-L346
    bad_words: list[str] | None = None
    _bad_words_token_ids: list[list[int]] | None = None

    # SOURCE: vllm/sampling_params.py:L348-L350
    skip_reading_prefix_cache: bool | None = None
    # SOURCE: vllm/sampling_params.py:L349-L350
    thinking_token_budget: int | None = None

    # SOURCE: vllm/sampling_params.py:L352-L358
    repetition_detection: RepetitionDetectionParams | None = None

    # SOURCE: vllm/sampling_params.py:L360-L455 —— from_optional 逐字
    @staticmethod
    def from_optional(
        n: int | None = 1,
        presence_penalty: float | None = 0.0,
        frequency_penalty: float | None = 0.0,
        repetition_penalty: float | None = 1.0,
        temperature: float | None = 1.0,
        top_p: float | None = 1.0,
        top_k: int = 0,
        min_p: float = 0.0,
        seed: int | None = None,
        stop: str | list[str] | None = None,
        stop_token_ids: list[int] | None = None,
        bad_words: list[str] | None = None,
        thinking_token_budget: int | None = None,
        include_stop_str_in_output: bool = False,
        ignore_eos: bool = False,
        max_tokens: int | None = 16,
        min_tokens: int = 0,
        logprobs: int | None = None,
        prompt_logprobs: int | None = None,
        detokenize: bool = True,
        skip_special_tokens: bool = True,
        spaces_between_special_tokens: bool = True,
        output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE,
        stream_interval: int | None = None,
        structured_outputs: StructuredOutputsParams | None = None,
        logit_bias: dict[int, float] | dict[str, float] | None = None,
        allowed_token_ids: list[int] | None = None,
        extra_args: dict[str, Any] | None = None,
        skip_clone: bool = False,
        repetition_detection: RepetitionDetectionParams | None = None,
        logprob_token_ids: list[int] | None = None,
    ) -> "SamplingParams":
        if logit_bias is not None:
            # Fast path uses a dict comprehension; on failure we iterate once
            # to identify the exact offending entry for the error message.
            try:
                logit_bias = {
                    int(token): min(100.0, max(-100.0, bias))
                    for token, bias in logit_bias.items()
                }
            except (ValueError, TypeError):
                invalid_keys = []
                converted_logit_bias = {}
                for token, bias in logit_bias.items():
                    try:
                        token_id = int(token)
                    except (ValueError, TypeError):
                        invalid_keys.append(token)
                        continue
                    converted_logit_bias[token_id] = min(100.0, max(-100.0, bias))
                if invalid_keys:
                    raise VLLMValidationError(
                        f"logit_bias contains key(s) that cannot be "
                        f"converted to integer token IDs: {invalid_keys!r}",
                        parameter="logit_bias",
                        value=invalid_keys,
                    ) from None
                logit_bias = converted_logit_bias

        return SamplingParams(
            n=1 if n is None else n,
            presence_penalty=0.0 if presence_penalty is None else presence_penalty,
            frequency_penalty=0.0 if frequency_penalty is None else frequency_penalty,
            repetition_penalty=1.0
            if repetition_penalty is None
            else repetition_penalty,
            temperature=1.0 if temperature is None else temperature,
            top_p=1.0 if top_p is None else top_p,
            top_k=top_k,
            min_p=min_p,
            seed=seed,
            stop=stop,
            stop_token_ids=stop_token_ids,
            bad_words=bad_words,
            thinking_token_budget=thinking_token_budget,
            include_stop_str_in_output=include_stop_str_in_output,
            ignore_eos=ignore_eos,
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            logprob_token_ids=logprob_token_ids,
            detokenize=detokenize,
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
            output_kind=output_kind,
            stream_interval=stream_interval,
            structured_outputs=structured_outputs,
            logit_bias=logit_bias,
            allowed_token_ids=allowed_token_ids,
            extra_args=extra_args,
            skip_clone=skip_clone,
            repetition_detection=repetition_detection,
        )

    # SOURCE: vllm/sampling_params.py:L457-L513 —— __post_init__ 逐字
    def __post_init__(self) -> None:
        if 0 < self.temperature < _MAX_TEMP:
            logger.warning(
                "temperature %s is less than %s, which may cause numerical "
                "errors nan or inf in tensors. We have maxed it out to %s.",
                self.temperature,
                _MAX_TEMP,
                _MAX_TEMP,
            )
            self.temperature = max(self.temperature, _MAX_TEMP)

        if self.seed == -1:
            self.seed = None

        self.thinking_token_budget = validate_thinking_token_budget(
            self.thinking_token_budget
        )

        if self.stop is None:
            self.stop = []
        elif isinstance(self.stop, str):
            self.stop = [self.stop]

        if self.stop_token_ids is None:
            self.stop_token_ids = []

        if self.bad_words is None:
            self.bad_words = []

        if self.logprobs is True:
            self.logprobs = 1

        if self.prompt_logprobs is True:
            self.prompt_logprobs = 1

        # Number of characters to hold back for stop string evaluation
        # until sequence is finished.
        if self.stop and not self.include_stop_str_in_output:
            self.output_text_buffer_length = max(len(s) for s in self.stop) - 1

        self._verify_args()

        if self.temperature < _SAMPLING_EPS:
            # Zero temperature means greedy sampling.
            self.top_p = 1.0
            self.top_k = 0
            self.min_p = 0.0
            self._verify_greedy_sampling()

        # eos_token_id is added to this by the engine
        self._all_stop_token_ids.update(self.stop_token_ids)

        if self.skip_reading_prefix_cache is None:
            # If prefix caching is enabled,
            # the output of prompt logprobs may less than n_prompt_tokens,
            # we need to skip reading cache at this request.
            self.skip_reading_prefix_cache = self.prompt_logprobs is not None

    # SOURCE: vllm/sampling_params.py:L515-L638 —— _verify_args 逐字
    def _verify_args(self) -> None:
        if not isinstance(self.n, int):
            raise VLLMValidationError(
                f"n must be an int, but is of type {type(self.n)}"
            )
        if self.n < 1:
            raise VLLMValidationError(f"n must be at least 1, got {self.n}.")
        max_n = envs.VLLM_MAX_N_SEQUENCES
        if self.n > max_n:
            raise VLLMValidationError(
                f"n must be at most {max_n}, got {self.n}. "
                "To increase this limit, set the VLLM_MAX_N_SEQUENCES "
                "environment variable."
            )
        if not -2.0 <= self.presence_penalty <= 2.0:
            raise VLLMValidationError(
                f"presence_penalty must be in [-2, 2], got {self.presence_penalty}."
            )
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise VLLMValidationError(
                f"frequency_penalty must be in [-2, 2], got {self.frequency_penalty}."
            )
        if not math.isfinite(self.repetition_penalty):
            raise VLLMValidationError(
                "repetition_penalty must be a finite number, "
                f"got {self.repetition_penalty}."
            )
        if self.repetition_penalty <= 0.0:
            raise VLLMValidationError(
                "repetition_penalty must be greater than zero, got "
                f"{self.repetition_penalty}."
            )
        if not math.isfinite(self.temperature):
            raise VLLMValidationError(
                f"temperature must be a finite number, got {self.temperature}.",
                parameter="temperature",
                value=self.temperature,
            )
        if self.temperature < 0.0:
            raise VLLMValidationError(
                f"temperature must be non-negative, got {self.temperature}.",
                parameter="temperature",
                value=self.temperature,
            )
        if self.temperature > 2.0:
            raise VLLMValidationError(
                f"temperature must be in [0, 2], got {self.temperature}.",
                parameter="temperature",
                value=self.temperature,
            )
        if not 0.0 < self.top_p <= 1.0:
            raise VLLMValidationError(
                f"top_p must be in (0, 1], got {self.top_p}.",
                parameter="top_p",
                value=self.top_p,
            )
        # quietly accept -1 as disabled, but prefer 0
        if self.top_k < -1:
            raise VLLMValidationError(
                f"top_k must be 0 (disable), or at least 1, got {self.top_k}."
            )
        if not isinstance(self.top_k, int):
            raise VLLMValidationError(
                f"top_k must be an integer, got {type(self.top_k).__name__}"
            )
        if not 0.0 <= self.min_p <= 1.0:
            raise VLLMValidationError(f"min_p must be in [0, 1], got {self.min_p}.")
        if self.max_tokens is not None and self.max_tokens < 1:
            raise VLLMValidationError(
                f"max_tokens must be at least 1, got {self.max_tokens}.",
                parameter="max_tokens",
                value=self.max_tokens,
            )
        if self.min_tokens < 0:
            raise VLLMValidationError(
                f"min_tokens must be greater than or equal to 0, got {self.min_tokens}."
            )
        if self.max_tokens is not None and self.min_tokens > self.max_tokens:
            raise VLLMValidationError(
                f"min_tokens must be less than or equal to "
                f"max_tokens={self.max_tokens}, got {self.min_tokens}."
            )
        if self.stream_interval is not None and self.stream_interval < 1:
            raise VLLMValidationError(
                f"stream_interval must be at least 1, got {self.stream_interval}.",
                parameter="stream_interval",
                value=self.stream_interval,
            )
        if self.logprobs is not None and self.logprobs != -1 and self.logprobs < 0:
            raise VLLMValidationError(
                f"logprobs must be non-negative or -1, got {self.logprobs}.",
                parameter="logprobs",
                value=self.logprobs,
            )
        if (
            self.prompt_logprobs is not None
            and self.prompt_logprobs != -1
            and self.prompt_logprobs < 0
        ):
            raise VLLMValidationError(
                f"prompt_logprobs must be non-negative or -1, got "
                f"{self.prompt_logprobs}.",
                parameter="prompt_logprobs",
                value=self.prompt_logprobs,
            )
        assert isinstance(self.stop_token_ids, list)
        if not all(isinstance(st_id, int) for st_id in self.stop_token_ids):
            raise VLLMValidationError(
                f"stop_token_ids must contain only integers, got {self.stop_token_ids}."
            )
        assert isinstance(self.stop, list)
        if any(not stop_str for stop_str in self.stop):
            raise VLLMValidationError("stop cannot contain an empty string.")
        if self.stop and not self.detokenize:
            raise VLLMValidationError(
                "stop strings are only supported when detokenize is True. "
                "Set detokenize=True to use stop."
            )
        assert isinstance(self.bad_words, list)
        if any(not bad_word for bad_word in self.bad_words):
            raise VLLMValidationError(
                f"bad_words cannot contain an empty string. "
                f"Got bad_words={self.bad_words}"
            )

    # SOURCE: vllm/sampling_params.py:L640-L644 —— _verify_greedy_sampling 逐字
    def _verify_greedy_sampling(self) -> None:
        if self.n > 1:
            raise VLLMValidationError(
                f"n must be 1 when using greedy sampling, got {self.n}."
            )

    # SOURCE: vllm/sampling_params.py:L646-L674 —— update_from_generation_config 逐字
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

    # SOURCE: vllm/sampling_params.py:L676-L715 —— update_from_tokenizer
    # （bad_words 的 tokenizer 编码支路；HOST SEAM：tokenizer.encode 按
    # 真实签名消费，host 测试不触达 bad_words 路径）
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
                self._bad_words_token_ids.append(prompt_token_ids)

    # SOURCE: vllm/sampling_params.py:L717-L723 —— sampling_type 逐字
    @cached_property
    def sampling_type(self) -> SamplingType:
        if self.temperature < _SAMPLING_EPS:
            return SamplingType.GREEDY
        if self.seed is not None:
            return SamplingType.RANDOM_SEED
        return SamplingType.RANDOM

    # SOURCE: vllm/sampling_params.py:L725-L727 —— eos_token_id
    @property
    def eos_token_id(self) -> int | None:
        return self._eos_token_id

    # SOURCE: vllm/sampling_params.py:L729-L731 —— all_stop_token_ids
    @property
    def all_stop_token_ids(self) -> set[int]:
        return self._all_stop_token_ids

    # SOURCE: vllm/sampling_params.py:L733-L736 —— bad_words_token_ids
    @property
    def bad_words_token_ids(self) -> list[list[int]] | None:
        # For internal use only. Backward compatibility not guaranteed
        return self._bad_words_token_ids

    # SOURCE: vllm/sampling_params.py:L738-L746 —— num_logprobs 逐字
    @property
    def num_logprobs(self) -> int | None:
        """Number of sample logprobs to return per output token, or `None` if
        no sample logprobs were requested. Takes `logprob_token_ids` into
        account: when `logprobs` is unset but `logprob_token_ids` is set,
        returns `len(logprob_token_ids)`."""
        if self.logprobs is not None:
            return self.logprobs
        return len(self.logprob_token_ids) if self.logprob_token_ids else None

    # SOURCE: vllm/sampling_params.py:L748-L753 —— clone 逐字
    def clone(self) -> "SamplingParams":
        """If skip_clone is True, uses shallow copy instead of deep copy."""
        if self.skip_clone:
            return copy.copy(self)

        return copy.deepcopy(self)

    # SUBTRACTED: vllm/sampling_params.py:L755-L1086 verify() 与
    # _validate_logprobs/_validate_logit_bias/_validate_logits_processors/
    # _validate_allowed_token_ids/_validate_spec_decode/_validate_diffusion/
    # _validate_structured_outputs——引擎侧入队前校验族（ch3 参数域与
    # ch30/ch32 结构化输出域消费），OpenAI 层只构造不校验；真实调用链在
    # InputProcessor.process_inputs（ch6 边界）内。

    # SOURCE: vllm/sampling_params.py:L1088-L1113 —— __repr__ 逐字
    def __repr__(self) -> str:
        return (
            f"SamplingParams(n={self.n}, "
            f"presence_penalty={self.presence_penalty}, "
            f"frequency_penalty={self.frequency_penalty}, "
            f"repetition_penalty={self.repetition_penalty}, "
            f"temperature={self.temperature}, "
            f"top_p={self.top_p}, "
            f"top_k={self.top_k}, "
            f"min_p={self.min_p}, "
            f"seed={self.seed}, "
            f"stop={self.stop}, "
            f"stop_token_ids={self.stop_token_ids}, "
            f"bad_words={self.bad_words}, "
            f"thinking_token_budget={self.thinking_token_budget}, "
            f"include_stop_str_in_output={self.include_stop_str_in_output}, "
            f"ignore_eos={self.ignore_eos}, "
            f"max_tokens={self.max_tokens}, "
            f"min_tokens={self.min_tokens}, "
            f"logprobs={self.logprobs}, "
            f"prompt_logprobs={self.prompt_logprobs}, "
            f"skip_special_tokens={self.skip_special_tokens}, "
            "spaces_between_special_tokens="
            f"{self.spaces_between_special_tokens}, "
            f"structured_outputs={self.structured_outputs}, "
            f"extra_args={self.extra_args})"
        )

    # SUBTRACTED: vllm/sampling_params.py:L1116-L1132 for_sampler_warmup
    # ——sampler 预热参数（ch29 域）。

# SUBTRACTED: vllm/sampling_params.py:L1135-L1149 BeamSearchParams——
# delete[0] beam_search 全家（to_beam_search_params 与 BeamSearchOnlineMixin
# 调用链已删，无消费点）。
