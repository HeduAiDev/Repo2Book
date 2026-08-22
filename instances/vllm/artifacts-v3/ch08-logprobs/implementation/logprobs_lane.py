# Subtract-only companion for v3 ch08 «输出的另一个维度：logprobs» (Part II:
# 分而治之——进程边界与消息。L0 上行泳道的 logprobs 支路放大：与 ch7 主泳道
# token→text 并行同车的另一维度——概率。从 GPU 采样器的 raw 留底、gather 三件套、
# 同一次 D2H、调度切行、msgpack 过线，到 API 进程 LogprobsProcessor 的装配与
# OpenAI 层 token/logprob/bytes 三件套).
#
# FAITHFUL SUBSET of the real vLLM logprobs lane at pin v0.27.1 (6e448d0ea).
# It keeps vLLM's names, structure and control flow; it only DELETES branches
# approved in the dossier subtraction_plan (every deletion marked
# `# SUBTRACTED:` with its source span) plus the documented HOST SEAMs
# (external machinery this chapter treats as a black box — ch5/ch6/ch7
# products, CUDA streams, msgspec). Mapping rule: take the real vLLM source,
# drop every SUBTRACTED branch, and you should get (approximately) this file.
#
# Fully real (the chapter's product):
# - the whole of vllm/logprobs.py (Logprob / FlatLogprobs 6 parallel primitive
#   lists / create_* containers / append_logprobs_for_next_position rank chain)
#   and vllm/v1/engine/logprobs.py (LogprobsProcessor: from_new_request,
#   _update_sample_logprobs, _update_prompt_logprobs, pop_prompt_logprobs,
#   _get_sampled_context_ids, _correct_decoded_token, _verify_tokens,
#   update_from_output) — both files verbatim except the two approved
#   FlatLogprobs deletions;
# - convert_ids_list_to_tokens + the leading-space restore helpers (the
#   NON-incremental detokenization strategy, vs ch7's incremental one);
# - Sampler's logprobs steps only: the raw snapshot (NOTE(woosuk): before any
#   penalties/temperature), gather_logprobs (topk + sampled + count-rank),
#   the greedy fast path where processed_* modes materialize (m16), and the
#   logprob_token_ids sparse gather;
# - LogprobsTensors/LogprobsLists NamedTuples (to_cpu_nonblocking / tolists /
#   empty_cpu / slice_request) and the AsyncGPUModelRunnerOutput D2H wrap;
# - _get_prompt_logprobs_dict (the prompt sub-branch: re-run compute_logits
#   over prompt hidden states, chunked accumulation, last-chunk delivery);
# - the scheduler's logprobs lines of update_from_output (slice per request,
#   the two EngineCoreOutput fields) and the msgpack ndarray/tensor hooks;
# - the output_processor logprobs segments (process_outputs step 3,
#   _new_completion_output DELTA tail slicing + cumulative, prompt pop) and
#   the OpenAI three-field record (_create_chat_logprobs / _get_top_logprobs).
#
# Runs on a CPU host WITHOUT the vllm package. Every def/class carries a
# `# SOURCE: vllm/...:Lxxx` ref into the pinned tree (line numbers re-verified
# against v0.27.1, not copied from v2's v0.21.0 assets). HOST SEAMs stand in
# for: msgspec (wire-compatible shim in _msgspec_seam.py, real msgpack
# bytes), CUDA stream/event objects, the ch6-product SamplingParams field
# face beyond the logprobs params, the ch7-product detokenizer face and the
# model/config faces _get_prompt_logprobs_dict touches.

from __future__ import annotations

import enum
from enum import Enum  # SOURCE: vllm/sampling_params.py:L9（from enum import Enum, IntEnum）
import itertools
import json
import logging
from collections import defaultdict
from collections.abc import Iterable, Iterator, MutableSequence
from dataclasses import dataclass, field
from inspect import isclass
from typing import (
    Any,
    ClassVar,
    Literal,
    NamedTuple,
    TypeAlias,
    cast,
    overload,
)

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

import _msgspec_seam
from _msgspec_seam import seam_msgspec

# The pinned vLLM does `import msgspec`; the name below is the HOST SEAM
# namespace (see _msgspec_seam.py) exposing Struct / msgpack / convert.
msgspec = seam_msgspec

# The pinned vLLM does `from msgspec import msgpack` (serial_utils.py:L21).
msgpack = seam_msgspec.msgpack  # HOST SEAM: real-msgpack-backed Encoder/Decoder

# SOURCE: vllm/tokenizers/__init__.py TokenizerLike — protocol face seam: the
# real file builds a Protocol over the transformers AnyTokenizer union; the
# kept code only calls .decode/.convert_ids_to_tokens/.backend_tokenizer.
TokenizerLike: TypeAlias = Any


# ============================================================================
# §0 Host seams — stdlib stand-ins so the module runs without the vllm
# package. Each mirrors the real interface subset the kept code touches.
# ============================================================================


# SOURCE: vllm/logger.py init_logger — logging seam (v1/engine/logprobs.py:L24
# and logprobs-side files only name it; nothing logs on the kept paths)
def init_logger(name: str):  # SOURCE: vllm/v1/engine/logprobs.py:L24 logger = init_logger(__name__) — HOST SEAM
    log = logging.getLogger(name)
    if not log.handlers:
        log.addHandler(logging.NullHandler())
    return log


logger = init_logger(__name__)  # SOURCE: vllm/v1/engine/logprobs.py:L24


# SOURCE: vllm/platforms/__init__.py current_platform.simple_compile_backend —
# the decorator target on batched_count_greater_than (ops/logprobs.py:L10).
# Real value: PlatformInterface.simple_compile_backend = "inductor"
# (vllm/platforms/interface.py:L165; CPU/CUDA do not override). HOST SEAM
# deviation: the host has no accelerator toolchain, so the seam passes
# "eager" — torch.compile still dynamo-traces but executes the same eager
# math (observable numerics identical; verified in tests against hand
# gathers).
class _CurrentPlatform:  # SOURCE: vllm/platforms/interface.py:L165 simple_compile_backend = "inductor" — HOST SEAM
    simple_compile_backend = "eager"


current_platform = _CurrentPlatform()


# SOURCE: vllm/utils/torch_utils.py:L72 PIN_MEMORY = is_pin_memory_available()
# — host has no CUDA, so pinning is off (real derivation preserved)
PIN_MEMORY = torch.cuda.is_available()  # HOST SEAM: is_pin_memory_available()


# SOURCE: vllm/v1/worker/gpu_model_runner.py:L276/L286-L306 torch.cuda
# Event/Stream — HOST SEAM: on a CUDA-less host the copy stream degenerates to
# an immediate no-op context (to_cpu_nonblocking on CPU tensors is already a
# no-op, outputs.py:L73-L75) and the event to trivial record/synchronize. On
# a CUDA host these delegate to the real torch.cuda objects.
def _cuda_event(blocking: bool = False):  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L276 torch.cuda.Event(blocking=True) — HOST SEAM
    if torch.cuda.is_available():
        return torch.cuda.Event(blocking=blocking)
    return SimpleEvent()


class SimpleEvent:  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L276 Event 的 record/synchronize 面 — HOST SEAM
    def record(self):  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L306 async_copy_ready_event.record()
        pass

    def synchronize(self):  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L313 async_copy_ready_event.synchronize()
        pass


class _CudaStreamCtx:  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L280-L281 with torch.cuda.stream(async_output_copy_stream) — HOST SEAM
    """Stand-in for `with torch.cuda.stream(s):` on a CUDA-less host."""

    def __init__(self, stream):  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L280 stream 上下文面
        self._stream = stream
        self._ctx = None

    def __enter__(self):  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L281 进入 copy stream
        if torch.cuda.is_available():
            self._ctx = torch.cuda.stream(self._stream)
            self._ctx.__enter__()

    def __exit__(self, *exc):  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L281 退出 copy stream
        if self._ctx is not None:
            return self._ctx.__exit__(*exc)
        return False


def _cuda_wait_stream(copy_stream, default_stream):  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L282 async_output_copy_stream.wait_stream(default_stream) — HOST SEAM
    if torch.cuda.is_available():
        copy_stream.wait_stream(default_stream)


def _cuda_current_stream():  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L278 default_stream = torch.cuda.current_stream() — HOST SEAM
    if torch.cuda.is_available():
        return torch.cuda.current_stream()
    return None


# SOURCE: vllm/entrypoints/openai/engine/protocol.py:L31-L35 OpenAIBaseModel —
# pydantic base allowing extra fields (host pydantic plays the same role).
class OpenAIBaseModel(BaseModel):  # SOURCE: vllm/entrypoints/openai/engine/protocol.py:L31-L35 — HOST SEAM (real config kept verbatim)
    # OpenAI API does allow extra fields
    model_config = ConfigDict(extra="allow")

    # Cache class field names
    field_names: ClassVar[set[str] | None] = None


# SimpleNamespace lives behind the seams above; import here to keep one face.
from types import SimpleNamespace  # noqa: E402  (HOST SEAM helper)


# ============================================================================
# §1 vllm/sampling_params.py — the logprobs params + RequestOutputKind
# ============================================================================


# SOURCE: vllm/sampling_params.py:L182-L188 RequestOutputKind — verbatim
class RequestOutputKind(Enum):
    # Return entire output so far in every RequestOutput
    CUMULATIVE = 0
    # Return only deltas in each RequestOutput
    DELTA = 1
    # Do not return intermediate RequestOutput
    FINAL_ONLY = 2


# SOURCE: vllm/sampling_params.py SamplingParams — 本章域 = logprobs 四参数
# （docstring 逐字）；其余字段为 HOST SEAM 字段面（参数域归 ch06——本章触达的
# 字段 + 真实默认值），校验体按 delete 项 9 删。
class SamplingParams:
    def __init__(
        self,
        logprobs: int | None = None,  # SOURCE: vllm/sampling_params.py:L267
        prompt_logprobs: int | None = None,  # SOURCE: vllm/sampling_params.py:L275
        logprob_token_ids: list[int] | None = None,  # SOURCE: vllm/sampling_params.py:L278
        flat_logprobs: bool = False,  # SOURCE: vllm/sampling_params.py:L284
        detokenize: bool = True,  # SOURCE: vllm/sampling_params.py:L293（m20 的开关）
        output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE,  # SOURCE: vllm/sampling_params.py:L182-L188
    ):
        # Number of log probabilities to return per output token. When set to
        # `None`, no probability is returned. If set to a non-`None` value, the
        # result includes the log probabilities of the specified number of most
        # likely tokens, as well as the chosen tokens. Note that the implementation
        # follows the OpenAI API: The API will always return the log probability of
        # the sampled token, so there may be up to `logprobs+1` elements in the
        # response. When set to -1, return all `vocab_size` log probabilities.
        self.logprobs = logprobs
        # Number of log probabilities to return per prompt token.
        # When set to -1, return all `vocab_size` log probabilities.
        self.prompt_logprobs = prompt_logprobs
        # Specific token IDs to return logprobs for. More efficient than
        # logprobs=-1 when you only need logprobs for a small set of tokens.
        # When set, logprobs for exactly these token IDs will be returned,
        # in addition to the sampled token. This is useful for scoring tasks
        # where you want to compare probabilities of specific label tokens.
        self.logprob_token_ids = logprob_token_ids
        # Whether to return logprobs in flatten format (i.e. FlatLogprob)
        # for better performance.
        # NOTE: GC costs of FlatLogprobs is significantly smaller than
        # list[dict[int, Logprob]]. After enabled, PromptLogprobs and
        # SampleLogprobs would populated as FlatLogprobs.
        self.flat_logprobs = flat_logprobs
        self.detokenize = detokenize
        self.output_kind = output_kind
        self.skip_reading_prefix_cache: bool | None = None
        self.__post_init__()

    # SOURCE: vllm/sampling_params.py:L457-L513 __post_init__ — 只留 logprobs
    # 相关两段；校验体/温度归一/stop 缓冲等按 delete 项 9 删
    def __post_init__(self) -> None:
        # SUBTRACTED: temperature 下限告警 / seed / thinking_budget / stop 与
        # stop_token_ids 归一 / _verify_args / greedy 归一（L458-L505,L514 起
        # 的校验体）——delete 项 9，非本章域
        if self.logprobs is True:  # SOURCE: vllm/sampling_params.py:L486-L487
            self.logprobs = 1

        if self.prompt_logprobs is True:  # SOURCE: vllm/sampling_params.py:L489-L490
            self.prompt_logprobs = 1

        # SUBTRACTED: stop string holdback 缓冲长度（L494-L495）——ch7 域
        if self.skip_reading_prefix_cache is None:  # SOURCE: vllm/sampling_params.py:L509-L513
            # If prefix caching is enabled,
            # the output of prompt logprobs may less than n_prompt_tokens,
            # we need to skip reading cache at this request.
            self.skip_reading_prefix_cache = self.prompt_logprobs is not None

    # SOURCE: vllm/sampling_params.py:L360-L455 from_optional — 只留 logprobs
    # 相关构造行；其余参数与 logit_bias 换算按 delete 项 9 删（ch06 域）
    @staticmethod
    def from_optional(
        logprobs: int | None = None,
        prompt_logprobs: int | None = None,
        logprob_token_ids: list[int] | None = None,
        detokenize: bool = True,
        output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE,
    ) -> "SamplingParams":
        # SUBTRACTED: logit_bias 字符键换算（L392-L419）——delete 项 9
        return SamplingParams(  # SOURCE: vllm/sampling_params.py:L421-L455 尾部构造
            # SUBTRACTED: n/penalties/temperature/top_p/top_k/min_p/seed/stop/
            # stop_token_ids/bad_words/thinking_budget/max_tokens/min_tokens/
            # structured_outputs/logit_bias/allowed_token_ids/extra_args/
            # skip_clone/repetition_detection 等行（L422-L440,L445-L454）——
            # delete 项 9
            logprobs=logprobs,
            prompt_logprobs=prompt_logprobs,
            logprob_token_ids=logprob_token_ids,
            detokenize=detokenize,
            output_kind=output_kind,
        )

    # SOURCE: vllm/sampling_params.py:L738-L746 num_logprobs property — verbatim
    @property
    def num_logprobs(self) -> int | None:  # SOURCE: vllm/sampling_params.py:L738-L746
        """Number of sample logprobs to return per output token, or `None` if
        no sample logprobs were requested. Takes `logprob_token_ids` into
        account: when `logprobs` is unset but `logprob_token_ids` is set,
        returns `len(logprob_token_ids)`."""
        if self.logprobs is not None:
            return self.logprobs
        return len(self.logprob_token_ids) if self.logprob_token_ids else None


# SOURCE: vllm/config/model.py:L99-L105 — verbatim（m16 四态契约）
LogprobsMode = Literal[
    "raw_logits", "raw_logprobs", "processed_logits", "processed_logprobs"
]
PROCESSED_LOGPROBS_MODES: tuple[LogprobsMode, ...] = (
    "processed_logits",
    "processed_logprobs",
)


# ============================================================================
# §2 vllm/logprobs.py — the whole file (chapter core): Logprob / FlatLogprobs
# / container factories / the rank chain
# ============================================================================


# We use dataclass for now because it is used for
# openai server output, and msgspec is not serializable.
# TODO(sang): Fix it.
@dataclass  # SOURCE: vllm/logprobs.py:L9-L24
class Logprob:
    """Infos for supporting OpenAI compatible logprobs and token ranks.

    Attributes:
        logprob: The logprob of chosen token
        rank: The vocab rank of chosen token (>=1)
        decoded_token: The decoded chosen token index
    """

    logprob: float
    rank: int | None = None
    decoded_token: str | None = None


LogprobsOnePosition = dict[int, Logprob]  # SOURCE: vllm/logprobs.py:L27


@dataclass  # SOURCE: vllm/logprobs.py:L30-L93
class FlatLogprobs(MutableSequence[LogprobsOnePosition | None]):
    """
    Flat logprobs of a request into multiple primitive type lists.

    Compared to list[dict[int, Logprob]], this data structure reduced GC
    overhead significantly. As it flattened logprob information for
    all positions and ranks in to multiple primitive type lists (i.e.
    logprobs, token_ids, ranks per token_ids, decoded_tokens).
    So regardless of the sequence length and top_logprobs setup,
    FlatLogprobs would only introduce a constant amount of objects.

    As each position might contains different amount of ranks,
    start_indices_per_position would be used to access the logprob ranges
    for different positions.

    NOTE: To reduce the migration overhead and improve backward compatibility,
    we support the key Sequence APIs of list, so it could act as
    list[LogprobsOnePosition]
    """

    # Start / end indices to indicate the range of logprobs for each position.
    start_indices: list[int] = field(default_factory=list)
    end_indices: list[int] = field(default_factory=list)

    # Flatten Logprob information for (each position, rank).
    # For position <i>, the logprobs are ranged
    # from self.start_indices[i] to self.end_indices[i] (exclusive).
    token_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    ranks: list[int | None] = field(default_factory=list)
    decoded_tokens: list[str | None] = field(default_factory=list)

    def append(self, logprobs_one_position: LogprobsOnePosition | None) -> None:
        """Appends the container with logprobs for the next position"""  # SOURCE: vllm/logprobs.py:L63-L72
        self.start_indices.append(len(self.logprobs))
        if logprobs_one_position:
            for token_id, logprob in logprobs_one_position.items():
                self.token_ids.append(token_id)
                self.logprobs.append(logprob.logprob)
                self.ranks.append(logprob.rank)
                self.decoded_tokens.append(logprob.decoded_token)
        self.end_indices.append(len(self.logprobs))

    def append_fast(  # SOURCE: vllm/logprobs.py:L74-L93
        self,
        token_ids: list[int],
        logprobs: list[float],
        ranks: itertools.chain[int],
        decoded_tokens: Iterable[str | None],
    ) -> None:
        """
        Appends logprobs for the next position without creating
        the intermediate logprob dictionary.
        """
        self.start_indices.append(len(self.logprobs))
        for token_id, logprob, rank, decoded_token in zip(
            token_ids, logprobs, ranks, decoded_tokens
        ):
            self.token_ids.append(token_id)
            self.logprobs.append(logprob)
            self.ranks.append(rank)
            self.decoded_tokens.append(decoded_token)
        self.end_indices.append(len(self.logprobs))

    # SUBTRACTED: extend 覆写（L95-L98）——delete 项 8：MutableSequence 基类
    # 继承面等效（逐位置调 append），本章核心路径不触达

    def __len__(self) -> int:
        """Gets number of positions stored in the container"""  # SOURCE: vllm/logprobs.py:L100-L102
        return len(self.start_indices)

    @overload
    def __getitem__(self, position: int) -> LogprobsOnePosition: ...  # SOURCE: vllm/logprobs.py:L104-L105

    @overload
    def __getitem__(self, s: slice, /) -> "FlatLogprobs": ...  # SOURCE: vllm/logprobs.py:L107-L108

    def __getitem__(self, index: int | slice):  # SOURCE: vllm/logprobs.py:L110-L135
        """Extracts logprobs of a given position or slice"""
        if isinstance(index, int):
            return {
                self.token_ids[i]: Logprob(
                    logprob=self.logprobs[i],
                    rank=self.ranks[i],
                    decoded_token=self.decoded_tokens[i],
                )
                for i in range(self.start_indices[index], self.end_indices[index])
            }
        elif isinstance(index, slice):
            min_index = self.start_indices[index][0]
            max_index = self.end_indices[index][-1]
            return FlatLogprobs(
                # Shift updated start_indices and end_indices to
                # be 0-indexed
                start_indices=[i - min_index for i in self.start_indices[index]],
                end_indices=[i - min_index for i in self.end_indices[index]],
                token_ids=self.token_ids[min_index:max_index],
                logprobs=self.logprobs[min_index:max_index],
                ranks=self.ranks[min_index:max_index],
                decoded_tokens=self.decoded_tokens[min_index:max_index],
            )
        else:
            raise TypeError(f"Invalid index type: {type(index)}")

    # delete 项 8 说三方法「一句话注释即可」，但 MutableSequence ABC 缺它们
    # 无法实例化——三行即全部原文（无可删体），保留原样（见 impl-notes §删除台账）
    def __setitem__(self, item, value) -> None:  # SOURCE: vllm/logprobs.py:L137-L138
        raise TypeError("Cannot set logprobs in FlatLogprobs")

    def __delitem__(self, item) -> None:  # SOURCE: vllm/logprobs.py:L140-L141
        raise TypeError("Cannot delete logprobs from FlatLogprobs")

    def insert(self, index: int, value: dict[int, Logprob] | None) -> None:  # SOURCE: vllm/logprobs.py:L143-L144
        raise TypeError("Cannot insert logprobs to FlatLogprobs")

    # SUBTRACTED: __iter__ 覆写（L146-L152）——delete 项 8：Sequence 基类的
    # __getitem__ 整数递增迭代等效（行为一致：逐位置 yield 同样的 dict）


# {token_id -> logprob} per each sequence group. None if the corresponding
# sequence group doesn't require prompt logprob.
PromptLogprobs = FlatLogprobs | list[LogprobsOnePosition | None]  # SOURCE: vllm/logprobs.py:L155-L157
# {token_id -> logprob} for each sequence group.
SampleLogprobs = FlatLogprobs | list[LogprobsOnePosition]  # SOURCE: vllm/logprobs.py:L158-L159


def create_prompt_logprobs(flat_logprobs: bool) -> PromptLogprobs:  # SOURCE: vllm/logprobs.py:L162-L167
    """Creates a container to store prompt logprobs for a request"""
    logprobs: PromptLogprobs = FlatLogprobs() if flat_logprobs else []
    # NOTE: logprob of first prompt token is None.
    logprobs.append(None)
    return logprobs


def create_sample_logprobs(flat_logprobs: bool) -> SampleLogprobs:  # SOURCE: vllm/logprobs.py:L170-L172
    """Creates a container to store decode logprobs for a request"""
    return FlatLogprobs() if flat_logprobs else []


def append_logprobs_for_next_position(  # SOURCE: vllm/logprobs.py:L175-L206
    request_logprobs: PromptLogprobs | SampleLogprobs,
    token_ids: list[int],
    logprobs: list[float],
    decoded_tokens: Iterable[str | None],
    rank: int,
    num_logprobs: int,
) -> None:
    """Appends logprobs for the next position"""
    if num_logprobs == -1:
        num_logprobs = len(logprobs)
    # We do not need a special case for the sampled token
    # being in the topk, since inserting duplicated data
    # into a dictionary twice is the same as doing it once.
    topk_ranks = range(1, num_logprobs + 1)
    ranks = itertools.chain((rank,), topk_ranks)

    if isinstance(request_logprobs, FlatLogprobs):
        request_logprobs.append_fast(token_ids, logprobs, ranks, decoded_tokens)
    else:
        request_logprobs.append(
            {
                token_id: Logprob(
                    logprob=logprob,
                    rank=rank,
                    decoded_token=token,
                )
                for token_id, logprob, rank, token in zip(
                    token_ids, logprobs, ranks, decoded_tokens
                )
            }
        )


# typing.overload rides the top import block (vllm/logprobs.py:L6).


# ============================================================================
# §3 vllm/tokenizers/detokenizer_utils.py — the NON-incremental detokenizer
# helper the logprobs branch uses (vs ch7's incremental main lane)
# ============================================================================


_CACHED_MARKER_KEY = "_vllm_space_marker_cache"  # SOURCE: vllm/tokenizers/detokenizer_utils.py:L62
_NOT_CACHED = "__not_computed__"  # SOURCE: vllm/tokenizers/detokenizer_utils.py:L63


def _get_leading_space_marker(tokenizer: TokenizerLike) -> str | None:  # SOURCE: vllm/tokenizers/detokenizer_utils.py:L66-L102
    """Read the space marker from the tokenizer's pre_tokenizer config.

    Only Metaspace pre_tokenizers (used by SentencePiece-based models like
    Llama, Mistral, T5) have a replacement character whose leading instance
    gets stripped by decode(). ByteLevel (GPT-2), BertPreTokenizer (BERT),
    and others do not have this issue.

    Returns the marker character, or None if decode() is safe for single
    tokens.
    """
    cached = getattr(tokenizer, _CACHED_MARKER_KEY, _NOT_CACHED)
    if cached is not _NOT_CACHED:
        return cached  # type: ignore[return-value]

    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        result = None
    else:
        result = None
        try:
            config = json.loads(backend.to_str())
        except Exception:
            pass
        else:
            pre = config.get("pre_tokenizer") or {}
            pre_type = pre.get("type")
            if pre_type == "Metaspace":
                result = pre.get("replacement", "▁")
            elif pre_type == "Sequence":
                for sub in pre.get("pretokenizers", []):
                    if sub.get("type") == "Metaspace":
                        result = sub.get("replacement", "▁")
                        break

    setattr(tokenizer, _CACHED_MARKER_KEY, result)
    return result


def _restore_leading_spaces(raw_token: str, token_str: str, marker: str) -> str:  # SOURCE: vllm/tokenizers/detokenizer_utils.py:L105-L116
    """Restore leading spaces that decode() stripped from a raw vocab piece."""
    num_markers = 0
    for ch in raw_token:
        if ch != marker:
            break
        num_markers += 1
    if num_markers == 0:
        return token_str
    existing = len(token_str) - len(token_str.lstrip(" "))
    missing = num_markers - existing
    return " " * missing + token_str if missing > 0 else token_str


def convert_ids_list_to_tokens(  # SOURCE: vllm/tokenizers/detokenizer_utils.py:L143-L170
    tokenizer: TokenizerLike,
    token_ids: list[int],
) -> list[str]:
    """Detokenize the input ids individually.

    Uses decode() for human-readable output, then checks the raw vocab
    piece via convert_ids_to_tokens() to restore any leading spaces that
    decode() stripped (SentencePiece add_dummy_prefix inverse).

    Args:
      tokenizer: tokenizer used by model under test
      token_ids: convert these tokens (Python list form)

    Returns:
      Python list of token string representations

    """
    if not token_ids:
        return []
    marker = _get_leading_space_marker(tokenizer)
    if marker is None:
        return [tokenizer.decode([tid]) or "" for tid in token_ids]
    raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)
    return [
        _restore_leading_spaces(raw, tokenizer.decode([tid]) or "", marker)
        for tid, raw in zip(token_ids, raw_tokens)
    ]


# ============================================================================
# §4 vllm/v1/outputs.py — the two logprobs carriers + sampler/runner outputs
# ============================================================================


class LogprobsLists(NamedTuple):  # SOURCE: vllm/v1/outputs.py:L28-L50
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: np.ndarray
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: np.ndarray
    # [num_reqs x num_generated_tokens]
    sampled_token_ranks: np.ndarray
    # [num_reqs]
    # Used for slicing the logprobs in cases like speculative
    # decoding where the number of generated tokens may be
    # different for each request.
    cu_num_generated_tokens: list[int] | None = None

    def slice_request(self, req_idx: int, num_positions: int):  # SOURCE: vllm/v1/outputs.py:L41-L50
        if self.cu_num_generated_tokens is not None:
            req_idx = self.cu_num_generated_tokens[req_idx]
        end_idx = req_idx + num_positions
        return LogprobsLists(
            self.logprob_token_ids[req_idx:end_idx],
            self.logprobs[req_idx:end_idx],
            self.sampled_token_ranks[req_idx:end_idx],
            None,
        )


class LogprobsTensors(NamedTuple):  # SOURCE: vllm/v1/outputs.py:L53-L137
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprob_token_ids: torch.Tensor
    # [num_reqs x num_generated_tokens, max_num_logprobs + 1]
    logprobs: torch.Tensor
    # [num_reqs x num_generated_tokens]
    selected_token_ranks: torch.Tensor
    # [num_reqs]
    cu_num_generated_tokens: list[int] | None = None

    def tolists(self, cu_num_generated_tokens: list[int] | None = None):  # SOURCE: vllm/v1/outputs.py:L63-L71
        return LogprobsLists(
            self.logprob_token_ids.cpu().numpy(),
            self.logprobs.cpu().numpy(),
            self.selected_token_ranks.cpu().numpy(),
            cu_num_generated_tokens
            if cu_num_generated_tokens is not None
            else self.cu_num_generated_tokens,
        )

    def to_cpu_nonblocking(self) -> "LogprobsTensors":  # SOURCE: vllm/v1/outputs.py:L73-L81
        if self.logprob_token_ids.device.type == "cpu":
            return self
        return LogprobsTensors(
            self.logprob_token_ids.to("cpu", non_blocking=True),
            self.logprobs.to("cpu", non_blocking=True),
            self.selected_token_ranks.to("cpu", non_blocking=True),
            self.cu_num_generated_tokens,
        )

    # SUBTRACTED: filter（L83-L92）/ cat（L94-L118）——delete 项 2 的 spec
    # decode 域辅助（ch31 讲），本章路径不触达

    @staticmethod
    def empty_cpu(  # SOURCE: vllm/v1/outputs.py:L120-L137
        num_positions: int, num_tokens_per_position: int
    ) -> "LogprobsTensors":
        """Create empty LogprobsTensors on CPU."""

        logprob_token_ids = torch.empty(
            (num_positions, num_tokens_per_position), dtype=torch.int32, device="cpu"
        )
        logprobs = torch.empty_like(logprob_token_ids, dtype=torch.float32)
        selected_token_ranks = torch.empty(
            num_positions, dtype=torch.int32, device="cpu"
        )
        return LogprobsTensors(
            logprob_token_ids=logprob_token_ids,
            logprobs=logprobs,
            selected_token_ranks=selected_token_ranks,
        )


@dataclass  # SOURCE: vllm/v1/outputs.py:L212-L219
class SamplerOutput:
    # [num_reqs, max_num_generated_tokens]
    # Different requests can have different number of generated tokens.
    # All requests are padded to max_num_generated_tokens.
    # PLACEHOLDER_TOKEN_ID (-1 by default) is used for padding.
    sampled_token_ids: torch.Tensor
    logprobs_tensors: LogprobsTensors | None


# ModelRunnerOutput is serialized and sent to the scheduler process.
# This is expensive for torch.Tensor so prefer to use list instead.
@dataclass  # SOURCE: vllm/v1/outputs.py:L260-L308
class ModelRunnerOutput:
    # [num_reqs]
    req_ids: list[str]
    # req_id -> index
    req_id_to_index: dict[str, int]

    # num_reqs x num_generated_tokens
    # num_generated_tokens is the number of tokens
    # generated in the current step. It can be different for
    # each request due to speculative/jump decoding.
    sampled_token_ids: list[list[int]] = field(default_factory=list)

    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs, max_num_logprobs + 1]
    # [num_reqs]
    logprobs: LogprobsLists | None = None

    # req_id -> (token_ids, logprobs, ranks)
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len, num_prompt_logprobs]
    # [prompt_len]
    prompt_logprobs_dict: dict[str, LogprobsTensors | None] = field(
        default_factory=dict
    )

    # SUBTRACTED: pooler_output / kv_connector_output / ec_connector_output /
    # num_nans_in_logits / cudagraph_stats / routed_experts（L287-L308）——
    # delete 项 2/3 的非本章域字段


# ============================================================================
# §5 vllm/v1/sample/ops/logprobs.py — the count-based rank kernel
# ============================================================================


@torch.compile(backend=current_platform.simple_compile_backend)  # SOURCE: vllm/v1/sample/ops/logprobs.py:L10
def batched_count_greater_than(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:  # SOURCE: vllm/v1/sample/ops/logprobs.py:L11-L27
    """
    Counts elements in each row of x that are greater than the corresponding
    value in values.  Use torch.compile to generate an optimized kernel for
    this function. otherwise, it will create additional copies of the input
    tensors and cause memory issues.

    Args:
        x (torch.Tensor): A 2D tensor of shape (batch_size, n_elements).
        values (torch.Tensor): A 2D tensor of shape (batch_size, 1).

    Returns:
        torch.Tensor: A 1D tensor of shape (batch_size,) with the counts.
    """
    torch._check(x.shape[0] >= 1)
    torch._check(x.shape[0] == values.shape[0])
    return (x >= values).sum(-1)


# ============================================================================
# §6 vllm/v1/sample/metadata.py — SamplingMetadata field face (the logprobs
# fields the sampler reads; the sampling-side fields belong to Part VII)
# ============================================================================


@dataclass  # SOURCE: vllm/v1/sample/metadata.py:L14-L57 — logprobs 触达面
class SamplingMetadata:
    temperature: torch.Tensor | None
    all_greedy: bool
    all_random: bool

    # None means no logprobs, 0 means sampled token logprobs only
    max_num_logprobs: int | None

    # Specific token IDs to compute logprobs for (more efficient than full vocab)
    # When set, logprobs are computed only for these token IDs using gather
    # req_index -> list of token IDs to get logprobs for
    logprob_token_ids: dict[int, list[int]] | None = None

    # SUBTRACTED: top_p/top_k/generators/no_penalties/prompt_token_ids/
    # frequency/presence/repetition_penalties/output_token_ids/
    # allowed_token_ids_mask/bad_words_token_ids/logitsprocs/spec_token_ids/
    # thinking_budget_state_holder（L19-L57）——delete 项 1 的采样域


# ============================================================================
# §7 vllm/v1/sample/sampler.py — the logprobs steps of the 9-step pipeline
# ============================================================================


_SAMPLING_EPS = 1e-5  # SOURCE: vllm/v1/sample/sampler.py:L17


class Sampler:  # SOURCE: vllm/v1/sample/sampler.py:L20-L59（docstring 9 步管线，第 1/8 步是本章域）
    """
    A layer that samples the next tokens from the model's outputs
    with the following steps in order:

    1. If logprobs are requested:
        a) If `logprobs_mode` is `raw_logprobs`, compute logprobs
           as the final logprobs to return.
        b) If `logprobs_mode` is `raw_logits`, clone the logits
           as the final logprobs to return.
    2. Convert logits to float32.
    3. Apply allowed token ids whitelist.
    4. Apply bad words exclusion.
    5. Apply logit processors which are not argmax-invariant,
       i.e. that can impact greedy sampling.
        a) Min tokens processor
        b) Logit bias processor
    6. Apply penalties
        a) Repetition penalty
        b) Frequency penalty
        c) Presence penalty
    7. Sample the next tokens. `sample` method performs the following steps:
        a) If not `all_random`, perform greedy sampling. If `all_greedy`,
           return the greedily sampled tokens and final logprobs if requested.
        b) Apply temperature.
        c) Apply logit processors which are argmax-invariant, by default
           the min_p processor.
        d) Apply top_k and/or top_p.
        e) Sample the next tokens with the probability distribution.
        f) If `all_random` or temperature >= epsilon (1e-5), return the
           randomly sampled tokens and final logprobs if requested. Else,
           return the greedily sampled tokens and logprobs if requested.
    8. Gather the logprobs of the top `max_num_logprobs` and sampled token
       (if requested). Note that if the sampled token is within the top
       `max_num_logprobs`, the logprob will be eventually merged in
       `LogprobsProcessor` during output processing. Therefore, the
       final output may contain either `max_num_logprobs + 1` or
       `max_num_logprobs` logprobs.
    9. Return the final `SamplerOutput`.
    """

    def __init__(  # SOURCE: vllm/v1/sample/sampler.py:L61-L70
        self,
        logprobs_mode: LogprobsMode = "raw_logprobs",
        use_fp64_gumbel: bool = False,
    ):
        super().__init__()  # SUBTRACTED: nn.Module 基类（真实源继承 nn.Module；
        # 本章不触达模块机制，省 torch.nn 依赖面）
        # SUBTRACTED: topk_topp_sampler = TopKTopPSampler(...)（L67）——
        # delete 项 1：随机路径采样器归 Part VII
        self.pin_memory = PIN_MEMORY
        self.logprobs_mode = logprobs_mode
        # SUBTRACTED: use_fp64_gumbel（L70）——Gumbel 路径域

    def forward(  # SOURCE: vllm/v1/sample/sampler.py:L72-L149
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool = False,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> SamplerOutput:
        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        # NOTE(woosuk): Use the original logits (before any penalties or
        # temperature scaling) for the top-k logprobs.
        # This is different from the V0 sampler, which uses the logits that
        # is used for sampling (after penalties and temperature scaling).
        num_logprobs = sampling_metadata.max_num_logprobs
        raw_logprobs: torch.Tensor | None = None
        if num_logprobs is not None or sampling_metadata.logprob_token_ids:
            if logprobs_mode == "raw_logprobs":
                raw_logprobs = self.compute_logprobs(logits)
            elif logprobs_mode == "raw_logits":
                if logits.dtype == torch.float32:
                    raw_logprobs = logits.clone()
                else:
                    raw_logprobs = logits.to(torch.float32)

        # Use float32 for the logits.
        logits = logits.to(torch.float32)

        logits = self.apply_logits_processors(
            logits, sampling_metadata, predict_bonus_token
        )
        # Sample the next token.
        sampled, processed_logprobs = self.sample(logits, sampling_metadata)
        if processed_logprobs is not None:
            raw_logprobs = processed_logprobs
        # Convert sampled token ids to int64 (long) type to ensure compatibility
        # with subsequent operations that may use these values as indices.
        # This conversion is necessary because FlashInfer sampling operations
        # return int32 (while PyTorch argmax and topk return int64).
        sampled = sampled.long()

        # Handle logprob_token_ids if specified (more efficient than full vocab)
        # This is used by generative_scoring API to get logprobs for specific tokens
        logprob_token_ids_tensors = None
        if sampling_metadata.logprob_token_ids:
            assert raw_logprobs is not None
            logprob_token_ids_tensors = self.gather_specific_token_logprobs(
                raw_logprobs, sampling_metadata.logprob_token_ids, sampled
            )

        if num_logprobs is None:
            logprobs_tensors = logprob_token_ids_tensors
        elif num_logprobs == -1:
            # Return the full unsorted and unranked logprobs.
            logprobs_tensors = LogprobsTensors(
                torch.empty(0), raw_logprobs, torch.empty(0)
            )
        else:
            # Gather the logprobs and ranks of the topk and sampled token.
            logprobs_tensors = self.gather_logprobs(
                raw_logprobs, num_logprobs, token_ids=sampled
            )

        # If we have both num_logprobs and logprob_token_ids, prefer
        # logprob_token_ids as it's more specific
        if logprob_token_ids_tensors is not None and num_logprobs is not None:
            logprobs_tensors = logprob_token_ids_tensors

        # Use int32 to reduce the tensor size.
        sampled = sampled.to(torch.int32)

        # These are GPU tensors.
        sampler_output = SamplerOutput(
            # The sampled tokens are expanded to 2D tensor with shape
            # [num_requests, 1], where each row represents one generated
            # token per request.
            sampled_token_ids=sampled.unsqueeze(-1),
            logprobs_tensors=logprobs_tensors,
        )
        return sampler_output

    def gather_specific_token_logprobs(  # SOURCE: vllm/v1/sample/sampler.py:L151-L225
        self,
        logprobs: torch.Tensor,
        logprob_token_ids: dict[int, list[int]],
        sampled: torch.Tensor,
    ) -> LogprobsTensors | None:
        """Gather logprobs for specific token IDs requested per request.

        Used by the generative_scoring API to return logprobs for an explicit
        set of token ids rather than the top-k. Handles heterogeneous token
        id lists across requests by padding shorter lists to the max length.

        Args:
            logprobs: [batch_size, vocab_size] tensor of (raw) logprobs to
                gather from.
            logprob_token_ids: dict mapping req_index -> list of token IDs
            sampled: [batch_size] tensor of sampled token IDs

        Returns:
            LogprobsTensors with logprobs for the specified tokens, or None
            if no requests have logprob_token_ids.
        """
        if not logprob_token_ids:
            return None

        batch_size = logprobs.shape[0]
        device = logprobs.device

        # Find max number of tokens across all requests
        max_num_tokens = max(len(tids) for tids in logprob_token_ids.values())
        pin = self.pin_memory

        # Build the padded token_ids and valid_mask matrices on pinned CPU,
        # then upload non-blocking.
        # SUBTRACTED: pinned CPU 逐位填充循环（L185-L198）——delete 项 10：
        # worked example 只需『列 0=sampled、其余=指定 ids、无效位 -inf』的
        # 结构；此处按同一契约直接构造（padding + mask + gather 三步核心）
        token_ids_cpu = torch.zeros(
            batch_size, max_num_tokens + 1, dtype=torch.int64, pin_memory=pin
        )
        # Create mask for valid positions (True = valid, False = padded)
        valid_mask_cpu = torch.zeros(
            batch_size, max_num_tokens + 1, dtype=torch.bool, pin_memory=pin
        )
        valid_mask_cpu[:, 0] = True  # Sampled token is always valid
        for req_idx, token_ids in logprob_token_ids.items():
            num_tokens = len(token_ids)
            token_ids_cpu[req_idx, 1 : num_tokens + 1] = torch.as_tensor(
                token_ids, dtype=torch.int64
            )
            valid_mask_cpu[req_idx, 1 : num_tokens + 1] = True

        token_ids_tensor = token_ids_cpu.to(device, non_blocking=True)
        valid_mask = valid_mask_cpu.to(device, non_blocking=True)
        # Sampled token in column 0 — fill on-device from the sampled GPU
        # tensor so we don't need to D2H + re-upload.
        token_ids_tensor[:, 0] = sampled

        # Gather logprobs at the requested token ids.
        gathered_logprobs = logprobs.gather(-1, token_ids_tensor)

        # Mask invalid (padded) positions with -inf
        gathered_logprobs = gathered_logprobs.masked_fill(~valid_mask, float("-inf"))

        # Compute ranks for the sampled token. log_softmax is monotonic w.r.t.
        # the original logits, so ranks computed from logprobs are equivalent.
        sampled_logprobs = logprobs.gather(-1, sampled.unsqueeze(-1))
        # Avoid 0/1 specialization recompile on the batch dimension of the
        # compiled batched_count_greater_than. See gather_logprobs for context.
        torch._dynamo.decorators.mark_unbacked(logprobs, 0)
        torch._dynamo.decorators.mark_unbacked(sampled_logprobs, 0)
        token_ranks = batched_count_greater_than(logprobs, sampled_logprobs)

        return LogprobsTensors(
            logprob_token_ids=token_ids_tensor.to(torch.int32),
            logprobs=gathered_logprobs,
            selected_token_ranks=token_ranks,
        )

    # SUBTRACTED: apply_temperature（L227-L237）——delete 项 1：随机路径域

    @staticmethod
    def greedy_sample(logits: torch.Tensor) -> torch.Tensor:  # SOURCE: vllm/v1/sample/sampler.py:L239-L241
        return logits.argmax(dim=-1).view(-1)

    def sample(  # SOURCE: vllm/v1/sample/sampler.py:L243-L302
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        logprobs_mode_override: LogprobsMode | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Sample logits based on sampling metadata.

        The various logits processing functions called in this method
        may update the logits tensor in-place.
        """

        logprobs_mode = logprobs_mode_override or self.logprobs_mode
        assert not (sampling_metadata.all_greedy and sampling_metadata.all_random)
        if sampling_metadata.all_random:
            greedy_sampled = None
            # SUBTRACTED: 随机路径（L273-L302：apply_temperature / argmax
            # invariant processors / TopKTopPSampler / torch.where 合并）——
            # delete 项 1，归 Part VII ch29-33
            raise NotImplementedError(
                "random sampling path subtracted (Part VII)"
            )  # HOST 注：保住全随机批的可观察失败面（greedy 留底已算好）
        else:
            greedy_sampled = self.greedy_sample(logits)
            if sampling_metadata.all_greedy:
                processed_logprobs = None
                if (
                    sampling_metadata.max_num_logprobs is not None
                    or sampling_metadata.logprob_token_ids
                ):
                    if logprobs_mode == "processed_logits":
                        processed_logprobs = logits
                    elif logprobs_mode == "processed_logprobs":
                        processed_logprobs = self.compute_logprobs(logits)
                return greedy_sampled, processed_logprobs

        assert sampling_metadata.temperature is not None
        # SUBTRACTED: 随机路径尾部（L275-L302）——delete 项 1
        raise NotImplementedError("random sampling path subtracted (Part VII)")

    @staticmethod
    def compute_logprobs(logits: torch.Tensor) -> torch.Tensor:  # SOURCE: vllm/v1/sample/sampler.py:L304-L306
        return logits.log_softmax(dim=-1, dtype=torch.float32)

    @staticmethod
    def gather_logprobs(  # SOURCE: vllm/v1/sample/sampler.py:L308-L356
        logprobs: torch.Tensor,
        num_logprobs: int,
        token_ids: torch.Tensor,
    ) -> LogprobsTensors:
        """
        Gather logprobs for topk and sampled/prompt token.

        Args:
          logprobs: (num tokens) x (vocab) tensor
          num_logprobs: maximum number of logprobs to
                        retain per token
          token_ids: prompt tokens (if prompt logprobs)
                     or sampled tokens (if sampled
                     logprobs); 1D token ID tensor
                     with (num tokens) elements
                     Must be int64.

        Returns:
          Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
          Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
          Sampled token rank tensor, (num tokens)
        """
        assert token_ids.dtype == torch.int64
        # Find the topK values.
        topk_logprobs, topk_indices = torch.topk(logprobs, num_logprobs, dim=-1)

        # Get with the logprob of the prompt or sampled token.
        token_ids = token_ids.unsqueeze(-1)
        token_logprobs = logprobs.gather(-1, token_ids)

        # Compute the ranks of the actual token.
        # Avoid 0/1 specialization recompile on the batch dimension
        # of the compiled batched_count_greater_than. mark_unbacked makes
        # the size fully symbolic so dynamo doesn't specialize when
        # batch_size transitions from 1 to >=2.
        torch._dynamo.decorators.mark_unbacked(logprobs, 0)
        torch._dynamo.decorators.mark_unbacked(token_logprobs, 0)
        token_ranks = batched_count_greater_than(logprobs, token_logprobs)

        # Concatenate together with the topk.
        indices = torch.cat((token_ids, topk_indices), dim=1)
        logprobs = torch.cat((token_logprobs, topk_logprobs), dim=1)

        # Use int32 to reduce the tensor size.
        indices = indices.to(torch.int32)

        return LogprobsTensors(indices, logprobs, token_ranks)

    # SUBTRACTED: _combine_outputs_with_spec_tokens（L358-L369）——delete 项 1
    # 的 spec decode 域

    def apply_logits_processors(  # SOURCE: vllm/v1/sample/sampler.py:L371-L417（调用点保留，实现体 delete 项 1）
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
        predict_bonus_token: bool,
    ) -> torch.Tensor:
        # SUBTRACTED: 实现体（L377-L416：allowed_token_ids mask / bad_words /
        # min_tokens/logit_bias 处理器 / apply_all_penalties / thinking
        # budget）——delete 项 1：处理器实现归 Part VII ch29-33。空处理器批
        # （无惩罚/无约束）时真实实现同样原样返回 logits。
        return logits

    # SUBTRACTED: apply_penalties（L419-L436）——delete 项 1


# ============================================================================
# §8 vllm/v1/worker/gpu_input_batch.py — batch registration (station 2)
# ============================================================================


@dataclass  # SOURCE: vllm/v1/worker/gpu_input_batch.py:L34-L57 — 本章触达面
class CachedRequestState:
    req_id: str
    prompt_token_ids: list[int] | None
    sampling_params: Any | None
    num_computed_tokens: int
    # To accumulate prompt logprobs tensor chunks across prefill steps.
    in_progress_prompt_logprobs_cpu: LogprobsTensors | None = None
    # SUBTRACTED: mm_features/generator/block_ids/output_token_ids/mrope/
    # xdrope/lora/prompt_embeds/prompt_is_token_ids（L38-L43,L46-L59）——
    # 非 logprobs 域字段面


# SOURCE: vllm/v1/worker/gpu_input_batch.py InputBatch — 本章域 = logprobs
# 登记（num_logprobs 字典 + 批级 max + logprob_token_ids）；批的其余机制
# （persistent batch 重排/cudagraph/tensor 面）归 ch4/ch9 域，按 delete 项 3 精简
class InputBatch:
    def __init__(self, vocab_size: int = 32000):  # SOURCE: vllm/v1/worker/gpu_input_batch.py:L269-L273（字段声明）+ L303
        self.vocab_size = vocab_size
        self.num_logprobs: dict[str, int] = {}

        # req_id -> list of specific token IDs to compute logprobs for
        # More efficient than num_logprobs=-1 when only a few tokens are needed
        self.logprob_token_ids: dict[str, list[int]] = {}
        self.req_id_to_index: dict[str, int] = {}
        # SUBTRACTED: 批的 persistent 状态面（温度/top_p/top_k CPU 张量、
        # penalties、generators、allowed_token_ids、logitsprocs、
        # spec_token_ids、batch_update_builder 等 L241-L303）——ch4/ch9 域
        self.sampling_metadata = self._make_sampling_metadata()

    def add_request(  # SOURCE: vllm/v1/worker/gpu_input_batch.py:L435-L444（登记位）
        self,
        req_id: str,
        req_index: int,
        sampling_params: SamplingParams,
    ) -> None:
        # SUBTRACTED: penalties/generators/allowed_token_ids 登记段
        # （L412-L434, L446-L464）——delete 项 3 的非本章域
        if sampling_params.logprobs is not None:
            self.num_logprobs[req_id] = (
                self.vocab_size
                if sampling_params.logprobs == -1
                else sampling_params.logprobs
            )

        # Store specific token IDs to compute logprobs for (more efficient)
        if sampling_params.logprob_token_ids is not None:
            self.logprob_token_ids[req_id] = sampling_params.logprob_token_ids
        self.req_id_to_index[req_id] = req_index

    def remove_request(self, req_id: str) -> None:  # SOURCE: vllm/v1/worker/gpu_input_batch.py:L530-L574（def L530；logprobs 弹出位 L573-L574）
        # SUBTRACTED: greedy/random/top_p/top_k/penalties/generators 弹出段
        # （L565-L572）——ch9 域
        self.num_logprobs.pop(req_id, None)
        self.logprob_token_ids.pop(req_id, None)
        self.req_id_to_index.pop(req_id, None)

    @property
    def max_num_logprobs(self) -> int | None:  # SOURCE: vllm/v1/worker/gpu_input_batch.py:L1149-L1151
        return max(self.num_logprobs.values()) if self.num_logprobs else None

    def _make_sampling_metadata(self) -> SamplingMetadata:  # SOURCE: vllm/v1/worker/gpu_input_batch.py:L860-L956
        # SUBTRACTED: 温度/top_p/top_k/penalties/prompt_token_ids/
        # output_token_ids/allowed_token_ids 掩码准备段（L861-L933）——
        # delete 项 1/3 的采样域
        # HOST 注：全贪心面——简化批未登记温度（真实判据 greedy_reqs 全覆盖
        # 在被删的 L935-L941），恒贪心批是本章可观察行为面
        greedy = True

        # Build per-request logprob_token_ids mapping: req_index -> token_ids
        logprob_token_ids_by_index: dict[int, list[int]] | None = None
        if self.logprob_token_ids:
            logprob_token_ids_by_index = {}
            for req_id, token_ids in self.logprob_token_ids.items():
                if req_id in self.req_id_to_index:
                    req_index = self.req_id_to_index[req_id]
                    logprob_token_ids_by_index[req_index] = token_ids

        return SamplingMetadata(
            temperature=None,  # SUBTRACTED: 温度面（L862-L867）——简化批恒贪心
            all_greedy=greedy,
            all_random=not greedy,
            max_num_logprobs=self.max_num_logprobs,
            logprob_token_ids=logprob_token_ids_by_index,
        )


# ============================================================================
# §9 vllm/v1/engine/__init__.py — the wire structs (station 7)
# ============================================================================


# These are possible values for RequestOutput.finish_reason,
# so form part of the external API.
FINISH_REASON_STRINGS = ("stop", "length", "abort", "error", "repetition")  # SOURCE: vllm/v1/engine/__init__.py:L29-L31


class FinishReason(enum.IntEnum):  # SOURCE: vllm/v1/engine/__init__.py:L43-L65
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


# SOURCE: vllm/v1/engine/__init__.py:L97-L146 EngineCoreRequest — 字段全保留
# （线格式 schema 契约）；mm/lora/pooling 等 ch6/ch7 域字段类型放宽为 Any
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

    # Per-position mask for mixed-mode inputs (e.g chat completion with
    # prompt_embeds content parts). `True` means the position is a real
    # token ID; `False` means the position uses a pre-computed entry from
    # `prompt_embeds`. `None` for pure-tokens and pure-embeds requests.
    prompt_is_token_ids: list[bool] | None = None

    # Index of the client, used to ensure outputs are sent back to the same
    # client for this request when scaling out the front-end.
    client_index: int = 0

    # Used in DP case to indicate which wave of requests this is expected to
    # belong to, to cover a race condition where the request is sent before
    # a wave finished notification is received.
    current_wave: int = 0
    priority: int = 0

    trace_headers: Any | None = None
    resumable: bool = False

    # The user-provided request ID. This field is set internally,
    # copied from the provided request_id that's originally assigned
    # to the request_id field, see InputProcessor.assign_request_id().
    # Used in outputs and to support abort(req_id, internal=False).
    external_req_id: str | None = None

    # SUBTRACTED: reasoning_ended / reasoning_parser_kwargs /
    # abort_immediately（L154-L160）——ch7 域字段（线 schema 兼容：array_like
    # 位置编码下省尾字段对解码无害）


class EngineCoreOutput(  # SOURCE: vllm/v1/engine/__init__.py:L184-L215
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    request_id: str
    new_token_ids: list[int]

    new_logprobs: LogprobsLists | None = None
    new_prompt_logprobs_tensors: LogprobsTensors | None = None

    # SUBTRACTED: pooling_output/events/kv_transfer_params/ec_transfer_params/
    # trace_headers/prefill_stats/routed_experts/num_nans_in_logits
    # （L196-L211）——delete 项 3：ch7 已整体讲过此结构，本章只聚焦
    # L193-L194 两字段（线 schema 兼容：省尾字段对 array_like 解码无害）
    finish_reason: FinishReason | None = None
    stop_reason: int | str | None = None

    @property
    def finished(self) -> bool:  # SOURCE: vllm/v1/engine/__init__.py:L213-L215
        return self.finish_reason is not None


class EngineCoreOutputs(  # SOURCE: vllm/v1/engine/__init__.py:L230-L258
    msgspec.Struct,
    array_like=True,  # type: ignore[call-arg]
    omit_defaults=True,  # type: ignore[call-arg]
    gc=False,
):  # type: ignore[call-arg]
    # NOTE(Nick): We could consider ways to make this more compact,
    # e.g. columnwise layout

    engine_index: int = 0

    # [num_reqs]
    outputs: list[EngineCoreOutput] = []
    # SUBTRACTED: scheduler_stats/timestamp/utility_output/finished_requests/
    # wave_complete/start_wave（L243-L254）——delete 项 3；行式布局 NOTE 逐字
    # 保留（WC4 的『班车』注脚）


# ============================================================================
# §10 vllm/v1/serial_utils.py — the ndarray/tensor crossing hooks (m5)
# ============================================================================


CUSTOM_TYPE_PICKLE = 1  # SOURCE: vllm/v1/serial_utils.py:L41
CUSTOM_TYPE_CLOUDPICKLE = 2  # SOURCE: vllm/v1/serial_utils.py:L42
CUSTOM_TYPE_RAW_VIEW = 3  # SOURCE: vllm/v1/serial_utils.py:L43

bytestr: TypeAlias = bytes | bytearray | memoryview  # SOURCE: vllm/v1/serial_utils.py:L54（zmq.Frame 面省——无 zmq 依赖）


class MsgpackEncoder:  # SOURCE: vllm/v1/serial_utils.py:L136-L178
    """Encoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Encoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.

    By default, arrays below 256B are serialized inline Larger will get sent
    via dedicated messages. Note that this is a per-tensor limit.
    """

    def __init__(
        self,
        size_threshold: int | None = None,
    ):
        if size_threshold is None:
            size_threshold = 256  # SOURCE: vllm/envs.py VLLM_MSGPACK_ZERO_COPY_THRESHOLD（默认 256）
        self.encoder = msgpack.Encoder(enc_hook=self.enc_hook)
        # This is used as a local stash of buffers that we can then access from
        # our custom `msgspec` hook, `enc_hook`. We don't have a way to
        # pass custom data to the hook otherwise.
        self.aux_buffers: list[bytestr] | None = None
        self.size_threshold = size_threshold
        # SUBTRACTED: oob_tensor_consumer 与 VLLM_ALLOW_INSECURE_SERIALIZATION
        # 告警（L152,L162-L164）——delete 项 4：OOB 面省

    def encode(self, obj: Any) -> Sequence[bytestr]:  # SOURCE: vllm/v1/serial_utils.py:L166-L178
        try:
            self.aux_buffers = bufs = [b""]
            bufs[0] = self.encoder.encode(obj)
            # This `bufs` list allows us to collect direct pointers to backing
            # buffers of tensors and np arrays, and return them along with the
            # top-level encoded buffer instead of copying their data into the
            # new buffer.
            return bufs
        finally:
            self.aux_buffers = None

    # SUBTRACTED: encode_into（L180-L189）——delete 项 4：多帧零拷贝细节

    def enc_hook(self, obj: Any) -> Any:  # SOURCE: vllm/v1/serial_utils.py:L191-L197
        if isinstance(obj, torch.Tensor):
            return self._encode_tensor(obj)

        # Fall back to pickle for object or void kind ndarrays.
        if isinstance(obj, np.ndarray) and obj.dtype.kind not in ("O", "V"):
            return self._encode_ndarray(obj)

        # SUBTRACTED: slice 钩子（L199-L204，BatchUpdate 用）与多模态/
        # UtilityResult/pickle 回退分支（L206-L226）——delete 项 4
        raise TypeError(f"Cannot serialize type {type(obj)}")  # HOST 注：未知类型默认硬错误（真实面）

    def _encode_ndarray(  # SOURCE: vllm/v1/serial_utils.py:L237-L255
        self, obj: np.ndarray
    ) -> tuple[str, tuple[int, ...], int | memoryview]:
        assert self.aux_buffers is not None
        # If the array is non-contiguous, we need to copy it first
        arr_data = obj.data if obj.flags.c_contiguous else obj.tobytes()
        if not obj.shape or obj.nbytes < self.size_threshold:
            # Encode small arrays and scalars inline. Using this extension type
            # ensures we can avoid copying when decoding.
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, arr_data)
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr_data)

        # We serialize the ndarray as a tuple of native types.
        # The data is either inlined if small, or an index into a list of
        # backing buffers that we've stashed in `aux_buffers`.
        return obj.dtype.str, obj.shape, data

    def _encode_tensor(  # SOURCE: vllm/v1/serial_utils.py:L257-L273
        self, obj: torch.Tensor
    ) -> tuple[str, tuple[int, ...], int | dict | memoryview]:
        # view the tensor as a contiguous 1D array of bytes
        if obj.nbytes < self.size_threshold and obj.is_cpu:
            # Smaller tensors are encoded inline, just like ndarrays.
            data = msgpack.Ext(CUSTOM_TYPE_RAW_VIEW, tensor_data(obj))
        else:
            # Otherwise encode index of backing buffer to avoid copy.
            assert self.aux_buffers is not None
            data = len(self.aux_buffers)
            self.aux_buffers.append(tensor_data(obj))
        dtype = str(obj.dtype).removeprefix("torch.")
        return dtype, obj.shape, data

    # SUBTRACTED: _encode_mm_* 家族与 UtilityResult/pickle 回退（L275-L310）——
    # delete 项 4


# SOURCE: vllm/v1/utils.py:L777-L787 tensor_data — verbatim
def tensor_data(tensor: torch.Tensor) -> memoryview:
    """Get the raw data of a tensor as a uint8 memoryview, useful for
    serializing and hashing.

    Args:
        tensor: The input tensor.

    Returns:
        A memoryview of the tensor data as uint8.
    """
    return tensor.flatten().cpu().contiguous().view(torch.uint8).numpy().data


class MsgpackDecoder:  # SOURCE: vllm/v1/serial_utils.py:L313-L348
    """Decoder with custom torch tensor and numpy array serialization.

    Note that unlike vanilla `msgspec` Decoders, this interface is generally
    not thread-safe when encoding tensors / numpy arrays.
    """

    def __init__(  # SOURCE: vllm/v1/serial_utils.py:L323-L334
        self,
        t: Any | None = None,
        share_mem: bool = True,
    ):
        self.share_mem = share_mem
        self.pin_tensors = PIN_MEMORY
        args = () if t is None else (t,)
        self.decoder = msgpack.Decoder(
            *args, ext_hook=self.ext_hook, dec_hook=self.dec_hook
        )
        self.aux_buffers: Any = ()
        # SUBTRACTED: oob_tensor_provider 与 VLLM_ALLOW_INSECURE_SERIALIZATION
        # 告警（L327,L333-L334）——delete 项 4

    def ext_hook(self, code: int, data: memoryview) -> Any:  # SOURCE: vllm/v1/serial_utils.py:L473-L484
        if code == CUSTOM_TYPE_RAW_VIEW:
            return data

        # SUBTRACTED: CUSTOM_TYPE_PICKLE/CLOUDPICKLE 不安全回退分支
        # （L477-L482，VLLM_ALLOW_INSECURE_SERIALIZATION 门控）——delete 项 4
        raise NotImplementedError(f"Extension type code {code} is not supported")

    def decode(self, bufs: bytestr | Any) -> Any:  # SOURCE: vllm/v1/serial_utils.py:L340-L348
        if isinstance(bufs, (bytes, bytearray, memoryview)):
            return self.decoder.decode(bufs)

        self.aux_buffers = bufs
        try:
            return self.decoder.decode(bufs[0])
        finally:
            self.aux_buffers = ()

    def dec_hook(self, t: type, obj: Any) -> Any:  # SOURCE: vllm/v1/serial_utils.py:L350-L365
        # Given native types in `obj`, convert to type `t`.
        if isclass(t):
            if issubclass(t, np.ndarray):
                return self._decode_ndarray(obj)
            if issubclass(t, torch.Tensor):
                return self._decode_tensor(obj)
            # SUBTRACTED: slice/MultiModalKwargsItem/MultiModalKwargsItems/
            # UtilityResult 分支（L357-L364）——delete 项 4
        return obj

    def _decode_ndarray(self, arr: Any) -> np.ndarray:  # SOURCE: vllm/v1/serial_utils.py:L389-L397
        dtype, shape, data = arr
        # zero-copy decode. We assume the ndarray will not be kept around,
        # as it now locks the whole received message buffer in memory.
        buffer = self.aux_buffers[data] if isinstance(data, int) else data
        arr = np.frombuffer(buffer, dtype=dtype)
        if not self.share_mem:
            arr = arr.copy()
        return arr.reshape(shape)

    def _decode_tensor(self, arr: Any) -> torch.Tensor:  # SOURCE: vllm/v1/serial_utils.py:L399-L425
        dtype, shape, data = arr
        # SUBTRACTED: oob_tensor_provider 分支（L401-L405）——delete 项 4
        is_aux = isinstance(data, int)
        buffer = self.aux_buffers[data] if is_aux else data
        buffer = buffer if isinstance(buffer, memoryview) else memoryview(buffer)
        torch_dtype = getattr(torch, dtype)
        assert isinstance(torch_dtype, torch.dtype)
        if not buffer.nbytes:  # torch.frombuffer doesn't like empty buffers
            assert 0 in shape
            return torch.empty(shape, dtype=torch_dtype)
        # Create uint8 array
        arr = torch.frombuffer(buffer, dtype=torch.uint8)
        # Clone ensures tensor is backed by pytorch-owned memory for safe
        # future async CPU->GPU transfer.
        # Pin larger tensors for more efficient CPU->GPU transfer.
        if not is_aux:
            arr = arr.clone()
        elif not self.share_mem:
            arr = arr.pin_memory() if self.pin_tensors else arr.clone()
        # Convert back to proper shape & type
        return arr.view(torch_dtype).view(shape)


# Sequence face for the encoder return type (real file imports it from
# collections.abc at serial_utils.py:L8).
from collections.abc import Sequence  # noqa: E402  # SOURCE: vllm/v1/serial_utils.py:L8


# ============================================================================
# §11 vllm/v1/worker/gpu_model_runner.py — D2H wrap + prompt sub-branch
# ============================================================================


# Wrapper for ModelRunnerOutput to support overlapped execution.
class AsyncGPUModelRunnerOutput:  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L258-L344（AsyncModelRunnerOutput ABC 面）
    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        logprobs_tensors: LogprobsTensors | None,
        invalid_req_indices: list[int],
        async_output_copy_stream: Any,  # torch.cuda.Stream 面（HOST SEAM）
        vocab_size: int,
        routed_experts: Any | None = None,  # SUBTRACTED 面：delete 项 2
        check_ep_fault: bool = False,  # SUBTRACTED 面：delete 项 2
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # Event on the copy stream so we can synchronize the non-blocking copy.
        # Blocking (sleep) event to avoid busy-polling the CUDA driver lock.
        self.async_copy_ready_event = _cuda_event(blocking=True)  # HOST SEAM（torch.cuda.Event）

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids
        self.vocab_size = vocab_size
        self._logprobs_tensors = logprobs_tensors
        # SUBTRACTED: routed_experts / _has_fault 字段（L283-L284）——delete 项 2

        # Initiate the copy on a separate stream, but do not synchronize it.
        default_stream = _cuda_current_stream()  # HOST SEAM（torch.cuda.current_stream）
        with _CudaStreamCtx(async_output_copy_stream):  # HOST SEAM（torch.cuda.stream）
            _cuda_wait_stream(async_output_copy_stream, default_stream)  # HOST SEAM（wait_stream）
            self.sampled_token_ids_cpu = self._sampled_token_ids.to(
                "cpu", non_blocking=True
            )
            self._logprobs_tensors_cpu = (
                self._logprobs_tensors.to_cpu_nonblocking()
                if self._logprobs_tensors
                else None
            )
            # SUBTRACTED: routed_experts 拷贝与 EP fault 查询（L298-L305）——
            # delete 项 2
            self.async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L308-L344
        """Copy the device tensors to the host and return a ModelRunnerOutput.

        This function blocks until the copy is finished.
        """
        max_gen_len = self.sampled_token_ids_cpu.shape[-1]
        self.async_copy_ready_event.synchronize()

        # Release the device tensors once the copy has completed.
        del self._logprobs_tensors
        del self._sampled_token_ids
        if max_gen_len == 1:
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            for i in self._invalid_req_indices:
                valid_sampled_token_ids[i].clear()
            logprobs_lists = None
            if self._logprobs_tensors_cpu is not None:
                logprobs_lists = self._logprobs_tensors_cpu.tolists()
        else:
            # SUBTRACTED: spec decode 的 RejectionSampler.parse_output 分支
            # （L326-L332）——delete 项 2，归 ch31；多 token 批按同一
            # tolists() 面处理
            valid_sampled_token_ids = self.sampled_token_ids_cpu.tolist()
            for i in self._invalid_req_indices:
                valid_sampled_token_ids[i].clear()
            logprobs_lists = None
            if self._logprobs_tensors_cpu is not None:
                logprobs_lists = self._logprobs_tensors_cpu.tolists()

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        output.logprobs = logprobs_lists

        # SUBTRACTED: routed_experts 装载与 EP fault 判定尾部（L338-L344）——
        # delete 项 2
        return output


# SOURCE: vllm/utils/torch_utils.py:L573-L584 async_tensor_h2d — verbatim
def async_tensor_h2d(
    data: list | np.ndarray | torch.Tensor,
    device: str | torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Copy list/numpy array/tensor async from host to device."""
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        t = data.pin_memory() if PIN_MEMORY else data
    else:
        t = torch.tensor(data, dtype=dtype, pin_memory=PIN_MEMORY, device="cpu")
    assert t.is_cpu
    return t.to(device=device, dtype=dtype, non_blocking=True)


# SOURCE: vllm/v1/worker/gpu_model_runner.py GPUModelRunner — 本章域 =
# _get_prompt_logprobs_dict（站 12）触达面；runner 的其余机制（KV/调度/
# cudagraph）归 ch4/ch9/ch13 域，按 delete 项 3/6 精简为字段面构造器
class GPUModelRunner:
    def __init__(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L676/L679（字段声明）+ L5620（方法面）
        self,
        requests: Any,
        num_prompt_logprobs: dict[str, int],
        model: Any,  # 两方法契约面：compute_logits(hidden_states)（WC2）
        model_config: Any,  # logprobs_mode 字段面
        sampler: Sampler,
        device: str = "cpu",
        query_start_loc: Any = None,  # .np 面的批内偏移账
        input_batch: Any = None,  # req_id_to_index 面
    ):
        self.requests: Any = requests
        self.num_prompt_logprobs: dict[str, int] = num_prompt_logprobs
        # SUBTRACTED: num_prompt_logprobs 的登记位在 _update_states
        # （L1312-L1317：prompt_logprobs 非 None 则 -1→vocab_size）——ch4 域，
        # 此处由构造面注入
        self.model = model
        self.model_config = model_config
        self.sampler = sampler
        self.device = device
        self.query_start_loc = query_start_loc
        self.input_batch = input_batch if input_batch is not None else SimpleNamespace(
            req_id_to_index={}
        )

    def _sync_device(self):  # SOURCE: vllm/v1/worker/gpu_model_runner.py:_sync_device 面（L5725 调用位）
        if torch.cuda.is_available():  # HOST SEAM：CPU 面无设备同步
            torch.cuda.synchronize()

    def _get_prompt_logprobs_dict(  # SOURCE: vllm/v1/worker/gpu_model_runner.py:L5620-L5727
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                # This can happen if the request was preempted in prefill stage.
                continue

            # Get metadata for this request.
            request = self.requests[req_id]
            # SUBTRACTED: prompt_embeds 不兼容 continue 分支（L5642-L5644）——
            # delete 项 6（保留 num_tokens is None 一个防御分支示意抢占）

            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = async_tensor_h2d(
                request.prompt_token_ids, device=self.device
            )

            # Set up target LogprobsTensors object.
            logprobs_tensors = request.in_progress_prompt_logprobs_cpu
            if logprobs_tensors is None:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                request.in_progress_prompt_logprobs_cpu = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            # SUBTRACTED: num_logits <= 0 continue 分支（L5677-L5681）——
            # delete 项 6（精确末块的边缘防御）

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # Compute prompt scores respecting logprobs_mode.
            # NOTE: prompt tokens skip sampling processors, so
            # processed_* and raw_* yield the same scores here.
            if self.model_config.logprobs_mode in ("raw_logits", "processed_logits"):
                scores = logits.to(torch.float32)
            else:
                scores = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks, _ = self.sampler.gather_logprobs(
                scores, num_prompt_logprobs, tgt_token_ids
            )

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True
            )
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            self.requests[req_id].in_progress_prompt_logprobs_cpu = None

        # SUBTRACTED: _sync_device 尾部（L5723-L5725）——delete 项 6（CPU 面
        # non_blocking 拷贝即时完成）
        return prompt_logprobs_dict


# ============================================================================
# §12 vllm/v1/core/sched/scheduler.py — the logprobs lines of
# update_from_output (station 6; the full body is ch9's chapter)
# ============================================================================


# SOURCE: vllm/v1/core/sched/scheduler.py:L69 Scheduler — 本章域 =
# update_from_output 的 logprobs 切行与装车行；调度器其余机制归 ch9，
# 按 delete 项 3 精简为 requests 字典面
class Scheduler:
    def __init__(self, requests: Any):  # SOURCE: vllm/v1/core/sched/scheduler.py:L69-L70 class Scheduler(SchedulerInterface) — HOST 字段面（真实构造参数 vllm_config 等归 ch9 域）
        self.requests: Any = requests

    def update_from_output(  # SOURCE: vllm/v1/core/sched/scheduler.py:L1670-L2059
        self,
        scheduler_output: Any,  # num_scheduled_tokens 字段面
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        sampled_token_ids = model_runner_output.sampled_token_ids
        logprobs = model_runner_output.logprobs
        prompt_logprobs_dict = model_runner_output.prompt_logprobs_dict
        num_scheduled_tokens = scheduler_output.num_scheduled_tokens
        # SUBTRACTED: pooler_outputs/num_nans/kv_connector/cudagraph_stats
        # 头部（L1679-L1682）与 deferred-free/perf_stats/kv-connector/routed
        # experts 准备段（L1684-L1726）——delete 项 3

        # NOTE(woosuk): As len(num_scheduled_tokens) can be up to 1K or more,
        # the below loop can be a performance bottleneck. We should do our best
        # to avoid expensive operations inside the loop.
        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
        # SUBTRACTED: spec_decoding_stats 与 stopped_reqs 集（L1695-L1732）
        for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
            # SUBTRACTED: in-flight 账/stale 输出/KV 失败跳过/已完成跳过
            # （L1734-L1759）——delete 项 3，ch9 域
            request = self.requests.get(req_id)

            req_index = model_runner_output.req_id_to_index[req_id]
            generated_token_ids = (
                sampled_token_ids[req_index] if sampled_token_ids else []
            )

            # SUBTRACTED: spec decode 回滚账与 encoder 释放（L1766-L1795）——
            # delete 项 3
            stopped = False
            new_logprobs = None
            new_token_ids = generated_token_ids
            # SUBTRACTED: pooler/kv/ec/prefill_stats 初始化（L1800-L1803）
            finish_reason = None
            # SUBTRACTED: 停止判定与状态更新（L1807-L1815）、structured
            # output 前进（L1817-L1843）、routed experts（L1845-L1883）——
            # delete 项 3：ch9 将完整展开 update_from_output
            # SUBTRACTED: stopped 收尾（_handle_stopped_request / KV 释放 /
            # stopped_running/preempted 集合，L1895-L1907）——delete 项 3

            # Extract sample logprobs if needed.
            if (
                request.sampling_params is not None
                and request.sampling_params.num_logprobs is not None
                and logprobs
            ):
                new_logprobs = logprobs.slice_request(req_index, len(new_token_ids))

            # SUBTRACTED: num_nans_in_logits 登记（L1917-L1918）——delete 项 3

            # Get prompt logprobs for this request.
            prompt_logprobs_tensors = prompt_logprobs_dict.get(req_id)
            should_emit_output = bool(
                new_token_ids or stopped
            )  # SUBTRACTED: pooler_output 项（L1885-L1887）——delete 项 3
            if should_emit_output:
                # Add EngineCoreOutput for this Request.
                outputs[request.client_index].append(
                    EngineCoreOutput(
                        request_id=req_id,
                        new_token_ids=new_token_ids,
                        finish_reason=finish_reason,
                        new_logprobs=new_logprobs,
                        new_prompt_logprobs_tensors=prompt_logprobs_tensors,
                        # SUBTRACTED: pooling_output/stop_reason/events/
                        # prefill_stats/kv_transfer_params/ec_transfer_params/
                        # trace_headers/routed_experts/num_nans_in_logits
                        # 字段行（L1931-L1939）——delete 项 3
                    )
                )

        # SUBTRACTED: 错误请求追加/KV 连接器收尾/stats/events 尾部
        # （L1946-L2011）——delete 项 3

        # Create EngineCoreOutputs for all clients that have requests with
        # outputs in this step.
        engine_core_outputs = {
            client_index: EngineCoreOutputs(outputs=outs)
            for client_index, outs in outputs.items()
        }
        # SUBTRACTED: finished_requests 附加（L2019-L2033）——delete 项 3
        return engine_core_outputs


# ============================================================================
# §13 vllm/outputs.py — the public output carriers (ch7 kept these too;
# ch08 rides the logprobs fields)
# ============================================================================


@dataclass  # SOURCE: vllm/outputs.py:L21-L63
class CompletionOutput:
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
    token_ids: Any
    cumulative_logprob: float | None
    logprobs: SampleLogprobs | None
    routed_experts: np.ndarray | None = None  # [seq_len,layer_num,topk]
    finish_reason: str | None = None
    stop_reason: int | str | None = None
    lora_request: Any | None = None

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


class RequestOutput:  # SOURCE: vllm/outputs.py:L85-L197（字段面全保留）
    """The output data of a completion request to the LLM.

    Args:
        request_id: The unique ID of the request.
        prompt: The prompt string of the request.
        prompt_token_ids: The token IDs of the prompt.
        prompt_logprobs: The log probabilities to return per prompt token.
        outputs: The output sequences of the request.
        finished: Whether the whole request is finished.
    """

    def __init__(
        self,
        request_id: str,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        prompt_logprobs: PromptLogprobs | None,
        outputs: list[CompletionOutput],
        finished: bool,
        metrics: Any | None = None,
        lora_request: Any | None = None,
        encoder_prompt: str | None = None,
        encoder_prompt_token_ids: list[int] | None = None,
        num_cached_tokens: int | None = None,
        num_cache_creation_tokens: int | None = None,
        *,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            logger.warning(  # SOURCE: vllm/outputs.py:L133-L136 warning_once 面
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


# ============================================================================
# §14 vllm/v1/engine/output_processor.py — arrival dispatch + exit loading
# (the logprobs segments; ch7 owns the main-swimlane segments)
# ============================================================================


class RequestState:  # SOURCE: vllm/v1/engine/output_processor.py:L129-L423（本章触达面）
    def __init__(
        self,
        request_id: str,
        external_req_id: str,
        request_index: int,
        output_kind: RequestOutputKind,
        prompt: str | None,
        prompt_token_ids: list[int] | None,
        logprobs_processor: LogprobsProcessor | None,
        detokenizer: Any | None,  # ch7 产品面：get_next_output_text/output_token_ids
        queue: Any | None = None,
    ):
        self.request_id = request_id
        self.external_req_id = external_req_id
        self.request_index = request_index
        self.output_kind = output_kind
        self.prompt = prompt
        self.prompt_token_ids = prompt_token_ids
        self.logprobs_processor = logprobs_processor
        self.detokenizer = detokenizer
        # SUBTRACTED: parent_req/lora/stream_interval/sent_tokens_offset/
        # stats/routed_experts_chunks/streaming_input（L134-L190）——ch7 域
        # （三道闸与 n>1 归 ch7 精简版）
        self.queue = queue

    @classmethod
    def from_new_request(  # SOURCE: vllm/v1/engine/output_processor.py:L211-L274
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: Any | None,
        request_index: int,
        queue: Any | None,
        log_stats: bool,
        stream_interval: int,
        detokenizer: Any | None = None,  # HOST SEAM 注入面（真实从 IncrementalDetokenizer.from_new_request 构造——ch7 域）
        output_kind: RequestOutputKind | None = None,
    ) -> "RequestState":
        if sampling_params := request.sampling_params:
            if not sampling_params.detokenize:
                tokenizer = None  # SOURCE: vllm/v1/engine/output_processor.py:L224-L225（m20 的源头）
            output_kind = sampling_params.output_kind
            # SUBTRACTED: stream_interval clamp（L227-L229）——ch7 域
            logprobs_processor = LogprobsProcessor.from_new_request(
                tokenizer=tokenizer,
                request=request,
            )
            # SUBTRACTED: detokenizer = IncrementalDetokenizer.from_new_request
            # （L234-L237）——ch7 产品，HOST SEAM 注入；max_tokens/top_p/n/
            # temperature（L238-L241）——ch7 域
        else:
            logprobs_processor = None
            assert request.pooling_params is not None  # SUBTRACTED 面：pooling 域（L242-L250）
            output_kind = output_kind

        assert request.external_req_id is not None
        return cls(
            request_id=request.request_id,
            external_req_id=request.external_req_id,
            request_index=request_index,
            output_kind=output_kind,
            prompt=prompt,
            prompt_token_ids=request.prompt_token_ids,
            logprobs_processor=logprobs_processor,
            detokenizer=detokenizer,
            # SUBTRACTED: lora/max_tokens_param/top_p/n/temperature/stats/
            # stream_interval/stream_input（L258-L273）——ch7 域
            queue=queue,
        )

    def make_request_output(  # SOURCE: vllm/v1/engine/output_processor.py:L276-L341（本章走直线；三道闸 ch7 域）
        self,
        new_token_ids: list[int],
        pooling_output: Any | None,
        finish_reason: FinishReason | None,
        stop_reason: int | str | None,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
    ) -> RequestOutput | None:
        finished = finish_reason is not None
        # SUBTRACTED: FINAL_ONLY 未完零构造闸与 stream_interval 节流闸
        # （L286-L313）——ch7 域三道闸
        # SUBTRACTED: pooling 分支（L317-L322）——delete 项 5

        output = self._new_completion_output(new_token_ids, finish_reason, stop_reason)

        # SUBTRACTED: n>1 父聚合分支（L326-L332）——ch7 域
        return self._new_request_output(
            self.external_req_id,
            [output],
            finished,
            kv_transfer_params,
            ec_transfer_params,
        )

    def _new_request_output(  # SOURCE: vllm/v1/engine/output_processor.py:L342-L386
        self,
        external_req_id: str,
        outputs: list[CompletionOutput],
        finished: bool,
        kv_transfer_params: dict[str, Any] | None = None,
        ec_transfer_params: dict[str, Any] | None = None,
    ) -> RequestOutput:
        # If prompt embeds were used, put placeholder prompt token ids
        prompt_token_ids = self.prompt_token_ids
        # SUBTRACTED: prompt_embeds 占位分支（L352-L353）——delete 项 5
        assert prompt_token_ids is not None

        # SUBTRACTED: PoolingRequestOutput 分支（L356-L365）——delete 项 5
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
            outputs=cast(list[CompletionOutput], outputs),
            finished=finished,
            kv_transfer_params=kv_transfer_params,
            ec_transfer_params=ec_transfer_params,
            # SUBTRACTED: lora/num_cached_tokens/num_cache_creation_tokens/
            # metrics 行（L375,L383-L385）——ch7 域
        )

    def _new_completion_output(  # SOURCE: vllm/v1/engine/output_processor.py:L388-L423
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

        # SUBTRACTED: routed experts 拼接（L409-L412）——delete 项 2/5

        return CompletionOutput(
            index=self.request_index,
            text=text,
            token_ids=token_ids,
            # SUBTRACTED: routed_experts 行（L418）——delete 项 2
            logprobs=logprobs,
            cumulative_logprob=self.logprobs_processor.cumulative_logprob,
            finish_reason=str(finish_reason) if finished else None,
            stop_reason=stop_reason if finished else None,
        )


class OutputProcessor:  # SOURCE: vllm/v1/engine/output_processor.py:L429-L836（本章 = 单循环骨架 + 第 3 步）
    """Process EngineCoreOutputs into RequestOutputs."""

    def __init__(  # SOURCE: vllm/v1/engine/output_processor.py:L432-L447
        self,
        tokenizer: TokenizerLike | None,
        *,
        log_stats: bool,
        stream_interval: int = 1,
    ):
        self.log_stats = log_stats
        self.tokenizer = tokenizer
        # SUBTRACTED: stream_interval/parent_requests/external_req_ids/
        # lora_states/tracing（L442-L447）——ch7 域
        self.request_states: dict[str, RequestState] = {}

    def add_request(  # SOURCE: vllm/v1/engine/output_processor.py add_request 面（本章触达的登记行）
        self,
        request: EngineCoreRequest,
        prompt: str | None,
        parent_req: Any | None,
        request_index: int,
        queue: Any | None,
        detokenizer: Any | None = None,
    ) -> None:
        self.request_states[request.request_id] = RequestState.from_new_request(
            tokenizer=self.tokenizer,
            request=request,
            prompt=prompt,
            parent_req=parent_req,
            request_index=request_index,
            queue=queue,
            log_stats=self.log_stats,
            stream_interval=1,
            detokenizer=detokenizer,
        )

    def process_outputs(  # SOURCE: vllm/v1/engine/output_processor.py:L589-L711（循环骨架逐字；主泳道步骤按 delete 项 5 占位）
        self,
        engine_core_outputs: list[EngineCoreOutput],
        engine_core_timestamp: float | None = None,
        iteration_stats: Any | None = None,
    ) -> Any:
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

        request_outputs: list[RequestOutput] = []
        reqs_to_abort: list[str] = []
        for engine_core_output in engine_core_outputs:
            req_id = engine_core_output.request_id
            req_state = self.request_states.get(req_id)
            if req_state is None:
                # Ignore output for already-aborted request.
                continue

            # SUBTRACTED: 1) stats（L626-L629）——ch7 域

            new_token_ids = engine_core_output.new_token_ids
            # SUBTRACTED: pooling/kv/ec/routed_experts 字段读取（L631-L640）——
            # delete 项 5
            finish_reason = engine_core_output.finish_reason
            stop_reason = engine_core_output.stop_reason
            kv_transfer_params = None
            ec_transfer_params = None
            # SUBTRACTED: is_prefilling/prefill_stats 段（L642-L650）——ch7 域

            # SUBTRACTED: 2) detokenize+stop 判定（L652-L661，主泳道——ch7
            # 精简版全量实现，本章占位：detokenizer 面由 ch07 模块承担）
            assert req_state.logprobs_processor is not None

            # 3) Compute sample and prompt logprobs for request,
            # if required.
            req_state.logprobs_processor.update_from_output(engine_core_output)

            # 4) Create and handle RequestOutput objects.
            if request_output := req_state.make_request_output(
                new_token_ids,
                None,  # SUBTRACTED: pooling_output（L670）——delete 项 5
                finish_reason,
                stop_reason,
                kv_transfer_params,
                ec_transfer_params,
            ):
                if req_state.queue is not None:
                    # AsyncLLM: put into queue for handling by generate().
                    req_state.queue.put(request_output)
                else:
                    # LLMEngine: return list of RequestOutputs.
                    request_outputs.append(request_output)

            # SUBTRACTED: streaming_input/完成清理/reqs_to_abort/stats/tracing
            # 尾段（L685-L702）——ch7 域

        return SimpleNamespace(  # HOST 注：OutputProcessorOutput 面（L708-L711）
            request_outputs=request_outputs, reqs_to_abort=reqs_to_abort
        )


# ============================================================================
# §15 vllm/v1/engine/logprobs.py — the whole file (chapter protagonist)
# ============================================================================


NONES = itertools.repeat(None)  # SOURCE: vllm/v1/engine/logprobs.py:L26


@dataclass  # SOURCE: vllm/v1/engine/logprobs.py:L29-L352 — 全类逐字
class LogprobsProcessor:
    # Tokenizer for this request,
    # None if detokenization is disabled.
    tokenizer: TokenizerLike | None

    # Logprobs for this request
    logprobs: SampleLogprobs | None
    prompt_logprobs: PromptLogprobs | None
    cumulative_logprob: float | None
    num_logprobs: int | None
    num_prompt_logprobs: int | None

    @classmethod
    def from_new_request(  # SOURCE: vllm/v1/engine/logprobs.py:L43-L67（@classmethod L42）
        cls,
        tokenizer: TokenizerLike | None,
        request: EngineCoreRequest,
    ) -> "LogprobsProcessor":
        sampling_params = request.sampling_params
        assert sampling_params is not None
        num_logprobs = sampling_params.num_logprobs
        num_prompt_logprobs = sampling_params.prompt_logprobs
        return cls(
            tokenizer=tokenizer,
            cumulative_logprob=(None if num_logprobs is None else 0.0),
            logprobs=(
                None
                if num_logprobs is None
                else create_sample_logprobs(sampling_params.flat_logprobs)
            ),
            prompt_logprobs=(
                None
                if num_prompt_logprobs is None
                else create_prompt_logprobs(sampling_params.flat_logprobs)
            ),
            num_prompt_logprobs=num_prompt_logprobs,
            num_logprobs=num_logprobs,
        )

    def _update_sample_logprobs(self, logprobs_lists: LogprobsLists) -> None:  # SOURCE: vllm/v1/engine/logprobs.py:L69-L119
        """Update with sample logprobs from EngineCore.

        Outer lists are only of len > 1 if EngineCore made
        >1 tokens in prior step (e.g. in spec decoding).

        Args:
          logprobs_lists: the lists of logprob tokens, logprobs, and ranks.

        """

        assert self.num_logprobs is not None
        assert self.logprobs is not None
        assert self.cumulative_logprob is not None

        token_ids_lst, logprobs_lst, ranks_lst, _ = logprobs_lists

        for rank_np, logprobs_np, token_ids_np in zip(
            ranks_lst, logprobs_lst, token_ids_lst
        ):
            rank = rank_np.tolist()
            logprobs = logprobs_np.tolist()
            token_ids = token_ids_np.tolist()
            # Detokenize (non-incrementally).
            decoded_tokens: list[str] | Iterable[None]
            if self.tokenizer is None:
                decoded_tokens = NONES
            else:
                decoded_tokens_list = convert_ids_list_to_tokens(
                    self.tokenizer, token_ids
                )
                context_token_ids = self._get_sampled_context_ids(self.logprobs)
                decoded_tokens = self._verify_tokens(
                    decoded_tokens_list=decoded_tokens_list,
                    tokens=token_ids,
                    context_token_ids=context_token_ids,
                )

            # Sampler puts the sampled logprob in first.
            sampled_token_logprob = logprobs[0]
            self.cumulative_logprob += sampled_token_logprob

            # Update with the Logprob container for this pos.
            append_logprobs_for_next_position(
                self.logprobs,
                token_ids,
                logprobs,
                decoded_tokens,
                rank,
                self.num_logprobs,
            )

    def _update_prompt_logprobs(  # SOURCE: vllm/v1/engine/logprobs.py:L121-L187
        self,
        prompt_logprobs_tensors: LogprobsTensors,
    ) -> None:
        """Update with prompt logprobs from EngineCore.

        Args:
          prompt_logprobs_tensors: tuple containing the prompt logprobs
                                   tensors.

        """

        # Prompt logprobs are enabled.
        assert self.num_prompt_logprobs is not None
        assert self.prompt_logprobs is not None

        token_ids, logprobs, ranks, _ = prompt_logprobs_tensors

        # Recover shapes.
        num_prompt_tokens, num_logprobs = logprobs.shape

        # Detokenize non-incrementally.
        # Output is flat: [num_tok, num_lps] -> [num_tok * num_lps]
        all_decoded_tokens: list[str] | None = (
            None
            if self.tokenizer is None
            else convert_ids_list_to_tokens(
                self.tokenizer, token_ids.flatten().tolist()
            )
        )

        # Pythonize the torch tensors.
        prompt_token_ranks = ranks.tolist()
        prompt_logprobs = logprobs.tolist()
        token_ids_list = token_ids.tolist()

        # Make Logprob for each position.
        for pos in range(num_prompt_tokens):
            # Handle flattening and UTF-8 correction per position
            offset = pos * num_logprobs
            offset_end = offset + num_logprobs

            decoded_tokens_for_pos: list[str] | Iterable[None]
            if all_decoded_tokens is None:
                decoded_tokens_for_pos = NONES
            else:
                # Extract decoded tokens for this position
                decoded_tokens_slice = all_decoded_tokens[offset:offset_end]
                # Context: preceding prompt tokens accumulated in
                # self.prompt_logprobs from previous loop iterations.
                context_token_ids = self._get_sampled_context_ids(self.prompt_logprobs)
                # Apply UTF-8 correction within this position's token boundaries
                decoded_tokens_for_pos = self._verify_tokens(
                    decoded_tokens_list=decoded_tokens_slice,
                    tokens=token_ids_list[pos],
                    context_token_ids=context_token_ids,
                )

            # Update with the Logprob container for this pos.
            append_logprobs_for_next_position(
                self.prompt_logprobs,
                token_ids_list[pos],
                prompt_logprobs[pos],
                decoded_tokens_for_pos,
                prompt_token_ranks[pos],
                self.num_prompt_logprobs,
            )

    def pop_prompt_logprobs(self) -> PromptLogprobs | None:  # SOURCE: vllm/v1/engine/logprobs.py:L189-L206
        """Pop and return all request prompt logprobs

        The logprobs processor aggregates prompt chunk logprobs
        over one or more prefill chunks. This method returns
        all prompt logprobs at once and then forgets them.
        Ensures correct RequestOutputKind.DELTA semantics
        wherein all prompt logprobs are returned at once at
        the end of prefill.

        Returns:
          None if prompt logprobs are disabled for this request.
          List of all prompt logprobs, otherwise.
        """
        plp = self.prompt_logprobs
        if plp:
            self.prompt_logprobs = []
        return plp

    @staticmethod
    def _get_sampled_context_ids(  # SOURCE: vllm/v1/engine/logprobs.py:L209-L247（@staticmethod L208）
        logprobs_source: SampleLogprobs | PromptLogprobs | None,
        max_context: int = 4,
    ) -> list[int]:
        """Extract recent sampled token IDs from a logprobs source.

        The sampled (or prompt) token at each position is the first
        entry, since it is always inserted first by
        append_logprobs_for_next_position.

        Args:
            logprobs_source: The logprobs container to extract from.
            max_context: Maximum number of preceding tokens to return.
                4 is sufficient for any UTF-8 multi-byte sequence.

        Returns:
            List of sampled token IDs, oldest first, most recent last.
        """
        if not logprobs_source:
            return []

        n = len(logprobs_source)
        start = max(0, n - max_context)

        # Efficient path for FlatLogprobs: access token_ids directly.
        if isinstance(logprobs_source, FlatLogprobs):
            return [
                logprobs_source.token_ids[logprobs_source.start_indices[i]]
                for i in range(start, n)
                if logprobs_source.start_indices[i] < logprobs_source.end_indices[i]
            ]

        # list[dict] path
        result: list[int] = []
        for i in range(start, n):
            entry = logprobs_source[i]
            if entry is not None:
                result.append(next(iter(entry)))
        return result

    def _correct_decoded_token(  # SOURCE: vllm/v1/engine/logprobs.py:L249-L310
        self, token_id: int, context_token_ids: list[int]
    ) -> str:
        """Correct a decoded token that contains the replacement character.

        When byte-fallback tokenization splits multi-byte UTF-8
        characters across tokens, individual token decoding produces
        the replacement character U+FFFD. This method uses preceding
        sampled tokens as context to reconstruct the correct text.

        Args:
            token_id: The single token ID to correct.
            context_token_ids: Preceding sampled token IDs in sequential
                order (oldest first). These are the actual tokens in
                the generated sequence, NOT top-k alternatives.

        Returns:
            The corrected decoded string, or empty string if the byte
            sequence is genuinely incomplete at this point.
        """
        assert self.tokenizer is not None

        max_ctx = min(len(context_token_ids), 4)

        for num_ctx in range(1, max_ctx + 1):
            context = context_token_ids[-num_ctx:]
            full_decoded = self.tokenizer.decode(context + [token_id])

            if full_decoded.endswith("�"):
                continue

            # Find the boundary between "clean" context tokens and
            # byte-fallback tokens that are part of the same incomplete
            # sequence. Byte-fallback context tokens returned "" when
            # they were processed, so their text must be attributed to
            # this completing token.
            clean_end = len(context)
            for j in range(len(context) - 1, -1, -1):
                if self.tokenizer.decode([context[j]]).endswith("�"):
                    clean_end = j
                else:
                    break

            # Decode only the clean (non-byte-fallback) prefix.
            if clean_end > 0:
                clean_prefix = self.tokenizer.decode(context[:clean_end])
            else:
                clean_prefix = ""

            if full_decoded.startswith(clean_prefix):
                return full_decoded[len(clean_prefix) :]

            # Tokenizer normalization may cause prefix mismatch.
            # Find the longest common prefix between them.
            common_len = 0
            for a, b in zip(clean_prefix, full_decoded):
                if a != b:
                    break
                common_len += 1
            return full_decoded[common_len:]

        return ""

    def _verify_tokens(  # SOURCE: vllm/v1/engine/logprobs.py:L312-L346
        self,
        decoded_tokens_list: list[str],
        tokens: list[int],
        context_token_ids: list[int] | None = None,
    ) -> list[str]:
        """Verify and correct decoded tokens with replacement characters.

        Args:
            decoded_tokens_list: Decoded token strings to verify.
            tokens: Token IDs corresponding to decoded_tokens_list.
                These are alternatives at the SAME position (e.g.
                [sampled, top1, top2]), NOT sequential tokens.
            context_token_ids: Preceding sampled token IDs providing
                sequential context. If None, extracted from
                self.logprobs.
        """
        if context_token_ids is None:
            context_token_ids = self._get_sampled_context_ids(self.logprobs)

        corrected_decoded_token_map = dict()
        for idx, text in enumerate(decoded_tokens_list):
            if text.endswith("�"):
                # Replacement char at the end means a potential
                # unfinished byte sequence from byte-fallback
                # tokenization. Correct each token independently
                # using only the sequential context.
                corrected_decoded_token_map[idx] = self._correct_decoded_token(
                    tokens[idx], context_token_ids
                )

        for idx, text in corrected_decoded_token_map.items():
            decoded_tokens_list[idx] = text

        return decoded_tokens_list

    def update_from_output(self, output: EngineCoreOutput) -> None:  # SOURCE: vllm/v1/engine/logprobs.py:L348-L352
        if output.new_logprobs is not None:
            self._update_sample_logprobs(output.new_logprobs)
        if output.new_prompt_logprobs_tensors is not None:
            self._update_prompt_logprobs(output.new_prompt_logprobs_tensors)


# ============================================================================
# §16 vllm/entrypoints/openai — the three-field record (station 14)
# ============================================================================


class ChatCompletionLogProb(OpenAIBaseModel):  # SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:L81-L84
    token: str
    logprob: float = -9999.0
    bytes: list[int] | None = None


class ChatCompletionLogProbsContent(ChatCompletionLogProb):  # SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:L87-L91
    # Workaround: redefine fields name cache so that it's not
    # shared with the super class.
    field_names: ClassVar[set[str] | None] = None
    top_logprobs: list[ChatCompletionLogProb] = Field(default_factory=list)


class ChatCompletionLogProbs(OpenAIBaseModel):  # SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:L94-L95
    content: list[ChatCompletionLogProbsContent] | None = None


# SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:L212 ChatCompletionRequest
# — 本章域 = logprobs 参数面与 to_sampling_params 的 logprobs 段；请求体其余
# 字段（messages/model/penalties/…）按 delete 项 9/5 精简为字段面
class ChatCompletionRequest(OpenAIBaseModel):
    messages: list[Any] = Field(default_factory=list)  # SUBTRACTED 面：必填 schema 归 ch2
    model: str | None = None
    logprobs: bool | None = False  # SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:L219
    top_logprobs: int | None = 0  # SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:L220
    prompt_logprobs: int | None = None  # SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:L285
    logprob_token_ids: list[int] | None = Field(  # SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:L286-L296
        default=None,
        description=(
            "Specific vocab token IDs to return logprobs for at each generated "
            "position, in addition to the sampled token. More efficient than "
            "`top_logprobs=-1` when only a small fixed label set is needed "
            "(e.g. multilabel scoring "
            "where each label corresponds to a known vocab id). When set, "
            "this explicit token selection takes precedence over the natural "
            "top-k selected by `top_logprobs`. Requires `logprobs=True`."
        ),
    )
    echo: bool = Field(  # SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:L303-L310
        default=False,
    )
    stream: bool | None = False  # SUBTRACTED 面：SSE 域归 ch2/ch7
    return_tokens_as_token_ids: bool = Field(  # SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:L398-L403
        default=False,
    )
    # SUBTRACTED: 请求体其余字段（L213-L283,L297-L302,L311-L398,L404+）——
    # delete 项 5/9：SSE/工具/结构化输出域

    def to_sampling_params(  # SOURCE: vllm/entrypoints/openai/chat_completion/protocol.py:L646-L734
        self,
        max_tokens: int,
        default_sampling_params: dict,
    ) -> SamplingParams:
        # SUBTRACTED: repetition_penalty/temperature/top_p/top_k/min_p 缺省
        # 推导（L652-L672）与 stop_token_ids 合并（L674-L684）——ch2 域

        prompt_logprobs = self.prompt_logprobs
        if prompt_logprobs is None and self.echo:
            prompt_logprobs = self.top_logprobs

        # SUBTRACTED: extra_args/kv_transfer 组装（L690-L696）——ch7 域
        return SamplingParams.from_optional(
            # SUBTRACTED: 非本章参数行（L698-L708,L716-L721,L725-L733）——
            # delete 项 5/9
            logprobs=(
                self.top_logprobs
                if self.logprobs and not self.logprob_token_ids
                else None
            ),
            prompt_logprobs=prompt_logprobs,
            logprob_token_ids=self.logprob_token_ids or None,
            output_kind=(
                RequestOutputKind.DELTA if self.stream else RequestOutputKind.FINAL_ONLY
            ),
        )


# SOURCE: vllm/entrypoints/generate/base/serving.py:L113 GenerateBaseServing — 本章域 =
# _get_decoded_token/format_token_id_placeholder；基座其余机制按 delete 项 5 精简
class GenerateBaseServing:
    def __init__(
        self,
        engine_client: Any | None = None,
        models: Any | None = None,
        request_logger: Any | None = None,
        return_tokens_as_token_ids: bool = False,
    ):  # SOURCE: vllm/entrypoints/generate/base/serving.py:L118-L133
        self.engine_client = engine_client
        self.models = models
        self.request_logger = request_logger
        self.return_tokens_as_token_ids = return_tokens_as_token_ids
        # SUBTRACTED: 基座其余初始化（L134-L248）——ch2/ch7 域

    @staticmethod
    def _get_decoded_token(  # SOURCE: vllm/entrypoints/generate/base/serving.py:L252-L270
        logprob: Logprob,
        token_id: int,
        tokenizer: TokenizerLike | None,
        return_as_token_id: bool = False,
    ) -> str:
        if return_as_token_id:
            return format_token_id_placeholder(token_id)

        if logprob.decoded_token is not None:
            return logprob.decoded_token

        if tokenizer is None:
            raise ValueError(
                "Unable to get tokenizer because `skip_tokenizer_init=True`"
            )

        return tokenizer.decode([token_id])


def format_token_id_placeholder(token_id: int) -> str:  # SOURCE: vllm/entrypoints/generate/base/serving.py:L273-L274
    return f"token_id:{token_id}"


# SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L110 OpenAIServingChat
# — 本章域 = _create_chat_logprobs/_get_top_logprobs 两函数；SSE 生成器全貌按
# delete 项 7 删（ch2/ch7 域）
class OpenAIServingChat(GenerateBaseServing):
    def __init__(
        self,
        engine_client: Any | None = None,
        models: Any | None = None,
        response_role: str = "assistant",
        *,
        return_tokens_as_token_ids: bool = False,
    ):  # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L111-L138
        super().__init__(
            engine_client=engine_client,
            models=models,
            request_logger=None,
            return_tokens_as_token_ids=return_tokens_as_token_ids,
        )
        self.response_role = response_role
        # SUBTRACTED: online_renderer/chat_template/reasoning/tool 配置面
        # （L139-L180）——delete 项 7

    # SUBTRACTED: chat_completion_stream_generator 的调用位（L587-L600）——
    # delete 项 7：SSE 帧序 ch2 已立、流式生成器 ch7 已讲；本章按
    # `_create_chat_logprobs(token_ids=output.token_ids, top_logprobs=
    # output.logprobs, tokenizer=tokenizer, num_output_top_logprobs=
    # request.top_logprobs, logprob_token_ids=request.logprob_token_ids,
    # return_as_token_id=request.return_tokens_as_token_ids)` 直连两个函数

    def _get_top_logprobs(  # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L1140-L1165
        self,
        logprobs: dict[int, Logprob],
        top_logprobs: int | None,
        tokenizer: TokenizerLike | None,
        should_return_as_token_id: bool,
        return_all: bool = False,
    ) -> list[ChatCompletionLogProb]:
        return [
            ChatCompletionLogProb(
                token=(
                    token := self._get_decoded_token(
                        p[1],
                        p[0],
                        tokenizer,
                        return_as_token_id=should_return_as_token_id,
                    )
                ),
                logprob=max(p[1].logprob, -9999.0),
                bytes=list(token.encode("utf-8", errors="replace")),
            )
            for i, p in enumerate(logprobs.items())
            if return_all
            or top_logprobs == -1
            or (top_logprobs is not None and i < top_logprobs)
        ]

    def _create_chat_logprobs(  # SOURCE: vllm/entrypoints/openai/chat_completion/serving.py:L1167-L1231
        self,
        token_ids: Any,
        top_logprobs: Any,
        tokenizer: TokenizerLike | None,
        num_output_top_logprobs: int | None = None,
        logprob_token_ids: list[int] | None = None,
        return_as_token_id: bool | None = None,
    ) -> ChatCompletionLogProbs:
        """Create OpenAI-style logprobs."""
        logprobs_content: list[ChatCompletionLogProbsContent] = []

        should_return_as_token_id = (
            return_as_token_id
            if return_as_token_id is not None
            else self.return_tokens_as_token_ids
        )
        for i, token_id in enumerate(token_ids):
            step_top_logprobs = top_logprobs[i]
            if step_top_logprobs is None or step_top_logprobs.get(token_id) is None:
                if should_return_as_token_id:
                    token = format_token_id_placeholder(token_id)
                else:
                    if tokenizer is None:
                        raise ValueError(
                            "Unable to get tokenizer because `skip_tokenizer_init=True`"
                        )

                    token = tokenizer.decode(token_id)

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=token,
                        bytes=list(token.encode("utf-8", errors="replace")),
                    )
                )
            else:
                step_token = step_top_logprobs[token_id]
                step_decoded = step_token.decoded_token

                logprobs_content.append(
                    ChatCompletionLogProbsContent(
                        token=self._get_decoded_token(
                            step_token,
                            token_id,
                            tokenizer,
                            should_return_as_token_id,
                        ),
                        logprob=max(step_token.logprob, -9999.0),
                        bytes=(
                            None
                            if step_decoded is None
                            else list(step_decoded.encode("utf-8", errors="replace"))
                        ),
                        top_logprobs=self._get_top_logprobs(
                            step_top_logprobs,
                            num_output_top_logprobs,
                            tokenizer,
                            should_return_as_token_id,
                            return_all=bool(logprob_token_ids),
                        ),
                    )
                )

        return ChatCompletionLogProbs(content=logprobs_content)
