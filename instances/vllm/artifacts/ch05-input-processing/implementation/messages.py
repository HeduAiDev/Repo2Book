"""Stage 1 输入处理精简版 —— 数据结构与依赖的忠实最小替身。

本文件汇集 InputProcessor / ParentRequest 主线代码所**依赖**的数据结构。它们在真实
vLLM 中分散于多个文件（见每处 # SOURCE）。为了让精简版**不 import vllm 也能跑、能打断点、
能数值追踪**，这里保留它们的**字段、控制流与可观察行为**，只删去与本章主线无关的字段/分支
（均标 # SUBTRACTED）。这些结构是『被处理的对象』，本章主角是处理它们的 InputProcessor /
ParentRequest（见 input_processor.py / parallel_sampling.py）。
"""

from __future__ import annotations

import copy as _copy
import enum
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# utils：random_uuid / length_from_prompt_token_ids_or_embeds / json_iter_leaves
# ---------------------------------------------------------------------------

_MASK_64_BITS = (1 << 64) - 1


# SOURCE: vllm/utils/__init__.py:L11 (random_uuid)
def random_uuid() -> str:
    return f"{uuid.uuid4().int & _MASK_64_BITS:016x}"  # 16 hex chars


# SOURCE: vllm/utils/__init__.py:L15 (length_from_prompt_token_ids_or_embeds)
def length_from_prompt_token_ids_or_embeds(prompt_token_ids, prompt_embeds) -> int:
    """Calculate the request length (in number of tokens) given either
    prompt_token_ids or prompt_embeds."""
    prompt_token_len = None if prompt_token_ids is None else len(prompt_token_ids)
    prompt_embeds_len = None if prompt_embeds is None else len(prompt_embeds)

    if prompt_token_len is None:
        if prompt_embeds_len is None:
            raise ValueError("Neither prompt_token_ids nor prompt_embeds were defined.")
        return prompt_embeds_len
    else:
        if prompt_embeds_len is not None and prompt_embeds_len != prompt_token_len:
            raise ValueError(
                "prompt_token_ids and prompt_embeds have different lengths: "
                f"{prompt_token_len} != {prompt_embeds_len}"
            )
        return prompt_token_len


# SOURCE: vllm/utils/jsontree.py:L36 (json_iter_leaves)
def json_iter_leaves(value):
    """Iterate through each leaf in a nested JSON structure."""
    if isinstance(value, dict):
        for v in value.values():
            yield from json_iter_leaves(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from json_iter_leaves(v)
    else:
        yield value


# ---------------------------------------------------------------------------
# 采样 / 池化参数
# ---------------------------------------------------------------------------

# SOURCE: vllm/sampling_params.py:L151 (class RequestOutputKind)
class RequestOutputKind(enum.Enum):
    CUMULATIVE = 0
    DELTA = 1
    FINAL_ONLY = 2


# SOURCE: vllm/sampling_params.py (class SamplingParams)
@dataclass
class SamplingParams:
    """采样参数的忠实最小替身。

    保留本章主线触及的字段与方法：n / seed / max_tokens / output_kind / clone() /
    update_from_generation_config() / update_from_tokenizer() / verify()。
    """

    n: int = 1
    seed: int | None = None
    max_tokens: int | None = None
    ignore_eos: bool = False
    stop_token_ids: list[int] | None = None
    bad_words: list[str] | None = None
    skip_clone: bool = False
    output_kind: RequestOutputKind = RequestOutputKind.CUMULATIVE
    thinking_token_budget: int | None = None

    # SUBTRACTED: 数十个采样字段（temperature/top_p/top_k/logprobs/logit_bias/
    #   structured_outputs/...）— 与 tokenize/多模态/id/fan-out 主线无关，
    #   原 vllm/sampling_params.py:L160-L350+。
    _eos_token_id: int | None = field(default=None, repr=False)
    _all_stop_token_ids: set[int] = field(default_factory=set, repr=False)
    _bad_words_token_ids: list[list[int]] | None = field(default=None, repr=False)

    def __post_init__(self):
        # SOURCE: vllm/sampling_params.py (SamplingParams.__post_init__)
        if self.stop_token_ids is None:
            self.stop_token_ids = []

    # SOURCE: vllm/sampling_params.py:L645 (clone)
    def clone(self) -> "SamplingParams":
        """If skip_clone is True, uses shallow copy instead of deep copy."""
        if self.skip_clone:
            return _copy.copy(self)
        return _copy.deepcopy(self)

    # SOURCE: vllm/sampling_params.py:L543 (update_from_generation_config)
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
                eos_ids.discard(eos_token_id)
            if eos_ids:
                self._all_stop_token_ids.update(eos_ids)
                if not self.ignore_eos:
                    assert self.stop_token_ids is not None
                    eos_ids.update(self.stop_token_ids)
                    self.stop_token_ids = list(eos_ids)

    # SOURCE: vllm/sampling_params.py:L573 (update_from_tokenizer)
    def update_from_tokenizer(self, tokenizer) -> None:
        if not self.bad_words:
            return
        self._bad_words_token_ids = []
        for bad_word in self.bad_words:
            for add_prefix_space in [False, True]:
                prefix = " " if add_prefix_space else ""
                prompt = prefix + bad_word.lstrip()
                prompt_token_ids = tokenizer.encode(
                    text=prompt, add_special_tokens=False
                )
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
            raise ValueError(
                f"The model vocabulary size is {tokenizer.max_token_id + 1},"
                f" but the following tokens were specified as bad:"
                f" {invalid_token_ids}."
            )

    @property
    def eos_token_id(self) -> int | None:
        # SOURCE: vllm/sampling_params.py:L622 (eos_token_id)
        return self._eos_token_id

    @property
    def all_stop_token_ids(self) -> set[int]:
        # SOURCE: vllm/sampling_params.py:L626 (all_stop_token_ids)
        return self._all_stop_token_ids

    @property
    def bad_words_token_ids(self) -> list[list[int]] | None:
        # SOURCE: vllm/sampling_params.py:L630 (bad_words_token_ids)
        return self._bad_words_token_ids

    # SOURCE: vllm/sampling_params.py:L652 (verify)
    def verify(self, model_config, speculative_config, structured_outputs_config,
               tokenizer) -> None:
        # SUBTRACTED: 6 个 _validate_* 子校验（logprobs/logit_bias/logits_processors/
        #   allowed_token_ids/spec_decode/structured_outputs）的内部细节
        #   — 与本章 tokenize/多模态/id 主线无关，原 vllm/sampling_params.py:L666+。
        #   保留 verify 入口本身，体现『_validate_params 调 params.verify』控制流。
        pass


# SOURCE: vllm/pooling_params.py (class PoolingParams)
@dataclass
class PoolingParams:
    """池化参数的忠实最小替身（保留 task / clone / verify）。"""

    task: str | None = None

    # SOURCE: vllm/pooling_params.py (clone)
    def clone(self) -> "PoolingParams":
        return _copy.deepcopy(self)

    # SOURCE: vllm/pooling_params.py (verify)
    def verify(self, model_config) -> None:
        # SUBTRACTED: pooling 维度等内部校验细节 — 非本章主线。
        pass


# ---------------------------------------------------------------------------
# 多模态：PlaceholderRange / MultiModalFeatureSpec / argsort_mm_positions
# ---------------------------------------------------------------------------

# SOURCE: vllm/multimodal/inputs.py:L119 (class PlaceholderRange)
@dataclass
class PlaceholderRange:
    """Placeholder location information for multi-modal data."""

    offset: int
    length: int
    is_embed: Any | None = None  # 真实为 torch.Tensor bool mask

    # SOURCE: vllm/multimodal/inputs.py (PlaceholderRange.get_num_embeds)
    def get_num_embeds(self) -> int:
        if self.is_embed is None:
            return self.length
        # 真实版用 is_embed.cumsum 末值；这里 is_embed 为 bool 列表替身。
        return int(sum(1 for x in self.is_embed if x))


@dataclass
class MultiModalFeatureSpec:
    """Represents a single multimodal input with its processed data and metadata."""

    # SOURCE: vllm/multimodal/inputs.py:L302 (class MultiModalFeatureSpec)
    data: Any | None
    modality: str
    identifier: str
    mm_position: PlaceholderRange
    mm_hash: str | None = None


# SOURCE: vllm/multimodal/utils.py:L112 (argsort_mm_positions)
def argsort_mm_positions(mm_positions: Mapping[str, Sequence[PlaceholderRange]]):
    """Given multimodal placeholders, output a sequence of (modality, idx) keys
    sorted by `offset` (starting index in the input sequence) in ascending order."""
    flat_items = (
        (modality, idx, item)
        for modality, items in mm_positions.items()
        for idx, item in enumerate(items)
    )
    sorted_flat_items = sorted(flat_items, key=lambda x: x[2].offset)
    return [(modality, idx) for modality, idx, _ in sorted_flat_items]


# ---------------------------------------------------------------------------
# 已渲染输入：EngineInput TypedDict 家族 + split_enc_dec_input
# ---------------------------------------------------------------------------
# 真实为 typing TypedDict（vllm/inputs/engine.py），运行期就是普通 dict。
# 精简版直接用 dict 携带 'type' 字段（'token'/'embeds'/'multimodal'/'enc_dec'）。


# SOURCE: vllm/inputs/engine.py:L365 (split_enc_dec_input)
def split_enc_dec_input(inputs):
    if inputs["type"] == "enc_dec":
        return inputs["encoder_prompt"], inputs["decoder_prompt"]
    return None, inputs


# ---------------------------------------------------------------------------
# EngineCoreRequest —— 本阶段产物（Stage 1 的终点）
# ---------------------------------------------------------------------------

# SOURCE: vllm/v1/engine/__init__.py:L80 (class EngineCoreRequest)
@dataclass
class EngineCoreRequest:
    """进入 EngineCore 的请求载荷。

    真实版是 msgspec.Struct(array_like, omit_defaults, gc=False)，为跨进程 IPC 序列化
    优化（array_like+omit_defaults 减体积、gc=False 降 GC 开销）。精简版用 dataclass
    保留全部本章字段与 .params 属性，便于 host 直接构造/断言。
    """

    request_id: str
    prompt_token_ids: list[int] | None
    mm_features: list[MultiModalFeatureSpec] | None
    sampling_params: SamplingParams | None
    pooling_params: PoolingParams | None
    arrival_time: float
    lora_request: Any | None
    cache_salt: str | None
    data_parallel_rank: int | None
    prompt_embeds: Any | None = None
    prompt_is_token_ids: list[bool] | None = None
    priority: int = 0
    trace_headers: Mapping[str, str] | None = None
    resumable: bool = False
    # 用户提供的原始 request_id；由 assign_request_id() 内部回填。
    external_req_id: str | None = None
    # SUBTRACTED: client_index/current_wave/reasoning_ended/reasoning_parser_kwargs
    #   等跨进程/推理特性字段 — 非本章主线，原 vllm/v1/engine/__init__.py:L103-L123。

    @property
    def params(self):
        # SOURCE: vllm/v1/engine/__init__.py:L131 (params)
        """Return the processed params (sampling or pooling)."""
        if self.sampling_params is not None:
            return self.sampling_params
        assert self.pooling_params is not None
        return self.pooling_params


# ---------------------------------------------------------------------------
# LoRARequest 替身（仅本章用到 lora_name）
# ---------------------------------------------------------------------------

@dataclass
class LoRARequest:
    # SOURCE: vllm/lora/request.py (class LoRARequest)
    lora_name: str
    lora_int_id: int = 0
    lora_path: str = ""


# ---------------------------------------------------------------------------
# CompletionOutput 替身（ParentRequest.get_outputs 用到 index/finished()）
# ---------------------------------------------------------------------------

# SOURCE: vllm/outputs.py (class CompletionOutput)
@dataclass
class CompletionOutput:
    index: int
    text: str = ""
    token_ids: list[int] = field(default_factory=list)
    _finished: bool = False

    # SOURCE: vllm/outputs.py (CompletionOutput.finished)
    def finished(self) -> bool:
        return self._finished


# 任务集合：与真实 vllm/tasks.py 一致的最小子集。
# SOURCE: vllm/tasks.py (GENERATION_TASKS / POOLING_TASKS)
GENERATION_TASKS = ("generate", "transcription")
POOLING_TASKS = ("encode", "embed", "classify", "score",
                 "token_embed", "token_classify", "plugin")
