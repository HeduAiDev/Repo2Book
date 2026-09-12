# SOURCE: vllm/v1/structured_output/request.py
# 只做减法的忠实精简版——本文件整体逐字保留（请求级容器三态演进 m12 +
# structured_output_key 归一 m2，未列任何删除项；仅插入 # SOURCE 行标记）。
# SUBTRACTED: SPDX 版权头。
import dataclasses
import functools
import json
from concurrent.futures import Future
from concurrent.futures._base import TimeoutError
from typing import TYPE_CHECKING, Any, cast

from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.structured_output.backend_types import (
    StructuredOutputGrammar,
    StructuredOutputKey,
    StructuredOutputOptions,
)

if TYPE_CHECKING:
    # （运行时永不执行——vllm.reasoning 包不进精简版也不受影响；reasoner
    #   字段的字符串注解 "ReasoningParser | None" 不经运行时求值。）
    from vllm.reasoning import ReasoningParser


# SOURCE: vllm/v1/structured_output/request.py:L21-L48 —— 逐字
@dataclasses.dataclass
class StructuredOutputRequest:
    params: StructuredOutputsParams
    _grammar: (
        Future[StructuredOutputGrammar] | StructuredOutputGrammar | Exception | None
    ) = None
    reasoning_ended: bool | None = None
    # Absolute index into the request's all_token_ids of the last reasoning
    # token (the reasoning-end marker). Tokens at or before this index are
    # reasoning content and must never be fed to the grammar. Only set when
    # reasoning ends in a step whose tokens the scheduler advances immediately
    # (structural tags + speculative decoding, see #42452).
    reasoning_end_token_index: int | None = None
    reasoning_parser_kwargs: dict[str, Any] | None = None
    # Cached per request; do not share reasoning parsers across requests because
    # their behavior can depend on reasoning_parser_kwargs.
    reasoner: "ReasoningParser | None" = None

    @staticmethod
    def from_sampling_params(
        sampling_params: SamplingParams | None,
    ) -> "StructuredOutputRequest | None":
        # SOURCE: vllm/v1/structured_output/request.py:L39-L48 —— 逐字
        if sampling_params is None:
            return None
        params = sampling_params.structured_outputs
        if not params or params.all_constraints_none():
            return None
        return StructuredOutputRequest(params=params)

    def _check_grammar_completion(self) -> bool:
        # SOURCE: vllm/v1/structured_output/request.py:L50-L59 —— 逐字
        #   （100µs 非阻塞探测 + Future→成品/Exception 原地替换）
        if isinstance(self._grammar, Future):
            try:
                # We will check whether the future is ready within 100 us
                self._grammar = self._grammar.result(timeout=0.0001)
            except TimeoutError:
                return False
            except Exception as e:
                self._grammar = e
        return True

    @property
    def is_grammar_ready(self) -> bool:
        # SOURCE: vllm/v1/structured_output/request.py:L61-L63 —— 逐字
        return self._check_grammar_completion()

    @property
    def grammar(self) -> StructuredOutputGrammar | Exception | None:
        # SOURCE: vllm/v1/structured_output/request.py:L65-L69 —— 逐字
        if not self._check_grammar_completion():
            return None
        return cast(StructuredOutputGrammar | Exception | None, self._grammar)

    @grammar.setter
    def grammar(
        self, grammar: StructuredOutputGrammar | Future[StructuredOutputGrammar]
    ) -> None:
        # SOURCE: vllm/v1/structured_output/request.py:L71-L75 —— 逐字
        self._grammar = grammar

    @functools.cached_property
    def structured_output_key(self) -> StructuredOutputKey:
        # SOURCE: vllm/v1/structured_output/request.py:L77-L79 —— 逐字
        return get_structured_output_key(self.params)


# SOURCE: vllm/v1/structured_output/request.py:L82-L103 get_structured_output_key
#   —— 逐字（六形态→(枚举,字符串) 二元组归一，含 json.dumps 归一细节）
# SOURCE: vllm/v1/structured_output/request.py:L82-L103
def get_structured_output_key(params: StructuredOutputsParams) -> StructuredOutputKey:
    if params.json is not None:
        if not isinstance(params.json, str):
            json_str = json.dumps(params.json)
        else:
            json_str = params.json
        return StructuredOutputOptions.JSON, json_str
    if params.json_object:
        return StructuredOutputOptions.JSON_OBJECT, ""
    if params.regex is not None:
        return StructuredOutputOptions.REGEX, params.regex
    if params.choice is not None:
        if not isinstance(params.choice, str):
            json_str = json.dumps(params.choice)
        else:
            json_str = params.choice
        return StructuredOutputOptions.CHOICE, json_str
    if params.grammar is not None:
        return StructuredOutputOptions.GRAMMAR, params.grammar
    if params.structural_tag is not None:
        return StructuredOutputOptions.STRUCTURAL_TAG, params.structural_tag
    raise ValueError("No valid structured output parameter found")
