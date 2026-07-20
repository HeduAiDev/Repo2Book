# SOURCE: vllm/v1/structured_output/request.py
# 只做减法的忠实精简版（pin ad7125a4 / v0.21.0）。文件改名 so_request.py 以免与
# 本章同目录下镜像 vllm/v1/request.py 的 request.py 撞名（真实仓库两者分属不同包，
# 无此冲突；同一惯例见 ch31 impl-notes）。
#
# 本章需要 reasoning_ended / reasoner / reasoning_parser_kwargs 三个字段——它们是
# StructuredOutputManager.should_fill_bitmask / should_advance / _get_reasoner
# 的门控状态位（m16 推理模型耦合），ch31 只讲了六选一互斥与 grammar 门控，未涉及
# 这三个字段，故本章原样保留（不是重新发明——它们就是 backend_types.py 同目录下
# request.py 里真实存在的 dataclass 字段）。
#
# SUBTRACTED: SPDX 版权头。
import dataclasses
import functools
import json
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import TYPE_CHECKING, Any

from backend_types import (
    StructuredOutputGrammar,
    StructuredOutputKey,
    StructuredOutputOptions,
)

# SUBTRACTED: `from vllm.sampling_params import SamplingParams, StructuredOutputsParams`
# 改为本章内 sampling_params 模块的同名精简版。
from sampling_params import SamplingParams, StructuredOutputsParams

if TYPE_CHECKING:
    from reasoning import ReasoningParser


@dataclasses.dataclass
class StructuredOutputRequest:
    # SOURCE: vllm/v1/structured_output/request.py:L21-29
    params: StructuredOutputsParams
    _grammar: "Future[StructuredOutputGrammar] | StructuredOutputGrammar | None" = None
    reasoning_ended: bool | None = None
    reasoning_parser_kwargs: dict[str, Any] | None = None
    # Cached per request; do not share reasoning parsers across requests because
    # their behavior can depend on reasoning_parser_kwargs.
    reasoner: "ReasoningParser | None" = None

    @staticmethod
    def from_sampling_params(
        sampling_params: "SamplingParams | None",
    ) -> "StructuredOutputRequest | None":
        # SOURCE: vllm/v1/structured_output/request.py:L31-40
        if sampling_params is None:
            return None
        params = sampling_params.structured_outputs
        if not params or params.all_constraints_none():
            return None
        return StructuredOutputRequest(params=params)

    def _check_grammar_completion(self) -> bool:
        # SOURCE: vllm/v1/structured_output/request.py:L42-53
        # NOTE: We have to lazy import to gate circular imports
        from request import RequestStatus

        if isinstance(self._grammar, Future):
            try:
                # We will check whether the future is ready within 100 us
                self._grammar = self._grammar.result(timeout=0.0001)
                self.status = RequestStatus.WAITING
            except FutureTimeoutError:
                return False
        return True

    @property
    def is_grammar_ready(self) -> bool:
        # SOURCE: vllm/v1/structured_output/request.py:L55-57
        return self._check_grammar_completion()

    @property
    def grammar(self) -> "StructuredOutputGrammar | None":
        # SOURCE: vllm/v1/structured_output/request.py:L59-64
        completed = self._check_grammar_completion()
        return self._grammar if completed else None  # type: ignore[return-value]

    @grammar.setter
    def grammar(
        self, grammar: "StructuredOutputGrammar | Future[StructuredOutputGrammar]"
    ) -> None:
        # SOURCE: vllm/v1/structured_output/request.py:L66-70
        self._grammar = grammar

    @functools.cached_property
    def structured_output_key(self) -> StructuredOutputKey:
        # SOURCE: vllm/v1/structured_output/request.py:L72-74
        return get_structured_output_key(self.params)


def get_structured_output_key(params: StructuredOutputsParams) -> StructuredOutputKey:
    # SOURCE: vllm/v1/structured_output/request.py:L77-98
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
