# SOURCE: vllm/v1/structured_output/request.py
# 只做减法的忠实精简版。文件改名 so_request.py 以免与本章同目录下镜像
# vllm/v1/request.py 的 request.py 撞名（真实仓库两者分属不同包，无此冲突）。
#
# SUBTRACTED: SPDX 版权头。
import dataclasses
import functools
import json
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError

from backend_types import (
    StructuredOutputGrammar,
    StructuredOutputKey,
    StructuredOutputOptions,
)

# SUBTRACTED: `from vllm.sampling_params import SamplingParams, StructuredOutputsParams`
# 改为本章内 sampling_params 模块的同名精简版；`if TYPE_CHECKING: from vllm.reasoning
# import ReasoningParser`（reasoning 相关，subtraction_plan 批准项3 一并删除）。
from sampling_params import SamplingParams, StructuredOutputsParams


@dataclasses.dataclass
class StructuredOutputRequest:
    # SOURCE: vllm/v1/structured_output/request.py:L21-40
    params: StructuredOutputsParams
    _grammar: "Future[StructuredOutputGrammar] | StructuredOutputGrammar | None" = None
    # SUBTRACTED: reasoning_ended / reasoning_parser_kwargs / reasoner 三个字段——
    # 服务于推理模型跳过语法约束，reasoning 相关，subtraction_plan 批准项3。

    @staticmethod
    def from_sampling_params(
        sampling_params: "SamplingParams | None",
    ) -> "StructuredOutputRequest | None":
        # SOURCE: vllm/v1/structured_output/request.py:L29-40
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
        # 【死契约，v0.21.0 全仓零 in-tree 调用者】门控真正的落点是下面的 grammar
        # property——调度器读的是它，不是这个属性。见 request.py:L59-70。
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
