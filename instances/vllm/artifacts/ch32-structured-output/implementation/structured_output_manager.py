# SOURCE: vllm/v1/structured_output/__init__.py
# 只做减法的忠实精简版（pin ad7125a4 / v0.21.0）。与 vLLM 同名、同结构、同控制流；
# 只删不增。本章从「语法已就绪」起步（ch31 已讲透语法编译门与后端契约），这里的
# StructuredOutputManager 直接接受构造好的 grammar 对象，聚焦「每步怎么装配一整批
# 位掩码」。
#
# SUBTRACTED: SPDX 版权头、__init__ 中与掩码装配无关的部分——tokenizer 构建、
# reasoning_parser_plugin 动态导入、_use_async_grammar_compilation 的
# external_launcher 判定、编译线程池 self.executor 的构造（__init__.py:L36-59,
# L70-97）——本章接受一个已构造好的 grammar 对象，这些属语法编译门（ch31 主题）。
# SUBTRACTED: grammar_init/_create_grammar 两个方法（四种后端的选择与构造分支，
# __init__.py:L114-183）——ch31 §31.4 已讲透，本章只依赖 fill_bitmask/accept_tokens/
# rollback/is_terminated/validate_tokens 这几个契约方法，后端身份对本章控制流不可见。
# SUBTRACTED: clear_backend（析构清理，与本章控制流无关）。
import itertools
import multiprocessing
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch

from backend_types import StructuredOutputGrammar

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from reasoning import ReasoningParser
    from request import Request


class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    def __init__(self, max_num_seqs: int, max_num_spec_tokens: int = 0):
        # SOURCE: vllm/v1/structured_output/__init__.py:L34-68（精简到本章相关字段；
        # 真实签名接受 vllm_config，这里直接接受调度侧已算好的两个数字——同一简化
        # 已见于 backend_xgrammar.py 的 XgrammarBackend）
        # SOURCE: vllm/v1/structured_output/__init__.py:L36
        self.backend: "object | None" = None
        self.reasoner_cls: "type[ReasoningParser] | None" = None
        self.enable_in_reasoning = False

        self._grammar_bitmask: "torch.Tensor | None" = None
        self._full_mask = torch.tensor(-1, dtype=torch.int32)

        # SOURCE: vllm/v1/structured_output/__init__.py:L60-68
        # 结构性前提：只有 max_num_seqs 严格大于 128 时才会构造并行填充所需的
        # executor_for_fillmask——max_num_seqs <= 128 时并行分支是结构性死代码
        # （grammar_bitmask 里 `len(...) > self.fill_bitmask_parallel_threshold`
        # 永远为假，因为参与掩码装配的请求数不会超过 max_num_seqs）。
        self.fill_bitmask_parallel_threshold = 128
        if self.fill_bitmask_parallel_threshold < max_num_seqs:
            self.fill_bitmask_parallel_batch_size = 16
            # Use:
            # - at least 1 CPU
            # - at most half the number of CPUs or 8, whichever is less
            max_workers = max(1, min(multiprocessing.cpu_count() // 2, 8))
            self.executor_for_fillmask = ThreadPoolExecutor(max_workers=max_workers)

        # 精简版把 vllm_config.scheduler_config.max_num_seqs 与
        # vllm_config.speculative_config.num_speculative_tokens 两个派生量直接存成
        # 私有字段——真实代码在 grammar_bitmask 每次调用时从 self.vllm_config 现算，
        # 数值等价，只是省去 vllm_config 这层间接。
        self._max_num_seqs = max_num_seqs
        self._max_num_spec_tokens = max_num_spec_tokens

    def _get_reasoner(self, request: "Request") -> "ReasoningParser | None":
        # SOURCE: vllm/v1/structured_output/__init__.py:L99-112
        structured_req = request.structured_output_request
        if structured_req is None or self.reasoner_cls is None:
            return None

        if structured_req.reasoner is None:
            # Lazily build the request-local parser so the structured-output
            # gate observes the same template kwargs used by the frontend.
            parser_kwargs = structured_req.reasoning_parser_kwargs or {}
            structured_req.reasoner = self.reasoner_cls(**parser_kwargs)
        return structured_req.reasoner

    def _fill_bitmasks(
        self, batch: "Iterable[tuple[StructuredOutputGrammar, int, bool]]"
    ) -> None:
        # SOURCE: vllm/v1/structured_output/__init__.py:L185-196
        assert self._grammar_bitmask is not None
        for grammar, index, apply_bitmask in batch:
            if apply_bitmask and not grammar.is_terminated():
                grammar.fill_bitmask(self._grammar_bitmask, index)
            else:
                # Note that for thinking support, we will need to
                # reset the relevant part of the bitmask for consequent
                # requests here.
                self._grammar_bitmask[index].fill_(self._full_mask)

    def _async_submit_fill_bitmask(
        self, batch: "list[tuple[StructuredOutputGrammar, int, bool]]"
    ) -> Future:
        # SOURCE: vllm/v1/structured_output/__init__.py:L198-201
        return self.executor_for_fillmask.submit(self._fill_bitmasks, batch)

    def grammar_bitmask(
        self,
        requests: "dict[str, Request]",
        structured_output_request_ids: "list[str]",
        scheduled_spec_decode_tokens: "dict[str, list[int]]",
    ) -> "npt.NDArray[np.int32] | None":
        # SOURCE: vllm/v1/structured_output/__init__.py:L203-299
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        max_num_spec_tokens = self._max_num_spec_tokens

        if self._grammar_bitmask is None:
            assert self.backend is not None

            # Allocate a bitmask for each token needing to be checked:
            # one for each speculative position, and one more for the
            # bonus token / non-speculative token.
            self._grammar_bitmask = self.backend.allocate_token_bitmask(
                self._max_num_seqs * (1 + max_num_spec_tokens)
            )

        # Generate a batched bitmask for all structured output requests.
        # When speculative decoding is enabled, we need to include multiple
        # masks for each request, one for each possible bonus token position.
        # These are stored inline in the tensor and unpacked by the gpu runner.
        cumulative_index = 0

        # Optimized parallel filling of bitmasks for
        # non-spec, large-batch-size cases
        if (
            len(structured_output_request_ids) > self.fill_bitmask_parallel_threshold
            and max_num_spec_tokens == 0
        ):
            promises = []
            batch = []
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request
                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                grammar = structured_output_request.grammar

                apply_bitmask = self.should_fill_bitmask(request)
                batch.append((grammar, cumulative_index, apply_bitmask))
                if len(batch) == self.fill_bitmask_parallel_batch_size:
                    promises.append(self._async_submit_fill_bitmask(batch))
                    batch = []

                cumulative_index += 1
            if batch:
                promises.append(self._async_submit_fill_bitmask(batch))

            # Wait for all bitmask filling tasks to complete.
            for promise in promises:
                promise.result()
        else:
            # Fallback to serial filling of bitmasks for small-batch-size cases
            for req_id in structured_output_request_ids:
                request = requests[req_id]
                structured_output_request = request.structured_output_request

                if TYPE_CHECKING:
                    assert structured_output_request is not None
                    assert structured_output_request.grammar is not None
                grammar = structured_output_request.grammar
                apply_bitmask = self.should_fill_bitmask(request)

                state_advancements = 0
                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
                for token in itertools.chain(req_tokens, (-1,)):
                    self._fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))
                    if token == -1:
                        # Stop advancing the grammar once we hit a padding token.
                        apply_bitmask = False
                    if apply_bitmask and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        assert accepted, (token, req_id, scheduled_spec_decode_tokens)
                        state_advancements += 1
                    cumulative_index += 1
                if state_advancements > 0:
                    grammar.rollback(state_advancements)

        bitmask_tensor = self._grammar_bitmask
        if cumulative_index < bitmask_tensor.shape[0]:
            bitmask_tensor = bitmask_tensor[:cumulative_index]

        # After finishing with the xgrammar operations, we convert to
        # np.ndarray, because that is much more efficient for serialization
        # and deserialization when sending this to the GPU workers.
        return bitmask_tensor.numpy()

    def should_fill_bitmask(self, request: "Request") -> bool:
        # SOURCE: vllm/v1/structured_output/__init__.py:L301-319
        # NOTE (Hanchen) if enable_in_reasoning is True, it means that
        # the model needs to be constrained in reasoning. So we should always
        # enable the bitmask filling.
        reasoner = self._get_reasoner(request)
        if reasoner is not None:
            if self.enable_in_reasoning:
                return True
            assert request.structured_output_request is not None
            if request.structured_output_request.reasoning_ended is None:
                request.structured_output_request.reasoning_ended = (
                    reasoner.is_reasoning_end(request.prompt_token_ids or [])
                )
            return request.structured_output_request.reasoning_ended
        return True

    def should_advance(self, request: "Request") -> bool:
        # SOURCE: vllm/v1/structured_output/__init__.py:L321-357
        if not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        if TYPE_CHECKING:
            assert request.structured_output_request is not None
            assert request.structured_output_request.grammar is not None
        # by default, we should always advance
        # for cases that don't use thinking mode.
        reasoner = self._get_reasoner(request)
        if reasoner is None:
            return True

        # if the model needs structured in reasoning, we should advance
        if self.enable_in_reasoning:
            return True

        structured_req = request.structured_output_request
        if structured_req.reasoning_ended:
            return True

        # Check if reasoning ends in *this* step
        delta_from = request.num_computed_tokens - request.num_output_placeholders
        all_token_ids = request.all_token_ids
        start = (
            delta_from if delta_from >= 0 else max(len(all_token_ids) + delta_from, 0)
        )
        if reasoner.is_reasoning_end_streaming(
            all_token_ids, itertools.islice(all_token_ids, start, None)
        ):
            # Reasoning just ended, so we shouldn't advance til
            # next pass
            structured_req.reasoning_ended = True

        return False
