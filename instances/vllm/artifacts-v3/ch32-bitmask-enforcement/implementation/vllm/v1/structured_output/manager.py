# SOURCE: vllm/v1/structured_output/__init__.py
# v3 ch31 脊柱①：StructuredOutputManager 的批装配切面——掩码缓冲/填充线程池
# （L57-L68）、单行语义 _fill_bitmasks（L194-L205）、批装配主函数 grammar_bitmask
# （L212-L359：预算/并行分支/串行 spec 窗口/裁剪/.numpy() 出发）、思考门控三件套
# （L361-L486）。编译侧全部删除（grammar_init/_create_grammar/四后端构造分支
# L114-L192、编译线程池+tokenizer+reasoning parser 装配 L70-L97、
# external_launcher 判定 L46-L55——dossier.delete[0]，ch30 主题）：
# 本章从『grammar 已就绪（StructuredOutputGrammar 成品挂在请求上）』起步，
# grammar 对象直接注入；reasoner_cls/tokenizer 同口径经属性注入
# （_get_reasoner 的惰性构造消费这两个属性，见 impl-notes 注入面说明）。
# TYPE_CHECKING 守卫断言与 LazyLoader 样板删（delete[7]——torch 改顶部显式
# import；串行分支的 AssertionError 是语义断言，原样保留）。
# 载体：真实类定义于 vllm/v1/structured_output/__init__.py（包 __init__ 承载）；
# 本镜像文件名为 manager.py（同包），全部锚点指向真实 __init__.py 行号。
import itertools
import multiprocessing
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
)

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from vllm.v1.request import Request
# SUBTRACTED: L14-L38 的 GuidanceBackend/XgrammarBackend 导入与
#   LazyLoader(torch) 样板（delete[0] 后端构造消费 / delete[7]）——
#   torch 改顶部显式 import（行为等价：LazyLoader 仅延迟加载）。


class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    # SOURCE: vllm/v1/structured_output/__init__.py:L38-L44 __init__ 头段 —— 逐字
    def __init__(self, vllm_config: VllmConfig):
        self.backend: StructuredOutputBackend | None = None
        # We only store the class of the reasoner in the manager.
        # The parser instance is request-scoped because some reasoning parsers
        # depend on per-request chat-template kwargs.
        self.reasoner_cls: "type[ReasoningParser] | None" = None
        self.vllm_config = vllm_config

        # SUBTRACTED: vllm/v1/structured_output/__init__.py:L46-L55
        #   _use_async_grammar_compilation（external_launcher 判定）——delete[0]
        #   编译侧装配（异步编译流归 ch30；本章 grammar 均已就绪）。

        # SOURCE: vllm/v1/structured_output/__init__.py:L57-L68 掩码缓冲 +
        #   填充线程池 —— 逐字
        self._grammar_bitmask: torch.Tensor | None = None
        self._full_mask = torch.tensor(-1, dtype=torch.int32)

        max_batch_size = self.vllm_config.scheduler_config.max_num_seqs
        self.fill_bitmask_parallel_threshold = 128
        if self.fill_bitmask_parallel_threshold < max_batch_size:
            self.fill_bitmask_parallel_batch_size = 16
            # Use:
            # - at least 1 CPU
            # - at most half the number of CPUs or 8, whichever is less
            max_workers = max(1, min(multiprocessing.cpu_count() // 2, 8))
            self.executor_for_fillmask = ThreadPoolExecutor(max_workers=max_workers)

        # SUBTRACTED: vllm/v1/structured_output/__init__.py:L70-L93 编译线程池
        #   self.executor + tokenizer + reasoning parser 插件/装配（含
        #   reasoner_cls = ReasoningParserManager.get_reasoning_parser(...)）
        #   ——delete[0] 编译侧装配。注意：本章思考门控（m14/m15）消费
        #   self.reasoner_cls / self.tokenizer 两个属性（_get_reasoner 的
        #   惰性构造位），按『grammar 直接注入』的同一口径由外部注入
        #   （manager.reasoner_cls = ... / manager.tokenizer = ...），
        #   见 impl-notes 注入面。

        # SOURCE: vllm/v1/structured_output/__init__.py:L95-L97 —— 逐字
        self.enable_in_reasoning = (
            self.vllm_config.structured_outputs_config.enable_in_reasoning
        )

    # SOURCE: vllm/v1/structured_output/__init__.py:L99-L112 _get_reasoner —— 逐字
    def _get_reasoner(self, request: "Request") -> "ReasoningParser | None":
        structured_req = request.structured_output_request
        if structured_req is None or self.reasoner_cls is None:
            return None

        if structured_req.reasoner is None:
            # Lazily build the request-local parser so the structured-output
            # gate observes the same template kwargs used by the frontend.
            parser_kwargs = structured_req.reasoning_parser_kwargs or {}
            structured_req.reasoner = self.reasoner_cls(
                tokenizer=self.tokenizer,
                **parser_kwargs,
            )
        return structured_req.reasoner

    # SUBTRACTED: vllm/v1/structured_output/__init__.py:L114-L192 grammar_init +
    #   _create_grammar（四后端构造分支/异步编译 Future/异常载具）——delete[0]
    #   编译侧全部，ch30 主题。

    # SOURCE: vllm/v1/structured_output/__init__.py:L194-L205 _fill_bitmasks —— 逐字
    def _fill_bitmasks(
        self, batch: Iterable[tuple[StructuredOutputGrammar, int, bool]]
    ) -> None:
        assert self._grammar_bitmask is not None
        for grammar, index, apply_bitmask in batch:
            if apply_bitmask and not grammar.is_terminated():
                grammar.fill_bitmask(self._grammar_bitmask, index)
            else:
                # Note that for thinking support, we will need to
                # reset the relevant part of the bitmask for consequent
                # requests here.
                self._grammar_bitmask[index].fill_(self._full_mask)

    # SOURCE: vllm/v1/structured_output/__init__.py:L207-L210 _async_submit_fill_bitmask —— 逐字
    def _async_submit_fill_bitmask(
        self, batch: list[tuple[StructuredOutputGrammar, int, bool]]
    ) -> Future:
        return self.executor_for_fillmask.submit(self._fill_bitmasks, batch)

    #   （仅删两处 TYPE_CHECKING 守卫断言，delete[7]）
    # SOURCE: vllm/v1/structured_output/__init__.py:L212-L359 grammar_bitmask —— 逐字
    def grammar_bitmask(
        self,
        requests: dict[str, "Request"],
        structured_output_request_ids: list[str],
        scheduled_spec_decode_tokens: dict[str, list[int]],
    ) -> "npt.NDArray[np.int32] | None":
        # Prepare the structured output bitmask for this batch.
        if not structured_output_request_ids:
            return None

        # Covers both speculative decoding and diffusion LLMs (canvas_length).
        max_num_spec_tokens = self.vllm_config.num_speculative_tokens

        if self._grammar_bitmask is None:
            assert self.backend is not None
            max_batch_size = self.vllm_config.scheduler_config.max_num_seqs

            # Allocate a bitmask for each token needing to be checked:
            # one for each speculative position, and one more for the
            # bonus token / non-speculative token.
            self._grammar_bitmask = self.backend.allocate_token_bitmask(
                max_batch_size * (1 + max_num_spec_tokens)
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
                # SUBTRACTED: L253-L257 两处 TYPE_CHECKING 守卫断言（delete[7]）
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

                # SUBTRACTED: L278-L282 两处 TYPE_CHECKING 守卫断言（delete[7]）
                grammar = structured_output_request.grammar
                apply_bitmask = self.should_fill_bitmask(request)

                reasoner = self._get_reasoner(request)
                detect_reasoning_end = (
                    not apply_bitmask
                    and reasoner is not None
                    and not self.enable_in_reasoning
                )
                simulated_buf: list[int] | None = None
                history_len = 0

                state_advancements = 0
                post_reasoning_end_in_window = False
                req_tokens = scheduled_spec_decode_tokens.get(req_id, ())
                for i, token in enumerate(req_tokens):
                    self._fill_bitmasks(((grammar, cumulative_index, apply_bitmask),))
                    advance_grammar = apply_bitmask
                    if token == -1:
                        apply_bitmask = False
                        advance_grammar = False
                    elif (
                        detect_reasoning_end
                        and reasoner is not None
                        and not apply_bitmask
                    ):
                        if simulated_buf is None:
                            history = list(request.all_token_ids)
                            history_len = len(history)
                            simulated_buf = history + list(req_tokens)
                        simulated = simulated_buf[: history_len + i + 1]
                        if reasoner.is_reasoning_end_streaming(simulated, [token]):
                            # Reasoning ended mid-window. Constrain the rest
                            # of the window via bitmask. Skip grammar advance
                            # through the marker (it is reasoning content);
                            # try to advance through subsequent drafts so the
                            # next bitmask row reflects the post-advance state,
                            # but tolerate rejection since those drafts predate
                            # the bitmask and are not guaranteed valid.
                            apply_bitmask = True
                            advance_grammar = False
                            post_reasoning_end_in_window = True
                    if advance_grammar and not grammar.is_terminated():
                        accepted = grammar.accept_tokens(req_id, [token])
                        if accepted:
                            state_advancements += 1
                        elif not post_reasoning_end_in_window:
                            raise AssertionError(
                                (token, req_id, scheduled_spec_decode_tokens)
                            )
                    cumulative_index += 1
                # Diffusion LLMs don't sample a bonus token after the
                # scheduled positions, so skip its bitmask in that case.
                if not (self.vllm_config.model_config.is_diffusion and req_tokens):
                    # bonus_apply must be True when the bonus-row position
                    # should be grammar-constrained. Two triggers:
                    # - should_fill_bitmask(request): reasoning was already
                    #   over at step start (or no reasoner /
                    #   enable_in_reasoning).
                    # - apply_bitmask: reasoning ended mid-window in this
                    #   call and was flipped True after the marker;
                    #   should_fill_bitmask still returns False here because
                    #   reasoning_ended is only persisted later by
                    #   should_advance.
                    bonus_apply = self.should_fill_bitmask(request) or apply_bitmask
                    self._fill_bitmasks(((grammar, cumulative_index, bonus_apply),))
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

    # SOURCE: vllm/v1/structured_output/__init__.py:L361-L379 should_fill_bitmask —— 逐字
    def should_fill_bitmask(self, request: "Request") -> bool:
        # NOTE (Hanchen) if enable_in_reasoning is True, it means that
        # the model needs to be constrained in reasoning. So we should always
        # enable the bitmask filling.
        reasoner = self._get_reasoner(request)
        if reasoner is not None:
            if self.enable_in_reasoning:
                return True
            assert request.structured_output_request is not None
            if request.structured_output_request.reasoning_ended is None:
                # This should be removed here, but since `openai_gptoss`
                # is an independent code path, it is kept for now.
                # After unifying the `openai_gptoss` and non-`openai_gptoss` styles,
                # it can be removed.
                request.structured_output_request.reasoning_ended = (
                    reasoner.is_reasoning_end(request.prompt_token_ids or [])
                )
            return request.structured_output_request.reasoning_ended
        return True

    #   （仅删 TYPE_CHECKING 守卫断言，delete[7]）
    # SOURCE: vllm/v1/structured_output/__init__.py:L381-L439 should_advance —— 逐字
    def should_advance(
        self,
        request: "Request",
        new_token_ids: list[int] | None = None,
    ) -> bool:
        if not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        # SUBTRACTED: L391-L393 两处 TYPE_CHECKING 守卫断言（delete[7]）
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

        # Check if reasoning ends in *this* step.
        # When the caller passes new_token_ids (the tokens that were just
        # appended this step), use it directly as the delta window. The
        # placeholder-derived fallback assumes num_output_placeholders ==
        # len(new_token_ids), which breaks under async scheduling + spec
        # decode when some drafts are rejected (#43388): the placeholder
        # count remains > 0 after the step and the computed delta window
        # starts past the reasoning-end marker.
        all_token_ids = request.all_token_ids
        if new_token_ids:
            # The tokens were already appended this step, so the step window
            # starts exactly len(new_token_ids) from the end.
            start = len(all_token_ids) - len(new_token_ids)
            delta_ids: Iterable[int] = new_token_ids
        else:
            delta_from = request.num_computed_tokens - request.num_output_placeholders
            start = (
                delta_from
                if delta_from >= 0
                else max(len(all_token_ids) + delta_from, 0)
            )
            delta_ids = itertools.islice(all_token_ids, start, None)
        if reasoner.is_reasoning_end_streaming(all_token_ids, delta_ids):
            structured_req.reasoning_ended = True

            # Record the boundary so the scheduler can exclude reasoning tokens.
            end_index = self._find_reasoning_end_index(reasoner, all_token_ids, start)

            structured_req.reasoning_end_token_index = end_index
            return True

        return False

    @staticmethod
    # SOURCE: vllm/v1/structured_output/__init__.py:L441-L460 _find_reasoning_end_index —— 逐字
    def _find_reasoning_end_index(
        reasoner: "ReasoningParser", all_token_ids: Sequence[int], start: int
    ) -> int:
        """Locates the last reasoning token within ``all_token_ids[start:]``.

        Returns:
            The absolute index of the token at which
            ``is_reasoning_end_streaming`` first fires. Falls back to the
            final index when no single token triggers the detection (e.g.
            a multi-token marker only recognized on the full delta), which
            conservatively treats the whole step as reasoning content.
        """
        prefix = list(itertools.islice(all_token_ids, start))
        for idx in range(start, len(all_token_ids)):
            token = all_token_ids[idx]
            prefix.append(token)
            if reasoner.is_reasoning_end_streaming(prefix, [token]):
                return idx
        return len(all_token_ids) - 1

    # SOURCE: vllm/v1/structured_output/__init__.py:L462-L486 trim_reasoning_for_advance —— 逐字
    def trim_reasoning_for_advance(
        self, request: "Request", new_token_ids: list[int]
    ) -> list[int]:
        """Drops reasoning content from tokens about to advance the grammar.

        When reasoning ends mid-step (see should_advance), the step's output
        still contains reasoning tokens up to and including the end marker.
        Those are not grammar content: feeding them to accept_tokens makes
        the grammar reject the marker and kills the request (#44006).

        Returns:
            The suffix of ``new_token_ids`` that follows the reasoning-end
            marker. Steps fully after the boundary are returned unchanged.
        """
        structured_req = request.structured_output_request
        if structured_req is None:
            return new_token_ids
        end_idx = structured_req.reasoning_end_token_index
        if end_idx is None:
            return new_token_ids
        first_idx = len(request.all_token_ids) - len(new_token_ids)
        num_reasoning = end_idx + 1 - first_idx
        if num_reasoning <= 0:
            return new_token_ids
        return new_token_ids[num_reasoning:]

    # SOURCE: vllm/v1/structured_output/__init__.py:L488-L490 clear_backend —— 逐字
    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()
