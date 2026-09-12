# SOURCE: vllm/v1/structured_output/__init__.py
# 只做减法的忠实精简版（pin v0.27.1 / 6e448d0ea）——引擎级编排者。
# 删除项（subtraction_plan.delete）：
#   [3] 批掩码装配整块（grammar_bitmask/_fill_bitmasks/_async_submit_fill_
#       bitmask/executor_for_fillmask 装配/_grammar_bitmask/_full_mask/
#       should_fill_bitmask）——ch31 主场；
#   [4] grammar_init 内 outlines/lm-format-enforcer 两 elif 分支；
#   [5] reasoning parser 装配段（四方法本体按 must_keep 原样保留）；
#   [7] logger 调用/TYPE_CHECKING 断言块。
# SUBTRACTED: SPDX 版权头；模块级 `if TYPE_CHECKING: import numpy/torch/
#   from vllm.reasoning import ReasoningParser/from vllm.v1.request import
#   Request` 与 else 位 torch=LazyLoader(...)（L21-L29——torch/numpy 的消费者
#   已随 delete[3] 删除；剩余注解全是字符串形式、不经运行时求值）；
# `from vllm.utils.import_utils import LazyLoader`（只服务上述 torch 位）。
import itertools
import multiprocessing
from collections.abc import Iterable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.tokenizers import cached_tokenizer_from_config
from vllm.v1.structured_output.backend_guidance import GuidanceBackend
from vllm.v1.structured_output.backend_types import (
    StructuredOutputBackend,
    StructuredOutputGrammar,
)
from vllm.v1.structured_output.backend_xgrammar import XgrammarBackend

logger = init_logger(__name__)


# SOURCE: vllm/v1/structured_output/__init__.py:L35-L490 StructuredOutputManager
class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    # SOURCE: vllm/v1/structured_output/__init__.py:L38-L97 __init__
    def __init__(self, vllm_config: VllmConfig):
        # SOURCE: vllm/v1/structured_output/__init__.py:L39-L44
        self.backend: StructuredOutputBackend | None = None
        # We only store the class of the reasoner in the manager.
        # The parser instance is request-scoped because some reasoning parsers
        # depend on per-request chat-template kwargs.
        self.reasoner_cls: "type | None" = None
        self.vllm_config = vllm_config

        # SOURCE: vllm/v1/structured_output/__init__.py:L46-L55 —— 逐字
        #   external_launcher 回退同步开关（m14）
        # When in external_launcher mode, async grammar compilation causes deadlocks
        # due to external_launcher mode having a scheduler for each TP rank.
        # Async grammar compilation causes the
        # WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR → WAITING transition to
        # happen at different times on different TP ranks,
        # breaking the determinism assumption that external_launcher relies on.
        self._use_async_grammar_compilation = (
            vllm_config.parallel_config.distributed_executor_backend
            != "external_launcher"
        )

        # SUBTRACTED: L57-L58 _grammar_bitmask/_full_mask 字段与 L60-L68
        #   fill_bitmask 并行线程池装配（fill_bitmask_parallel_threshold/
        #   batch_size/executor_for_fillmask）——delete[3]，ch31 主场。

        # SOURCE: vllm/v1/structured_output/__init__.py:L70-L80 —— 逐字
        #   （半 CPU 线程池：编译是 CPU-bound 非 IO-bound）
        if not self.vllm_config.model_config.skip_tokenizer_init:
            # The default max_workers if not specified is the number of
            # CPUs * 5, which is way too high since these tasks are CPU-bound,
            # not I/O bound. We also know we would never dominate CPU usage
            # with just grammar compilation, so we set it to half the number
            # of CPUs.
            max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            self.tokenizer = cached_tokenizer_from_config(
                model_config=self.vllm_config.model_config
            )
            # SUBTRACTED: L81-L93 reasoning parser 装配段
            #   （reasoning_parser_plugin import + reasoning_parser 查表构造
            #   self.reasoner_cls = ReasoningParserManager.get_reasoning_
            #   parser(...)——delete[5]。reasoner_cls 停留在 L43 的 None 初值，
            #   即源码真实路径（不配 reasoning_parser 的模型即此）。

        # SOURCE: vllm/v1/structured_output/__init__.py:L95-L97
        self.enable_in_reasoning = (
            self.vllm_config.structured_outputs_config.enable_in_reasoning
        )

    # SOURCE: vllm/v1/structured_output/__init__.py:L99-L112 _get_reasoner
    #   —— 逐字（should_advance 的第一跳；reasoner None 快返回的常量路径）
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

    # SOURCE: vllm/v1/structured_output/__init__.py:L114-L175 grammar_init
    #   —— 命门：惰性建唯一后端 + 提交线程池（IO 线程独占调用）
    def grammar_init(self, request: "Request") -> None:
        if request.structured_output_request is None:
            return

        # SUBTRACTED: L118-L122 `if TYPE_CHECKING: assert(...)` 块（delete[7]）

        # Initialize the backend the first time it is needed.
        #
        # NOTE: We only support a single backend. We do NOT support different
        # backends on a per-request basis in V1 (for now, anyway...).
        # _backend is set in Processor._validate_structured_output
        if self.backend is None:
            assert request.sampling_params is not None
            backend = request.sampling_params.structured_outputs._backend
            vocab_size = self.vllm_config.model_config.get_vocab_size()
            if backend == "xgrammar":
                self.backend = XgrammarBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            elif backend == "guidance":
                self.backend = GuidanceBackend(
                    self.vllm_config,
                    tokenizer=self.tokenizer,
                    vocab_size=vocab_size,
                )
            # SUBTRACTED: L145-L162 outlines/lm-format-enforcer 两 elif 分支
            #   （含函数内惰性 import）——delete[4]：两后端整体删除。
            else:
                # SOURCE: vllm/v1/structured_output/__init__.py:L163-L164
                #   非法后端防御（保留）
                raise ValueError(f"Unsupported structured output backend: {backend}")

        # SOURCE: vllm/v1/structured_output/__init__.py:L166-L175 —— 逐字
        #   （external_launcher 回退同步、异常也包 Future 保持下游同形）
        grammar: Future[StructuredOutputGrammar] | StructuredOutputGrammar
        if self._use_async_grammar_compilation:
            grammar = self.executor.submit(self._create_grammar, request)
        else:
            try:
                grammar = self._create_grammar(request)
            except Exception as e:
                grammar = Future()
                grammar.set_exception(e)
        request.structured_output_request.grammar = grammar

    # SOURCE: vllm/v1/structured_output/__init__.py:L177-L192 _create_grammar
    #   —— 线程池工作函数：key→compile_grammar 一跳 + 失败经 Future 传调度器
    def _create_grammar(self, request: "Request") -> StructuredOutputGrammar:
        struct_request = request.structured_output_request
        assert struct_request is not None
        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request. Grammar
        # compilation may still fail; the Future carries that error to the
        # scheduler so it can fail only this request.
        try:
            request_type, grammar_spec = struct_request.structured_output_key
            assert self.backend is not None
            return self.backend.compile_grammar(request_type, grammar_spec)
        except Exception:
            # SUBTRACTED: logger.exception("Failed to compile grammar for
            #   request %s", request.request_id)（L189-L191——delete[7]；
            #   异常仍经 Future 原样传回调度器）
            raise

    # SUBTRACTED: L194-L210 _fill_bitmasks/_async_submit_fill_bitmask、
    #   L212-L359 grammar_bitmask（批装配/并行填充/spec 窗口预推进+rollback/
    #   ndarray 转换）、L361-L379 should_fill_bitmask（fill 侧思考门）——
    #   delete[3]，ch31 主场。本章终点=『FSM 编译好、能被调度器推进』。

    # SOURCE: vllm/v1/structured_output/__init__.py:L381-L439 should_advance
    #   —— 推进侧思考门（m17）：reasoner None 快返回 True 的常量路径 +
    #   reasoning_ended 缓存 + is_reasoning_end_streaming 探测（#43388）
    def should_advance(
        self,
        request: "Request",
        new_token_ids: list[int] | None = None,
    ) -> bool:
        if not request.use_structured_output:
            return False

        # To determine whether we can advance the FSM.
        # Supports thinking usage where we skip the reasoning components.
        # SUBTRACTED: L391-L393 `if TYPE_CHECKING: assert(...)` 块（delete[7]）
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

    # SOURCE: vllm/v1/structured_output/__init__.py:L441-L460 _find_reasoning_end_index
    #   —— 逐字（定位思考结束 token；无单 token 触发时保守回退末位）
    @staticmethod
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

    # SOURCE: vllm/v1/structured_output/__init__.py:L462-L486 trim_reasoning_for_
    #   advance —— 逐字（剔除混在同一步的思考 token，#44006）
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

    # SOURCE: vllm/v1/structured_output/__init__.py:L488-L490 clear_backend
    def clear_backend(self) -> None:
        if self.backend is not None:
            self.backend.destroy()
