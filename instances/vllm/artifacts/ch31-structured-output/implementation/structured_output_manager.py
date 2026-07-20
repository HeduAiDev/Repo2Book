# SOURCE: vllm/v1/structured_output/__init__.py（内容对应真实的包 __init__.py；
# 精简版改名 structured_output_manager.py——本工厂约定 __init__.py 只作包标记，
# 不参与保真度扫描，见 ch13/ch07 等既有章节同一约定）。
#
# StructuredOutputManager：引擎级单例，惰性构造唯一后端、把编译提交线程池、
# 每步装配 bitmask（批装配部分本章不实现，见下方 SUBTRACTED）。
#
# SUBTRACTED: SPDX 版权头。
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

from backend_guidance import GuidanceBackend
from backend_types import StructuredOutputBackend, StructuredOutputGrammar
from backend_xgrammar import XgrammarBackend


class StructuredOutputManager:
    """Engine-level manager for structured output requests."""

    def __init__(self, vllm_config, tokenizer=None, external_launcher: bool = False):
        # SOURCE: vllm/v1/structured_output/__init__.py:L38-97
        self.backend: "StructuredOutputBackend | None" = None
        self.vllm_config = vllm_config

        # SOURCE: vllm/v1/structured_output/__init__.py:L46-55 —— external_launcher
        # 模式下每个 TP rank 各有一个 scheduler，异步编译会让
        # WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR → WAITING 的跃迁在不同 rank 上发生在
        # 不同时刻，破坏 external_launcher 依赖的确定性假设，进而死锁——因此该模式
        # 退回同步编译。
        self._use_async_grammar_compilation = not external_launcher

        # SUBTRACTED: `self._grammar_bitmask` / `self._full_mask` 字段（批装配用，
        # 批准项4）；`fill_bitmask_parallel_threshold` / `executor_for_fillmask`
        # 并行填充线程池（批准项4，__init__.py:L60-68）。

        # SOURCE: vllm/v1/structured_output/__init__.py:L70-80
        #
        # The default max_workers if not specified is the number of
        # CPUs * 5, which is way too high since these tasks are CPU-bound,
        # not I/O bound. We also know we would never dominate CPU usage
        # with just grammar compilation, so we set it to half the number
        # of CPUs.
        max_workers = max(1, (multiprocessing.cpu_count() + 1) // 2)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tokenizer = tokenizer

        # SUBTRACTED: reasoning_parser_plugin / reasoner_cls 装配、
        # `_get_reasoner`、`enable_in_reasoning`（__init__.py:L40-43, L81-97, L99-112）
        # ——reasoning 相关，批准项3整体删除。

    def grammar_init(self, request) -> None:
        # SOURCE: vllm/v1/structured_output/__init__.py:L114-170
        #
        # 【命门】编译扔进线程池，绝不阻塞调度循环；请求进阻塞等待态
        # （见 request.py 里 WAITING_FOR_STRUCTURED_OUTPUT_GRAMMAR 的置位）。
        if request.structured_output_request is None:
            return

        # Initialize the backend the first time it is needed.
        #
        # NOTE: We only support a single backend. We do NOT support different
        # backends on a per-request basis in V1 (for now, anyway...).
        # _backend is set in Processor._validate_structured_output
        if self.backend is None:
            backend = request.sampling_params.structured_outputs._backend
            # SUBTRACTED: 真实取值路径是
            # `self.vllm_config.model_config.get_vocab_size()`——ModelConfig 不在本章
            # 范围，用 getattr 兜底代替。
            vocab_size = getattr(self.vllm_config, "vocab_size", 128)
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
            # SUBTRACTED: `elif backend == "outlines": ...` /
            # `elif backend == "lm-format-enforcer": ...`（批准项5，两个后端文件本身
            # 已整体删除；这两个 elif 只是 new 对应类，删掉调用点是删除后端文件的
            # 必然连带，不是独立判断）。
            else:
                raise ValueError(f"Unsupported structured output backend: {backend}")

        if self._use_async_grammar_compilation:
            grammar = self.executor.submit(self._create_grammar, request)
        else:
            grammar = self._create_grammar(request)  # type: ignore[assignment]
        request.structured_output_request.grammar = grammar  # type: ignore[assignment]

    def _create_grammar(self, request) -> StructuredOutputGrammar:
        # SOURCE: vllm/v1/structured_output/__init__.py:L172-183
        key = request.structured_output_request.structured_output_key

        # Note that the request was validated in the engine core client,
        # so at this point we know it is a supported type of request.
        request_type, grammar_spec = key

        assert self.backend is not None
        return self.backend.compile_grammar(request_type, grammar_spec)

    # SUBTRACTED: `_fill_bitmasks` / `_async_submit_fill_bitmask` / `grammar_bitmask`
    # 主体 / `should_advance` / `should_fill_bitmask`（__init__.py:L185-357）——
    # 批装配/并行填充（批准项4）与 reasoning 门控（批准项3）均归下一章
    # （约束解码 II）。本章到"语法对象造好、能 fill_bitmask"为止。
