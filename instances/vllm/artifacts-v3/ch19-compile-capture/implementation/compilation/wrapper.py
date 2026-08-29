# Subtract-only companion for v3 ch19 — vllm/compilation/wrapper.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions (each marked `# SUBTRACTED:`), plus
# 章范围外域段以 SUBTRACTED+归属注记收窄（impl-notes §范围裁剪）。
#
# Kept surface: TorchCompileWithNoGuardsWrapper (m10) — 丢 guard +
# fullgraph=True + dynamic=False 一次编译永不回看。SUBTRACTED: nvtx 覆写
# 观测面（_call_with_optional_nvtx_range/layerwise_nvtx_tracing——观测域）、
# bytecode_hook 直执行路径与 aot_compile/cleanup（L156-L290——torch<2.8
# 兼容与 AOT 实验态）、reset_compile_wrapper（弹性 EP 域）。
from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Generator
from contextlib import contextmanager, nullcontext
from typing import Any, ParamSpec, TypeVar

import torch

from .._host_seams import (
    _apply_constrain_to_fx_strides_patch,
    envs,
    init_logger,
)
from ..config import CompilationMode, get_current_vllm_config
from ..config.compilation import DynamicShapesType

logger = init_logger(__name__)

R = TypeVar("R")
P = ParamSpec("P")


# SOURCE: vllm/compilation/wrapper.py:L26-L44 _compilation_context —— 编译期
#   调高 dynamo cache 上限（一次编译期间防 cache 抖动）
@contextmanager
def _compilation_context() -> Generator[None, None, None]:  # SOURCE: vllm/compilation/wrapper.py:L26-L44
    """Context manager for compilation settings.

    This manager sets higher dynamo cache limits for compilation.
    (Needed for qwen2_5_vl see test_qwen2_5_vl_evs_functionality).
    Generally, a recompilation can happen whenever we use a new
    backend instance in torch.compile.
    """
    original_cache_size = torch._dynamo.config.cache_size_limit
    original_accumulated_cache = torch._dynamo.config.accumulated_cache_size_limit

    try:
        torch._dynamo.config.cache_size_limit = 2048
        torch._dynamo.config.accumulated_cache_size_limit = 8192
        yield
    finally:
        torch._dynamo.config.cache_size_limit = original_cache_size
        torch._dynamo.config.accumulated_cache_size_limit = original_accumulated_cache


# SOURCE: vllm/compilation/wrapper.py:L47-L54 TorchCompileWithNoGuardsWrapper
#   —— 丢全部 guard → 首次 __call__ 触发一次编译、此后 Dynamo 永不再 trace
class TorchCompileWithNoGuardsWrapper:
    """
    A wrapper class for torch.compile, it ensures that all guards are dropped
    when CompilationMode is not CompilationMode.STOCK_TORCH_COMPILE.
    When guards are dropped, the first time __call__ is invoked, a single
    compilation is triggered. Dynamo should never be traced again after that
    since we drop all guards.
    """

    # SUBTRACTED: _call_with_optional_nvtx_range（L56-L70——nvtx 观测面，
    #   layerwise_nvtx_marker_context 为 vllm/utils/nvtx_pytorch_hooks 域）。

    # SOURCE: vllm/compilation/wrapper.py:L72-L160 __init__（nvtx 开关行随
    #   观测面删；bytecode_hook 注册尾段 L156-L160 随实验态删）
    def __init__(
        self,
        compile_prefix: str = "",
        is_encoder: bool = False,
    ) -> None:
        self.compiled = False
        self._compile_prefix = compile_prefix
        self._is_encoder = is_encoder

        vllm_config = get_current_vllm_config()
        self.vllm_config = vllm_config
        mode = vllm_config.compilation_config.mode
        # SUBTRACTED: layerwise_nvtx_tracing_enabled 开关行（L84-L86——观测域）。
        if mode is None:
            raise RuntimeError("Compilation mode cannot be NO_COMPILATION")

        backend = vllm_config.compilation_config.init_backend(
            vllm_config, prefix=compile_prefix, is_encoder=is_encoder
        )
        options = {}

        if isinstance(backend, str) and backend == "inductor":
            options = vllm_config.compilation_config.inductor_compile_config

        self.first_compile = True
        self.evaluate_guards = (
            vllm_config.compilation_config.dynamic_shapes_config.evaluate_guards
        )

        ds_type = vllm_config.compilation_config.dynamic_shapes_config.type

        # SOURCE: vllm/compilation/wrapper.py:L105-L125 丢全部 guard ——
        #   非 STOCK 模式下 guard_filter_fn 把 guard 表清空
        if mode != CompilationMode.STOCK_TORCH_COMPILE:
            # Drop all the guards.
            if self.evaluate_guards:
                assert not envs.VLLM_USE_BYTECODE_HOOK, (
                    "compilation_config.dynamic_shapes_config.evaluate_guards "
                    "requires VLLM_USE_BYTECODE_HOOK=0. "
                )
                assert ds_type != DynamicShapesType.UNBACKED, (
                    "UNBACKED dynamic shapes do not add guards"
                )

                options["guard_filter_fn"] = lambda x: [
                    entry.guard_type == "SHAPE_ENV" for entry in x
                ]
            else:
                if hasattr(torch.compiler, "skip_all_guards_unsafe"):
                    # Torch 2.10+ provides skip_all_guards_unsafe
                    options["guard_filter_fn"] = torch.compiler.skip_all_guards_unsafe
                else:
                    # Equivalent fallback for older PyTorch: skip all guards
                    options["guard_filter_fn"] = lambda x: [False for _ in x]

        compiled_ptr: Any = self.forward
        # Validate that unbacked dynamic shapes require VLLM_USE_BYTECODE_HOOK=False

        # Apply the constrain_to_fx_strides patch before first compilation.
        # This covers STOCK_TORCH_COMPILE and DYNAMO_ONCE paths. The VLLM
        # compile paths call this from their own compile() methods too.
        # SOURCE: vllm/compilation/wrapper.py:L130-L135
        _apply_constrain_to_fx_strides_patch()

        # SOURCE: vllm/compilation/wrapper.py:L137-L154 一次编译：
        #   fullgraph=True + dynamic=False + backend
        aot_context = nullcontext()
        if envs.VLLM_USE_AOT_COMPILE:
            if hasattr(torch._dynamo.config, "enable_aot_compile"):
                aot_context = torch._dynamo.config.patch(enable_aot_compile=True)
            else:
                msg = "torch._dynamo.config.enable_aot_compile is not "
                msg += "available. AOT compile is disabled and please "
                msg += "upgrade PyTorch version to use AOT compile."
                logger.warning(msg)

        with aot_context:
            self._compiled_callable = torch.compile(
                compiled_ptr,
                fullgraph=True,
                dynamic=False,
                backend=backend,
                options=options,
            )

        # SUBTRACTED: bytecode_hook 注册尾段（L156-L160——VLLM_USE_BYTECODE_
        #   HOOK 实验态，默认关）。

    # SUBTRACTED: aot_compile（L162-L169——AOT 实验态）。

    # SOURCE: vllm/compilation/wrapper.py:L171-L201 __call__（bytecode_hook
    #   支随实验态删，保 else 主支：首编后 fail_on_recompile 立场）
    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # SOURCE: vllm/compilation/wrapper.py:L171-L201
        # SUBTRACTED: bytecode_hook 直执行支（L172-L190——实验态默认关）。
        ctx = (
            nullcontext()
            if self.first_compile or not self.evaluate_guards
            else torch.compiler.set_stance("fail_on_recompile")
        )
        self.first_compile = False
        with _compilation_context(), ctx:
            # SUBTRACTED: nvtx range 包装（L199-L201——观测域）。
            return self._compiled_callable(*args, **kwargs)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any: ...  # SOURCE: vllm/compilation/wrapper.py:L203-L204

    # SUBTRACTED: original_code_object/bytecode_hook/_dispatch_to_compiled_code/
    #   cleanup（L206-L290——bytecode-hook 实验态）与 reset_compile_wrapper
    #   （L293-L346——弹性 EP 域）。
