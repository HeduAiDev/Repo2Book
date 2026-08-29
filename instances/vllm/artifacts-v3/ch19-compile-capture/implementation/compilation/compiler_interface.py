# Subtract-only companion for v3 ch19 — vllm/compilation/compiler_interface.py
# (pin v0.27.1 / 6e448d0ea). Kept surface: the compiler adaptor protocol that
# PiecewiseBackend reaches through CompilerManager (EagerAdaptor verbatim —
# the host-CPU face the companion exercises; InductorAdaptor kept as a shell
# whose compile body is out of the ch19 surface — it runs in the vllm
# container) and trigger_inductor_lazy_init (m18 startup-tail call face).
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.fx as fx

from .._host_seams import init_logger
from .counter import compilation_counter

logger = init_logger(__name__)

# SUBTRACTED: vllm/compilation/compiler_interface.py 的缓存装配与 torch
#   monkey-patch 族（get_inductor_factors/set_inductor_config/set_functorch_
#   config/_patch_standalone_compile_atomic_save 等，L27-L248——Inductor 配置
#   与序列化域，随 delete[3] 缓存块删除）；InductorStandaloneAdaptor
#   （L251-L470——VLLM_USE_STANDALONE_COMPILE 实验态）。


# SOURCE: vllm/compilation/compiler_interface.py:L27-L110 CompilerInterface
#   —— 编译器适配协议（compile 返回 (compiled_graph, handle)）
class CompilerInterface:
    """
    The interface for a compiler that can be used by vLLM.
    """

    # The name of the compiler, e.g. inductor.
    # This is a class-level attribute.
    name: str

    def initialize_cache(  # SOURCE: vllm/compilation/compiler_interface.py:L33-L53
        self, cache_dir: str, disable_cache: bool = False, prefix: str = ""
    ) -> None:
        """
        when the vllm process uses `cache_dir` as the cache directory,
        the compiler should initialize itself with the cache directory,
        e.g. by re-directing its own cache directory to a sub-directory.
        """
        pass

    def compute_hash(self, vllm_config: Any) -> str:  # SOURCE: vllm/compilation/compiler_interface.py:L54-L66
        """
        Gather all of the relevant information from the vllm config,
        to compute a hash so that we can cache the compiled model.
        """
        return ""

    def compile(  # SOURCE: vllm/compilation/compiler_interface.py:L66-L110
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Any,
        key: str | None = None,
    ) -> tuple[Callable[..., Any] | None, Any | None]:
        raise NotImplementedError


# SOURCE: vllm/compilation/compiler_interface.py:L449 InductorAdaptor —— 壳
#   保留（make_compiler 的 inductor 分支返回它）；compile 体（L482-L795，
#   ~230 行 set_inductor_config/artifact 序列化）属 torch-internal 装配域，
#   SUBTRACTED：宿主伴读走 eager 适配器，inductor 路线在 vllm 容器内跑。
class InductorAdaptor(CompilerInterface):
    name = "inductor"

    def compile(  # SOURCE: vllm/compilation/compiler_interface.py:L482 起（体删）
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Any,
        key: str | None = None,
    ) -> tuple[Callable[..., Any] | None, Any | None]:
        # SUBTRACTED: L482-L795 compile 体（torch._inductor 配置/序列化——
        #   Inductor 内部装配域；宿主经 backend="eager" 走 EagerAdaptor）。
        raise NotImplementedError(
            "InductorAdaptor.compile is torch-internal assembly (vllm "
            "container domain); the ch19 host companion exercises the eager "
            "adaptor path."
        )


# SOURCE: vllm/compilation/compiler_interface.py:L796-L810 EagerAdaptor ——
#   不编译，返回图本体；无缓存 handle
class EagerAdaptor(CompilerInterface):  # SOURCE: vllm/compilation/compiler_interface.py:L796-L810
    name = "eager"

    def compile(  # SOURCE: vllm/compilation/compiler_interface.py:L796-L810
        self,
        graph: fx.GraphModule,
        example_inputs: list[Any],
        compiler_config: dict[str, Any],
        compile_range: Any,
        key: str | None = None,
    ) -> tuple[Callable[..., Any] | None, Any | None]:
        compilation_counter.num_eager_compiles += 1
        # we don't need to compile the graph, just return the graph itself.
        # It does not support caching, return None for the handle.
        return graph, None


# SOURCE: vllm/compilation/compiler_interface.py:L768-L794 trigger_inductor_
#   lazy_init —— 启动期预热 inductor 的进程级惰性初始化（m18 尾四连之一）
def trigger_inductor_lazy_init(device: torch.device | None = None) -> None:
    """Eagerly trigger inductor's once-per-process lazy inits (SFDP pattern
    matcher, pad_mm, misc patterns).

    These normally fire on the first torch.compile invocation and include
    CUDA syncs. If warmup hits the on-disk compile cache, no compile actually
    runs so these never fire during warmup, and they'd blow up on the first
    real-request cache miss once the sync-check gate is on.

    Private torch API; best-effort. Newer torch versions take an
    `input_device` argument and cache per-device, so pass the current CUDA
    device to ensure the cache key matches later compile calls.
    """
    # SOURCE: vllm/compilation/compiler_interface.py:L781-L794 try 体（逐字）
    try:
        import inspect

        from torch._inductor.fx_passes.joint_graph import (
            lazy_init as _inductor_lazy_init,
        )

        if inspect.signature(_inductor_lazy_init).parameters:
            _inductor_lazy_init(device)
        else:
            _inductor_lazy_init()
    except Exception as e:  # noqa: BLE001
        logger.info("Skipping inductor lazy_init pre-trigger: %s", e)


# SUBTRACTED: is_compile_cache_enabled（L193-L?——缓存开关判读，消费点随
#   delete[3] 缓存块删除；伴读版 CompilerManager 恒 disable_cache）。
