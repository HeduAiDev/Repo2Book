# Subtract-only companion for v3 ch19 — vllm/compilation/counter.py
# (pin v0.27.1 / 6e448d0ea). Verbatim: the whole file is the compilation
# observation counter used by backends/cuda_graph/runner spans.
from __future__ import annotations

import copy
import dataclasses
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any


# SOURCE: vllm/compilation/counter.py:L11-L55 CompilationCounter
@dataclasses.dataclass
class CompilationCounter:
    num_models_seen: int = 0
    num_graphs_seen: int = 0
    # including the splitting ops
    num_piecewise_graphs_seen: int = 0
    # not including the splitting ops
    num_piecewise_capturable_graphs_seen: int = 0
    num_backend_compilations: int = 0
    # Number of gpu_model_runner attempts to trigger CUDAGraphs capture
    num_gpu_runner_capture_triggers: int = 0
    # Number of CUDAGraphs captured
    num_cudagraph_captured: int = 0
    # InductorAdapter.compile calls
    num_inductor_compiles: int = 0
    # EagerAdapter.compile calls
    num_eager_compiles: int = 0
    # The number of time vLLM's compiler cache entry was updated
    num_cache_entries_updated: int = 0
    # The number of standalone_compile compiled artifacts saved
    num_compiled_artifacts_saved: int = 0
    # The number of standalone_compile compiled artifacts loaded from cache
    num_compiled_artifacts_loaded: int = 0
    # The number of AOT compile invocations
    num_aot_compiles: int = 0
    # The number of AOT compiled artifacts saved to disk
    num_aot_artifacts_saved: int = 0
    # The number of AOT compiled artifacts loaded from disk
    num_aot_artifacts_loaded: int = 0
    # Number of times a model was loaded with CompilationMode.STOCK_TORCH_COMPILE
    stock_torch_compile_count: int = 0

    def clone(self) -> "CompilationCounter":  # SOURCE: vllm/compilation/counter.py:L43-L44
        return copy.deepcopy(self)

    @contextmanager
    def expect(self, **kwargs: Any) -> Generator[None, None, None]:  # SOURCE: vllm/compilation/counter.py:L46-L55
        old = self.clone()
        yield
        for k, v in kwargs.items():
            assert getattr(self, k) - getattr(old, k) == v, (
                f"{k} not as expected, before it is {getattr(old, k)}"
                f", after it is {getattr(self, k)}, "
                f"expected diff is {v}"
            )


# SOURCE: vllm/compilation/counter.py:L58 compilation_counter（模块级单例）
compilation_counter = CompilationCounter()
