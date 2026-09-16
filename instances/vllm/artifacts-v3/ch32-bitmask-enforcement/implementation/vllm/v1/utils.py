# SOURCE: vllm/v1/utils.py
# HOST SEAM：本章消费面一件——record_function_or_nullcontext
# （sample_tokens 的采样段护栏 gpu_model_runner:L4588——默认 nullcontext）。
# 逐字（_PROFILER_FUNC 快路径 + envs 两判据）。
from __future__ import annotations

import contextlib
from contextlib import AbstractContextManager

import vllm.envs as envs

_PROFILER_FUNC = None


# SOURCE: vllm/v1/utils.py:L756-L773 record_function_or_nullcontext —— 逐字
def record_function_or_nullcontext(name: str) -> AbstractContextManager:
    global _PROFILER_FUNC

    # fast path assume it is set
    if _PROFILER_FUNC is not None:
        return _PROFILER_FUNC(name)

    func = contextlib.nullcontext
    if envs.VLLM_CUSTOM_SCOPES_FOR_PROFILING:
        from torch.profiler import record_function

        func = record_function
    elif envs.VLLM_NVTX_SCOPES_FOR_PROFILING:
        import nvtx

        func = nvtx.annotate

    _PROFILER_FUNC = func
    return func(name)
