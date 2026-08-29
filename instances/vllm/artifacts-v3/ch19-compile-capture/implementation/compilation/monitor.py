# Subtract-only companion for v3 ch19 — vllm/compilation/monitor.py
# (pin v0.27.1 / 6e448d0ea). Kept surface: the cudagraph capture window
# tripwire (set/validate — capture_model 与 CUDAGraphWrapper 捕获路径的先验)
# plus the shared dynamo timing global. The depyf/profiling timing context
# managers are cache/observation domain (delete[3] family), SUBTRACTED.
from __future__ import annotations

from .._host_seams import init_logger

logger = init_logger(__name__)

# SUBTRACTED: monitor_torch_compile / monitor_profiling_run（L17-L84——
#   deyf 转储与 profiling 计时上下文，缓存/观测域，随 delete[3] 一族删除；
#   torch_compile_start_time 全局量保留（VllmBackend.__call__ 的计数伴读）。

# Shared global so backends.py can read the start time for Dynamo timing.
# SOURCE: vllm/compilation/monitor.py:L13-L14 torch_compile_start_time
torch_compile_start_time: float = 0.0

# SUBTRACTED: depyf 计时段（L17-L84）。

# SOURCE: vllm/compilation/monitor.py:L87 cudagraph_capturing_enabled ——
#   捕获窗口全局开关（默认开；capture_model 收尾关掉）
cudagraph_capturing_enabled: bool = True


# SOURCE: vllm/compilation/monitor.py:L90-L99 validate_cudagraph_capturing_
#   enabled —— 意外捕获 tripwire：窗口关闭后任何捕获直接 RuntimeError
def validate_cudagraph_capturing_enabled() -> None:  # SOURCE: vllm/compilation/monitor.py:L90-L99
    # used to monitor whether a cudagraph capturing is legal at runtime.
    # should be called before any cudagraph capturing.
    # if an illegal cudagraph capturing happens, raise an error.
    global cudagraph_capturing_enabled
    if not cudagraph_capturing_enabled:
        raise RuntimeError(
            "CUDA graph capturing detected at an inappropriate "
            "time. This operation is currently disabled."
        )


# SOURCE: vllm/compilation/monitor.py:L102-L104 set_cudagraph_capturing_enabled
def set_cudagraph_capturing_enabled(enabled: bool) -> None:
    global cudagraph_capturing_enabled
    cudagraph_capturing_enabled = enabled
