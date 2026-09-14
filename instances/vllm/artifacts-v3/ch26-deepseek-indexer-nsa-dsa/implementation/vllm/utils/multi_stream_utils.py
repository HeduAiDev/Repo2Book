# SOURCE: vllm/utils/multi_stream_utils.py
# ch26 切面（站 13/m11）：maybe_execute_in_parallel 逐字（DeepseekV4Indexer
# 的 wq_b+量化 与 compressor 双流重叠；aux_stream=None → 顺序执行、join 保证
# compressor 写 K 先于 indexer_op——host 与 ROCm 无流形态同型）。
# SUBTRACTED：execute_in_parallel（attention.py 的三路 GEMM 重叠——V4 base
# 装配域，本章不携带；→ ch28）。
from __future__ import annotations

from typing import Any, Callable

import torch


# SOURCE: vllm/utils/multi_stream_utils.py:L20-L67 maybe_execute_in_parallel
#   —— 逐字
def maybe_execute_in_parallel(
    fn0: Callable[[], Any],
    fn1: Callable[[], Any],
    event0: torch.cuda.Event,
    event1: torch.cuda.Event,
    aux_stream: torch.cuda.Stream | None = None,
) -> tuple[Any, Any]:
    """Run two functions potentially in parallel on separate CUDA streams.

    When aux_stream is provided, fn0 runs on the current (default) stream and
    fn1 runs on aux_stream, synchronized via CUDA events. When aux_stream is
    None or a breakable CUDA graph capture is active, both functions execute
    sequentially on the current stream.

    This design follows TensorRT-LLM's maybe_execute_in_parallel pattern
    (tensorrt_llm/_torch/modules/multi_stream_utils.py).

    Args:
        fn0: Callable for the default stream.
        fn1: Callable for the auxiliary stream.
        event0: CUDA event recorded before fn0 so aux_stream can wait.
        event1: CUDA event recorded after fn1 so default stream can wait.
        aux_stream: The second CUDA stream for fn1.
            Multi-stream is disabled when aux_stream is None or a breakable
            CUDA graph capture is active.

    Returns:
        Tuple of (fn0_result, fn1_result).
    """
    # SOURCE: vllm/utils/multi_stream_utils.py:L20-L67（锚点双置：声明上方同文）
    if aux_stream is not None:
        from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphCapture

        if BreakableCUDAGraphCapture.is_active():
            aux_stream = None

    if aux_stream is not None:
        event0.record()
        result0 = fn0()
        with torch.cuda.stream(aux_stream):
            event0.wait()
            result1 = fn1()
            event1.record()
        event1.synchronize()
        return result0, result1
    else:
        result0 = fn0()
        result1 = fn1()
        return result0, result1
