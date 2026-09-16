# SOURCE: vllm/v1/worker/gpu/async_utils.py
# HOST SEAM：本章消费面一件——async_copy_to_np（DraftTokensHandler 在
# copy_stream 上的 GPU→CPU 异步拷贝出口，spec_decode/utils.py:L38）。
# 逐字（该文件其余的 stream 上下文管理器等归 ch12 的 D2H 域）。
from __future__ import annotations

import numpy as np
import torch


# SOURCE: vllm/v1/worker/gpu/async_utils.py:L137-L138 async_copy_to_np —— 逐字
def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    return x.to("cpu", non_blocking=True).numpy()
