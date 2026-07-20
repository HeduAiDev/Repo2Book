# SOURCE: vllm/v1/worker/gpu/async_utils.py
# 只做减法的忠实精简版。本章只需要 async_copy_to_np——DraftTokensHandler 用它把
# 草稿 token 从 GPU 异步搬回 CPU，供调度器做语法校验。
#
# SUBTRACTED: SPDX 版权头、stream 上下文管理器（与本章草稿回传控制流无关）。
import numpy as np
import torch


def async_copy_to_np(x: torch.Tensor) -> np.ndarray:
    # SOURCE: vllm/v1/worker/gpu/async_utils.py:L107-108
    return x.to("cpu", non_blocking=True).numpy()
