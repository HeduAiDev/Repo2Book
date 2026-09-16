# SOURCE: vllm/distributed/device_communicators/quick_all_reduce.py
# HOST SEAM：QuickAllReduce —— ROCm MI3* 的 quickreduce 补充 AR（派发链第二级）。
# 宿主非 ROCm；seam 只承载构造/判定面，never enabled。

from __future__ import annotations

import torch


# SOURCE: vllm/distributed/device_communicators/quick_all_reduce.py:L44-L367 QuickAllReduce
class QuickAllReduce:  # HOST SEAM
    # SOURCE: vllm/distributed/device_communicators/quick_all_reduce.py:L64-L166 __init__
    def __init__(self, group, device):
        self.group = group
        self.device = device
        self.disabled = True  # HOST SEAM：非 ROCm 宿主

    # SOURCE: vllm/distributed/device_communicators/quick_all_reduce.py
    #   should_quick_all_reduce（seam：恒 False）
    def should_quick_allreduce(self, input_: torch.Tensor) -> bool:
        # SOURCE: vllm/distributed/device_communicators/quick_all_reduce.py:L314-L337（锚点双置）
        return False

    # SOURCE: vllm/distributed/device_communicators/quick_all_reduce.py
    #   quick_all_reduce（seam：不可达）
    def quick_all_reduce(self, input_: torch.Tensor) -> torch.Tensor | None:
        # SOURCE: vllm/distributed/device_communicators/quick_all_reduce.py:L339-L348（锚点双置）
        return None
