# SOURCE: vllm/distributed/device_communicators/aiter_custom_all_reduce.py
# HOST SEAM：AiterCustomAllreduce —— ROCm aiter AR（派发链第四级）。宿主非 ROCm；
# seam 只承载构造/判定面，never enabled。

from __future__ import annotations

import torch


# SOURCE: vllm/distributed/device_communicators/aiter_custom_all_reduce.py:L19-L95
#   AiterCustomAllreduce
class AiterCustomAllreduce:  # HOST SEAM
    # SOURCE: vllm/distributed/device_communicators/aiter_custom_all_reduce.py:L30-L43 __init__
    def __init__(self, group, device):
        self.group = group
        self.device = device
        self.disabled = True  # HOST SEAM：非 ROCm 宿主

    # SOURCE: vllm/distributed/device_communicators/aiter_custom_all_reduce.py
    #   should_custom_ar（seam：恒 False）
    def should_custom_ar(self, input_: torch.Tensor) -> bool:
        # SOURCE: vllm/distributed/device_communicators/aiter_custom_all_reduce.py:L53-L54（锚点双置）
        return False

    # SOURCE: vllm/distributed/device_communicators/aiter_custom_all_reduce.py
    #   custom_all_reduce（seam：不可达）
    def custom_all_reduce(self, input_: torch.Tensor) -> torch.Tensor | None:
        # SOURCE: vllm/distributed/device_communicators/aiter_custom_all_reduce.py:L56-L57（锚点双置）
        return None

    # SOURCE: vllm/distributed/device_communicators/aiter_custom_all_reduce.py:L62-L63 close
    def close(self):  # HOST SEAM
        return None
