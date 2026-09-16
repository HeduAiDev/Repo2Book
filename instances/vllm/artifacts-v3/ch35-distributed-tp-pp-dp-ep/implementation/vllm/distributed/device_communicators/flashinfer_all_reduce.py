# SOURCE: vllm/distributed/device_communicators/flashinfer_all_reduce.py
# HOST SEAM：FlashInferAllReduce —— flashinfer trtllm AR（派发链第三级）。
# 宿主无 flashinfer；seam 只承载构造/判定面，never enabled。

from __future__ import annotations

import torch


# SOURCE: vllm/distributed/device_communicators/flashinfer_all_reduce.py:L322-L427
#   FlashInferAllReduce
class FlashInferAllReduce:  # HOST SEAM
    # SOURCE: vllm/distributed/device_communicators/flashinfer_all_reduce.py:L323-L364 __init__
    def __init__(self, group, device):
        self.group = group
        self.device = device
        self.disabled = True  # HOST SEAM：无 flashinfer

    # SOURCE: vllm/distributed/device_communicators/flashinfer_all_reduce.py
    #   should_use_fi_ar（seam：恒 False）
    def should_use_fi_ar(self, input_: torch.Tensor) -> bool:
        # SOURCE: vllm/distributed/device_communicators/flashinfer_all_reduce.py:L384-L405（锚点双置）
        return False

    # SOURCE: vllm/distributed/device_communicators/flashinfer_all_reduce.py
    #   all_reduce（seam：不可达）
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor | None:
        # SOURCE: vllm/distributed/device_communicators/flashinfer_all_reduce.py:L407-L423（锚点双置）
        return None

    # SOURCE: vllm/distributed/device_communicators/flashinfer_all_reduce.py:L425-L427 destroy
    def destroy(self):  # HOST SEAM
        return None
