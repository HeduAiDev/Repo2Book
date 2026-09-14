# SOURCE: vllm/distributed/device_communicators/symm_mem.py
# HOST SEAM：SymmMemCommunicator —— torch symmetric-memory AR（派发链第六级）。
# 宿主无 symmetric memory；seam 只承载构造/判定面，never enabled。

from __future__ import annotations

import torch


# SOURCE: vllm/distributed/device_communicators/symm_mem.py:L25-L156 SymmMemCommunicator
class SymmMemCommunicator:  # HOST SEAM
    # SOURCE: vllm/distributed/device_communicators/symm_mem.py:L33-L115 __init__
    def __init__(self, group, device):
        self.group = group
        self.device = device
        self.disabled = True  # HOST SEAM：无 symmetric memory

    # SOURCE: vllm/distributed/device_communicators/symm_mem.py:L117-L125 should_use_symm_mem
    def should_use_symm_mem(self, input_: torch.Tensor) -> bool:
        return False

    # SOURCE: vllm/distributed/device_communicators/symm_mem.py:L127-L156 all_reduce
    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor | None:
        return None
