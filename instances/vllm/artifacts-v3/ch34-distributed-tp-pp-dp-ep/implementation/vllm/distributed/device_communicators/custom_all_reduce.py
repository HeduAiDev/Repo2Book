# SOURCE: vllm/distributed/device_communicators/custom_all_reduce.py
# HOST SEAM：CustomAllreduce —— vLLM 自研低延迟小张量 AR（CUDA kernel + IPC
# buffer 注册）。宿主无 CUDA；seam 只承载 CudaCommunicator.__init__ 的构造面与
# 派发链要触的判定面（disabled/should_custom_ar/custom_all_reduce），never
# enabled——真实部署由 _ENABLE_CUSTOM_ALL_REDUCE 与平台决定。

from __future__ import annotations

import torch


# SOURCE: vllm/distributed/device_communicators/custom_all_reduce.py:L56-L550 CustomAllreduce
class CustomAllreduce:  # HOST SEAM
    # SOURCE: vllm/distributed/device_communicators/custom_all_reduce.py:L70-L259 __init__
    #   （真实版：IPC handle 全员交换 + buffer 注册；seam：不可用态）
    def __init__(self, group, device, symm_mem_enabled: bool = False):
        # SOURCE: vllm/distributed/device_communicators/custom_all_reduce.py:L70-L259（锚点双置）
        self.group = group
        self.device = device
        self.disabled = True  # HOST SEAM：宿主无 CUDA 上下文，恒走禁用语义

    # SOURCE: vllm/distributed/device_communicators/custom_all_reduce.py:L348-L361
    #   should_custom_ar —— 门槛判定（seam：禁用态恒 False）
    def should_custom_ar(self, input_: torch.Tensor) -> bool:
        # SOURCE: vllm/distributed/device_communicators/custom_all_reduce.py:L348-L361（锚点双置）
        return False

    # SOURCE: vllm/distributed/device_communicators/custom_all_reduce.py:L382-L398
    #   custom_all_reduce（seam：不可达）
    def custom_all_reduce(self, input_: torch.Tensor) -> torch.Tensor | None:
        # SOURCE: vllm/distributed/device_communicators/custom_all_reduce.py:L382-L398（锚点双置）
        return None

    # SUBTRACTED: capture/registered graph 面与 IPC 细节——CUDA kernel 域。
