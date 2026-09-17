# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""平台抽象（本章消费面：is_cuda_alike / is_xpu / Stream 三件套）。

# SOURCE: vllm/platforms/__init__.py:L1-L260 + vllm/platforms/interface.py Platform 谓词族
# SUBTRACTED: 平台插件发现与 CUDA/ROCm/XPU/TPU 子类——本章 host 无加速器。
#
# SEAM（本章 host 替身，与 ch13/ch37 的 HOST SEAM 同款）：
#   真源里 current_platform 经 Platform.__getattr__ 把 Stream / current_stream /
#   stream 委托给 torch.cuda（interface.py:L1131-L1152：device = getattr(torch,
#   self.device_type) 后取同名属性）。host 无 CUDA，这里给出语义等价的替身：
#     | 真 CUDA 平台                                | 本替身                       |
#     |---------------------------------------------|------------------------------|
#     | Stream() 新建独立异步流                     | HostStream：无序对象占位     |
#     | stream.wait_stream(计算流) 排队等待         | no-op（host 搬运本就同步）   |
#     | with current_platform.stream(s) 切流        | no-op 上下文                 |
#     | torch.Event 在真流上 record/query/synchronize| host 上即记即完（无设备）   |
#   偏离面：GPU→CPU store 须等计算流的**时序**在 host 上观察不到——DMA 三戒的
#   论证以源码注释为准（gpu_worker.py:L376-L389），控制流与字节流不变。
"""

from contextlib import contextmanager
from typing import Any, Iterator


# SOURCE: vllm/platforms/interface.py Platform 谓词族（L100-L430）
# SUBTRACTED: CUDA/ROCm/XPU/TPU 子类与 device_communicator 解析——本章 host 无加速器。
class _HostPlatform:
    """宿主平台：无加速器，device_type = "cpu"。"""

    device_type: str = "cpu"
    device_name: str = "cpu"
    dispatch_key: str = "CPU"

    # SOURCE: vllm/platforms/interface.py:L210-L230（is_* 谓词族）
    def is_cuda(self) -> bool:
        # SOURCE: vllm/platforms/interface.py:L210-L230
        return False

    def is_cuda_alike(self) -> bool:
        # SOURCE: vllm/platforms/interface.py:L220-L226
        return False

    def is_rocm(self) -> bool:
        # SOURCE: vllm/platforms/interface.py:L210-L230
        return False

    def is_xpu(self) -> bool:
        # SOURCE: vllm/platforms/interface.py:L210-L230
        return False

    def is_tpu(self) -> bool:
        # SOURCE: vllm/platforms/interface.py:L210-L230
        return False

    def is_cpu(self) -> bool:
        # SOURCE: vllm/platforms/interface.py:L210-L230
        return True

    # ── SEAM：Stream 三件套（真源经 __getattr__ 委托 torch.cuda） ──

    def Stream(self) -> "HostStream":
        return HostStream()

    def current_stream(self) -> "HostStream":
        return _DEFAULT_STREAM

    @contextmanager
    def stream(self, stream: "HostStream") -> Iterator[None]:
        yield


# SEAM: 真 torch.cuda.Stream 的 host 占位——排队等待 no-op（见模块 docstring）。
class HostStream:
    """host 占位流：wait_stream/wait_event 不排队（无异步执行体）。"""

    def wait_stream(self, other: Any) -> None:
        return None

    def wait_event(self, event: Any) -> None:
        return None


_DEFAULT_STREAM = HostStream()

# SOURCE: vllm/platforms/__init__.py:L250-L262（current_platform 解析）
current_platform = _HostPlatform()

__all__ = ["current_platform", "HostStream"]
