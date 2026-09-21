# SOURCE: vllm/platforms/__init__.py
# HOST SEAM：current_platform 的本机承载。真实 platforms/__init__.py 按
# 可见加速器实例化 Platform 子类（CUDA/ROCm/XPU/CPU/TPU…）；本章精简版
# 消费面只有四处——TopKTopPSampler 构造期绑定的 is_cuda() 裁决
# （topk_topp_sampler.py:L93）、flashinfer_sampler_supported 的
# is_cuda()/get_device_capability()（L43/L53）、_load_custom_logitsprocs 的
# is_tpu()（logits_processor/__init__.py:L177）、batched_count_greater_than
# 的 torch.compile 后端（ops/logprobs.py:L10）与 Triton wrapper 的
# num_compute_units（topk_topp_triton.py:L909 经 utils/platform_utils）。
# is_cuda 用 torch.cuda.is_available() 等价探测（真实为加速器检测选类）；
# get_device_capability 真实走 NVML（cuda.py:L734-L742），此处用
# torch.cuda.get_device_capability() 同源取值。
from __future__ import annotations

from typing import Any, NamedTuple

import torch


# SOURCE: vllm/platforms/interface.py:L89-L122 DeviceCapability —— 逐字
#   （flashinfer_sampler_supported 的能力比较与 as_version_str 报错串消费）
class DeviceCapability(NamedTuple):
    major: int
    minor: int

    # SOURCE: vllm/platforms/interface.py:L93-L96 DeviceCapability.__lt__
    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) < (other.major, other.minor)

    # SOURCE: vllm/platforms/interface.py:L98-L101 DeviceCapability.__le__
    def __le__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) <= (other.major, other.minor)

    # SOURCE: vllm/platforms/interface.py:L103-L106 DeviceCapability.__eq__
    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) == (other.major, other.minor)

    # SOURCE: vllm/platforms/interface.py:L108-L111 DeviceCapability.__ge__
    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) >= (other.major, other.minor)

    # SOURCE: vllm/platforms/interface.py:L113-L116 DeviceCapability.__gt__
    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, DeviceCapability):
            return NotImplemented
        return (self.major, self.minor) > (other.major, other.minor)

    # SOURCE: vllm/platforms/interface.py:L118-L119 DeviceCapability.__hash__
    def __hash__(self) -> int:
        return hash((self.major, self.minor))

    # SOURCE: vllm/platforms/interface.py:L121-L122 DeviceCapability.as_version_str
    def as_version_str(self) -> str:
        return f"{self.major}.{self.minor}"


# SOURCE: vllm/platforms/interface.py Platform —— HOST SEAM：消费面方法位
class _CurrentPlatform:
    # SOURCE: vllm/platforms/__init__.py 加速器检测选类 —— HOST SEAM：
    #   torch.cuda.is_available() 等价探测（有可见 CUDA → CUDA 平台语义）
    @classmethod
    def is_cuda(cls) -> bool:
        return torch.cuda.is_available()

    # SOURCE: vllm/platforms/interface.py Platform.is_tpu 基类默认 False
    #   —— HOST SEAM（TPU 域不在本章范围）
    @classmethod
    def is_tpu(cls) -> bool:
        return False

    # SOURCE: vllm/platforms/interface.py:L165 simple_compile_backend = "inductor"
    #   —— HOST SEAM 取 "eager"（ch8 同款：host 无 inductor 工具链，
    #   torch.compile 仍 dynamo-trace、执行同一套 eager 数学，数值不变）
    simple_compile_backend: str = "eager"

    # SOURCE: vllm/platforms/cuda.py:L734-L742 get_device_capability
    #   —— HOST SEAM：NVML 探测的同源退化（torch 直询驱动）
    @classmethod
    def get_device_capability(cls, device_id: int = 0) -> "DeviceCapability | None":
        if not torch.cuda.is_available():
            return None
        major, minor = torch.cuda.get_device_capability(device_id)
        return DeviceCapability(major=major, minor=minor)

    # SOURCE: vllm/platforms/cuda.py:L678-L679 num_compute_units —— 逐字
    #   （CUDA 位；Triton wrapper 的 NUM_PROGRAMS=min(SM 数, batch) 消费）
    @classmethod
    def num_compute_units(cls, device_id: int = 0) -> int:
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(device_id).multi_processor_count
        # SOURCE: vllm/platforms/cpu.py:L423-L424 num_compute_units
        #   —— CPU 位逐字（torch.get_num_threads()）
        return torch.get_num_threads()

    # SOURCE: vllm/platforms/cuda.py:L286 起 is_pin_memory_available
    #   —— HOST SEAM：CUDA 可用即 True / 否则 False（cpu.py:L396-L397
    #   CPUPlatform 返回 False 的两极合并）
    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return torch.cuda.is_available()


# SOURCE: vllm/platforms/__init__.py current_platform 实例位 —— HOST SEAM
current_platform = _CurrentPlatform()
