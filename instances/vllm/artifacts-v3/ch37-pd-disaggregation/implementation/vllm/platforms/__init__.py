# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""平台抽象（本章消费面：device_type / NIXL 内存类型 / 设备设置）。

# SOURCE: vllm/platforms/__init__.py:L1-L200（current_platform 解析与 Platform 谓词族）
# SUBTRACTED: 平台插件发现（`_init_platform_plugins` / 各平台子类）——本章只需
#   「宿主是 CPU」这一个事实；真实部署里 current_platform 决定 kv_buffer_device
#   默认值、NIXL 内存类型（VRAM/DRAM）与 NVLink 可用的 host-buffer 旁路。
"""

from typing import Any


# SOURCE: vllm/platforms/interface.py Platform 谓词族（L100-L400）
# SUBTRACTED: CUDA/ROCm/XPU/TPU 子类与 device_communicator 解析——本章 host 无加速器。
class _HostPlatform:
    """宿主平台：无加速器，device_type = "cpu"。"""

    device_type: str = "cpu"
    device_name: str = "cpu"
    dispatch_key: str = "CPU"

    # SOURCE: vllm/platforms/interface.py:L400-L430（is_* 谓词族）
    def is_cuda(self) -> bool:
        return False

    def is_rocm(self) -> bool:
        return False

    def is_xpu(self) -> bool:
        return False

    def is_tpu(self) -> bool:
        return False

    def is_cpu(self) -> bool:
        return True

    # SOURCE: vllm/platforms/interface.py:L540-L570（get_nixl_supported_devices）
    def get_nixl_supported_devices(self) -> dict:
        return {"cpu": ("cpu",)}

    # SOURCE: vllm/platforms/interface.py:L520-L540（get_nixl_memory_type）
    def get_nixl_memory_type(self) -> str | None:
        return None

    # SOURCE: vllm/platforms/interface.py:L480-L500（set_device）
    def set_device(self, device: Any) -> None:
        return None

    # SOURCE: vllm/platforms/interface.py:L700-L730（discover_numa_topology）
    def discover_numa_topology(self) -> list[list[int]]:
        return []


current_platform = _HostPlatform()

__all__ = ["current_platform"]
