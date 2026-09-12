# SOURCE: vllm/platforms/__init__.py
# HOST SEAM：本章消费面一件——current_platform.is_pin_memory_available()
# （torch_utils PIN_MEMORY 的派生源）与 is_rocm()（default_v2_model_runner_
# architectures 的 ROCm 排除表判据）。真实经平台注册表多态（CUDA/ROCm/NPU…，
# 归 ch17 平台域）；HOST 以 torch 可用性承载 CUDA 平台的两个查询面。
from __future__ import annotations

import torch


class _HostCudaPlatform:
    """CUDA 平台两个查询面的 HOST 承载。"""

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        # SOURCE: vllm/platforms/interface.py is_pin_memory_available 契约面
        #   —— HOST SEAM：无 CUDA → False（真实 CPU 平台同值；CUDA 平台
        #   还有 WSL 内核版本校验，host 测试不受其影响）
        return torch.cuda.is_available()

    @classmethod
    def is_rocm(cls) -> bool:
        # SOURCE: vllm/platforms/interface.py is_rocm 契约面 —— HOST SEAM
        return False


current_platform = _HostCudaPlatform()
