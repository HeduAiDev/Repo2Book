# Subtract-only companion for v3 ch17 — vllm/platforms/cuda.py (pin v0.27.1
# / 6e448d0ea)。本章只取 check_and_update_config 的 worker_cls='auto' 解析点
# （L307-L313）：硬件适配轴的分发不在执行器而在平台插件——CUDA 平台给出
# gpu_worker.Worker，ROCm/XPU/CPU 平台各给各的。
#
# SUBTRACTED: vllm/platforms/cuda.py 的其余全部（设备能力探测/attention 后端
#   选择/pinned memory/WSL2 探测等 L208-L306 与 L313 之后，含 Nvml/
#   NonNvml 二分与 `CudaPlatform = NvmlCudaPlatform if nvml_available else …`
#   的 L1014 别名——平台插件全量面不在本章范围；_host_seams.current_platform
#   以最小接口面代行消费侧）。

from __future__ import annotations


# SOURCE: vllm/platforms/cuda.py:L208 CudaPlatformBase — CUDA 平台基类（骨架）
class CudaPlatformBase:
    # SUBTRACTED: Platform 设备能力/后端/记忆面（cuda.py:L209-L305）——本档只
    #   保 check_and_update_config 的 worker_cls 解析段。

    @classmethod
    # SOURCE: vllm/platforms/cuda.py:L306-L313 worker_cls='auto' 解析点
    def check_and_update_config(cls, vllm_config) -> None:
        parallel_config = vllm_config.parallel_config
        model_config = vllm_config.model_config

        if parallel_config.worker_cls == "auto":
            parallel_config.worker_cls = "vllm.v1.worker.gpu_worker.Worker"

        # SUBTRACTED: mm 前缀语言模型/多模态调度器校验等其余段
        #   （cuda.py:L314 之后——平台校验面，本章不展开）。
