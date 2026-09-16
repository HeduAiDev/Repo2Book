# SOURCE: vllm/envs.py
# HOST SEAM：本章消费面三件环境变量——VLLM_USE_V2_MODEL_RUNNER
# （config/vllm.py:L579 use_v2_model_runner 的 env 优先位）、
# VLLM_CUSTOM_SCOPES_FOR_PROFILING / VLLM_NVTX_SCOPES_FOR_PROFILING
# （v1/utils.py record_function_or_nullcontext 的裁剪判据）。解析语义逐字
# （maybe_convert_bool 三态：未设=None / "0"/"false"=False / 其余=True）；
# 真实 envs 经 environmental_variables 装配为模块属性，HOST SEAM 以 PEP 562
# 模块 __getattr__ 承载同一读取面（monkeypatch 直接 setattr 模块属性即可
# 覆盖——测试对 env 优先位的做法）。
from __future__ import annotations

import os


def maybe_convert_bool(value: str) -> bool:
    # SOURCE: vllm/envs.py maybe_convert_bool —— 逐字语义
    return value.lower() not in ("0", "false")


def __getattr__(name: str):
    # SOURCE: vllm/envs.py:L279 + L1954-L1956 VLLM_USE_V2_MODEL_RUNNER —— 逐字
    if name == "VLLM_USE_V2_MODEL_RUNNER":
        raw = os.getenv("VLLM_USE_V2_MODEL_RUNNER", None)
        if raw is None:
            return None
        return maybe_convert_bool(raw)
    # SOURCE: vllm/envs.py VLLM_CUSTOM_SCOPES_FOR_PROFILING —— 逐字
    if name == "VLLM_CUSTOM_SCOPES_FOR_PROFILING":
        return maybe_convert_bool(
            os.getenv("VLLM_CUSTOM_SCOPES_FOR_PROFILING", "0")
        )
    # SOURCE: vllm/envs.py VLLM_NVTX_SCOPES_FOR_PROFILING —— 逐字
    if name == "VLLM_NVTX_SCOPES_FOR_PROFILING":
        return maybe_convert_bool(os.getenv("VLLM_NVTX_SCOPES_FOR_PROFILING", "0"))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
