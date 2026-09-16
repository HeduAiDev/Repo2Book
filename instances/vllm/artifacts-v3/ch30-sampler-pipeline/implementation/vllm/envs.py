# SOURCE: vllm/envs.py
# HOST SEAM：envs 面的最小承载——本章消费面只剩 VLLM_USE_FLASHINFER_SAMPLER
# （topk_topp_sampler.py:L45/L64，FlashInfer 采样开关，真实默认 True）。
# 真实机制 = environment_variables 字典 + 模块级 __getattr__（L2059 起），
# 值经 functools.cache 缓存（L2092）；HOST SEAM 简化为一阶 lambda 直读——
# 数值语义一致（默认 True、显式 0/1 取 bool），仅去掉缓存以便测试切换环境。
from __future__ import annotations

import os

# SOURCE: vllm/envs.py:L49 VLLM_USE_FLASHINFER_SAMPLER: bool = True
#   （默认值）+ L848-L851 的 lambda 取值式
environment_variables = {
    "VLLM_USE_FLASHINFER_SAMPLER": lambda: (
        bool(int(os.environ["VLLM_USE_FLASHINFER_SAMPLER"]))
        if "VLLM_USE_FLASHINFER_SAMPLER" in os.environ
        else True
    ),
}


# SOURCE: vllm/envs.py:L2059 __getattr__ —— HOST SEAM（无 functools.cache）
def __getattr__(name: str):
    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# SOURCE: vllm/envs.py:L2115-L2119 is_set —— 逐字
def is_set(name: str):
    """Check if an environment variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
