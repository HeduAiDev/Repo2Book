# SOURCE: vllm/envs.py
# HOST SEAM：本章消费面只有两个结构化输出环境变量（backend_xgrammar.py:L69
# 的 VLLM_XGRAMMAR_CACHE_MB 编译缓存字节预算 / utils.py:L64 的
# VLLM_REGEX_COMPILATION_TIMEOUT_S ReDoS 超时）。机制=字典+__getattr__（真实
# vllm/envs.py:L2059 起同构），但去掉 functools.cache（真实 L2092）——
# 测试需要切换超时值，取值语义逐字一致（lambda 默认值原样）。
# SUBTRACTED: SPDX 版权头；其余数百个环境变量项。
import os

_environment_variables: dict = {
    # SOURCE: vllm/envs.py:L1558-L1561 —— 逐字（docstring 语义：512MB ≈ 1000
    # 个 JSON schema）
    # Control the cache sized used by the xgrammar compiler. The default
    # of 512 MB should be enough for roughly 1000 JSON schemas.
    # It can be changed with this variable if needed for some reason.
    "VLLM_XGRAMMAR_CACHE_MB": lambda: int(os.getenv("VLLM_XGRAMMAR_CACHE_MB", "512")),
    # SOURCE: vllm/envs.py:L1562-L1568 —— 逐字
    # Maximum time in seconds allowed for regex compilation in structured
    # output backends (xgrammar, outlines). Prevents ReDoS attacks where
    # adversarial patterns cause exponential DFA state-space explosion.
    # Set to 0 to disable the timeout (not recommended in production).
    "VLLM_REGEX_COMPILATION_TIMEOUT_S": lambda: int(
        os.getenv("VLLM_REGEX_COMPILATION_TIMEOUT_S", "5")
    ),
}


# SOURCE: vllm/envs.py:L2059 __getattr__ —— HOST SEAM 同构（逐项取值）
def __getattr__(name: str):
    if name in _environment_variables:
        return _environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
