# SOURCE: vllm/envs.py
# 本章消费面：VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS——CUDA graph 内存
# 估计入账的开关（m3；默认 True，envs.py:L295 类声明 + L2015-L2020 的
# getenv lambda 读法合一）。
# SUBTRACTED: 其余 env 项（VLLM_PREFIX_CACHE_RETENTION_INTERVAL 稀疏驻留
#   → ch15 等）——本章链路不用。
import os


# SOURCE: vllm/envs.py:L295 envs（类属性面；读法 = L2018-L2020 lambda）
class envs:
    """环境变量账位（本章只消费 CUDA graph 估计开关）。"""

    # SOURCE: vllm/envs.py:L295 + L2018-L2020（getenv 默认 "1" → True）
    # If set to 1, enable CUDA graph memory estimation during memory profiling.
    # This profiles CUDA graph memory usage to provide more accurate KV cache
    # memory allocation. Enabled by default as of v0.21.0
    VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS: bool = bool(
        int(os.getenv("VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS", "1"))
    )
