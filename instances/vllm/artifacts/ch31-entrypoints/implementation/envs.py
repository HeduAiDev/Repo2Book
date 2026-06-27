"""本章用到的环境标志（站位真实 vllm/envs.py）。

【关键澄清的源码锚点】VLLM_ENABLE_V1_MULTIPROCESSING 默认 True —— 正是它让 LLMEngine.from_engine_args
把 enable_multiprocessing 强翻为 True，于是离线默认走 SyncMPClient（后台进程 + ZMQ），
而非 InprocClient（进程内）。把环境变量设 '0' 才回退 InprocClient（测试/调试/V0 风格）。
"""
import os

# SOURCE: vllm/envs.py:L129,L1109-L1110 (VLLM_ENABLE_V1_MULTIPROCESSING)
# SUBTRACTED: 真实在 environment_variables 字典里以 lambda 惰性读 os.environ 并 bool 化（默认 "1"）。
#   本章用模块级常量等价呈现：默认 True，可经环境变量覆盖。原 vllm/envs.py:L1112-L1113。
VLLM_ENABLE_V1_MULTIPROCESSING: bool = (
    os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING", "1") == "1"
)
