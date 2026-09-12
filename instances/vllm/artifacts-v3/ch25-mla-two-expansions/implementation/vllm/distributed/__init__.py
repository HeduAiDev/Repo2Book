# SOURCE: vllm/distributed/parallel_state.py + vllm/distributed/utils.py
# HOST SEAM：分布式面的单进程退化承载（TP=1/DCP=1——delete[0] 删掉全部
# DCP/PCP 分支后本章只剩三个消费位）：
#   get_tensor_model_parallel_world_size → 1（deepseek_v2.py 切头算术）
#   get_dcp_group → 与真实「测试环境未初始化组」同型抛 AssertionError——
#     MLACommonMetadataBuilder.__init__ 的 try/except 捕获后退化 1（源码
#     原生路径，mla_attention.py:L1911-L1915 注释原话 "DCP might not be
#     initialized in testing"）
#   is_global_first_rank → True（ROCm 预编译循环的进度条位已删，保名面）
from __future__ import annotations


# SOURCE: vllm/distributed/parallel_state.py get_tensor_model_parallel_world_size
#   —— HOST SEAM：TP=1 退化
def get_tensor_model_parallel_world_size() -> int:
    # SOURCE: vllm/distributed/parallel_state.py —— HOST SEAM
    return 1


# SOURCE: vllm/distributed/parallel_state.py get_dcp_group —— HOST SEAM：
#   未初始化组同型抛错（builder 捕获后 dcp_world_size=1）
def get_dcp_group():
    # SOURCE: vllm/distributed/parallel_state.py get_dcp_group —— HOST SEAM
    raise AssertionError("Default process group has not been initialized.")


# SOURCE: vllm/distributed/parallel_state.py get_tp_group —— HOST SEAM 位
#   （delete[0] 已删全部调用点；保名面防外部 import）
def get_tp_group():
    # SOURCE: vllm/distributed/parallel_state.py get_tp_group —— HOST SEAM
    raise AssertionError("Default process group has not been initialized.")


# SOURCE: vllm/distributed/parallel_state.py is_global_first_rank —— HOST SEAM
def is_global_first_rank() -> bool:
    # SOURCE: vllm/distributed/parallel_state.py is_global_first_rank —— HOST SEAM
    return True
