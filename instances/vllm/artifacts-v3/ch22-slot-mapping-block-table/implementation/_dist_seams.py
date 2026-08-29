# HOST SEAM（叶模块）：分布式组探测的最小承载——block_table.py 的
# try/except 探测面（避免 _host_seams ↔ block_table 循环导入而单置）。
# SOURCE: vllm/distributed/parallel_state.py get_pcp_group/get_dcp_group 的
#   host 替身：host 无分布式初始化——与真实「测试环境未初始化组」同型抛
#   AssertionError，由 BlockTable/FlashAttentionImpl 的 try/except 捕获退化
#   为单卡 world_size=1/rank=0（block_table.py:L121-L134 的原设计路径）。
from __future__ import annotations


# SOURCE: vllm/distributed/parallel_state.py get_pcp_group —— HOST SEAM 未初始化分支
def get_pcp_group():
    raise AssertionError("PCP group is not initialized (HOST SEAM)")


# SOURCE: vllm/distributed/parallel_state.py get_dcp_group —— HOST SEAM 未初始化分支
def get_dcp_group():
    raise AssertionError("DCP group is not initialized (HOST SEAM)")
