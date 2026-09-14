# SOURCE: vllm/distributed/__init__.py
# HOST SEAM：单进程退化承载（ch25 同款骨架）——本章消费面：
#   get_tensor_model_parallel_world_size → 1（wq_b「复制不切 TP」的退化面）
#   get_pcp_group → world_size=1 的恒等组（builder compress_ratio>1 的 PCP
#     all-gather 支在单进程下不触发；all_gather 为恒等）
#   get_dcp_group → rank_in_group=0（SparseAttnIndexer.__init__ 的 DCP 标量位；
#     dcp_world_size>1 恒假 → 短路不触达）
from __future__ import annotations


# SOURCE: vllm/distributed/parallel_state get_tensor_model_parallel_world_size
#   —— HOST SEAM：单进程=1
def get_tensor_model_parallel_world_size() -> int:
    # SOURCE: vllm/distributed/parallel_state.py —— HOST SEAM
    return 1


# SOURCE: vllm/distributed/parallel_state.GroupCoordinator —— HOST SEAM：
#   单成员组（rank=0 / world_size=1 / all_gather 恒等）
class _SingleGroup:
    rank_in_group = 0
    world_size = 1

    # SOURCE: vllm/distributed/parallel_state all_gather —— HOST SEAM：恒等
    def all_gather(self, tensor, dim=-1):
        # SOURCE: vllm/distributed/parallel_state.py —— HOST SEAM
        return tensor


# SOURCE: vllm/distributed/parallel_state get_pcp_group —— HOST SEAM
def get_pcp_group():
    # SOURCE: vllm/distributed/parallel_state.py —— HOST SEAM
    return _SingleGroup()


# SOURCE: vllm/distributed/parallel_state get_dcp_group —— HOST SEAM
def get_dcp_group():
    # SOURCE: vllm/distributed/parallel_state.py —— HOST SEAM
    return _SingleGroup()
