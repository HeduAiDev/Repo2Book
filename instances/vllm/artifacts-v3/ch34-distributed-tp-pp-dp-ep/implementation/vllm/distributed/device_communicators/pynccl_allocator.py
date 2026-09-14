# SOURCE: vllm/distributed/device_communicators/pynccl_allocator.py
# HOST SEAM：symmetric-memory 分配器谓词面。真实版管理 NCCL symmetric memory
# 池/上下文；宿主无 NCCL——两个谓词按『特性关闭』语义返回，all_reduce 派发链
# 的 symm-mem 分支因此恒不可达（真实部署默认亦多为关）。

from __future__ import annotations


# SOURCE: vllm/distributed/device_communicators/pynccl_allocator.py:L48-L50 is_symmetric_memory_enabled
def is_symmetric_memory_enabled() -> bool:  # HOST SEAM
    return False


# SUBTRACTED: nccl_symm_mem_context / is_symmetric_memory_tensor / 分配器本体
#   ——symm-mem 传输域（cuda_communicator 的 _get_symm_scratch 家族按删除项 2
#   注为省略）。
