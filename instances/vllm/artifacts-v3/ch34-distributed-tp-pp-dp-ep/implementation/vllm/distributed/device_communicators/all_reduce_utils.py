# SOURCE: vllm/distributed/device_communicators/all_reduce_utils.py
# HOST SEAM：NCCL symm-mem AR 判定谓词面。真实版按 symmetric-memory 使能 +
# 张量门槛 + world_size 档位表判定；宿主 is_symmetric_memory_enabled 恒 False
# → 谓词恒 False，派发链 symm-mem 分支不可达（与真实默认部署同形）。
# SUBTRACTED: NCCL_SYMM_MEM_ALL_REDUCE_CONFIG 档位表（L107-L110）与 producer/
#   consumer/can_actually_p2p/gpu_p2p_access_check 探测族（L161-L424）——唯一
#   消费者 _log_all_reduce_backend_selection 已按删除项 2 裁除；P2P 探测归
#   CustomAllreduce 域（CUDA IPC 面）。

from __future__ import annotations

import torch


# SOURCE: vllm/distributed/device_communicators/all_reduce_utils.py
#   should_nccl_symm_mem_allreduce —— 恒 False（symmetric memory 关闭语义）
def should_nccl_symm_mem_allreduce(world_size: int, tensor: torch.Tensor) -> bool:
    # HOST SEAM
    # SOURCE: vllm/distributed/device_communicators/all_reduce_utils.py:L112-L146（锚点双置）
    return False


# SOURCE: vllm/distributed/device_communicators/all_reduce_utils.py
#   should_nccl_symm_mem_ag_rs —— 恒 False（同上）
def should_nccl_symm_mem_ag_rs() -> bool:
    # HOST SEAM
    # SOURCE: vllm/distributed/device_communicators/all_reduce_utils.py:L149-L158（锚点双置）
    return False
