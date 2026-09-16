# SOURCE: vllm/distributed/device_communicators/pynccl.py
# HOST SEAM：PyNcclCommunicator —— 真实类经 pynccl_wrapper 直调 NCCL C API
# (pynccl.py:L78+)。宿主无 NCCL 库；seam 用 torch.distributed（组内 gloo）实现
# 同一组集合语义（all_reduce/all_gather/all_gatherv/reduce_scatter/reduce_scatterv
# 逐 op 对标真实实现的通信图：all_gatherv=按 root 逐段 broadcast、reduce_scatterv
# =按 root 逐段 reduce）。契约偏离仅一处并在此声明：异步 stream 语义退化为同步
# 完成（gloo 无 stream；调用方随后必然同步，观察结果等价）。

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed import ReduceOp


# SOURCE: vllm/distributed/device_communicators/pynccl.py:L31-L57 register_nccl_symmetric_ops
def register_nccl_symmetric_ops(pynccl_comm) -> None:  # HOST SEAM
    """真实版：把 NCCL symmetric-memory 算子注册进 torch 命名空间。seam：no-op
    （is_symmetric_memory_enabled 恒 False，注册面不可达）。"""
    return None


# SOURCE: vllm/distributed/device_communicators/pynccl.py:L60-L434 PyNcclCommunicator
class PyNcclCommunicator:  # HOST SEAM
    # SOURCE: vllm/distributed/device_communicators/pynccl.py:L61-L146 __init__
    #   （真实版：unique_id 广播 + ncclCommInitRank + warmup all_reduce；seam：
    #   记组与世界，恒可用、非禁用——集合语义由 gloo 承载）
    def __init__(self, group, device=None):
        # SOURCE: vllm/distributed/device_communicators/pynccl.py:L61-L146（锚点双置）
        self.group = group
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)
        self.ranks = dist.get_process_group_ranks(group)
        self.device = torch.device("cpu") if device is None else torch.device(device)
        self.disabled = False  # 真实语义位保留（with_parallel_state 切换）
        self.available = True

    # SOURCE: vllm/distributed/device_communicators/pynccl.py:L166-L197 all_reduce
    #   —— out-of-place AR（ncclAllReduce → seam: dist.all_reduce）
    def all_reduce(
        self,
        in_tensor: torch.Tensor,
        out_tensor: torch.Tensor = None,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ) -> torch.Tensor:
        # SOURCE: vllm/distributed/device_communicators/pynccl.py:L166-L197（锚点双置）
        if self.disabled:
            return None
        if out_tensor is None:
            out_tensor = torch.empty_like(in_tensor)
        out_tensor.copy_(in_tensor)
        dist.all_reduce(out_tensor, op=op, group=self.group)
        return out_tensor

    # SOURCE: vllm/distributed/device_communicators/pynccl.py:L199-L220 all_gather
    def all_gather(self, output_tensor: torch.Tensor, input_tensor: torch.Tensor,
                   stream=None):
        if self.disabled:
            return
        dist.all_gather_into_tensor(output_tensor, input_tensor, group=self.group)

    # SOURCE: vllm/distributed/device_communicators/pynccl.py:L222-L255 all_gatherv
    #   —— 逐 root ncclBroadcast 写 output 切片（seam: dist.broadcast，同通信图）
    def all_gatherv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        stream=None,
    ):
        # SOURCE: vllm/distributed/device_communicators/pynccl.py:L222-L255（锚点双置）
        if self.disabled:
            return
        assert output_tensor.shape[0] == sum(sizes)
        split_offset = 0
        for root, split_size in enumerate(sizes):
            dst_slice = output_tensor[split_offset : split_offset + split_size]
            buf = input_tensor if root == self.rank else dst_slice
            dist.broadcast(buf, src=self.ranks[root], group=self.group)
            split_offset += split_size

    # SOURCE: vllm/distributed/device_communicators/pynccl.py:L257-L283 reduce_scatter
    def reduce_scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        if self.disabled:
            return
        dist.reduce_scatter_tensor(output_tensor, input_tensor, group=self.group)

    # SOURCE: vllm/distributed/device_communicators/pynccl.py:L285-L320 reduce_scatterv
    #   —— 逐 root ncclReduce（seam: dist.reduce，同通信图）
    def reduce_scatterv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        # SOURCE: vllm/distributed/device_communicators/pynccl.py:L285-L320（锚点双置）
        if self.disabled:
            return
        split_offset = 0
        for root, split_size in enumerate(sizes):
            chunk = input_tensor[split_offset : split_offset + split_size, ...]
            buf = output_tensor if root == self.rank else chunk.clone()
            dist.reduce(buf, dst=self.ranks[root], op=op, group=self.group)
            split_offset += split_size

    # SOURCE: vllm/distributed/device_communicators/pynccl.py:L148-L164 destroy
    #   （真实版：daemon 线程 ncclCommAbort；seam：置禁用态）
    def destroy(self):
        # SOURCE: vllm/distributed/device_communicators/pynccl.py:L148-L164（锚点双置）
        self.available = False
        self.disabled = True
        return None

    # SOURCE: vllm/distributed/device_communicators/pynccl.py:L376-L400 broadcast
    #   （真实版：ncclBroadcast；seam: dist.broadcast）
    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        # SOURCE: vllm/distributed/device_communicators/pynccl.py:L376-L400（锚点双置）
        if self.disabled:
            return
        dist.broadcast(tensor, src=self.ranks[src], group=self.group)

    # SOURCE: vllm/distributed/device_communicators/pynccl.py:L402-L406 group_start/group_end
    def group_start(self):
        return None

    def group_end(self):
        # SOURCE: vllm/distributed/device_communicators/pynccl.py:L405-L406（锚点双置）
        return None

    # SUBTRACTED: send/recv/batch_isend_irecv 等 NCCL 专有面——本章消费面之外
    #   （P2P 走 torch.distributed.isend/irecv）。
