# vllm_ascend/patch/platform/patch_distributed.py —— subtract-only 精简版（ch17 横切回收：distributed）
#
# 横切回收点（ch06 点过的伏笔在此收口）：310P 的 HCCL 对 device 上的 broadcast /
# int64 all_reduce 支持受限，于是在模块尾用 is_310p 守卫，把 torch.distributed 的
# broadcast / all_reduce 猴补成 all_gather 模拟——
#   broadcast：all_gather 后取 src 项写回；
#   int64 all_reduce：all_gather 后本地 sum/max。
# CPU 张量 / 非 int64 张量走原生 fn（不模拟）。
#
# 全段是纯 Python（给个假 group/单卡即可），host 可跑（用 gloo 单进程 group 或桩掉
# torch.distributed 的 rank/world_size/all_gather，见 ../tests）。模块尾的 if 守卫在
# host 默认 A2、不会触发——所以 import 本文件不会污染全局 torch.distributed。
import torch

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


# SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L25-L30
class NullHandle:
    def __init__(self):
        # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L26-L27
        pass

    def wait(self):
        # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L29-L30
        pass


# SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L33-L85
def communication_adaptation_310p():
    def broadcast310p_wrapper(fn):
        # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L34
        def broadcast310p(tensor, src=0, group=None, async_op=False, group_src=None):
            # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L35
            root = group_src if group_src is not None else src

            if tensor.device == torch.device("cpu"):
                return fn(tensor, src=root, group=group, async_op=async_op)
            rank = torch.distributed.get_rank(group)
            world_size = torch.distributed.get_world_size(group)
            tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
            tensor_list[rank] = tensor
            torch.distributed.all_gather(tensor_list, tensor, group=group)
            tensor[...] = tensor_list[src]
            if async_op:
                return NullHandle()
            else:
                return None

        return broadcast310p

    torch.distributed.broadcast = broadcast310p_wrapper(torch.distributed.broadcast)
    torch.distributed.distributed_c10d.broadcast = broadcast310p_wrapper(torch.distributed.distributed_c10d.broadcast)

    def all_reduce_wrapper_310p(fn):
        # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L56
        def all_reduce(
            tensor,
            op=torch.distributed.ReduceOp.SUM,
            group=None,
            async_op=False,
        ):
            # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L57-L78
            if tensor.dtype != torch.int64:
                return fn(tensor, op, group, async_op)
            rank = torch.distributed.get_rank(group)
            world_size = torch.distributed.get_world_size(group)
            tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
            tensor_list[rank] = tensor
            torch.distributed.all_gather(tensor_list, tensor, group=group)
            if op == torch.distributed.ReduceOp.SUM:
                return torch.stack(tensor_list).sum(0)
            elif op == torch.distributed.ReduceOp.MAX:
                return torch.tensor(
                    torch.stack(tensor_list).cpu().numpy().max(0),
                    device=tensor.device,
                )
            else:
                raise RuntimeError(f"not implement op {op}")

        return all_reduce

    torch.distributed.all_reduce = all_reduce_wrapper_310p(torch.distributed.all_reduce)
    torch.distributed.distributed_c10d.all_reduce = all_reduce_wrapper_310p(
        torch.distributed.distributed_c10d.all_reduce
    )


# SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L88-L89
if get_ascend_device_type() == AscendDeviceType._310P:
    communication_adaptation_310p()
