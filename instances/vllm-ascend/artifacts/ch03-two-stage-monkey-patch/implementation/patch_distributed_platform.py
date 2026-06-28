# 技法④ 库函数 wrapper + 技法⑤ from-import 缓存陷阱修复
#   —— vllm_ascend/patch/platform/patch_distributed.py（subtract-only）
#
# wrapper：闭包捕获原 fn，在前后补 310p 张量对齐逻辑，不满足条件时回落 fn(...)。
# 缓存陷阱：同一函数在 torch.distributed.broadcast 与 torch.distributed.distributed_c10d.broadcast
#   两个名字下都被重绑——因为调用方可能 `from ...distributed_c10d import broadcast` 缓存了引用，
#   只改顶层名字会漏网。
#
# SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L20-L22
import torch

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type


class NullHandle:
    # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L25-L30
    def __init__(self):
        # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L26-L27
        pass

    def wait(self):
        # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L29-L30
        pass


def communication_adaptation_310p():
    # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L33-L86
    def broadcast310p_wrapper(fn):
        # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L34-L51
        def broadcast310p(tensor, src=0, group=None, async_op=False, group_src=None):
            # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L35-L49
            root = group_src if group_src is not None else src

            if tensor.device == torch.device("cpu"):
                return fn(tensor, src=root, group=group, async_op=async_op)  # 回落原 fn
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

    # 技法⑤：顶层名字 + distributed_c10d 子模块别名「同时」重绑，堵住 from-import 缓存漏网。
    torch.distributed.broadcast = broadcast310p_wrapper(torch.distributed.broadcast)
    torch.distributed.distributed_c10d.broadcast = broadcast310p_wrapper(torch.distributed.distributed_c10d.broadcast)

    def all_reduce_wrapper_310p(fn):
        # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L56-L83
        def all_reduce(tensor, op=torch.distributed.ReduceOp.SUM, group=None, async_op=False):
            # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L57-L81
            if tensor.dtype != torch.int64:
                return fn(tensor, op, group, async_op)  # 仅对 int64 走自定义路径，否则透传原 fn
            # SUBTRACTED: int64 路径的 all_gather + SUM/MAX 归约实现 (patch_distributed.py:L64-L79)。
            ...

        return all_reduce

    torch.distributed.all_reduce = all_reduce_wrapper_310p(torch.distributed.all_reduce)
    torch.distributed.distributed_c10d.all_reduce = all_reduce_wrapper_310p(
        torch.distributed.distributed_c10d.all_reduce
    )


# 运行期条件加载：整个 patch 被设备判定包住——非 310P 设备 import 本模块后什么都不做。
if get_ascend_device_type() == AscendDeviceType._310P:
    communication_adaptation_310p()
