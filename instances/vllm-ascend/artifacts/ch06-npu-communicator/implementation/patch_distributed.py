"""换底座线④：仅 310P 把 broadcast / int64 all_reduce 猴补成 all_gather 模拟。

只做减法的忠实精简版。复用 ch03 讲过的「闭包 wrapper(fn) 捕获原函数 + 模块属性赋值
替换」两段式猴补技法（本章不重讲机制，只讲补什么缺口）：Atlas 300I(310P) 硬件不支持
原生 broadcast 与 int64 all_reduce，用 all_gather（310P 支持）在 host 侧拼出等价语义。
降级直通：broadcast 对 cpu tensor、all_reduce 对非 int64 dtype 直接调原函数 fn，
只拦截 310P 真正缺能力的场景。

host 可跑：communication_adaptation_310p() 里的 wrapper 只用 torch.distributed.all_gather
（gloo 即支持），可用单进程 gloo 组单测 SUM/直通路径（见 tests）。原模块尾的
import 期 310P 守卫已 SUBTRACTED，避免在非昇腾 host 全局猴补 torch.distributed。
"""
# SUBTRACTED: 文件头 Apache-2.0 许可证注释块（原 patch_distributed.py:L1-L18）
import torch

# SUBTRACTED: from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type
#   —— host 无 vllm_ascend；这两个符号仅用于文件尾的「仅 310P」守卫（见末尾 SUBTRACTED 守卫）。
#   原 import: vllm_ascend/patch/platform/patch_distributed.py:L22


# SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L25-L30
class NullHandle:
    # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L25-L30
    def __init__(self):
        # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L26-L27
        pass

    def wait(self):
        # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L29-L30
        pass


# SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L33-L85
def communication_adaptation_310p():
    def broadcast310p_wrapper(fn):
        # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L34-L51
        def broadcast310p(tensor, src=0, group=None, async_op=False, group_src=None):
            # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L35-L49
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
    # SUBTRACTED: torch.distributed.distributed_c10d.broadcast 的第二次镜像赋值（原 L54）——
    #   与上一行同一 wrapper 的重复挂载点；精简版演示「猴补一个入口」即可（plan 批准，真实代码两处都需要）。

    def all_reduce_wrapper_310p(fn):
        # SOURCE: vllm_ascend/patch/platform/patch_distributed.py:L56-L80
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
            # SUBTRACTED: ReduceOp.MAX 的 numpy 往返实现 + else raise（原 L72-L78）——
            #   MAX 与 SUM 是平行分支，删 MAX 不影响「all_gather→stack→归约」范式展示（plan 批准）。

        return all_reduce

    torch.distributed.all_reduce = all_reduce_wrapper_310p(torch.distributed.all_reduce)
    # SUBTRACTED: torch.distributed.distributed_c10d.all_reduce 的第二次镜像赋值（原 L83-L85）——同上（plan 批准）。


# SUBTRACTED: import 期「仅 310P」守卫（原 patch_distributed.py:L88-L89）——
#   原文件结尾是：
#       if get_ascend_device_type() == AscendDeviceType._310P:
#           communication_adaptation_310p()
#   精简版不在 import 时全局猴补 torch.distributed（非昇腾 host 会破坏全局状态），
#   改由测试显式调用 communication_adaptation_310p()。守卫语义：A2/A3/A5 不触发，仅 310P 补。
