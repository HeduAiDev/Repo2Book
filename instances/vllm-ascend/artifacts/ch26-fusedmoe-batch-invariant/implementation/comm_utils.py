"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/ops/fused_moe/comm_utils.py

回收 f3（埋于 ch06）的底层落点：ch06 NPUCommunicator.all_to_all 只讲形状代数，
真正『按专家把 token 跨 EP 卡重分发』在这里——async_all_to_all 用不等长 split 的
dist.all_to_all_single（all2all-v）给每张卡发不同条数的 token。
"""
import torch
import torch.distributed as dist

# SUBTRACTED: `import torch_npu`（仅多流分支 COMM_STREAM 用）；host 无 NPU，本精简版不走多流分支。
#             原 vllm_ascend/ops/fused_moe/comm_utils.py:L18-L23

COMM_STREAM = None


# SOURCE: vllm_ascend/ops/fused_moe/comm_utils.py:L26
def async_all_to_all(input_, output_split_sizes, input_split_sizes, group, event=None):
    if output_split_sizes is None:
        # Equal split (all2all)
        a2a_out = torch.empty_like(input_)
    else:
        # Unequal split (all2all-v)
        a2a_out = input_.new_empty(
            size=[sum(output_split_sizes)] + list(input_.size()[1:]),
            dtype=input_.dtype,
            # SUBTRACTED: device=torch.npu.current_device() → 用 input_.device，host 无 npu 设备。
            #             不改变形状代数，仅落盘设备不同。原 comm_utils.py:L35
            device=input_.device,
        )

    # SUBTRACTED: event!=None 时在独立 COMM_STREAM(torch_npu.npu.Stream) 上 event.wait() 再发起的
    #             多流重叠分支（原 comm_utils.py:L38-L52）——纯性能优化、不改数据流；
    #             删后仍是同一个 dist.all_to_all_single(async_op=True) + handle.wait() 的等价顺序流。
    handle = dist.all_to_all_single(
        a2a_out,
        input_.contiguous(),
        output_split_sizes=output_split_sizes,
        input_split_sizes=input_split_sizes,
        group=group,
        async_op=True,
    )
    return input_, a2a_out, handle


# SUBTRACTED: _gather_along_first_dim / gather_from_sequence_parallel_region（原 comm_utils.py:L65-L104）
#             供 All2AllV._preprocess 统计各专家全局 token 数用；本精简版 _preprocess 已整体省略
#             （见 token_dispatcher.py），故其依赖的 all-gather 封装一并删除，不影响 async_all_to_all 主线。
