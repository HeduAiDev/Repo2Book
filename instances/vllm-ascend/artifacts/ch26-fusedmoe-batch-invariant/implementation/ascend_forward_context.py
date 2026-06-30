"""只做减法的精简版 —— 活动实例 vllm-ascend
真实文件：vllm_ascend/ascend_forward_context.py（仅截 MoECommType + select_moe_comm_method）

f10 的起点（埋于 ch15）：每拍前向开始时按 soc/EP/token 数选定一个 MoECommType 枚举。
本章在 moe_comm_method.py 把每个枚举落地成真正的 *CommImpl 实例（f10 回收）。
"""
from enum import Enum

from vllm.config import VllmConfig
from vllm.distributed import get_ep_group

from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type, is_moe_model

# SUBTRACTED: 该文件其余 ~350 行（set_ascend_forward_context / _EXTRA_CTX 容器 / DP 同步 / aclgraph
#             batch descriptor / set_mc2_mask 等）属 ch15 主题；本精简版只截出 MoE 通信方式『选择点』。

# 由 ch15 启动期设置；本章只读它做容量判断。
_mc2_tokens_capacity = 512


# SOURCE: vllm_ascend/ascend_forward_context.py:L26
class MoECommType(Enum):
    ALLGATHER = 0
    MC2 = 1
    ALLTOALL = 2
    FUSED_MC2 = 3


# SOURCE: vllm_ascend/ascend_forward_context.py:L213
def get_mc2_tokens_capacity():
    return _mc2_tokens_capacity


# SOURCE: vllm_ascend/ascend_forward_context.py:L233
def select_moe_comm_method(num_tokens: int, vllm_config: VllmConfig, is_draft_model=False) -> "MoECommType | None":
    """Select the MoE communication method according to parallel settings,
    device generation, token count, and quantization.

    1. Non-MoE models return `None`.
    2. Without expert parallel, fall back to all-gather.
    3. On A2/A3/A5 with expert parallel, pick MC2 / FUSED_MC2 / ALLTOALL by token
       count and capacity; 310P always all-gather.
    """
    if not is_moe_model(vllm_config):
        return None
    mc2_tokens_capacity = get_mc2_tokens_capacity()
    soc_version = get_ascend_device_type()
    # SUBTRACTED: quant_type = getattr(hf_text_config, 'moe_quantize'/'quantize', None)
    #             仅 A3 fused_mc2==2 的 'w8a8_dynamic' 判用，随该分支细判一并省略（原 fwd_context.py:L262-L266）。

    if not vllm_config.parallel_config.enable_expert_parallel or get_ep_group().world_size == 1:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A2}:
        # SUBTRACTED: num_experts_per_device<=24 且 ep_world_size>=16 且 num_tokens<=capacity → MC2
        #             的细粒度阈值（原 fwd_context.py:L271-L283）——抽象通信方式到硬件代际的映射细节，
        #             host 无 NPU 无法触发；保留『按条件二选一』骨架。
        moe_comm_type = MoECommType.MC2 if num_tokens <= mc2_tokens_capacity else MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A3}:
        # SUBTRACTED: fused_mc2_enable / dispatch_ffn_combine EP<=32 守卫 / quant=='w8a8_dynamic' / MTP
        #             守卫等 fused_decode/prefill 细判（原 fwd_context.py:L285-L308）。保留主干：
        #             容量内 MC2(或 FUSED_MC2)、容量外 ALLTOALL(或 FUSED_MC2) 二选一。
        if num_tokens <= mc2_tokens_capacity:
            moe_comm_type = MoECommType.MC2
        else:
            moe_comm_type = MoECommType.ALLTOALL
    elif soc_version in {AscendDeviceType._310P}:
        moe_comm_type = MoECommType.ALLGATHER
    elif soc_version in {AscendDeviceType.A5}:
        # SUBTRACTED: A5 的 world_size / num_experts_per_tok 阈值细判（原 fwd_context.py:L310-L317）。
        moe_comm_type = MoECommType.MC2 if num_tokens <= mc2_tokens_capacity else MoECommType.ALLTOALL
    else:
        raise ValueError(f"Unsupported soc_version: {soc_version}")
    return moe_comm_type
