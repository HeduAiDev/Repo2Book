# vllm_ascend/ops/fused_moe/moe_comm_method.py —— subtract-only 精简版（ch15 配角）
#
# 本章只用到 get_moe_comm_method：由 forward_context 注入前选定的 MoECommType
# 取出对应的 MoECommMethod 实例（下游 MoE 层 forward 时据此发起 all_to_all/MC2/all_gather）。
# 各 MoECommMethod 的真实算子实现留 ch26，本文件只保留「按 type 取 method」的工厂查表。
from vllm_ascend.ascend_forward_context import MoECommType

# SUBTRACTED: AlltoAllCommImpl / AllGatherCommImpl / MC2CommImpl / FusedMC2CommImpl 各通信
#   原语实现与 token_dispatcher / quantization 大段 import（moe_comm_method.py:L1-L48）
#   —— MoE 通信原语实现留 ch26，本章只讲「在 forward context 里选定并取出」。

_MoECommMethods: dict["MoECommType | None", object] = {}


# SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L51
def get_moe_comm_method(moe_comm_type: "MoECommType | None"):
    return _MoECommMethods.get(moe_comm_type)


# SOURCE: vllm_ascend/ops/fused_moe/moe_comm_method.py:L55
def setup_moe_comm_method(moe_config):
    # SUBTRACTED: 真实实现按 ep_size 把 ALLTOALL/ALLGATHER/MC2/FUSED_MC2 的 *CommImpl 实例
    #   注册进 _MoECommMethods（moe_comm_method.py:L56-L64）—— 实例构造依赖 NPU 运行时，
    #   实现留 ch26；本章只需 get_moe_comm_method 的查表语义（测试预填登记表）。
    raise NotImplementedError("Real MoECommMethod impls are covered in ch26.")
