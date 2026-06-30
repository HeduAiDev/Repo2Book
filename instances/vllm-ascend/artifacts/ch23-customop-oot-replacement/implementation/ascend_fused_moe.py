# 只做减法的精简版 —— 活动实例 vllm-ascend
# 真实文件：vllm_ascend/ops/fused_moe/fused_moe.py
#
# 第四则代表项：FusedMoE 也只是 REGISTERED_ASCEND_OPS 表里同构的一条「类名→Ascend子类」映射，
# 顶替机制与上面的标本完全一致。其完整 forward_oot（专家选路/通信/融合）是 Part VI ch26 的主题，
# 本章只保留类头以呈现「它也走同一张表」。
import torch
from vllm.model_executor.layers.fused_moe.layer import FusedMoE


class AscendFusedMoE(FusedMoE):
    # SOURCE: vllm_ascend/ops/fused_moe/fused_moe.py:L41
    def forward_oot(self, *args, **kwargs):
        # SUBTRACTED: 完整 MoE forward_oot（select_experts / moe_comm / npu 融合专家算子等整条管线）——
        #             FusedMoE 顶替是 ch26 专题；本章仅需它作为注册表的第四则代表项参与「建表→register_oot」演示。
        raise NotImplementedError("see ch26")
