# 只做减法的精简版 —— 活动实例 vllm-ascend
# 真实文件：vllm_ascend/ops/activation.py
#
# 标本一：最简「只换头」—— 继承基座激活算子，只覆写 forward_oot，一行 torch_npu 算子顶替 CUDA 实现。
import torch
import torch_npu
from vllm.model_executor.layers.activation import QuickGELU, SiluAndMul


class AscendQuickGELU(QuickGELU):
    # SOURCE: vllm_ascend/ops/activation.py:L25
    def forward_oot(self, x: torch.tensor) -> torch.Tensor:
        out = torch_npu.npu_fast_gelu(x)
        return out


class AscendSiluAndMul(SiluAndMul):
    # SOURCE: vllm_ascend/ops/activation.py:L31
    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        # SUBTRACTED: weight_prefetch_method.maybe_prefetch_mlp_weight_pre/postprocess（L33-34,L36）——
        #             权重预取是性能优化，与「forward_oot 顶替为 npu_swiglu」的语义/数值无关。
        out = torch_npu.npu_swiglu(x)
        return out

# SUBTRACTED: AscendSiluAndMulWithClamp(L40-50) / AscendSwigluOAIAndMul(L53-61)——
#             与上两则标本同模式，不在本章主线。
