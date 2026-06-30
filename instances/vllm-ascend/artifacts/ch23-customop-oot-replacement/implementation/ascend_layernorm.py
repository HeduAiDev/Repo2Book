# 只做减法的精简版 —— 活动实例 vllm-ascend
# 真实文件：vllm_ascend/ops/layernorm.py
#
# 标本二（本章核心）：AscendRMSNorm 继承基座 RMSNorm（身不变），只覆写 forward_oot（换头），
# forward_oot 内 enable_custom_op() 真二分：融合算子 vs 原子算子回退。
import torch
from vllm.config import get_current_vllm_config
from vllm.model_executor.layers.layernorm import RMSNorm

from vllm_ascend.utils import enable_custom_op


class AscendRMSNorm(RMSNorm):
    # SOURCE: vllm_ascend/ops/layernorm.py:L28
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: int | None = None,
        has_weight: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        # super().__init__ 复用基座 RMSNorm 的 weight/variance_epsilon 等（身不变）。
        super().__init__(hidden_size, eps, var_hidden_size, has_weight, dtype)
        get_current_vllm_config()
        self.bias = None
        self.bias_loaded = False
        # SUBTRACTED: quant_config anti_method m4 的 norm.bias 探测分支（L42-47）与 _bias_weight_loader（L49-61）——
        #             量化边缘场景；精简版直接 bias=None/bias_loaded=False，保留 self.bias 字段供 forward_oot 二分引用即可。

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # SOURCE: vllm_ascend/ops/layernorm.py:L63
        import torch_npu

        if residual is not None:
            residual = torch.ops.vllm.maybe_chunk_residual(x, residual)
            if enable_custom_op():
                # 融合分支：一颗 AscendC 融合 kernel 同时算 add + rms_norm + bias。
                x, _, residual = torch.ops._C_ascend.npu_add_rms_norm_bias(
                    x, residual, self.weight, self.bias, self.variance_epsilon
                )
            else:
                # 回退分支：torch_npu 原子算子拼（add_rms_norm 一颗 + 单独 x.add_(bias)）。
                x, _, residual = torch_npu.npu_add_rms_norm(x, residual, self.weight, self.variance_epsilon)
                if self.bias is not None:
                    x.add_(self.bias)
            return x, residual

        x, residual = torch_npu.npu_rms_norm(x, self.weight, self.variance_epsilon)
        if self.bias_loaded:
            x.add_(self.bias)
        # SUBTRACTED: 无 residual 分支末尾 weight_prefetch_method.maybe_prefetch_mlp_weight_postprocess(L86-87)——
        #             权重预取性能优化，与数值/二分无关。
        return x

# SUBTRACTED: AscendGemmaRMSNorm / AscendRMSNormGated / LayerNormFn（layernorm.py:L91-202）——
#             与 AscendRMSNorm 同模式扩展，不在本章主线。
