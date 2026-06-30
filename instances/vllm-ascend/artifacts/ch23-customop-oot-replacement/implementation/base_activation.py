# 只做减法的精简版 —— 对照基座 vLLM v0.21.0
# 真实文件：vllm/model_executor/layers/activation.py
#
# 这两个基座激活算子是「身」：接口/注册键不变。昇腾子类继承它们、只新增 forward_oot（换头）。
# 保留 forward_native（回退/对照基线）+ forward_cuda（被顶替的 CUDA 实现）以呈现「换头不换身」。
import torch
import torch.nn.functional as F
from vllm.platforms import current_platform

from vllm.model_executor.custom_op import CustomOp


@CustomOp.register("silu_and_mul")
class SiluAndMul(CustomOp):
    # SOURCE: vllm/model_executor/layers/activation.py:L118
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.
    """

    def __init__(self, *, compile_native: bool = True):
        # SOURCE: vllm/model_executor/layers/activation.py:L130
        super().__init__(compile_native=compile_native)
        if current_platform.is_cuda_alike() or current_platform.is_xpu():
            self.op = torch.ops._C.silu_and_mul
        elif current_platform.is_cpu():
            self._forward_method = self.forward_native

    @staticmethod
    def forward_native(x: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/activation.py:L137
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/activation.py:L143
        # 被昇腾 forward_oot 顶替掉的 CUDA 实现（self.op == torch.ops._C.silu_and_mul）。
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    # SUBTRACTED: forward_xpu(L150-151) —— 非本章关心平台。


@CustomOp.register("quick_gelu")
class QuickGELU(CustomOp):
    # SOURCE: vllm/model_executor/layers/activation.py:L505

    def __init__(self):
        # SOURCE: vllm/model_executor/layers/activation.py:L509
        super().__init__()
        if (
            current_platform.is_cuda_alike()
            or current_platform.is_cpu()
            or current_platform.is_xpu()
        ):
            self.op = torch.ops._C.gelu_quick

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/activation.py:L518
        """PyTorch-native implementation equivalent to forward()."""
        return x * torch.sigmoid(1.702 * x)

    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/activation.py:L522
        out = torch.empty_like(x)
        self.op(out, x)
        return out

    # SUBTRACTED: forward_xpu(L527-528) —— 非本章关心平台。

# SUBTRACTED: FatreluAndMul / SiluAndMulWithClamp / MulAndSilu / GELU 系列 / SwigluOAIAndMul 等
#             其余 ~20 个激活算子（activation.py 余下）—— 与本章两则标本同构，不在主线。
