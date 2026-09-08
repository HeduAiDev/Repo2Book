# SOURCE: vllm/model_executor/layers/activation.py
# ch23 消费面：SiluAndMul（LlamaMLP 的门控激活，must_keep）。
# SUBTRACTED：SituAndMul/SiluAndMulWithClamp/其余激活族与 CUDA op 绑定面
#   （activation.py 其余）——MoE/平台 kernel 域。
from __future__ import annotations

import torch
import torch.nn.functional as F

from vllm.model_executor.custom_op import CustomOp
from vllm.platforms import current_platform


# SOURCE: vllm/model_executor/layers/activation.py:L117
#   @CustomOp.register("silu_and_mul") 注册装饰
@CustomOp.register("silu_and_mul")
# SOURCE: vllm/model_executor/layers/activation.py:L118 SiluAndMul
class SiluAndMul(CustomOp):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    # SOURCE: vllm/model_executor/layers/activation.py:L130-L137 __init__
    #   —— 减法子集（平台 op 绑定逐字；host 非 cuda/cpu/xpu → 不绑 op）
    def __init__(self, *, compile_native: bool = True):
        super().__init__(compile_native=compile_native)
        if (
            current_platform.is_cuda_alike()
            or current_platform.is_cpu()
            or current_platform.is_xpu()
        ):
            # SOURCE: vllm/model_executor/layers/activation.py:L137 op 绑定
            #   ——CUDA _C.silu_and_mul；host 平台面全 False 不进（直调
            #   forward_native 的真实派发分支）
            self.op = torch.ops._C.silu_and_mul

    # SOURCE: vllm/model_executor/layers/activation.py:L139-L143 forward_native
    #   （逐字——门控数学本体）
    @staticmethod
    def forward_native(x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        # SOURCE: vllm/model_executor/layers/activation.py:L139-L143 forward_native
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]

    # SOURCE: vllm/model_executor/layers/activation.py:L144-L149 forward_cuda
    #   （逐字——kernel 派发面；host 不达）
    def forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/activation.py:L144-L149 forward_cuda
        d = x.shape[-1] // 2
        output_shape = x.shape[:-1] + (d,)
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        self.op(out, x)
        return out

    # SOURCE: vllm/model_executor/layers/activation.py:L152-L153 forward_xpu（逐字）
    def forward_xpu(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_cuda(x)

    # SOURCE: vllm/model_executor/layers/activation.py:L155-L158 forward_cpu
    #   —— 减法子集（POWERPC arch 分支删除——平台特例；主支直调 native）
    def forward_cpu(self, x: torch.Tensor) -> torch.Tensor:
        # SOURCE: vllm/model_executor/layers/activation.py:L155-L158 forward_cpu
        return self.forward_native(x)
