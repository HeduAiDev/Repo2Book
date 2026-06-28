# 技法④ 极简样本 —— vllm_ascend/patch/worker/patch_triton.py（subtract-only）
#
# 招式：给库模块直接补/换一个函数，不涉及类或 self。
# 台账原因：torch_npu 捆绑的 Triton 没有 next_power_of_2，而 vLLM/ascend 在 94+ 处调它。
#
# SOURCE: vllm_ascend/patch/worker/patch_triton.py:L1-L10
import vllm.model_executor.layers.mamba.ops.causal_conv1d
from vllm.triton_utils import HAS_TRITON, triton  # noqa: F401
from vllm.utils.math_utils import next_power_of_2

from vllm_ascend.ops.triton.mamba.causal_conv1d import causal_conv1d_fn, causal_conv1d_update_npu
# SUBTRACTED: fla.ops / gumbel 等 import，及 fla 层的 LayerNormFn / chunk_gated_delta_rule /
#   fused_recurrent 重绑（同构「给库模块换函数」，不增信息）(patch_triton.py:L1-L9, L15-L20)。

# 招式核心一行：把 vLLM 自带的 next_power_of_2 直接挂到 triton 模块上。
triton.next_power_of_2 = next_power_of_2

vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_update = causal_conv1d_update_npu
vllm.model_executor.layers.mamba.ops.causal_conv1d.causal_conv1d_fn = causal_conv1d_fn

# SUBTRACTED: 文件后半段 HAS_TRITON 为 False 时注入的两个纯 PyTorch 回退算子
#   (_fused_post_conv_prep_pytorch / _fused_recurrent_packed_decode_pytorch) —— 是回退算子
#   的数学实现，与技法④无关 (patch_triton.py:L22+)。
