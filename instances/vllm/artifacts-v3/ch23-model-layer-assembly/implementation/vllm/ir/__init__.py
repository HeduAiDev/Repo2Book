# SOURCE: vllm/ir/__init__.py
# ch23 精简版：只承载 layernorm 两个 IR 算子（RMSNorm 的 forward_native 消费）。
# SUBTRACTED: ir/op.py 的 IrOp 注册体系（torch.library 定义/编译 lowering/
# 优先级 impl）——ch19 编译域；精简版直调 native 实现（IrOpInplaceOverload
# 未启 torch wrap 时同样走 _inner_call → native，见 op.py:L530-L534）。
from .ops import fused_add_rms_norm, rms_norm  # noqa: F401
