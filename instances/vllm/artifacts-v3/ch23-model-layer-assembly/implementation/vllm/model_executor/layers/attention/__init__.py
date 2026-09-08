# SOURCE: vllm/model_executor/layers/attention/__init__.py
# ch23 精简版：attention 包门面——只导出 Attention（EncoderOnlyAttention 等
# 家族随 delete[3] 删除）。
from vllm.model_executor.layers.attention.attention import Attention  # noqa: F401
