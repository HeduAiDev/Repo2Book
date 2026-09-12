# SOURCE: vllm/model_executor/layers/attention/__init__.py —— 减法门面
#   （本章只 re-export MLAAttention；Attention/RSWA/ChunkedLocal 等通用
#   插座族归 ch21/ch23 域）
from vllm.model_executor.layers.attention.mla_attention import MLAAttention

__all__ = [
    "MLAAttention",
]
