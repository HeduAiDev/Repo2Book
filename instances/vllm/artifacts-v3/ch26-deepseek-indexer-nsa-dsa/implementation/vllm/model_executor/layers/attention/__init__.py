# SOURCE: vllm/model_executor/layers/attention/__init__.py
# ch26 切面：再导出消费面（MLAAttention——mla.py 的插座 import 位）。
from vllm.model_executor.layers.attention.mla_attention import MLAAttention  # noqa: F401

__all__ = ["MLAAttention"]
