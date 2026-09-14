# SOURCE: vllm/model_executor/layers/quantization/__init__.py
# HOST SEAM：QuantizationConfig 类型位（Indexer/_try_load 形参的类型注解消费）。
from __future__ import annotations


# SOURCE: vllm/model_executor/layers/quantization/config.py QuantizationConfig
#   —— HOST SEAM 类型位（本章无量化方法消费——quant_config 恒 None 传递）
class QuantizationConfig:
    pass
