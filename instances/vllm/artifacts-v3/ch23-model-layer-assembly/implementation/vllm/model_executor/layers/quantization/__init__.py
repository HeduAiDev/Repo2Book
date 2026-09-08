# SOURCE: vllm/model_executor/layers/quantization/__init__.py
# ch23 精简版：quantization 包门面——只承载 base_config 的 QuantizationConfig/
# QuantizeMethodBase/method_has_implemented_embedding 注解面（量化本体归 ch27）。
from vllm.model_executor.layers.quantization.base_config import (  # noqa: F401
    QuantizationConfig,
    QuantizeMethodBase,
    method_has_implemented_embedding,
)
