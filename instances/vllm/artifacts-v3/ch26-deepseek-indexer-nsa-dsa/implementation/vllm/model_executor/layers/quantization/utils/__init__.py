# SOURCE: vllm/model_executor/layers/quantization/utils/__init__.py
# —— 包标记（消费面在 fp8_utils / quant_utils）
from vllm.model_executor.layers.quantization.utils.fp8_utils import (  # noqa: F401
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (  # noqa: F401
    GroupShape,
    get_fp8_min_max,
    scaled_dequantize,
)
