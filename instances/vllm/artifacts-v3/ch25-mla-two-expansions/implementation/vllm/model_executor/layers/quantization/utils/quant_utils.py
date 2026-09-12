# SOURCE: vllm/model_executor/layers/quantization/utils/quant_utils.py
# ch25 消费面：get_and_maybe_dequant_weights（process_weights_after_loading
# 的权重取回，L436-L470 的非量化支）——HOST SEAM 子集：本包 seam Linear 无
# 量化 method 面，取「unquantized: 直接返回 [out, in] 权重」的真实支；
# fp8/LoRA 解包支归 ch27（delete[2] 删其消费点）。
from __future__ import annotations

import torch


# SOURCE: vllm/model_executor/layers/quantization/utils/quant_utils.py:L436
#   get_and_maybe_dequant_weights —— 减法子集（非量化支逐字）
def get_and_maybe_dequant_weights(
    layer, out_dtype: torch.dtype = torch.float32
):
    """Return layer's unquantized weights in [out, in] layout"""
    # SOURCE: vllm/model_executor/layers/quantization/utils/quant_utils.py:L456-L461
    #   Unquantized layer: just return base weights
    weight = getattr(layer, "weight", None)
    return weight.to(out_dtype)
