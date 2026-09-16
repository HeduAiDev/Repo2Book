# SOURCE: vllm/model_executor/layers/fused_moe/config.py
# ch34 切面：FusedMoEQuantConfig 类型位（naive_dp_ep 的签名面；量化细节按删除项 9
# 压成两行注释，归 ch27 域）。

from __future__ import annotations


# SOURCE: vllm/model_executor/layers/fused_moe/config.py FusedMoEQuantConfig
#   —— 载体（真实为 dataclass 配置族；本章只作类型位）
class FusedMoEQuantConfig:  # HOST/ch27 SEAM
    # SOURCE: vllm/model_executor/layers/fused_moe/config.py:L214-L589（锚点双置）
    quant_dtype: str | None = None
