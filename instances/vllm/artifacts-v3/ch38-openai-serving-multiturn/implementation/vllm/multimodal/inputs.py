# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/multimodal/inputs.py —— HOST SEAM（最小承载）：本章消费面
# （EngineCoreRequest.mm_features 注解、inputs 引擎 schema 的 mm 键类型）
# 只需要这两个形状；多模态处理全链归 ch6。
from typing import Any, NamedTuple


# SOURCE: vllm/multimodal/inputs.py —— HOST SEAM：PlaceholderRange 形状
class PlaceholderRange(NamedTuple):
    # SOURCE: vllm/multimodal/inputs.py —— HOST SEAM 字段位
    offset: int = 0
    # SOURCE: vllm/multimodal/inputs.py —— HOST SEAM 字段位（length 是
    # placeholder 的 prompt token 跨度，_get_mm_token_counts 曾按它求和）
    length: int = 0


# SOURCE: vllm/multimodal/inputs.py —— HOST SEAM：MultiModalFeatureSpec 形状
class MultiModalFeatureSpec(NamedTuple):
    # SOURCE: vllm/multimodal/inputs.py —— HOST SEAM 字段位
    modality: str = ""
    data: Any = None


# SOURCE: vllm/multimodal/inputs.py —— HOST SEAM：MultiModalPlaceholders 别名
MultiModalPlaceholders = dict[str, list[PlaceholderRange]]

# SUBTRACTED: vllm/multimodal/inputs.py 其余（MultiModalKwargs 族、
# MultiModalInputs、batch spec）——多模态数据面归 ch6；本章纯文本主线
# （delete[7] 已把多模态 usage 细分采集从精简版删去）。
