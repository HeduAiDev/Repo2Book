# SPDX-License-Identifier: Apache-2.0
# SOURCE: vllm/entrypoints/openai/models/protocol.py —— HOST SEAM（最小
# 承载）：BaseModelPath 逐字（init_app_state 的 served_model_names 装配）；
# LoRAModulePath 由 api_utils.process_lora_modules 消费，同字面承载。
from dataclasses import dataclass


# SOURCE: vllm/entrypoints/openai/models/protocol.py:L8-L11 —— BaseModelPath 逐字
@dataclass
class BaseModelPath:
    name: str
    model_path: str


# SOURCE: vllm/entrypoints/openai/models/protocol.py:L14-L21 —— LoRAModulePath
# 字段位（HOST SEAM：真实含 is_3d_lora_weight 可选字段）
@dataclass
class LoRAModulePath:
    # SOURCE: vllm/entrypoints/openai/models/protocol.py —— name/path/base_model_name
    name: str
    path: str
    base_model_name: str | None = None
