# SOURCE: vllm/config/cache.py
# HOST SEAM：CacheDType 的 Literal 面（真实为 --kv-cache-dtype 的取值
# Literal；本章作为类型注解消费——selector 的入参校验位）。
from __future__ import annotations

from typing import Literal

# SOURCE: vllm/config/cache.py CacheDType —— HOST SEAM Literal 面
CacheDType = Literal[
    "auto",
    "fp8",
    "fp8_e5m2",
    "fp8_e4m3",
    "fp8_ds_mla",
    "bfloat16",
    "float16",
    "int4_per_token_head",
    "int8_per_token_head",
    "fp8_per_token_head",
    "nvfp4",
]
