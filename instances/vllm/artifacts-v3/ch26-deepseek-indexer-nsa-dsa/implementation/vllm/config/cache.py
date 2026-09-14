# SOURCE: vllm/config/cache.py
# HOST SEAM（ch25 同款）：CacheDType 的 Literal 字符串面（"auto"/"bfloat16"/
# "fp8"/"fp8_ds_mla"/…——flashmla_sparse 的 supported_kv_cache_dtypes 消费）。
from __future__ import annotations

CacheDType = str  # HOST SEAM：Literal 字符串面
