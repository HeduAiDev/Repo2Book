# SOURCE: vllm/config/attention.py
# ch26 切面：AttentionConfig 的 indexer 消费字段（FP4 变体总开关 + indexer KV
# dtype + 显式 backend 名 + prefill backend 槽——selector/MLAAttention 消费位）。
# 真实文件为完整 dataclass；本切面按消费字段承载，默认值逐字。
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# SOURCE: vllm/config/attention.py IndexerKVDtype —— 逐字 Literal 面
IndexerKVDType = Literal["bf16", "fp8", "mxfp4", "nvfp4"]

# SOURCE: vllm/config/attention.py MLAPrefillBackendEnum 的名字位（枚举体
#   与优先级选择面归 ch21/ch25——本切面只承载 mla_prefill_backend 槽的类型名）


# SOURCE: vllm/config/attention.py AttentionConfig —— 减法子集（本章消费字段）
@dataclass
class AttentionConfig:
    # SOURCE: vllm/config/attention.py backend —— 显式后端名（selector 消费）
    backend: str | None = None

    # SOURCE: vllm/config/attention.py mla_prefill_backend —— MLA prefill 家族
    #   选择槽（get_mla_prefill_backend 消费；默认 None=自动）
    mla_prefill_backend: str | None = None

    # SOURCE: vllm/config/attention.py:L68-L69 use_fp4_indexer_cache —— 逐字
    use_fp4_indexer_cache: bool = False
    """If set, use fp4 indexer cache for dsv32 family model (not support yet)"""

    # SOURCE: vllm/config/attention.py:L71-L73 indexer_kv_dtype —— 逐字
    indexer_kv_dtype: IndexerKVDType = "bf16"
    """Data type for the sparse-attention indexer K cache. Quantized formats
    (fp8, mxfp4, nvfp4) require indexer kernel support in the backend."""

    # SOURCE: vllm/config/attention.py use_non_causal —— 逐字默认
    use_non_causal: bool = False
    """Whether to use non-causal (bidirectional) attention."""
