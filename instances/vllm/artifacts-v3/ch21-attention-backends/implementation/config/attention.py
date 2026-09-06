# Subtract-only companion for v3 ch21 — vllm/config/attention.py
# (pin v0.27.1 / 6e448d0ea). 本章主角用户面：AttentionConfig 的
# backend（全局后端）与 backend_per_kind（逐 KVCacheSpecKind 覆写——
# 『逐 KV 组混布』的用户面）+ 两 field_validator（"auto"→None 自动选择、
# backend_per_kind 字符串表解析与 kind 校验）+ 本章消费位 use_non_causal /
# flash_attn_max_num_splits_for_cuda_graph。其余字段以章界注记收窄。
from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic import field_validator

from .._host_seams import config
from ..v1.attention.backends.registry import AttentionBackendEnum

# SUBTRACTED: MLAPrefillBackendEnum import（L10）——MLA prefill 域 → ch24。
# SUBTRACTED: IndexerKVDType / MiniMaxM3MSADecodeBackend 类型别名（L13-L14）
#   ——量化 indexer 与 MiniMax MSA 域（delete[4] 同域）。


# SOURCE: vllm/config/attention.py:L17-L36 AttentionConfig ——（逐字：backend
#   与 backend_per_kind 两字段及其 docstring）
@config
class AttentionConfig:
    """Configuration for attention mechanisms in vLLM."""

    backend: AttentionBackendEnum | None = None
    """Attention backend to use. Use "auto" or None for automatic selection."""

    backend_per_kind: dict[str, AttentionBackendEnum] = field(default_factory=dict)
    """Per-KV-cache-group attention backend overrides, keyed by
    `KVCacheSpecKind` (e.g. `{"mla_attention": "FLASHINFER_MLA",
    "sliding_window_mla": "TRITON_MLA"}`). This lets a model that splits its
    layers across multiple KV-cache groups (e.g. interleaved full and
    sliding-window attention) use a different backend per group.

    An entry overrides `backend` for layers of the matching kind; kinds not
    listed fall back to `backend` (or automatic selection). A selected backend
    that is invalid for that kind raises at startup."""

    # SOURCE: vllm/config/attention.py:L75-L76 use_non_causal ——（逐字）本章
    #   消费位：selector L152 收进 AttentionSelectorConfig 进优先级表分档
    use_non_causal: bool = False
    """Whether to use non-causal (bidirectional) attention."""

    # SOURCE: vllm/config/attention.py:L46-L47 flash_attn_max_num_splits_
    #   for_cuda_graph ——（逐字）本章消费位：FA Builder 的 scheduler_metadata
    #   预分配把 num_splits 上界定在此（捕获期中间缓冲尺寸恒定）
    flash_attn_max_num_splits_for_cuda_graph: int = 32
    """Flash Attention max number splits for cuda graph decode."""

    # SUBTRACTED: minimax_m3_msa_decode_backend / flash_attn_version /
    #   use_prefill_decode_attention / tq_max_kv_splits_for_cuda_graph /
    #   use_trtllm_attention / disable_flashinfer_q_quantization /
    #   mla_prefill_backend / use_prefill_query_quantization /
    #   use_fp4_indexer_cache / indexer_kv_dtype / sparse_mla_force_mqa /
    #   flex_attn_block_m / flex_attn_block_n / flex_attn_q_block_size /
    #   flex_attn_kv_block_size（L24-L104）——各后端/模型专属配置域
    #   （flex_attn_block_m 的消费支随 attention.py delete[9] 删）。

    # SUBTRACTED: __post_init__ 的 MSA 别名块（L106-L115）——delete[4]：
    #   消费已删的 CUTLASS_MSA/TRITON_MSA 枚举成员（MiniMax MSA 域）。

    # SUBTRACTED: compute_hash（L117-L129）——编译缓存 hash 域（ch19）。


    @field_validator("backend", mode="before")  # SOURCE: vllm/config/attention.py:L131-L143
    @classmethod
    def validate_backend_before(cls, value: Any) -> Any:  # SOURCE: vllm/config/attention.py
        """Enable parsing of the `backend` enum type from string.

        The special value "auto" is treated as None, which triggers
        automatic backend selection.
        """
        if isinstance(value, str):
            if value.lower() == "auto":
                return None
            return AttentionBackendEnum[value.upper()]
        return value


    @field_validator("backend_per_kind", mode="before")  # SOURCE: vllm/config/attention.py:L153-L177
    @classmethod
    def validate_backend_per_kind_before(cls, value: Any) -> Any:  # SOURCE: vllm/config/attention.py
        """Parse the `backend_per_kind` map from strings.

        Keys must be valid `KVCacheSpecKind` values; values are parsed like
        `backend` (enum name, case-insensitive).
        """
        from ..v1.kv_cache_interface import KVCacheSpecKind

        if not isinstance(value, dict):
            return value
        valid_kinds = {kind.value for kind in KVCacheSpecKind}
        parsed: dict[str, AttentionBackendEnum] = {}
        for kind, backend in value.items():
            if kind not in valid_kinds:
                raise ValueError(
                    f"Unknown KV cache group kind '{kind}' in "
                    f"backend_per_kind. Valid kinds are: "
                    f"{', '.join(sorted(valid_kinds))}."
                )
            if isinstance(backend, str):
                backend = AttentionBackendEnum[backend.upper()]
            parsed[kind] = backend
        return parsed

    # SUBTRACTED: validate_mla_prefill_backend_before（L145-L151）——MLA
    #   prefill 域（ch24）。
