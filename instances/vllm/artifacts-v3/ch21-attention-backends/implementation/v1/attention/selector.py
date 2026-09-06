# Subtract-only companion for v3 ch21 — vllm/v1/attention/selector.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions (delete[3], each marked `# SUBTRACTED:`).
#
# 本章主角文件之一（站 1-2）：选后端入口——get_attn_backend 把零散维度收进
# AttentionSelectorConfig NamedTuple、backend_per_kind 按 KVCacheSpecKind 逐组
# 覆写、@cache 内层 _cached_get_attn_backend 惰性 import 后端类 +
# set_kv_cache_layout 全局副作用。
from __future__ import annotations

from functools import cache
from typing import Literal, NamedTuple, cast, get_args

import torch

from ..._host_seams import envs, init_logger
from .backend import AttentionBackend, AttentionType
from .backends.registry import AttentionBackendEnum

logger = init_logger(__name__)

# SOURCE: vllm/config/cache.py:L19-L34 CacheDType ——（逐字）HOST SEAM 类型面
#   （合法 kv_cache_dtype 的全集 Literal）
CacheDType = Literal[
    "auto",
    "float16",
    "bfloat16",
    "fp8",
    "fp8_e4m3",
    "fp8_e5m2",
    "fp8_inc",
    "fp8_ds_mla",
    "turboquant_k8v4",
    "turboquant_4bit_nc",
    "turboquant_k3v4_nc",
    "turboquant_3bit_nc",
    "int4_per_token_head",
    "int8_per_token_head",
    "fp8_per_token_head",
    "nvfp4",
]


# SOURCE: vllm/v1/attention/selector.py:L24-L58 AttentionSelectorConfig ——（逐字）
#   选后端的可哈希键：@cache 同配置只解一次、几十层共享结果
class AttentionSelectorConfig(NamedTuple):
    head_size: int
    dtype: torch.dtype
    kv_cache_dtype: str | None
    block_size: int | None
    use_mla: bool = False
    has_sink: bool = False
    use_sparse: bool = False
    use_mm_prefix: bool = False
    use_per_head_quant_scales: bool = False
    attn_type: str = AttentionType.DECODER
    has_sliding_window: bool = False
    use_non_causal: bool = False
    use_batch_invariant: bool = False
    use_kv_connector: bool = False
    use_pcp: bool = False

    def __repr__(self):  # SOURCE: vllm/v1/attention/selector.py:L41-L58
        return (
            f"AttentionSelectorConfig(head_size={self.head_size}, "
            f"dtype={self.dtype}, "
            f"kv_cache_dtype={self.kv_cache_dtype}, "
            f"block_size={self.block_size}, "
            f"use_mla={self.use_mla}, "
            f"has_sink={self.has_sink}, "
            f"use_sparse={self.use_sparse}, "
            f"use_mm_prefix={self.use_mm_prefix}, "
            f"use_per_head_quant_scales={self.use_per_head_quant_scales}, "
            f"attn_type={self.attn_type}, "
            f"has_sliding_window={self.has_sliding_window}, "
            f"use_non_causal={self.use_non_causal}, "
            f"use_batch_invariant={self.use_batch_invariant}, "
            f"use_kv_connector={self.use_kv_connector}, "
            f"use_pcp={self.use_pcp})"
        )


# SOURCE: vllm/v1/attention/selector.py:L61-L98 get_attn_spec_kind ——（逐字）
#   层信号 → KVCacheSpecKind（backend_per_kind 覆写时按它对键）
def get_attn_spec_kind(  # SOURCE: vllm/v1/attention/selector.py
    use_mla: bool,
    has_sliding_window: bool,
    attn_type: str,
) -> "KVCacheSpecKind":
    """Derive the KV-cache group kind a layer belongs to from its signals.

    Mirrors ``get_kv_cache_spec_kind`` (which derives the kind from the
    produced ``KVCacheSpec``) so users can target groups by kind when
    setting ``AttentionConfig.backend_per_kind``.

    ``SINK_FULL_ATTENTION`` is intentionally not derived here: it is produced
    only by the ``StaticSinkAttention`` layer, whereas a plain ``Attention``
    layer with attention sinks (e.g. gpt-oss) still yields a
    ``FullAttentionSpec``/``SlidingWindowSpec``. Sinks therefore do not change
    the kind.

    Args:
        use_mla: Whether the layer uses multi-head latent attention.
        has_sliding_window: Whether the layer applies a sliding window.
        attn_type: The layer's ``AttentionType``.

    Returns:
        The ``KVCacheSpecKind`` the layer maps to.
    """
    from ..kv_cache_interface import KVCacheSpecKind

    if attn_type == AttentionType.ENCODER_ONLY:
        return KVCacheSpecKind.ENCODER_ONLY_ATTENTION
    if attn_type == AttentionType.ENCODER_DECODER:
        return KVCacheSpecKind.CROSS_ATTENTION
    if use_mla:
        if has_sliding_window:
            return KVCacheSpecKind.SLIDING_WINDOW_MLA
        return KVCacheSpecKind.MLA_ATTENTION
    if has_sliding_window:
        return KVCacheSpecKind.SLIDING_WINDOW
    return KVCacheSpecKind.FULL_ATTENTION


# SOURCE: vllm/v1/attention/selector.py:L101-L174 get_attn_backend ——（逐字）
#   公开入口：kv_cache_dtype 合法性 assert → 收维度 → 打包 →
#   backend_per_kind 逐组覆写 → 交 @cache 内层
def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
    use_mm_prefix: bool = False,
    use_per_head_quant_scales: bool = False,
    attn_type: str | None = None,
    num_heads: int | None = None,
    has_sliding_window: bool = False,
) -> type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""

    if kv_cache_dtype is not None:
        # SOURCE: vllm/v1/attention/selector.py:L116-L117（逐字；CacheDType
        #   是 vllm/config/cache.py 的 Literal 类型别名——HOST SEAM 以等值
        #   Literal 承载该类型面）
        valid_cache_dtypes = get_args(CacheDType)
        assert kv_cache_dtype in valid_cache_dtypes, (
            f"Invalid kv_cache_dtype: {kv_cache_dtype}. "
            f"Valid values are: {valid_cache_dtypes}"
        )

    from ..._host_seams import get_current_vllm_config

    vllm_config = get_current_vllm_config()

    cache_config = vllm_config.cache_config
    block_size: int | None
    if cache_config is not None and cache_config.user_specified_block_size:
        block_size = cache_config.block_size
    else:
        block_size = None

    kv_transfer_config = vllm_config.kv_transfer_config
    use_kv_connector = (
        kv_transfer_config is not None and kv_transfer_config.is_kv_transfer_instance
    )

    attn_type = attn_type or AttentionType.DECODER
    attn_selector_config = AttentionSelectorConfig(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=cast("str | None", kv_cache_dtype),
        block_size=block_size,
        use_mla=use_mla,
        has_sink=has_sink,
        use_sparse=use_sparse,
        use_mm_prefix=use_mm_prefix,
        use_per_head_quant_scales=use_per_head_quant_scales,
        attn_type=attn_type,
        has_sliding_window=has_sliding_window,
        use_non_causal=vllm_config.attention_config.use_non_causal,
        use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
        use_kv_connector=use_kv_connector,
        use_pcp=vllm_config.parallel_config.prefill_context_parallel_size > 1,
    )

    # A per-KV-group override (keyed by KVCacheSpecKind) takes precedence over
    # the global backend; kinds not present in the map fall back to it.
    attention_config = vllm_config.attention_config
    backend = attention_config.backend
    if attention_config.backend_per_kind:
        kind = get_attn_spec_kind(
            use_mla=use_mla,
            has_sliding_window=has_sliding_window,
            attn_type=attn_type,
        )
        backend = attention_config.backend_per_kind.get(kind.value, backend)

    return _cached_get_attn_backend(
        backend=backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )


# SOURCE: vllm/v1/attention/selector.py:L177-L208 _cached_get_attn_backend ——（逐字）
#   @cache 内层：平台 get_attn_backend_cls 给类路径字符串 → resolve_obj_by_
#   qualname 真 import → 后端要求特定 KV layout 则全局 set_kv_cache_layout
@cache
def _cached_get_attn_backend(  # SOURCE: vllm/v1/attention/selector.py
    backend,
    attn_selector_config: AttentionSelectorConfig,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    from ..._host_seams import current_platform

    attention_cls = current_platform.get_attn_backend_cls(
        backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )
    if not attention_cls:
        raise ValueError(
            f"Invalid attention backend for {current_platform.device_name}"
        )
    from ...utils.import_utils import resolve_obj_by_qualname

    backend = resolve_obj_by_qualname(attention_cls)

    # Adjust kv cache layout if the selected backend requires a specific one
    required_layout = backend.get_required_kv_cache_layout()
    if required_layout is not None:
        from .backends.utils import set_kv_cache_layout

        set_kv_cache_layout(required_layout)
        logger.info_once(
            "Using %s KV cache layout for %s backend.",
            required_layout,
            backend.get_name(),
        )

    return backend


# SUBTRACTED: get_mamba_attn_backend / _cached_get_mamba_attn_backend
#   （L211-L230）——delete[3]：Mamba 后端选择线本章不展开（ch22 只碰
#   SlotMappingMode）；MambaAttentionBackendEnum 随 registry 的 delete[4] 删。
