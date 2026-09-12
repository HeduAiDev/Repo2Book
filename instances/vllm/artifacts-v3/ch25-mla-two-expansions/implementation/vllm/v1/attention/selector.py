# SOURCE: vllm/v1/attention/selector.py
# ch25 切面（站 3 的 get_attn_backend(use_mla=True) 调用位）：选后端入口
# 的显式后端子集——AttentionSelectorConfig 打包 + _cached_get_attn_backend
# 的「平台交类路径 → resolve」主干逐字。
# SUBTRACTED：自动优先级选择面（平台分档/validate_configuration/回退循环）
#   ——ch21 域（dossier m10 明示「选择机制归 ch21」）；host 平台对 backend=None
#   如实抛 NotImplementedError（与真实 CPU 平台「无可用 MLA 后端」同型失败面）。
from __future__ import annotations

from typing import NamedTuple, cast

import torch

from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.attention.backend import AttentionBackend, AttentionType

logger = init_logger(__name__)


# SOURCE: vllm/v1/attention/selector.py:L24-L53 AttentionSelectorConfig ——
#   逐字 NamedTuple（本章消费字段）
class AttentionSelectorConfig(NamedTuple):
    # SOURCE: vllm/v1/attention/selector.py:L24-L53（锚点双置：声明上方注释同文）
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


# SOURCE: vllm/v1/attention/selector.py:L101-L174 get_attn_backend —— 减法
#   子集（显式后端支逐字；自动优先级面 → ch21）
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
    # SOURCE: vllm/v1/attention/selector.py:L101-L174（锚点双置：声明上方注释同文）
    """Selects which attention backend to use and lazily imports it."""

    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()

    cache_config = vllm_config.cache_config
    block_size: int | None
    if cache_config is not None and cache_config.user_specified_block_size:
        block_size = cache_config.block_size
    else:
        block_size = None

    attn_type = attn_type or AttentionType.DECODER
    attn_selector_config = AttentionSelectorConfig(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=cast(str | None, kv_cache_dtype),
        block_size=block_size,
        use_mla=use_mla,
        has_sink=has_sink,
        use_sparse=use_sparse,
        use_mm_prefix=use_mm_prefix,
        use_per_head_quant_scales=use_per_head_quant_scales,
        attn_type=attn_type,
        has_sliding_window=has_sliding_window,
        use_batch_invariant=vllm_config.attention_config.__dict__.get(
            "use_batch_invariant", False
        ),
        use_kv_connector=False,
        use_pcp=vllm_config.parallel_config.prefill_context_parallel_size > 1,
    )
    # SUBTRACTED: backend_per_kind 按 KVCacheSpecKind 的逐组覆写
    #   （selector.py:L165-L173）——ch21 域

    attention_config = vllm_config.attention_config
    backend = attention_config.backend

    return _cached_get_attn_backend(
        backend=backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )


def _cached_get_attn_backend(
    backend,
    attn_selector_config: AttentionSelectorConfig,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    # SOURCE: vllm/v1/attention/selector.py:L177-L208 _cached_get_attn_backend
    #   —— 减法子集（平台交类路径 → resolve 主干逐字；无缓存位）
    from vllm.platforms import current_platform

    attention_cls = current_platform.get_attn_backend_cls(
        backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )
    if not attention_cls:
        raise ValueError(
            f"Invalid attention backend for {current_platform.device_name}"
        )
    backend_cls = resolve_obj_by_qualname(attention_cls)

    return backend_cls
