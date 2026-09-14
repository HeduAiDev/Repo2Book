# SOURCE: vllm/v1/attention/selector.py
# ch26 切面：get_attn_backend 的显式后端支（attention_config.backend 名 →
# registry 解析 → current_platform.get_attn_backend_cls → qualname 解析）。
# MLAAttention(attn_backend=None) 在 host 上走此路径；显式注入 attn_backend
# 类的参数面与真实一致（L123-L130）。
# SUBTRACTED：自动优先级面（use_mla×算力代分档——→ ch21）；AttentionSelectorConfig
#   的长尾旗标按消费承载。
from __future__ import annotations

from functools import cache
from typing import cast

import torch


# SOURCE: vllm/v1/attention/selector.py AttentionSelectorConfig —— HOST SEAM
#   消费字段子集
# SOURCE: vllm/v1/attention/selector.py —— HOST SEAM（锚点双置）
class AttentionSelectorConfig:
    # SOURCE: vllm/v1/attention/selector.py —— HOST SEAM（锚点双置）
    def __init__(self, **kw):
        self.__dict__.update(kw)


# SOURCE: vllm/v1/attention/selector.py:L101-… get_attn_backend —— 减法子集
#   （显式 backend 支；None → ch21 域失败面）
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
):
    """Selects which attention backend to use and lazily imports it."""
    # SOURCE: vllm/v1/attention/selector.py:L101-…（锚点双置）
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    attn_type = attn_type or "decoder"
    attn_selector_config = AttentionSelectorConfig(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=cast(str | None, kv_cache_dtype),
        block_size=None,
        use_mla=use_mla,
        has_sink=has_sink,
        use_sparse=use_sparse,
        attn_type=attn_type,
    )

    attention_config = vllm_config.attention_config
    backend = attention_config.backend
    # SUBTRACTED: backend_per_kind 的 per-KV-group 覆盖（kind 映射面）——ch14 域
    return _cached_get_attn_backend(
        backend=backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )


@cache
def _cached_get_attn_backend(
    backend,
    attn_selector_config: AttentionSelectorConfig,
    num_heads: int | None = None,
) -> type:
    # SOURCE: vllm/v1/attention/selector.py:L170-… —— 减法子集
    from vllm.platforms import current_platform
    from vllm.utils.import_utils import resolve_obj_by_qualname

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

    # SUBTRACTED: get_required_kv_cache_layout 的布局回写联动——ch14 域
    return backend_cls
