"""ch24 精简版 — 后端选择器（只做减法，f18 回收：选后端的公开入口）。

对应 vllm/v1/attention/selector.py。get_attn_backend 把调用方零散参数收进可哈希的
AttentionSelectorConfig，连同用户显式指定的 backend 一起喂给 @cache 的
_cached_get_attn_backend；后者经平台层 get_attn_backend_cls 返回类路径字符串、
resolve_obj_by_qualname 懒加载成类，必要时设 KV layout。命名/控制流与真实一致。
"""

from __future__ import annotations

from functools import cache
from typing import NamedTuple

import torch

from backend import AttentionBackend, AttentionType
from registry import resolve_obj_by_qualname


# SOURCE: vllm/v1/attention/selector.py:L22 (AttentionSelectorConfig)
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
    use_non_causal: bool = False
    use_batch_invariant: bool = False


# SOURCE: vllm/v1/attention/selector.py:L53 (get_attn_backend)
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
    *,
    backend=None,
    block_size: int | None = None,
) -> type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""
    # SUBTRACTED: 开头对 kv_cache_dtype 的合法性 assert（selector.py:L69-L74）依赖
    # vllm.config.cache.CacheDType 类型集合，删之不改主路径。
    # SUBTRACTED: 真实从 get_current_vllm_config() 读 cache_config.user_specified_block_size
    # 决定 block_size、读 vllm_config.attention_config.backend 拿用户显式后端、读
    # VLLM_BATCH_INVARIANT 环境（selector.py:L76-L96）。host 无全局 VllmConfig，本精简版把
    # 这两项作为可选参数 backend/block_size 直接传入，语义等价。
    attn_selector_config = AttentionSelectorConfig(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        block_size=block_size,
        use_mla=use_mla,
        has_sink=has_sink,
        use_sparse=use_sparse,
        use_mm_prefix=use_mm_prefix,
        use_per_head_quant_scales=use_per_head_quant_scales,
        attn_type=attn_type or AttentionType.DECODER,
        use_non_causal=False,
        use_batch_invariant=False,
    )

    return _cached_get_attn_backend(
        backend=backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )


@cache
# SOURCE: vllm/v1/attention/selector.py:L106 (_cached_get_attn_backend)
def _cached_get_attn_backend(
    backend,
    attn_selector_config: AttentionSelectorConfig,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    # SUBTRACTED: 真实 `from vllm.platforms import current_platform` 再调
    # current_platform.get_attn_backend_cls（selector.py:L111-L117）；host 无平台注册，本精简版
    # 直接 import 本章的 cuda 平台替身，语义等价（按优先级/校验选后端、返回类路径字符串）。
    from platform_cuda import CudaPlatform

    attention_cls = CudaPlatform.get_attn_backend_cls(
        backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )
    if not attention_cls:
        raise ValueError("Invalid attention backend")
    backend = resolve_obj_by_qualname(attention_cls)

    # Adjust kv cache layout if the selected backend requires a specific one
    required_layout = backend.get_required_kv_cache_layout()
    if required_layout is not None:
        # SUBTRACTED: 真实调 vllm.v1.attention.backends.utils.set_kv_cache_layout 设全局
        # KV layout（selector.py:L128-L135）；host 无该全局状态，留判定分支、省副作用调用。
        from flash_attn import set_kv_cache_layout

        set_kv_cache_layout(required_layout)

    return backend
