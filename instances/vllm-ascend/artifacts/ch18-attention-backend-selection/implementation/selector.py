# vllm/v1/attention/selector.py（+ vllm/utils/import_utils.py）—— subtract-only 精简版
#
# OOT 后端「被接进来」的真正解析路径：
#   selector.get_attn_backend → _cached_get_attn_backend
#     → current_platform.get_attn_backend_cls(...) 拿到一个**点分类路径字符串**
#     → resolve_obj_by_qualname() importlib 解析成后端类
# 这解释了为何昇腾 get_attn_backend_cls 返回的是字符串而非类对象——selector 延迟解析，
# 避免在平台层 import 全部后端（mla/sfa/dsa 各自重依赖），做到按需加载。
import importlib
from functools import cache
from typing import Any


# SOURCE: vllm/utils/import_utils.py:L104-L110
def resolve_obj_by_qualname(qualname: str) -> Any:
    """Resolve an object by its fully-qualified class name."""
    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


# SOURCE: vllm/v1/attention/selector.py:L105-L136
@cache
def _cached_get_attn_backend(
    backend,
    attn_selector_config,
    num_heads: int | None = None,
) -> type:
    # SOURCE: vllm/v1/attention/selector.py:L105-L136
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
    backend = resolve_obj_by_qualname(attention_cls)

    # SUBTRACTED: required_layout = backend.get_required_kv_cache_layout(); if ...: set_kv_cache_layout(...)
    #   （selector.py:L124-L134）—— 选定后端后按需调整 KV cache layout 的旁支；dossier elide 明示
    #   「对昇腾路由立意非主线，可一句带过」。删除不影响「点分路径字符串 → resolve_obj_by_qualname → 后端类」主路。

    return backend
