"""ch24 精简版 — 后端注册表（只做减法）。

对应 vllm/v1/attention/backends/registry.py。AttentionBackendEnum 的每个成员值就是一段
默认类路径字符串；get_class() = 查覆盖表 _ATTN_OVERRIDES → 回退枚举值 →
resolve_obj_by_qualname 懒加载真正 import。register_backend 可运行时覆盖或注册第三方 CUSTOM
后端。命名、控制流与真实 vLLM 一致。
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum, EnumMeta
from typing import cast


# SOURCE: vllm/utils/import_utils.py (resolve_obj_by_qualname)
def resolve_obj_by_qualname(qualname: str):
    """Resolve an object by its fully qualified name."""
    # SUBTRACTED: 真实实现在 vllm/utils/import_utils.py，逻辑相同（rsplit 出 module 与
    # classname、importlib.import_module、getattr）。这里内联等价实现，避免拖入 vllm 包。
    import importlib

    module_name, obj_name = qualname.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, obj_name)


# SOURCE: vllm/v1/attention/backends/registry.py:L18 (_AttentionBackendEnumMeta)
class _AttentionBackendEnumMeta(EnumMeta):
    """Metaclass for AttentionBackendEnum to provide better error messages."""

    def __getitem__(cls, name: str):  # SOURCE: vllm/v1/attention/backends/registry.py:L21 (__getitem__)
        try:
            return super().__getitem__(name)
        except KeyError:
            members = cast("dict[str, Enum]", cls.__members__).keys()
            valid_backends = ", ".join(members)
            raise ValueError(
                f"Unknown attention backend: '{name}'. "
                f"Valid options are: {valid_backends}"
            ) from None


# SOURCE: vllm/v1/attention/backends/registry.py:L34 (AttentionBackendEnum)
class AttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    """Enumeration of all supported attention backends.

    The enum value is the default class path, but this can be overridden
    at runtime using register_backend().

    To get the actual backend class (respecting overrides), use:
        backend.get_class()
    """

    FLASH_ATTN = "flash_attn.FlashAttentionBackend"
    TRITON_ATTN = (
        "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    )
    FLASHINFER = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
    # SUBTRACTED: 真实枚举约二十余项（ROCM_*/XPU_*/各 MLA 后端/FLEX_ATTENTION/TREE_ATTN/
    # CPU_ATTN/TURBOQUANT/...，registry.py:L44-L88）——都是「名字 = 完整类路径字符串」的同构
    # 条目。本章只保留 FLASH_ATTN（贯穿全章的具体后端，类路径指向本精简版 flash_attn.py 以便
    # get_class() 真能懒加载到本章的 FlashAttentionBackend）+ 两个代表项。
    # NOTE: 真实里 FLASH_ATTN 值是 "vllm.v1.attention.backends.flash_attn.
    # FlashAttentionBackend"；本精简版把 implementation/ 放进 sys.path、值改成可顶层 import 的
    # "flash_attn.FlashAttentionBackend"，仅为让 get_class() 在 host 真能解析到本章后端类。

    # Placeholder for third-party/custom backends - must be registered before use.
    # set to None to avoid alias with other backend, whose value is an empty string.
    CUSTOM = None

    # SOURCE: vllm/v1/attention/backends/registry.py:L90 (get_path)
    def get_path(self, include_classname: bool = True) -> str:
        """Get the class path for this backend (respects overrides)."""
        path = _ATTN_OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"Backend {self.name} must be registered before use. "
                f"Use register_backend(Backend.{self.name}, 'your.module.YourClass')"
            )
        if not include_classname:
            path = path.rsplit(".", 1)[0]
        return path

    # SOURCE: vllm/v1/attention/backends/registry.py:L111 (get_class)
    def get_class(self) -> "type[AttentionBackend]":
        """Get the backend class (respects overrides)."""
        return resolve_obj_by_qualname(self.get_path())

    # SOURCE: vllm/v1/attention/backends/registry.py:L121 (is_overridden)
    def is_overridden(self) -> bool:
        """Check if this backend has been overridden."""
        return self in _ATTN_OVERRIDES

    # SOURCE: vllm/v1/attention/backends/registry.py:L129 (clear_override)
    def clear_override(self) -> None:
        """Clear any override for this backend, reverting to the default."""
        _ATTN_OVERRIDES.pop(self, None)


# SUBTRACTED: MambaAttentionBackendEnum 与 _MAMBA_ATTN_OVERRIDES（registry.py:L134-L208）
# 是与上表完全同构的 mamba/线性注意力后端注册表；本章只讲标准注意力后端，删之不影响主链。

# SOURCE: vllm/v1/attention/backends/registry.py:L207 (_ATTN_OVERRIDES)
_ATTN_OVERRIDES: dict[AttentionBackendEnum, str] = {}


# SOURCE: vllm/v1/attention/backends/registry.py:L211 (register_backend)
def register_backend(
    backend: AttentionBackendEnum,
    class_path: str | None = None,
    is_mamba: bool = False,
) -> Callable[[type], type]:
    """Register or override a backend implementation.

    Examples:
        # Override an existing attention backend
        @register_backend(AttentionBackendEnum.FLASH_ATTN)
        class MyCustomFlashAttn:
            ...

        # Register a custom third-party attention backend
        @register_backend(AttentionBackendEnum.CUSTOM)
        class MyCustomBackend:
            ...

        # Direct registration
        register_backend(AttentionBackendEnum.CUSTOM, "my.module.MyCustomBackend")
    """
    # SUBTRACTED: is_mamba 分支写 _MAMBA_ATTN_OVERRIDES（registry.py:L249-L262）——随 mamba
    # 注册表一并删去；本章只走 _ATTN_OVERRIDES 这条标准注意力路径。
    def decorator(cls: type) -> type:  # SOURCE: vllm/v1/attention/backends/registry.py:L249 (register_backend.decorator)
        _ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"
        return cls

    if class_path is not None:
        _ATTN_OVERRIDES[backend] = class_path
        return lambda x: x

    return decorator
