# SOURCE: vllm/v1/attention/backends/mla/prefill/registry.py
# ch25 切面（m10 / must_keep MLAPrefillBackendEnum）：prefill 家族注册表
# 全文——_MLAPrefillBackendEnumMeta + MLAPrefillBackendEnum（五成员 + CUSTOM
# 槽，类路径字符串逐字）+ get_path/get_class/is_overridden/clear_override +
# register_mla_prefill_backend（装饰器/直注两形态）逐字。
from __future__ import annotations

from collections.abc import Callable
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING

from vllm.utils.import_utils import resolve_obj_by_qualname

if TYPE_CHECKING:
    from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend


# SOURCE: vllm/v1/attention/backends/mla/prefill/registry.py:L19-L31
#   _MLAPrefillBackendEnumMeta —— 逐字
class _MLAPrefillBackendEnumMeta(EnumMeta):
    # SOURCE: vllm/v1/attention/backends/mla/prefill/registry.py:L19-L31（锚点双置：声明上方注释同文）
    """Metaclass for MLAPrefillBackendEnum to provide better error messages."""

    def __getitem__(cls, name):
        # SOURCE: vllm/v1/attention/backends/mla/prefill/registry.py:L19-L31（锚点双置：声明上方注释同文）
        try:
            return super().__getitem__(name)
        except KeyError:
            members = cls.__members__.keys()
            valid_backends = ", ".join(members)
            raise ValueError(
                f"Unknown MLA prefill backend: '{name}'. "
                f"Valid options are: {valid_backends}"
            ) from None


# SOURCE: vllm/v1/attention/backends/mla/prefill/registry.py:L34-L96
#   MLAPrefillBackendEnum —— 逐字（must_keep：第二套 prefill 家族注册表）
class MLAPrefillBackendEnum(Enum, metaclass=_MLAPrefillBackendEnumMeta):
    """Enumeration of all supported MLA prefill backends."""

    FLASH_ATTN = (
        "vllm.v1.attention.backends.mla.prefill.flash_attn.FlashAttnPrefillBackend"
    )
    FLASHINFER = (
        "vllm.v1.attention.backends.mla.prefill.flashinfer.FlashInferPrefillBackend"
    )
    TRTLLM_RAGGED = (
        "vllm.v1.attention.backends.mla.prefill.trtllm_ragged."
        "TrtllmRaggedPrefillBackend"
    )
    TOKENSPEED_MLA = (
        "vllm.v1.attention.backends.mla.prefill.tokenspeed_mla."
        "TokenspeedMLAPrefillBackend"
    )
    ROCM_AITER_FA = (
        "vllm.v1.attention.backends.mla.prefill.aiter_flash_attn."
        "AiterFlashAttnPrefillBackend"
    )
    # Placeholder for third-party/custom backends - must be registered before use
    # set to None to avoid alias with other backend, whose value is an empty string
    CUSTOM = None

    def get_path(self) -> str:
        """Get the class path for this backend (respects overrides).

        Returns:
            The fully qualified class path string

        Raises:
            ValueError: If Backend.CUSTOM is used without being registered
        """
        # SOURCE: vllm/v1/attention/backends/mla/prefill/registry.py:L59-L76
        path = _MLA_PREFILL_OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"MLA prefill backend {self.name} must be registered before "
                f"use. Use register_mla_prefill_backend("
                f"MLAPrefillBackendEnum.{self.name}, "
                f"'your.module.YourClass')"
            )
        return path

    def get_class(self) -> "type[MLAPrefillBackend]":
        """Get the backend class (respects overrides).

        Returns:
            The backend class

        Raises:
            ImportError: If the backend class cannot be imported
            ValueError: If CUSTOM is used without being registered
        """
        # SOURCE: vllm/v1/attention/backends/mla/prefill/registry.py:L78-L88
        return resolve_obj_by_qualname(self.get_path())

    def is_overridden(self) -> bool:
        """Check if this backend has been overridden."""
        # SOURCE: vllm/v1/attention/backends/mla/prefill/registry.py:L90-L92
        return self in _MLA_PREFILL_OVERRIDES

    def clear_override(self) -> None:
        """Clear any override for this backend, reverting to the default."""
        # SOURCE: vllm/v1/attention/backends/mla/prefill/registry.py:L94-L96
        _MLA_PREFILL_OVERRIDES.pop(self, None)


_MLA_PREFILL_OVERRIDES: dict[MLAPrefillBackendEnum, str] = {}


# SOURCE: vllm/v1/attention/backends/mla/prefill/registry.py:L102-L142
#   register_mla_prefill_backend —— 逐字
def register_mla_prefill_backend(
    backend: MLAPrefillBackendEnum,
    class_path: str | None = None,
) -> Callable[[type], type]:
    """Register or override an MLA prefill backend implementation.

    Args:
        backend: The MLAPrefillBackendEnum member to register
        class_path: Optional class path. If not provided and used as
            decorator, will be auto-generated from the class.

    Returns:
        Decorator function if class_path is None, otherwise a no-op.

    Examples:
        # Override an existing MLA prefill backend
        @register_mla_prefill_backend(MLAPrefillBackendEnum.FLASH_ATTN)
        class MyCustomFlashAttn(MLAPrefillBackend):
            ...

        # Register a custom third-party MLA prefill backend
        @register_mla_prefill_backend(MLAPrefillBackendEnum.CUSTOM)
        class MyCustomPrefillBackend(MLAPrefillBackend):
            ...

        # Direct registration
        register_mla_prefill_backend(
            MLAPrefillBackendEnum.CUSTOM,
            "my.module.MyCustomPrefillBackend"
        )
    """

    def decorator(cls: type) -> type:
        # SOURCE: vllm/v1/attention/backends/mla/prefill/registry.py:L134-L135
        _MLA_PREFILL_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"
        return cls

    if class_path is not None:
        _MLA_PREFILL_OVERRIDES[backend] = class_path
        return lambda x: x

    return decorator
