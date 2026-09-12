# SOURCE: vllm/v1/attention/backends/registry.py
# ch25 切面：decode 家族注册表的最小消费面——AttentionBackendEnum（本章
# 文件引用的成员 + CUSTOM 槽）+ _ATTN_OVERRIDES + register_backend 装饰器
#（第三方注册流：测试/host 参考后端经此登记，selector 的显式后端支解析）。
# SUBTRACTED：MambaAttentionBackendEnum 整族与其余 15+ 同构成员、
#   supports_batch_invariance 面——ch14/ch21 域（delete[4]/章界收窄）。
from __future__ import annotations

from enum import Enum
from typing import Callable


# SOURCE: vllm/v1/attention/backends/registry.py AttentionBackendEnum —— 减法
#   子集（值 = 真实类路径字符串逐字；成员面收窄到本章引用 + 全家族的
#   MLA 代表 FLASHMLA/TRITON_MLA/FLASHINFER 与通用 FLASH_ATTN）
class AttentionBackendEnum(Enum):
    FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    FLASHMLA = "vllm.v1.attention.backends.mla.flashmla.FlashMLABackend"
    TRITON_MLA = "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend"
    FLASHINFER = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
    # SUBTRACTED: ROCM/XPU/DSV4/CPU_ATTN 等 15+ 同构条目——ch21 域
    # Placeholder for third-party/custom backends - must be registered
    # before use, set to None to avoid alias with other backends.
    CUSTOM = None

    def get_path(self) -> str:
        """Get the class path for this backend (respects overrides).

        Returns:
            The fully qualified class path string

        Raises:
            ValueError: If Backend.CUSTOM is used without being registered
        """
        # SOURCE: vllm/v1/attention/backends/registry.py get_path —— 逐字
        path = _ATTN_OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"Attention backend {self.name} must be registered before "
                f"use. Use register_backend("
                f"AttentionBackendEnum.{self.name}, "
                f"'your.module.YourClass')"
            )
        return path

    def get_class(self) -> type:
        """Get the backend class (respects overrides).

        Returns:
            The backend class

        Raises:
            ImportError: If the backend class cannot be imported
            ValueError: If CUSTOM is used without being registered
        """
        # SOURCE: vllm/v1/attention/backends/registry.py get_class —— 逐字
        from vllm.utils.import_utils import resolve_obj_by_qualname

        return resolve_obj_by_qualname(self.get_path())


# SOURCE: vllm/v1/attention/backends/registry.py _ATTN_OVERRIDES —— 逐字位
_ATTN_OVERRIDES: dict = {}


# SOURCE: vllm/v1/attention/backends/registry.py register_backend —— 逐字
#   （装饰器/直注两形态）
def register_backend(
    backend: AttentionBackendEnum, class_path: str | None = None
) -> Callable[[type], type]:
    """Register or override an attention backend implementation.

    Args:
        backend: The AttentionBackendEnum member to register
        class_path: Optional class path. If not provided and used as
            decorator, will be auto-generated from the class.

    Returns:
        Decorator function if class_path is None, otherwise a no-op.

    Examples:
        # Override an existing backend
        @register_backend(AttentionBackendEnum.FLASH_ATTN)
        class MyCustomFlashAttn(AttentionBackend):
            ...

        # Register a custom third-party backend
        @register_backend(AttentionBackendEnum.CUSTOM)
        class MyCustomBackend(AttentionBackend):
            ...

        # Direct registration
        register_backend(
            AttentionBackendEnum.CUSTOM, "my.module.MyCustomBackend"
        )
    """

    def decorator(cls: type) -> type:
        # SOURCE: vllm/v1/attention/backends/registry.py register_backend.decorator
        _ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"
        return cls

    if class_path is not None:
        _ATTN_OVERRIDES[backend] = class_path
        return lambda x: x

    return decorator
