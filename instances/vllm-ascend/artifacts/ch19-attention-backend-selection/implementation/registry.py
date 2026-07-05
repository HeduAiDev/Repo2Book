# vllm/v1/attention/backends/registry.py —— subtract-only 精简版（后端注册表）
#
# 本章点(2)「注册进 vLLM」：vLLM 把 CUSTOM=None 预留为第三方/OOT 后端槽；
# register_backend() 把 override 写进 _ATTN_OVERRIDES。昇腾导入期用
#   @register_backend(AttentionBackendEnum.CUSTOM, "ASCEND")
# 占住 CUSTOM 槽（声明「CUSTOM 由昇腾接管」）。注意第二参形参名是 class_path——
# 传 "ASCEND" 会落进 _ATTN_OVERRIDES[CUSTOM] 并返回 no-op 装饰器（类本身不被改写）。
#
# 纯 Python，可在 host 跑：调 register_backend 后查 _ATTN_OVERRIDES 即可验证占位。
from collections.abc import Callable
from enum import Enum, EnumMeta
from typing import cast

# SUBTRACTED: init_logger / resolve_obj_by_qualname 的 import（registry.py:L9-L10）——
#   logger 与本章无关；resolve_obj_by_qualname 在 selector.py 精简版里另行保留。


# SOURCE: vllm/v1/attention/backends/registry.py:L18-L31
class _AttentionBackendEnumMeta(EnumMeta):
    """Metaclass for AttentionBackendEnum to provide better error messages."""

    def __getitem__(cls, name: str):
        # SOURCE: vllm/v1/attention/backends/registry.py:L21-L31
        try:
            return super().__getitem__(name)
        except KeyError:
            members = cast("dict[str, Enum]", cls.__members__).keys()
            valid_backends = ", ".join(members)
            raise ValueError(
                f"Unknown attention backend: '{name}'. "
                f"Valid options are: {valid_backends}"
            ) from None


# SOURCE: vllm/v1/attention/backends/registry.py:L34-L90
class AttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    """Enumeration of all supported attention backends.

    The enum value is the default class path, but this can be overridden
    at runtime using register_backend().
    """

    FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    # SUBTRACTED: 其余几十个内建后端成员（FLASH_ATTN_DIFFKV/TRITON_ATTN/ROCM_*/CPU_ATTN/...，
    #   registry.py:L45-L87）—— 本章只需 FLASH_ATTN（昇腾伪装 + FA3 早返回判定要它）与
    #   CUSTOM（OOT 占位槽），其余内建后端与昇腾路由立意正交，折叠。

    # Placeholder for third-party/custom backends - must be registered before use
    # set to None to avoid alias with other backend, whose value is an empty string
    CUSTOM = None

    # SUBTRACTED: get_path / get_class / is_overridden / clear_override 等方法
    #   （registry.py:L92-L198）—— 真正的后端解析在昇腾这里走 get_attn_backend_cls 返回的
    #   点分路径（见 selector.py 精简版），不依赖 CUSTOM.get_class() 去解析 "ASCEND"；
    #   故 register_backend 在此更像「声明/占位」。这些方法对本章「占位 vs 真正解析」对账非主线，折叠。


# SOURCE: vllm/v1/attention/backends/registry.py:L194-L200
_ATTN_OVERRIDES: dict[AttentionBackendEnum, str] = {}
# SUBTRACTED: _MAMBA_ATTN_OVERRIDES（registry.py:L200）—— mamba 注意力与本章无关。


# SOURCE: vllm/v1/attention/backends/registry.py:L203-L255
def register_backend(
    backend: AttentionBackendEnum,
    class_path: str | None = None,
    is_mamba: bool = False,
) -> Callable[[type], type]:
    """Register or override a backend implementation.

    class_path 非 None 时把字符串写进 _ATTN_OVERRIDES[backend] 并返回 no-op 装饰器
    （类本身不被改写）；class_path 为 None 时返回真正改写注册表的 decorator。
    """

    def decorator(cls: type) -> type:
        # SOURCE: vllm/v1/attention/backends/registry.py:L241-L246
        if is_mamba:
            # SUBTRACTED: _MAMBA_ATTN_OVERRIDES 分支 —— 本章无 mamba。
            raise NotImplementedError
        else:
            _ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"  # type: ignore[index]
        return cls

    if class_path is not None:
        if is_mamba:
            # SUBTRACTED: _MAMBA_ATTN_OVERRIDES 分支 —— 本章无 mamba。
            raise NotImplementedError
        else:
            _ATTN_OVERRIDES[backend] = class_path  # type: ignore[index]
        return lambda x: x

    return decorator
