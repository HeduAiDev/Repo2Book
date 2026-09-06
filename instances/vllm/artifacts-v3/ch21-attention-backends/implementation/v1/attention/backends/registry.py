# Subtract-only companion for v3 ch21 — vllm/v1/attention/backends/registry.py
# (pin v0.27.1 / 6e448d0ea). Same names, same structure, same control flow;
# only dossier-approved deletions (delete[4], each marked `# SUBTRACTED:`).
#
# 本章主角文件之一：后端注册表——枚举值=默认类路径字符串的静态表 +
# _ATTN_OVERRIDES 运行时覆盖表 + get_path/get_class 懒加载 +
# register_backend 第三方注册（CUSTOM=None 占位）。
from __future__ import annotations

from collections.abc import Callable
from enum import Enum, EnumMeta
from typing import TYPE_CHECKING, cast

from ....utils.import_utils import resolve_obj_by_qualname
from ...._host_seams import init_logger

if TYPE_CHECKING:
    from ..backend import AttentionBackend

logger = init_logger(__name__)


# SOURCE: vllm/v1/attention/backends/registry.py:L18-L31 _AttentionBackendEnumMeta
#   ——（逐字）未知名字的错误信息里列出全部合法成员
class _AttentionBackendEnumMeta(EnumMeta):
    """Metaclass for AttentionBackendEnum to provide better error messages."""

    def __getitem__(cls, name: str):  # SOURCE: vllm/v1/attention/backends/registry.py:L21-L31
        """Get backend by name with helpful error messages."""
        try:
            return super().__getitem__(name)
        except KeyError:
            members = cast("dict[str, Enum]", cls.__members__).keys()
            valid_backends = ", ".join(members)
            raise ValueError(
                f"Unknown attention backend: '{name}'. "
                f"Valid options are: {valid_backends}"
            ) from None


# SOURCE: vllm/v1/attention/backends/registry.py:L34-L128 AttentionBackendEnum ——
#   枚举值=完整类路径字符串（『名字→类路径』静态表）
class AttentionBackendEnum(Enum, metaclass=_AttentionBackendEnumMeta):
    """Enumeration of all supported attention backends.

    The enum value is the default class path, but this can be overridden
    at runtime using register_backend().

    To get the actual backend class (respecting overrides), use:
        backend.get_class()
    """

    # SOURCE: vllm/v1/attention/backends/registry.py:L44-L51 常用四项（逐字）
    FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"
    FLASH_ATTN_DIFFKV = (
        "vllm.v1.attention.backends.flash_attn_diffkv.FlashAttentionDiffKVBackend"
    )
    TRITON_ATTN = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
    TRITON_ATTN_DIFFKV = (
        "vllm.v1.attention.backends.triton_attn_diffkv.TritonAttentionDiffKVBackend"
    )

    # SUBTRACTED: ROCM 族六项（ROCM_ATTN/ROCM_AITER_MLA/ROCM_AITER_TRITON_MLA/
    #   ROCM_AITER_FA/ROCM_AITER_MLA_SPARSE/ROCM_AITER_UNIFIED_ATTN，L52-L62
    #   与 L120-L123）与 XPU_MLA_SPARSE（L63）——delete[4]：AMD/Intel 平台族
    #   同构条目，cuda.py 优先级表不引用。
    # SOURCE: vllm/v1/attention/backends/registry.py:L64 TORCH_SDPA ——（逐字）
    #   保留：ViT 专用接口（cuda.py get_supported_vit_attn_backends）仍引用
    TORCH_SDPA = ""  # this tag is only used for ViT
    # SOURCE: vllm/v1/attention/backends/registry.py:L65-L81 非 MLA 常用项与
    #   MLA 头部（逐字——cuda.py 优先级表引用的全部成员一个不删）
    FLASHINFER = "vllm.v1.attention.backends.flashinfer.FlashInferBackend"
    FLASHINFER_MLA = (
        "vllm.v1.attention.backends.mla.flashinfer_mla.FlashInferMLABackend"
    )
    TOKENSPEED_MLA = (
        "vllm.v1.attention.backends.mla.tokenspeed_mla.TokenspeedMLABackend"
    )
    FLASHINFER_MLA_SPARSE = (
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse."
        "FlashInferMLASparseTRTLLMBackend"
    )
    FLASHINFER_MLA_SPARSE_SM120 = (
        "vllm.v1.attention.backends.mla.flashinfer_mla_sparse."
        "FlashInferMLASparseSM120Backend"
    )
    TRITON_MLA = "vllm.v1.attention.backends.mla.triton_mla.TritonMLABackend"
    CUTLASS_MLA = "vllm.v1.attention.backends.mla.cutlass_mla.CutlassMLABackend"
    FLASHMLA = "vllm.v1.attention.backends.mla.flashmla.FlashMLABackend"
    # SOURCE: vllm/v1/attention/backends/registry.py:L83-L85 FLASHMLA_SPARSE（逐字）
    FLASHMLA_SPARSE = (
        "vllm.v1.attention.backends.mla.flashmla_sparse.FlashMLASparseBackend"
    )
    # SUBTRACTED: DSV4 sparse 三项（FLASHMLA_SPARSE_DSV4/FLASHINFER_MLA_SPARSE_
    #   DSV4/ROCM_FLASHMLA_SPARSE_DSV4，L87-L96）——delete[4]：模型驱动的
    #   DeepSeek V4 专属后端（ch24 域点名即可）。
    # SOURCE: vllm/v1/attention/backends/registry.py:L97-L100 FA 的 MLA 两项（逐字）
    FLASH_ATTN_MLA = "vllm.v1.attention.backends.mla.flashattn_mla.FlashAttnMLABackend"
    FLASH_ATTN_MLA_SPARSE = (
        "vllm.v1.attention.backends.mla.flashattn_mla_sparse.FlashAttnMLASparseBackend"
    )
    # SUBTRACTED: MiniMax MSA 三项（MINIMAX_M3_SPARSE/CUTLASS_MSA/TRITON_MSA，
    #   L101-L111）——delete[4]：模型驱动的同构条目（AttentionConfig.__post_
    #   init 的 MSA 别名块连带删——消费已删成员）。
    # SOURCE: vllm/v1/attention/backends/registry.py:L112-L125 其余常用项（逐字）
    NO_ATTENTION = "vllm.v1.attention.backends.no_attention.NoAttentionBackend"
    FLEX_ATTENTION = "vllm.v1.attention.backends.flex_attention.FlexAttentionBackend"
    # SUBTRACTED: HPC_ATTN（L114-L119）与 CPU_ATTN（L124）——delete[4]：
    #   Tencent hpc-ops 专属 / CPU 平台后端（CPU 平台域）。
    TURBOQUANT = "vllm.v1.attention.backends.turboquant_attn.TurboQuantAttentionBackend"
    # Placeholder for third-party/custom backends - must be registered before use
    # set to None to avoid alias with other backend, whose value is an empty string
    # SOURCE: vllm/v1/attention/backends/registry.py:L126-L128 CUSTOM（逐字）
    CUSTOM = None

    def get_path(self, include_classname: bool = True) -> str:  # SOURCE: vllm/v1/attention/backends/registry.py:L130-L147
        """Get the class path for this backend (respects overrides).

        Returns:
            The fully qualified class path string

        Raises:
            ValueError: If Backend.CUSTOM is used without being registered
        """
        path = _ATTN_OVERRIDES.get(self, self.value)
        if not path:
            raise ValueError(
                f"Backend {self.name} must be registered before use. "
                f"Use register_backend(Backend.{self.name}, 'your.module.YourClass')"
            )
        if not include_classname:
            path = path.rsplit(".", 1)[0]
        return path

    def get_class(self) -> "type[AttentionBackend]":  # SOURCE: vllm/v1/attention/backends/registry.py:L149-L159
        """Get the backend class (respects overrides).

        Returns:
            The backend class

        Raises:
            ImportError: If the backend class cannot be imported
            ValueError: If Backend.CUSTOM is used without being registered
        """
        return resolve_obj_by_qualname(self.get_path())

    def is_overridden(self) -> bool:  # SOURCE: vllm/v1/attention/backends/registry.py:L161-L167
        """Check if this backend has been overridden.

        Returns:
            True if this backend has a registered override
        """
        return self in _ATTN_OVERRIDES

    def clear_override(self) -> None:  # SOURCE: vllm/v1/attention/backends/registry.py:L169-L171
        """Clear any override for this backend, reverting to the default."""
        _ATTN_OVERRIDES.pop(self, None)


# SUBTRACTED: MambaAttentionBackendEnum 整类（L174-L234）与 _MAMBA_ATTN_
#   OVERRIDES（L238）——delete[4]：Mamba 后端选择线本章不展开（get_mamba_
#   attn_backend 随 selector 的 delete[3] 删）；register_backend 的 is_mamba
#   分支连带删（消费已删表）。


# SOURCE: vllm/v1/attention/backends/registry.py:L237 _ATTN_OVERRIDES ——（逐字）
#   覆盖表本体：运行时 register_backend 写入，get_path 感知
_ATTN_OVERRIDES: dict[AttentionBackendEnum, str] = {}


# SOURCE: vllm/v1/attention/backends/registry.py:L241-L293 register_backend ——
#   运行时覆盖/注册第三方（is_mamba 分支 delete[4] 连带删）
def register_backend(
    backend: AttentionBackendEnum,
    class_path: str | None = None,
) -> Callable[[type], type]:
    """Register or override a backend implementation.

    Args:
        backend: The AttentionBackendEnum member to register
        class_path: Optional class path. If not provided and used as
            decorator, will be auto-generated from the class.

    Returns:
        Decorator function if class_path is None, otherwise a no-op

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
        register_backend(
            AttentionBackendEnum.CUSTOM,
            "my.module.MyCustomBackend"
        )
    """

    def decorator(cls: type) -> type:  # SOURCE: vllm/v1/attention/backends/registry.py:L279-L284
        _ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"  # type: ignore[index]
        return cls

    if class_path is not None:  # SOURCE: vllm/v1/attention/backends/registry.py:L286-L291
        _ATTN_OVERRIDES[backend] = class_path  # type: ignore[index]
        return lambda x: x

    return decorator
