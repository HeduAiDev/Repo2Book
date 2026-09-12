# SOURCE: vllm/v1/attention/ops/flashmla.py
# ch25 切面：FlashMLA 可用性探测 + 接口门面全文减法——_is_flashmla_
# available / is_flashmla_dense_supported 逐字（host 无 CUDA 扩展 → False）；
# else 支的 stub 位逐字（flash_mla_with_kvcache/get_mla_metadata 转抛）。
# 真身由 third_party 接口承载（vllm/third_party/flashmla/flash_mla_interface.
# py 的 HOST SEAM 参考镜像——文件头 Data-Movement 伪码的精确数学）。
# SUBTRACTED：flash_mla_with_kvcache_fp8 / get_mla_metadata_dense_fp8 双
# kernel 分派（L109-L152）——delete[6]（单格式 bf16 即可演示 MQA kernel
# 接口契约）；sparse 变体族 → ch26。
from __future__ import annotations

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

# SOURCE: vllm/v1/attention/ops/flashmla.py:L10-L28 可用性探测 —— 逐字
if current_platform.is_cuda():
    try:
        import vllm._flashmla_C  # noqa: F401

        _flashmla_C_AVAILABLE = True
    except ImportError:
        _flashmla_C_AVAILABLE = False
else:
    _flashmla_C_AVAILABLE = False

if current_platform.is_cuda():
    try:
        import vllm._flashmla_extension_C  # noqa: F401

        _flashmla_extension_C_AVAILABLE = True
    except ImportError:
        _flashmla_extension_C_AVAILABLE = False
else:
    _flashmla_extension_C_AVAILABLE = False


# SOURCE: vllm/v1/attention/ops/flashmla.py:L31-L49 _is_flashmla_available
#   —— 逐字
def _is_flashmla_available() -> tuple[bool, str | None]:
    # SOURCE: vllm/v1/attention/ops/flashmla.py:L31-L49（锚点双置：声明上方注释同文）
    if not _flashmla_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_C is not available, likely was not "
            "compiled due to insufficient nvcc version or a supported arch "
            "was not in the list of target arches to compile for.",
        )
    if not _flashmla_extension_C_AVAILABLE:
        return (
            False,
            "vllm._flashmla_extension_C is not available, likely "
            "due to a build error.",
        )

    return True, None


# SOURCE: vllm/v1/attention/ops/flashmla.py:L51-L60 is_flashmla_dense_supported
#   —— 逐字
def is_flashmla_dense_supported() -> tuple[bool, str | None]:
    # SOURCE: vllm/v1/attention/ops/flashmla.py:L51-L60（锚点双置：声明上方注释同文）
    """
    Return: is_supported_flag, unsupported_reason (optional).
    """
    is_available, maybe_reason = _is_flashmla_available()
    if not is_available:
        return False, maybe_reason
    if not current_platform.is_device_capability_family(90):
        return False, "FlashMLA Dense is only supported on Hopper devices."
    return True, None


# SOURCE: vllm/v1/attention/ops/flashmla.py:L63-L70 _raise_flashmla_unavailable
#   —— 逐字
def _raise_flashmla_unavailable(*_args, **_kwargs):
    # SOURCE: vllm/v1/attention/ops/flashmla.py:L63-L70（锚点双置：声明上方注释同文）
    _, reason = _is_flashmla_available()
    raise RuntimeError(reason or "FlashMLA is not available")


# SOURCE: vllm/v1/attention/ops/flashmla.py:L91-L106 接口门面 —— HOST SEAM：
#   真实在 CUDA 上 re-export third_party 接口；host 走参考镜像（同一接口）
from vllm.third_party.flashmla.flash_mla_interface import (  # noqa: F401
    FlashMLASchedMeta,
    flash_mla_with_kvcache,
    get_mla_metadata,
)

# SUBTRACTED: flash_mla_with_kvcache_fp8 与 get_mla_metadata_dense_fp8 的
#   wrapper（L109-L152）——delete[6]（fp8 双 kernel 分派删，bf16 单格式）
