# SOURCE: vllm/models/deepseek_v4/__init__.py
# ch23 逐字收录（m9：v0.27 新布局的平台分发入口——hardware-isolated 布局的
# 门面；nvidia/amd/xpu 子包本体与 quant_config 归真实仓库/ch28 capstone，
# 本章不建——此文件仅作布局证据，不参与 import 链）。
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 model — hardware-isolated entry point.

The actual implementation lives under ``nvidia/`` and ``amd/``; this module
picks the right one for the current platform and re-exports the public
classes used by the model registry and quantization config lookup.
"""

from vllm.platforms import current_platform

# SUBTRACTED: from .quant_config import DeepseekV4FP8Config ——量化配置分发
#   归 ch27/ch28（子包本体不在精简版）

# Pick the per-platform implementation. The NVIDIA branch is the static
# default that mypy sees; the ROCm/XPU branches override at runtime and are
# kept type-compatible via ``# type: ignore[assignment]``.
if current_platform.is_rocm():
    # SUBTRACTED: from .amd.dspark/.amd.model/.amd.mtp ——子包本体归 ch28
    pass
elif current_platform.is_xpu():
    # SUBTRACTED: from .xpu.* ——子包本体归 ch28
    pass
else:
    # SUBTRACTED: from .nvidia.* ——子包本体归 ch28
    pass

# SUBTRACTED: __all__ 四类再导出（deepseek_v4/__init__.py:L34-L39）——
#   子包不在，无物可导；registry 条目对 "vllm.models.deepseek_v4" 的惰性
#   import 在真实仓库命中此文件的三个平台分支
