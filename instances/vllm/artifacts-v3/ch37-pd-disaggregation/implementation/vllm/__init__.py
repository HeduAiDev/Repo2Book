# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM 包根：本章只保留兼容 hash 需要的版本号。

# SOURCE: vllm/__init__.py:L1-L60（__version__ 定义段）
# SUBTRACTED: 包级重导出（EngineArgs/LLM/AsyncLLM... 全部入口类）——本章不跑真引擎；
#   保留 __version__ 是因为 compute_nixl_compatibility_hash 把它算进兼容 hash：
#   版本一变，握手就该炸（metadata.py:L114-L116）。
"""

__version__ = "0.27.1"

__all__ = ["__version__"]
