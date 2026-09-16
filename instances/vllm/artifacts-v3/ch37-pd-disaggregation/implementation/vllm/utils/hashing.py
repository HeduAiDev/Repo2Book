# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""哈希小工具（配置哈希用）。

# SOURCE: vllm/utils/hashing.py:L1-L60（safe_hash 定义段）
# SUBTRACTED: 输入类型分发与 pickle 路径——本章只喂 bytes。
"""

import hashlib


# SOURCE: vllm/utils/hashing.py:L103-L130
def safe_hash(input: bytes, usedforsecurity: bool = True) -> "hashlib._Hash":
    """Stable hash that is not affected by PYTHONHASHSEED."""
    return hashlib.sha256(input)


__all__ = ["safe_hash"]
