# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""配置层小工具：`@config` 装饰器与 hash_factors。

# SOURCE: vllm/config/utils.py:L1-L120（@config 装饰器）+ L381-L383（hash_factors）
# SUBTRACTED: 装饰器内部的增删字段校验/日志（L40-L200 的 schema 比对）——本章
#   的 KVTransferConfig 是静态字段集，不跑配置迁移检查。
"""

import hashlib
import json
from typing import Any, TypeVar

_T = TypeVar("_T")


# SOURCE: vllm/config/utils.py:L41-L200（重载声明 + 实现）
# SUBTRACTED: 字段增删比对与自动日志——本章只借它统一 dataclass 语义
#   （源码里 KVTransferConfig 靠它获得 dataclass 行为 + hash 接口）。
# SOURCE: vllm/config/utils.py:L41-L200（重载声明 + 实现）
def config(cls: _T) -> _T:
    # SOURCE: vllm/config/utils.py:L41-L200（重载声明 + 实现）
    """Mark a class as a vLLM config（源码语义：dataclass + 校验钩子）。"""
    from dataclasses import dataclass

    return dataclass(cls)  # type: ignore[return-value]


# SOURCE: vllm/config/utils.py:L381-L383
def hash_factors(items: dict[str, object]) -> str:
    """Return a SHA-256 hex digest of the canonical items structure."""
    return hashlib.sha256(json.dumps(items, sort_keys=True).encode()).hexdigest()


__all__ = ["config", "hash_factors"]
