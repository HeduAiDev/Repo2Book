# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""本章消费的环境变量（源码 envs.py 是懒求值映射，这里收敛成消费面）。

# SOURCE: vllm/envs.py（VLLM_RPC_BASE_PATH / VLLM_P2P_SIDE_CHANNEL_HOST/PORT /
#   VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES 的定义条目）
# SUBTRACTED: 其余 700+ 个环境变量与懒求值 lambda 映射——测试以模块属性覆写
#   代替环境重读（源码里这些值只在构造期读一次，语义等价；ch37 同款删法）。
"""

import os

# SOURCE: vllm/envs.py:VLLM_RPC_BASE_PATH 条目
VLLM_RPC_BASE_PATH: str = os.getenv("VLLM_RPC_BASE_PATH", "/tmp")

# SOURCE: vllm/envs.py:VLLM_P2P_SIDE_CHANNEL_HOST/PORT 条目
VLLM_P2P_SIDE_CHANNEL_HOST: str = os.getenv("VLLM_P2P_SIDE_CHANNEL_HOST", "localhost")
VLLM_P2P_SIDE_CHANNEL_PORT: int = int(os.getenv("VLLM_P2P_SIDE_CHANNEL_PORT", "5710"))

# SOURCE: vllm/envs.py:VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES 条目
VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES: bool = (
    os.getenv("VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES", "0").lower() in ("1", "true")
)

__all__ = [
    "VLLM_RPC_BASE_PATH",
    "VLLM_P2P_SIDE_CHANNEL_HOST",
    "VLLM_P2P_SIDE_CHANNEL_PORT",
    "VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES",
]
