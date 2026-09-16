# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""本章用到的 vLLM 环境变量（源码 envs.py 是懒求值映射，这里收敛成本章消费面）。

# SOURCE: vllm/envs.py:L211-L212 + L1591-L1596（VLLM_NIXL_SIDE_CHANNEL_HOST/PORT）
# SUBTRACTED: 其余 700+ 个环境变量——它们服务的是引擎其它子系统。
"""

import os

# SOURCE: vllm/envs.py:L211-L212
VLLM_NIXL_SIDE_CHANNEL_HOST: str = os.getenv(
    "VLLM_NIXL_SIDE_CHANNEL_HOST", "localhost"
)
VLLM_NIXL_SIDE_CHANNEL_PORT: int = int(os.getenv("VLLM_NIXL_SIDE_CHANNEL_PORT", "5600"))

# SOURCE: vllm/envs.py:L1595-L1596
# SUBTRACTED: 懒求值 lambda 映射（源码每读一次都重新查环境）——本章在 import 期定值，
#   测试用 `vllm.envs.VLLM_NIXL_SIDE_CHANNEL_PORT = <port>` 覆写模块属性（源码里
#   side channel 端口只在 connector 构造时读一次，语义等价）。

__all__ = ["VLLM_NIXL_SIDE_CHANNEL_HOST", "VLLM_NIXL_SIDE_CHANNEL_PORT"]
