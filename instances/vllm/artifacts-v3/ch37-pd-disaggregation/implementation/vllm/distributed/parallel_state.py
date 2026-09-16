# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""并行状态（本章只取 TP 身份两问）。

NIXL worker 用 tp_rank / tp_world_size 决定「本地 rank 向远端哪些 rank 收发」，
以及回执里 `tp_size` 的取值。

# SOURCE: vllm/distributed/parallel_state.py:L1-L1970
# SUBTRACTED: 全部集合通信组（GroupCoordinator / NCCL / 图捕获集成 / 5 维 rank
#   张量构造）——那是 ch35 的面；本章的传输由 NIXL（替身）承担，不经 torch 集合通信。
"""

# SOURCE: vllm/distributed/parallel_state.py:L1600-L1660（init_distributed_environment
#   建立的全局 group 状态）
_TP_RANK: int = 0
_TP_WORLD_SIZE: int = 1


# SOURCE: vllm/distributed/parallel_state.py:L2092-L2094
def get_tensor_model_parallel_rank() -> int:
    """Return my rank for the tensor model parallel group."""
    return _TP_RANK


# SOURCE: vllm/distributed/parallel_state.py:L2087-L2089
def get_tensor_model_parallel_world_size() -> int:
    """Return world size for the tensor model parallel group."""
    return _TP_WORLD_SIZE


# SOURCE: vllm/distributed/parallel_state.py:L1596-L1660（建组时写下这两值）
# SEAM: 真源码由 init_distributed_environment 建组时写下这两个值；本章的两台
#   引擎都在同一进程，故提供显式设置入口（测试装配用；生产路径里值来自环境初始化）。
# SOURCE: vllm/distributed/parallel_state.py:L1596-L1660（建组时写下这两值）
def set_tensor_model_parallel_state(rank: int = 0, world_size: int = 1) -> None:
    # SOURCE: vllm/distributed/parallel_state.py:L1596-L1660（建组时写下这两值）
    """SEAM 入口：设定本章进程的 TP 身份（rank/world_size）。"""
    global _TP_RANK, _TP_WORLD_SIZE
    _TP_RANK = rank
    _TP_WORLD_SIZE = world_size


__all__ = [
    "get_tensor_model_parallel_rank",
    "get_tensor_model_parallel_world_size",
    "set_tensor_model_parallel_state",
]
