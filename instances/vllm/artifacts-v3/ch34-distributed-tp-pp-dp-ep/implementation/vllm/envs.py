# SOURCE: vllm/envs.py —— HOST SEAM：精简版触碰的 flag 面（默认值对 pin v0.27.1）。
# 真实文件是 env-backed descriptor 表；seam 以模块级属性 stand-in（消费面是
# `import vllm.envs as envs` 后的属性访问）。

from __future__ import annotations

# SOURCE: vllm/envs.py VLLM_ALLREDUCE_USE_SYMM_MEM
VLLM_ALLREDUCE_USE_SYMM_MEM: bool = False
# SOURCE: vllm/envs.py VLLM_ALLREDUCE_USE_FLASHINFER
VLLM_ALLREDUCE_USE_FLASHINFER: bool = False
# SOURCE: vllm/envs.py VLLM_BATCH_INVARIANT
VLLM_BATCH_INVARIANT: bool = False
# SOURCE: vllm/envs.py LOCAL_RANK（torchrun/env:// 启动器注入）
LOCAL_RANK: int = 0
# SUBTRACTED: 其余 flag（safety/deprecated/观测面）——精简版不触达。


# SOURCE: vllm/envs.py enable_envs_cache — 真实表冻结钩子；seam 无表可冻。
def enable_envs_cache() -> None:  # HOST SEAM
    return None
