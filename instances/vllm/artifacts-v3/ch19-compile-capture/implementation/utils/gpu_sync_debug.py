# Subtract-only companion for v3 ch19 — vllm/utils/gpu_sync_debug.py
# (pin v0.27.1 / 6e448d0ea). Kept surface: the sync-check gate global +
# enable_gpu_sync_check (m18 运行期零惊喜三纠察之一的翻转入口——
# compile_or_warm_up_model 尾部调用). SUBTRACTED: with_gpu_sync_check 装饰器
# 与 _install_compile_time_sync_suppressors 的 torch 内部 patch（L38-L120
# ——观测/patch 域，VLLM_GPU_SYNC_CHECK 默认 None 时 enable 即早退 no-op）。
from __future__ import annotations

from .._host_seams import envs

# Global sync-check gate. Off during engine setup (model load, KV cache
# init, warmup/compile) so first-compile and lazy-init syncs pass through;
# flipped on by `enable_gpu_sync_check()` at the end of
# `GPUWorker.compile_or_warm_up_model`, after which `with_gpu_sync_check`-
# decorated functions activate the configured debug mode.
# SOURCE: vllm/utils/gpu_sync_debug.py:L23-L24 _sync_check_enabled（注释块
#   L18-L23 原文如上）
_sync_check_enabled: bool = False


def enable_gpu_sync_check() -> None:  # SOURCE: vllm/utils/gpu_sync_debug.py:L26-L34
    """Flip the sync-check gate on. Call once per worker, after warmup /
    first-compile is complete. No-op unless `VLLM_GPU_SYNC_CHECK` is set."""
    if envs.VLLM_GPU_SYNC_CHECK is None:
        return
    global _sync_check_enabled
    _sync_check_enabled = True
    # SUBTRACTED: _install_compile_time_sync_suppressors()（L33——torch
    #   inductor/aot 入口的 patch 族，观测域）。

# SUBTRACTED: with_gpu_sync_check / gpu_sync_allowed / _install_compile_
#   time_sync_suppressors（L38-L120——装饰器与 patch 域）。
