# Subtract-only companion for v3 ch17 — vllm/utils/gpu_sync_debug.py (pin
# v0.27.1 / 6e448d0ea)。运行期 GPU 同步纠察：compile_or_warm_up_model 尾部
# enable_gpu_sync_check() 开闸（启动期不查），此后 @with_gpu_sync_check 装饰
# 的 execute_model / sample_tokens 按已配置模式强制。
#
# SUBTRACTED: _install_compile_time_sync_suppressors（gpu_sync_debug.py:
#   L39-L89——inductor/aot 编译点的同步抑制补丁，ch19 编译域）。
# 宿主走真实代码自带的非 CUDA 分支（L158-L165 no-op）——本模块真实行为。

from __future__ import annotations

import functools
import sys
from contextlib import contextmanager

import torch

from .._host_seams import current_platform, envs

SYNC_ERROR_MESSAGE = (
    "GPU<->CPU sync detected - avoid it or wrap with gpu_sync_allowed()"
)

_GPU_SYNC_ALLOWED_FIRST_SEEN: set[tuple[str, int]] = set()

# Global sync-check gate. Off during engine setup (model load, KV cache
# init, warmup/compile) so first-compile and lazy-init syncs pass through;
# flipped on by `enable_gpu_sync_check()` at the end of
# `GPUWorker.compile_or_warm_up_model`, after which `with_gpu_sync_check`-
# decorated functions activate the configured debug mode.
_sync_check_enabled: bool = False


# SOURCE: vllm/utils/gpu_sync_debug.py:L26-L33 enable_gpu_sync_check
def enable_gpu_sync_check() -> None:
    """Flip the sync-check gate on. Call once per worker, after warmup /
    first-compile is complete. No-op unless `VLLM_GPU_SYNC_CHECK` is set."""
    if envs.VLLM_GPU_SYNC_CHECK is None:
        return
    global _sync_check_enabled
    _sync_check_enabled = True
    # SUBTRACTED: _install_compile_time_sync_suppressors()（L33——inductor
    #   编译点补丁，ch19 域）。

@contextmanager
# SOURCE: vllm/utils/gpu_sync_debug.py:L92-L98 _suppress_gpu_sync_check
def _suppress_gpu_sync_check(prev_mode: int):
    torch.cuda.set_sync_debug_mode(0)
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(prev_mode)

@contextmanager
# SOURCE: vllm/utils/gpu_sync_debug.py:L101-L103 _noop_cm
def _noop_cm():
    yield


if current_platform.is_cuda_alike():

    # SOURCE: vllm/utils/gpu_sync_debug.py:L108-L129 gpu_sync_allowed (CUDA)
    def gpu_sync_allowed(first_only: bool = False):
        """Context manager that suppresses `torch.cuda.set_sync_debug_mode` for the
        duration of the `with` block.

        If `first_only` is True, only the first entry from this call site
        suppresses the sync check; subsequent entries from the same site are
        no-ops so any further GPU syncs will be reported. The "site" is the
        caller's (filename, lineno), so different
        `with gpu_sync_allowed(first_only=True):` lines track independently.
        """
        if envs.VLLM_GPU_SYNC_CHECK is None or torch.compiler.is_compiling():
            return _noop_cm()
        prev_mode = torch.cuda.get_sync_debug_mode()
        if not prev_mode:
            return _noop_cm()
        if first_only:
            frame = sys._getframe(1)
            key = (frame.f_code.co_filename, frame.f_lineno)
            if key in _GPU_SYNC_ALLOWED_FIRST_SEEN:
                return _noop_cm()
            _GPU_SYNC_ALLOWED_FIRST_SEEN.add(key)
        return _suppress_gpu_sync_check(prev_mode)

    # SOURCE: vllm/utils/gpu_sync_debug.py:L131-L156 with_gpu_sync_check (CUDA)
    def with_gpu_sync_check(fn):
        """Decorator that enables `torch.cuda.set_sync_debug_mode` around `fn`
        when `VLLM_GPU_SYNC_CHECK` is set *and* the gate has been flipped by
        `enable_gpu_sync_check()`. Before the gate flips (i.e. during
        engine setup / warmup) the decorated function runs as-is.
        """
        mode = envs.VLLM_GPU_SYNC_CHECK
        if mode is None:
            return fn

        @functools.wraps(fn)
        # SOURCE: (见 impl-notes.md §Source Map——utils/gpu_sync_debug.py)
        def wrapper(*args, **kwargs):
            if not _sync_check_enabled:
                return fn(*args, **kwargs)
            prev_mode = torch.cuda.get_sync_debug_mode()
            torch.cuda.set_sync_debug_mode(mode)
            try:
                return fn(*args, **kwargs)
            except RuntimeError as re:
                if str(re) == "called a synchronizing CUDA operation":
                    raise RuntimeError(SYNC_ERROR_MESSAGE) from re
                raise re
            finally:
                torch.cuda.set_sync_debug_mode(prev_mode)

        return wrapper

else:
    # No-op the methods in non-CUDA cases.

    # SOURCE: vllm/utils/gpu_sync_debug.py:L161-L162 gpu_sync_allowed (non-CUDA)
    def gpu_sync_allowed(first_only: bool = False):
        return _noop_cm()

    # SOURCE: vllm/utils/gpu_sync_debug.py:L164-L165 with_gpu_sync_check (non-CUDA)
    def with_gpu_sync_check(fn):
        return fn
