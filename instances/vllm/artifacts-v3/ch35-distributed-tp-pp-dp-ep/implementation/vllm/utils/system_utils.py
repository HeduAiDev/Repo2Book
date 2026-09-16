# SOURCE: vllm/utils/system_utils.py —— 本章触碰面的真实子集：suppress_stdout /
# get_mp_context / set_process_title。其余（环境变量写回/NUMA/cgroup）不携带。

from __future__ import annotations

import multiprocessing
import os
import sys
from contextlib import contextmanager
from collections.abc import Iterator

import vllm.envs as envs


# SOURCE: vllm/utils/system_utils.py:L62-L88 suppress_stdout — 逐字
@contextmanager
def suppress_stdout():
    """
    Suppress stdout from C libraries at the file descriptor level.

    Only suppresses stdout, not stderr, to preserve error messages.
    Suppression is disabled when VLLM_LOGGING_LEVEL is set to DEBUG.

    Example:
        with suppress_stdout():
            # C library calls that would normally print to stdout
            torch.distributed.new_group(ranks, backend="gloo")
    """
    # SOURCE: vllm/utils/system_utils.py:L62-L91（锚点双置）
    # SUBTRACTED: VLLM_LOGGING_LEVEL==DEBUG 的豁免分支（L74-L77）——seam envs 面
    #   未携带该 flag，恒走压制路径。
    stdout_fd = sys.stdout.fileno()
    stdout_dup = os.dup(stdout_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)

    try:
        sys.stdout.flush()
        os.dup2(devnull_fd, stdout_fd)
        yield
    finally:
        sys.stdout.flush()
        os.dup2(stdout_dup, stdout_fd)
        os.close(stdout_dup)
        os.close(devnull_fd)


# SOURCE: vllm/utils/system_utils.py:L168-L181 get_mp_context —— 逐字 minus
#   _maybe_force_spawn/_sync_visible_devices_env_vars 两个内部修正钩子（L175-L179，
#   平台/可见设备同步域，seam envs 无相应 flag）。
def get_mp_context():
    """Get a multiprocessing context with a particular method (spawn or fork).
    By default we follow the value of the VLLM_WORKER_MULTIPROC_METHOD to
    determine the multiprocessing method (default is fork). However, under
    certain conditions, we may enforce spawn and override the value of
    VLLM_WORKER_MULTIPROC_METHOD.
    """
    # SOURCE: vllm/utils/system_utils.py:L168-L181（锚点双置）
    mp_method = envs.VLLM_WORKER_MULTIPROC_METHOD
    return multiprocessing.get_context(mp_method)


# SOURCE: vllm/utils/system_utils.py envs.VLLM_WORKER_MULTIPROC_METHOD —— HOST SEAM
#   （默认 fork；win32 宿主 spawn——ch09 win32 先例）
envs.VLLM_WORKER_MULTIPROC_METHOD = "spawn"  # HOST SEAM (win32)
envs.VLLM_PROCESS_NAME_PREFIX = ""  # HOST SEAM


# SOURCE: vllm/utils/system_utils.py:L184-L194 set_process_title —— 逐字
def set_process_title(
    name: str,
    suffix: str = "",
    prefix: str = envs.VLLM_PROCESS_NAME_PREFIX,
) -> None:
    """Set the current process title with optional suffix."""
    try:
        import setproctitle
    except ImportError:
        return
    setproctitle.setproctitle(f"{prefix}({os.getpid()}): {name} {suffix}")
