# SOURCE: vllm/utils/gpu_sync_debug.py
# 同步禁区 tripwire（m19）：VLLM_GPU_SYNC_CHECK 开启 + warmup 完成后
# enable_gpu_sync_check() 翻闸，execute_model/sample_tokens 被 with_gpu_sync_check
# 包住——任何 CPU↔GPU sync 直接抛。纪律从注释升级为运行期纠察。
# HOST 侧（CPU host，无 CUDA）：真实文件的 `else` 分支（L158-L165——非 CUDA
# 平台 no-op）即本机语义；CUDA 分支（L106-L156 set_sync_debug_mode 纠察）保留
# 注释存证——容器内（scripts/vllm_docker.sh）真 CUDA 环境由真 vllm 接管。
import functools
import os
from contextlib import contextmanager

# SOURCE: vllm/utils/gpu_sync_debug.py:L12-L14 SYNC_ERROR_MESSAGE（逐字）
SYNC_ERROR_MESSAGE = (
    "GPU<->CPU sync detected - avoid it or wrap with gpu_sync_allowed()"
)

# Global sync-check gate. Off during engine setup (model load, KV cache
# init, warmup/compile) so first-compile and lazy-init syncs pass through;
# flipped on by `enable_gpu_sync_check()` at the end of
# `GPUWorker.compile_or_warm_up_model`, after which `with_gpu_sync_check`-
# decorated functions activate the configured debug mode.
# SOURCE: vllm/utils/gpu_sync_debug.py:L23 _sync_check_enabled
_sync_check_enabled: bool = False


# SOURCE: vllm/utils/gpu_sync_debug.py:L26 enable_gpu_sync_check
def enable_gpu_sync_check() -> None:
    """Flip the sync-check gate on. Call once per worker, after warmup /
    first-compile is complete. No-op unless `VLLM_GPU_SYNC_CHECK` is set."""
    # SOURCE: vllm/utils/gpu_sync_debug.py:L29-L33
    if os.environ.get("VLLM_GPU_SYNC_CHECK") is None:
        return
    global _sync_check_enabled
    _sync_check_enabled = True
    # SUBTRACTED: _install_compile_time_sync_suppressors（L33/L39-L89——
    #   torch inductor 编译期抑制，CUDA 编译域）。


# SOURCE: vllm/utils/gpu_sync_debug.py:L101-L102 _noop_cm
@contextmanager
def _noop_cm():
    # SOURCE: vllm/utils/gpu_sync_debug.py:L102
    yield


# SOURCE: vllm/utils/gpu_sync_debug.py:L158-L165 非 CUDA 分支（HOST 侧即本机
# 语义：no-op；CUDA 分支 L106-L156 的 set_sync_debug_mode 纠察在真 GPU 环境
# 由真 vllm 执行——容器内）
# No-op the methods in non-CUDA cases.

# SOURCE: vllm/utils/gpu_sync_debug.py:L161-L162 gpu_sync_allowed（非 CUDA）
def gpu_sync_allowed(first_only: bool = False):
    # SOURCE: vllm/utils/gpu_sync_debug.py:L162
    return _noop_cm()


# SOURCE: vllm/utils/gpu_sync_debug.py:L164-L165 with_gpu_sync_check（非 CUDA）
def with_gpu_sync_check(fn):
    # SOURCE: vllm/utils/gpu_sync_debug.py:L165
    return fn
