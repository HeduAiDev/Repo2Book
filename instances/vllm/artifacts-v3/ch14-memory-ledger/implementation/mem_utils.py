# SOURCE: vllm/utils/mem_utils.py
# 显存测量的三件套（m1/m2）：MemorySnapshot（六类读数快照）、
# MemoryProfilingResult（峰值账载体）、memory_profiling（前后测差值的
# 上下文管理器）——non_kv_cache_memory = total_consumed +
# transient_peak_headroom（L317-L326）。format_gib 是 request_memory 报错
# 文案的读数格式化（mem_constants 的 KiB/MiB/GiB 折入，L17 引用面）。
# HOST SEAM：设备读数（memory_stats/get_memory_info/memory_reserved）是
#   torch.accelerator 的 CUDA 面——host 测试经 monkeypatch 注入
#   MemorySnapshot.measure / torch.accelerator（见 tests）；容器内真跑。
# SUBTRACTED（dossier.delete 批准项的落点 + 本章链路不用）：
#   psutil/CPU 面（get_cpu_memory L46-L48、release_device_memory_under_
#   pressure L51-L83——UMA 压力释放，平台域）；get_max_shared_memory_bytes
#   （L34-L43——attention kernel 共享内存，→ ch21）；DeviceMemoryProfiler
#   （L86-L105——旧版测量器，v1 主线用 memory_profiling）；
#   current_platform 依赖（device 解析与 UMA 分支 L125-L130、L148-L155
#   ——平台域；切面 device 直供、非 torch 差值走快照字段相减）。
import contextlib
import gc
import time
from collections.abc import Generator
from dataclasses import dataclass, field

import torch
import torch.types

# SOURCE: vllm/utils/mem_constants.py（折入：KiB/MiB/GiB 常量，mem_utils
#   L17 的引用面）
KiB_bytes = 1 << 10
MiB_bytes = 1 << 20
GiB_bytes = 1 << 30


# SOURCE: vllm/utils/mem_utils.py:L30 format_gib
def format_gib(b: int) -> str:
    # SOURCE: vllm/utils/mem_utils.py:L31
    return f"{round(b / GiB_bytes, 2)}"


# SOURCE: vllm/utils/mem_utils.py:L108 MemorySnapshot
@dataclass
class MemorySnapshot:
    """Memory snapshot."""

    # SOURCE: vllm/utils/mem_utils.py:L112-L119 六类读数字段
    torch_peak: int = 0
    torch_allocated: int = 0
    free_memory: int = 0
    total_memory: int = 0
    cuda_memory: int = 0
    torch_memory: int = 0
    non_torch_memory: int = 0
    timestamp: float = 0.0

    device: torch.types.Device = None
    auto_measure: bool = True

    # SOURCE: vllm/utils/mem_utils.py:L124 __post_init__
    def __post_init__(self) -> None:
        # SUBTRACTED: current_platform.current_device 的平台解析
        #   （L125-L128——平台域；切面 device 直供）
        self.device_ = torch.device(self.device)
        if self.auto_measure:
            self.measure()

    # SOURCE: vllm/utils/mem_utils.py:L135 measure
    def measure(self) -> None:
        # HOST SEAM：CUDA 设备读数面——容器内真跑；host 测试注入。
        device = self.device_

        # we measure the torch peak memory usage via allocated_bytes,
        # rather than `torch.accelerator.memory_reserved()` .
        # After `torch.accelerator.reset_peak_memory_stats()`,
        # `torch.accelerator.memory_reserved()` will keep growing, and only shrink
        # when we call `torch.accelerator.empty_cache()` or OOM happens.
        # SOURCE: vllm/utils/mem_utils.py:L143-L147
        stats = torch.accelerator.memory_stats(device)
        self.torch_peak = stats.get("allocated_bytes.all.peak", 0)
        self.torch_allocated = stats.get("allocated_bytes.all.current", 0)

        self.free_memory, self.total_memory = torch.accelerator.get_memory_info(device)
        # SUBTRACTED: is_integrated_gpu（UMA）psutil 修正分支（L148-L155
        #   ——平台域，GH200/DGX Spark 类集成显存）

        # SOURCE: vllm/utils/mem_utils.py:L157
        self.cuda_memory = self.total_memory - self.free_memory

        # torch.accelerator.memory_reserved() is how many bytes
        # PyTorch gets from cuda (by calling cudaMalloc, etc.)
        # this is used to measure the non-torch memory usage
        # SOURCE: vllm/utils/mem_utils.py:L162
        self.torch_memory = torch.accelerator.memory_reserved(device)

        # SOURCE: vllm/utils/mem_utils.py:L164
        self.non_torch_memory = self.cuda_memory - self.torch_memory
        self.timestamp = time.time()

    # SOURCE: vllm/utils/mem_utils.py:L167 __sub__
    def __sub__(self, other: "MemorySnapshot") -> "MemorySnapshot":
        # SOURCE: vllm/utils/mem_utils.py:L168-L172
        if self.device_ != other.device_:
            raise ValueError(
                "The two snapshots should be from the same device! "
                f"Found: {self.device_} vs. {other.device_}"
            )

        # SOURCE: vllm/utils/mem_utils.py:L174-L185
        return MemorySnapshot(
            torch_peak=self.torch_peak - other.torch_peak,
            torch_allocated=self.torch_allocated - other.torch_allocated,
            free_memory=self.free_memory - other.free_memory,
            total_memory=self.total_memory - other.total_memory,
            cuda_memory=self.cuda_memory - other.cuda_memory,
            torch_memory=self.torch_memory - other.torch_memory,
            non_torch_memory=self.non_torch_memory - other.non_torch_memory,
            timestamp=self.timestamp - other.timestamp,
            device=self.device_,
            auto_measure=False,
        )

    # SUBTRACTED: __repr__（L187-L198——观测面，dossier.delete 第 9 条
    #   观测旁路的同族；读数经 format_gib 已在报错文案可见）。


# SOURCE: vllm/utils/mem_utils.py:L201 MemoryProfilingResult
@dataclass
class MemoryProfilingResult:
    """Memory profiling result. All numbers are in bytes."""

    # SOURCE: vllm/utils/mem_utils.py:L205-L212
    non_kv_cache_memory: int = 0
    torch_peak_increase: int = 0
    non_torch_increase: int = 0
    total_consumed: int = 0
    transient_peak_headroom: int = 0
    weights_memory: int = 0
    before_create: MemorySnapshot = field(default_factory=MemorySnapshot)
    profile_time: float = 0.0

    # SOURCE: vllm/utils/mem_utils.py:L214 __post_init__
    def __post_init__(self) -> None:
        # SOURCE: vllm/utils/mem_utils.py:L215-L218
        device = self.before_create.device_

        self.before_profile = MemorySnapshot(device=device, auto_measure=False)
        self.after_profile = MemorySnapshot(device=device, auto_measure=False)

    # SUBTRACTED: __repr__（L220-L230——观测面，同上）。


# SOURCE: vllm/utils/mem_utils.py:L233 memory_profiling
@contextlib.contextmanager
def memory_profiling(
    baseline_snapshot: MemorySnapshot,
    weights_memory: int = 0,
) -> Generator[MemoryProfilingResult, None, None]:
    """
    Memory profiling context manager.

    baseline_snapshot: the memory snapshot before the current vLLM instance.
    weights_memory: memory used by PyTorch when loading the model weights.
        Note that, before loading the model weights, we also initialize the device
        and distributed environment, which may consume some memory. This part is not
        included in the weights_memory because PyTorch does not control it.

    The memory in one GPU can be classified into 3 categories:
    1. memory used by anything other than the current vLLM instance.
    2. memory used by torch in the current vLLM instance.
    3. memory used in the current vLLM instance, but not by torch.

    A quantitive example:

    Before creating the current vLLM instance:
        category 1: 1 GiB
        category 2: 0 GiB
        category 3: 0 GiB

    After creating the current vLLM instance and loading the model,
    (i.e., before profiling):
        category 1: 1 GiB
        category 2: 2 GiB (model weights take 2 GiB)
        category 3: 0.5 GiB (memory used by NCCL)

    During profiling (peak):
        category 1: 1 GiB
        category 2: 4 GiB (peak activation tensors take 2 GiB)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    After profiling:
        category 1: 1 GiB
        category 2: 3 GiB (after garbage-collecting activation tensors)
        category 3: 1 GiB (memory used by NCCL + buffers for some attention backends)

    In this case, non-kv cache takes 5 GiB in total, including:
    a. 2 GiB used by the model weights (category 2)
    b. 2 GiB reserved for the peak activation tensors (category 2)
    c. 1 GiB used by non-torch components (category 3)

    The memory used for loading weights (a.) is directly given from the
    argument `weights_memory`.

    The increase of `torch.accelerator.memory_stats()["allocated_bytes.all.peak"]`
    during profiling gives (b.).

    The increase of `non_torch_memory` from creating the current vLLM instance
    until after profiling to get (c.).
    """
    # SOURCE: vllm/utils/mem_utils.py:L289-L291（前测：GC + 清缓存 + 重置峰值）
    gc.collect()
    torch.accelerator.empty_cache()
    torch.accelerator.reset_peak_memory_stats(baseline_snapshot.device_)

    # SOURCE: vllm/utils/mem_utils.py:L293-L299
    result = MemoryProfilingResult(
        before_create=baseline_snapshot,
        # the part of memory used for holding the model weights
        weights_memory=weights_memory,
    )

    result.before_profile.measure()

    yield result

    # SOURCE: vllm/utils/mem_utils.py:L303-L306（后测）
    gc.collect()
    torch.accelerator.empty_cache()

    result.after_profile.measure()

    # SOURCE: vllm/utils/mem_utils.py:L308-L312
    diff_profile = result.after_profile - result.before_profile
    diff_from_create = result.after_profile - result.before_create
    result.torch_peak_increase = diff_profile.torch_peak
    result.non_torch_increase = diff_from_create.non_torch_memory
    result.profile_time = diff_profile.timestamp

    # Measure total consumption via mem_get_info() instead of
    # memory_reserved(), which goes negative when pluggable allocators
    # (e.g. cumem) bypass PyTorch's tracking.
    # SOURCE: vllm/utils/mem_utils.py:L317-L319
    result.total_consumed = (
        result.before_create.free_memory - result.after_profile.free_memory
    )

    # total_consumed already covers persistent torch allocations; add only the
    # transient peak headroom to avoid double-counting.
    # SOURCE: vllm/utils/mem_utils.py:L323-L325
    result.transient_peak_headroom = (
        result.after_profile.torch_peak - result.after_profile.torch_allocated
    )
    # SOURCE: vllm/utils/mem_utils.py:L326
    result.non_kv_cache_memory = result.total_consumed + result.transient_peak_headroom
