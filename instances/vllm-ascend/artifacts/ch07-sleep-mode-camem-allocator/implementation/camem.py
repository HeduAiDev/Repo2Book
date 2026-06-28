"""显存底座：sleep-mode 与 CANN 虚拟内存分配器（camem）—— 只做减法的忠实精简版。

控制流与 vllm_ascend/device_allocator/camem.py 逐行一致（它本身又是 vLLM
vllm/device_allocator/cumem.py 的「同构换符号」移植）。本精简版只做两处减法：
  ① 删除文件头 Apache-2.0 许可证块（原 camem.py:L1-L16）；
  ② host 无 CANN/vllm，把两个无法在 host 导入的模块级符号降级为忠实占位
     （acl.rt.memcpy → 占位 def memcpy；vllm.logger.logger → stdlib logging），
     其余一字不改。

host 可读可跑的是「纯 Python 状态机」：单例 + pointer_to_data 账本 + current_tag
打标签 + sleep/wake_up 的 offload/discard 路由。真正调 vllm_ascend_C 的虚拟内存
映射（create_and_map/unmap_and_release）与 aclrtMemcpy 需要 NPU/CANN，不在 host
跑——测试用 monkeypatch 把这几个模块级符号换成记录器后驱动状态机。
"""
# SUBTRACTED: 文件头 Apache-2.0 许可证注释块（原 camem.py:L1-L16）
# SUBTRACTED: CANN-mem-based pytorch pluggable allocator 顶部注释（原 camem.py:L17-L18）
import dataclasses
import gc
import os
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import torch

# SUBTRACTED: from acl.rt import memcpy（原 camem.py:L27）——host 无 CANN，给出忠实占位。
#   memcpy 是 must_keep：它是与 cudaMemcpy 的关键差异点（多 destMax 上界 + 显式 kind
#   方向枚举 ACL_MEMCPY_DEVICE_TO_HOST/HOST_TO_DEVICE）。sleep/wake_up 按原样调用它，
#   测试 monkeypatch camem.memcpy 来驱动状态机。
def memcpy(dst, dest_max, src, count, kind):  # type: ignore # noqa: F811
    # SOURCE: vllm_ascend/device_allocator/camem.py:L27 (from acl.rt import memcpy, 占位)
    raise RuntimeError("acl.rt.memcpy: host 无 CANN（精简版占位，原为 from acl.rt import memcpy）")


# SUBTRACTED: from vllm.logger import logger（原 camem.py:L28）——用 stdlib logging 顶替（host 无 vllm）。
import logging

logger = logging.getLogger(__name__)


def find_loaded_library(lib_name) -> str | None:
    # SOURCE: vllm_ascend/device_allocator/camem.py:L31-L53
    """
    According to according to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    a loaded library.
    """  # noqa
    found_line = None
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found_line = line
                break
    if found_line is None:
        # the library is not loaded in the current process
        return None
    # if lib_name is libcudart, we need to match a line with:
    # address /path/to/libcudart-hash.so.11.0
    start = found_line.index("/")
    path = found_line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(lib_name), f"Unexpected filename: {filename} for library {lib_name}"
    return path


# SOURCE: vllm_ascend/device_allocator/camem.py:L56-L72
# 条件导入 vllm_ascend_C 的三个 C 入口 + camem_available 开关。host 上 vllm_ascend_C
# 不存在 → ImportError → 走 except 兜底（init_module=None、camem_available=False），
# 这正是真实源码的「导入失败则禁用 sleep mode」降级路径，原样保留。
camem_available = False
try:
    from vllm_ascend.vllm_ascend_C import (  # type: ignore # noqa: F401
        init_module,
        python_create_and_map,
        python_unmap_and_release,
    )

    lib_name = find_loaded_library("vllm_ascend_C")
    camem_available = True
except ImportError as e:
    logger.warning("Failed to import vllm_ascend_C:%s. Sleep mode will be disabled. ", e)
    init_module = None
    python_create_and_map = None
    python_unmap_and_release = None
    lib_name = None
    # 保留：libcudart = None 是从 vLLM cumem.py 移植时残留的死变量（全文再无 libcudart
    # 引用）。原样保留以如实展示「移植=换符号」时未清理干净的痕迹。
    libcudart = None

# py_device, py_alignedSize, py_d_mem, py_p_memHandle
HandleType = tuple[int, int, int, int]


@dataclasses.dataclass
class AllocationData:
    # SOURCE: vllm_ascend/device_allocator/camem.py:L78-L82
    handle: HandleType
    tag: str
    cpu_backup_tensor: torch.Tensor | None = None


def create_and_map(allocation_handle: HandleType) -> None:
    # SOURCE: vllm_ascend/device_allocator/camem.py:L85-L86
    python_create_and_map(*allocation_handle)


def unmap_and_release(allocation_handle: HandleType) -> None:
    # SOURCE: vllm_ascend/device_allocator/camem.py:L89-L90
    python_unmap_and_release(*allocation_handle)


def get_pluggable_allocator(
    # SOURCE: vllm_ascend/device_allocator/camem.py:L93-L99
    python_malloc_fn: Callable[[tuple[int, int, int, int]], None],
    python_free_func: Callable[[int], tuple[int, int, int, int]],
) -> "torch.npu.memory.NPUPluggableAllocator":
    init_module(python_malloc_fn, python_free_func)
    new_alloc = torch.npu.memory.NPUPluggableAllocator(lib_name, "my_malloc", "my_free")
    return new_alloc


@contextmanager
def use_memory_pool_with_allocator(
    # SOURCE: vllm_ascend/device_allocator/camem.py:L102-L110
    python_malloc_fn: Callable[[tuple[int, int, int, int]], None],
    python_free_func: Callable[[int], tuple[int, int, int, int]],
):
    new_alloc = get_pluggable_allocator(python_malloc_fn, python_free_func)
    mem_pool = torch.npu.memory.MemPool(new_alloc._allocator)
    with torch.npu.memory.use_mem_pool(mem_pool):
        yield mem_pool, new_alloc


class CaMemAllocator:
    # SOURCE: vllm_ascend/device_allocator/camem.py:L113-L273
    """
    A singleton class that manages a memory pool for CANN tensors.
    The memory in this pool can be offloaded or discarded when the
    allocator sleeps.
    Inside the `use_memory_pool(tag)` context, all tensors created will
    be allocated in the memory pool, and has the same tag as the
    tag passed to the context.
    When we call `sleep`, all tensors with the specified tag will be
    offloaded to CPU memory, and the rest of the tensors will be discarded.
    When we call `wake_up`, all tensors that are previously offloaded
    will be loaded back to GPU memory, and the rest of the tensors will
    have empty memory.
    Why it needs to be a singleton?
    When allocated tensors are garbage collected, PyTorch will call
    the free callback, which will call the `python_free_callback` method.
    The C-extension uses a global variable to store the function of an
    instance of this class. If we create multiple instances of this class,
    the global variable will be overwritten and the free callback will
    not work as expected.
    """

    instance = None
    default_tag: str = "default"

    @staticmethod
    def get_instance() -> "CaMemAllocator":
        # SOURCE: vllm_ascend/device_allocator/camem.py:L138-L147
        """
        CaMemAllocator is a singleton class.
        We cannot call the constructor directly.
        Call this method to get the instance.
        """
        if CaMemAllocator.instance is None:
            CaMemAllocator.instance = CaMemAllocator()
        return CaMemAllocator.instance

    def __init__(self):
        # SOURCE: vllm_ascend/device_allocator/camem.py:L149-L159
        conf = os.environ.get("PYTORCH_NPU_ALLOC_CONF", "")
        assert "expandable_segments:True" not in conf, (
            "Expandable segments are not compatible with memory pool. "
            "Please track https://github.com/pytorch/pytorch/issues/147851 "
            "for the latest updates."
        )

        self.pointer_to_data: dict[int, AllocationData] = {}
        self.current_tag: str = CaMemAllocator.default_tag
        self.allocator_and_pools: dict[str, Any] = {}

    def python_malloc_callback(self, allocation_handle: HandleType) -> None:
        # SOURCE: vllm_ascend/device_allocator/camem.py:L161-L167
        """
        Internal method to store the allocation data
        when memory is allocated in the memory pool."""
        py_d_mem = allocation_handle[2]
        self.pointer_to_data[py_d_mem] = AllocationData(allocation_handle, self.current_tag)
        return

    def python_free_callback(self, ptr: int) -> HandleType:
        # SOURCE: vllm_ascend/device_allocator/camem.py:L169-L176
        """
        Internal method to look up the allocation data
        when memory is freed in the memory pool."""
        data = self.pointer_to_data.pop(ptr)
        if data.cpu_backup_tensor is not None:
            data.cpu_backup_tensor = None
        return data.handle

    def sleep(self, offload_tags: tuple[str, ...] | str | None = None) -> None:
        # SOURCE: vllm_ascend/device_allocator/camem.py:L178-L208
        """
        Put the allocator in sleep mode.
        All data in the memory allocation with the specified tag will be
        offloaded to CPU memory, and others will be discarded.
        :param offload_tags: The tags of the memory allocation that will be
            offloaded. The rest of the memory allocation will be discarded.
        """
        if offload_tags is None:
            # by default, allocated tensors are offloaded
            # when the allocator sleeps
            offload_tags = (CaMemAllocator.default_tag,)
        elif isinstance(offload_tags, str):
            offload_tags = (offload_tags,)

        assert isinstance(offload_tags, tuple)

        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            if data.tag in offload_tags:
                size_in_bytes = handle[1]
                cpu_backup_tensor = torch.empty(size_in_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
                cpu_ptr = cpu_backup_tensor.data_ptr()
                ACL_MEMCPY_DEVICE_TO_HOST = 2
                dest_max = cpu_ptr + size_in_bytes * 2
                memcpy(cpu_ptr, dest_max, ptr, size_in_bytes, ACL_MEMCPY_DEVICE_TO_HOST)
                data.cpu_backup_tensor = cpu_backup_tensor
            unmap_and_release(handle)

        gc.collect()
        torch.npu.empty_cache()

    def wake_up(self, tags: list[str] | None = None) -> None:
        # SOURCE: vllm_ascend/device_allocator/camem.py:L210-L227
        """
        Wake up the allocator from sleep mode.
        All data that is previously offloaded will be loaded back to GPU
        memory, and the rest of the data will have empty memory."""
        for ptr, data in self.pointer_to_data.items():
            if tags is None or data.tag in tags:
                handle = data.handle
                create_and_map(handle)
                if data.cpu_backup_tensor is not None:
                    cpu_backup_tensor = data.cpu_backup_tensor
                    if cpu_backup_tensor is not None:
                        size_in_bytes = cpu_backup_tensor.numel() * cpu_backup_tensor.element_size()
                        cpu_ptr = cpu_backup_tensor.data_ptr()
                        ACL_MEMCPY_HOST_TO_DEVICE = 1
                        dest_max = ptr + size_in_bytes * 2
                        memcpy(ptr, dest_max, cpu_ptr, size_in_bytes, ACL_MEMCPY_HOST_TO_DEVICE)
                        data.cpu_backup_tensor = None

    @contextmanager
    def use_memory_pool(self, tag: str | None = None):
        # SOURCE: vllm_ascend/device_allocator/camem.py:L229-L263
        """
        A context manager to use the memory pool.
        All memory allocation created inside the context will be allocated
        in the memory pool, and has the specified tag.
        :param tag: The tag of the memory allocation. If None, the default tag
            will be used.
        """
        if tag is None:
            tag = CaMemAllocator.default_tag

        assert isinstance(tag, str)

        old_tag = self.current_tag
        self.current_tag = tag
        with use_memory_pool_with_allocator(self.python_malloc_callback, self.python_free_callback) as data:
            # start to hit another PyTorch bug in PyTorch 2.6,
            # possibly because of gc-related issue w.r.t. the allocator and
            # the memory pool.
            # to avoid the issue, we keep a reference of the data.
            # see https://github.com/pytorch/pytorch/issues/146431 .
            self.allocator_and_pools[tag] = data
            yield
            # PyTorch's bug, calling torch.cuda.empty_cache() will error
            # when using pluggable allocator, see
            # https://github.com/pytorch/pytorch/issues/145168 .
            # if we have some memory allocated and then freed,
            # the memory will not be released.
            # right now it is fine, because we only use this allocator
            # during weight loading and kv cache creation, where we only
            # allocate memory.
            # TODO: we need to find a way to release the memory,
            # i.e. calling torch.cuda.empty_cache()
            self.current_tag = old_tag

    def get_current_usage(self) -> int:
        # SOURCE: vllm_ascend/device_allocator/camem.py:L265-L273
        """
        Get the total number of bytes allocated in the memory pool.
        """
        sum_bytes: int = 0
        for ptr, data in self.pointer_to_data.items():
            handle = data.handle
            sum_bytes += handle[1]
        return sum_bytes
