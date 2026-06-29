"""测试接缝桩（NOT subtract-only）—— 把 KV 卸载链路从 vllm / torch_npu 拉取的运行期
符号，在 host（无 NPU/CANN）上接住，让纯 Python 控制流（分层搬运节拍 / block 视图重建 /
DMA 拷贝调度 / 指针算术）可跑可断言。

边界（dossier 明示）：昇腾代码 host 无 NPU/CANN 不可跑——『分层搬运节拍 / block 视图重建 /
DMA 拷贝调度是纯 Python，可跑；实际 device↔host 搬运不真跑』。因此：
  * torch.npu.Stream/Event/stream/current_stream/set_device → 由本桩补丁到真 torch 上（no-op）。
  * torch.ops._C_ascend.swap_blocks_batch → 记录调用参数到 SWAP_CALLS（不真搬字节），
    供测试断言指针布局 (base + block_id * bytes_per_block) 与方向码。
  * 基座 vllm.v1.kv_offload.* / vllm.v1.simple_kv_offload.* 的抽象基类 → 极简 stub 接住。
凡指针算术 / stride/shape 取尺寸 / set_() 重建视图 / numpy 广播都走真 torch，可跑可验。
每个替身都标 # SOURCE 指向它顶替的真实符号。
"""
from __future__ import annotations

import contextlib
import types
from typing import NamedTuple

import torch


# ----------------------------------------------------------------------
# torch.npu host 补丁（record/query/synchronize/elapsed_time 均为 host no-op）
# ----------------------------------------------------------------------
class _NpuStream:  # SOURCE: torch.npu.Stream（host 无 NPU：仅作流序编排占位）
    def wait_stream(self, other: "_NpuStream") -> None:  # SOURCE: torch.npu.Stream.wait_stream
        # 真机上让本流等另一条流的所有已入队任务完成；host 上无设备队列，no-op。
        pass

    def wait_event(self, event: "_NpuEvent") -> None:  # SOURCE: torch.npu.Stream.wait_event
        # 真机上让本流等某个 Event 触发；host no-op。
        pass


class _NpuEvent:  # SOURCE: torch.npu.Event（host 无 NPU：搬运视为即时完成）
    def __init__(self, enable_timing: bool = False) -> None:  # SOURCE: torch.npu.Event.__init__
        self.enable_timing = enable_timing
        self._recorded = False

    def record(self, stream: "_NpuStream | None" = None) -> None:  # SOURCE: torch.npu.Event.record
        self._recorded = True

    def query(self) -> bool:  # SOURCE: torch.npu.Event.query
        # host 无异步设备：一旦 record 过即视为完成（演示『发了不等→轮询完成』节拍）。
        return self._recorded

    def synchronize(self) -> None:  # SOURCE: torch.npu.Event.synchronize
        pass

    def elapsed_time(self, other: "_NpuEvent") -> float:  # SOURCE: torch.npu.Event.elapsed_time
        # 真机返回两 Event 间毫秒数；host 给固定占位值，不影响控制流。
        return 1.0


_DEFAULT_STREAM = _NpuStream()


@contextlib.contextmanager
def _npu_stream_ctx(stream: _NpuStream):  # SOURCE: torch.npu.stream（流上下文，host 直通）
    yield


def _npu_current_stream() -> _NpuStream:  # SOURCE: torch.npu.current_stream
    return _DEFAULT_STREAM


def _npu_set_device(device) -> None:  # SOURCE: torch.npu.set_device（host no-op）
    pass


# 记录每次 swap_blocks_batch 调用的 (src, dst, sizes, direction)，供测试断言指针布局。
SWAP_CALLS: list[tuple] = []


def _swap_blocks_batch(batch_src, batch_dst, batch_sizes, direction):  # SOURCE: torch.ops._C_ascend.swap_blocks_batch（底层 aclrtMemcpyBatchAsync；host 仅记录不搬字节）
    SWAP_CALLS.append((batch_src, batch_dst, batch_sizes, direction))


def reset_swap_calls() -> None:  # SOURCE: 测试辅助（无对应真实符号）— 清空 SWAP_CALLS
    SWAP_CALLS.clear()


# 把 shim 安装到真 torch 上，使精简版逐字读 `torch.npu.*` / `torch.ops._C_ascend.*`。
torch.npu = types.SimpleNamespace(  # type: ignore[attr-defined]
    Stream=_NpuStream,
    Event=_NpuEvent,
    stream=_npu_stream_ctx,
    current_stream=_npu_current_stream,
    set_device=_npu_set_device,
)
try:
    torch.ops._C_ascend = types.SimpleNamespace(swap_blocks_batch=_swap_blocks_batch)  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - torch.ops 在个别版本只读
    pass


# ----------------------------------------------------------------------
# 基座 vllm 抽象（标准路径接 v1/kv_offload；极简路径接 v1/simple_kv_offload）
# ----------------------------------------------------------------------
def is_pin_memory_available() -> bool:  # SOURCE: vllm/utils/platform_utils.py:is_pin_memory_available
    # host 无 NPU pinned host 分配器：退化为非 pinned（不影响控制流，仅影响 DMA 带宽）。
    return False


class OffloadingHandler:  # SOURCE: vllm/v1/kv_offload/worker/worker.py:OffloadingHandler
    """搬运执行体抽象：transfer_async / get_finished / wait 三原语。"""


class OffloadingManager:  # SOURCE: vllm/v1/kv_offload/abstract.py:OffloadingManager
    """scheduler 侧卸载记账/分配/lookup 抽象。"""


class CPUOffloadingManager(OffloadingManager):  # SOURCE: vllm/v1/kv_offload/cpu/manager.py:CPUOffloadingManager
    def __init__(self, block_size: int, num_blocks: int, enable_events: bool = False):  # SOURCE: cpu/manager.py:CPUOffloadingManager.__init__
        # 卸载调度与硬件无关——标准路径 NPUOffloadingSpec 直接复用基座这个 Manager。
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.enable_events = enable_events


class OffloadingSpec:  # SOURCE: vllm/v1/kv_offload/spec.py:OffloadingSpec
    def __init__(self, vllm_config, kv_cache_config=None):  # SOURCE: spec.py:OffloadingSpec.__init__
        # 真实 OffloadingSpec 从 config 内部推出下列字段；host 桩直接读测试给定的 config。
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.extra_config = getattr(vllm_config, "kv_connector_extra_config", {}) or {}
        self.gpu_block_size = getattr(vllm_config, "gpu_block_size", [16])
        self.block_size_factor = getattr(vllm_config, "block_size_factor", 4)


class LoadStoreSpec:  # SOURCE: vllm/v1/kv_offload/abstract.py:LoadStoreSpec
    """一侧（GPU/CPU）的 block_ids 载体。"""


class GPULoadStoreSpec(LoadStoreSpec):  # SOURCE: vllm/v1/kv_offload/mediums.py:GPULoadStoreSpec
    medium = "GPU"

    def __init__(self, block_ids):  # SOURCE: mediums.py:GPULoadStoreSpec.__init__
        self.block_ids = block_ids


class CPULoadStoreSpec(LoadStoreSpec):  # SOURCE: vllm/v1/kv_offload/mediums.py:CPULoadStoreSpec
    medium = "CPU"

    def __init__(self, block_ids):  # SOURCE: mediums.py:CPULoadStoreSpec.__init__
        self.block_ids = block_ids


class TransferResult(NamedTuple):  # SOURCE: vllm/v1/kv_offload/worker/worker.py:TransferResult
    job_id: int
    success: bool
    transfer_size: int
    transfer_time: float
    transfer_type: tuple


# SOURCE: vllm/v1/kv_offload/worker/worker.py:TransferSpec（= (src_spec, dst_spec) 二元组）
TransferSpec = tuple


class AttentionBackend:  # SOURCE: vllm/v1/attention/backend.py:AttentionBackend（仅类型标注）
    pass


class VllmConfig:  # SOURCE: vllm/config.py:VllmConfig（仅类型标注/测试载体）
    pass


class KVCacheConfig:  # SOURCE: vllm/v1/kv_cache_interface.py:KVCacheConfig（仅类型标注）
    pass


class SimpleCPUOffloadWorker:  # SOURCE: vllm/v1/simple_kv_offload/worker.py:SimpleCPUOffloadWorker
    """极简路径基座 worker：本章只演示昇腾覆写的两处，其余 step 期入口由它提供。"""

    def __init__(self, vllm_config, kv_cache_config, cpu_capacity_bytes):  # SOURCE: simple_kv_offload/worker.py:SimpleCPUOffloadWorker.__init__
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self.cpu_capacity_bytes = cpu_capacity_bytes
        # 真实基座在此造 CUDA DmaCopyBackend；昇腾子类 __init__ 随后替换为 NPUDmaCopyBackend。
        self._backend = None
