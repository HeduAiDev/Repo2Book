"""驱动 sleep/wake_up 状态机 —— 把真实 NPU/CANN 原语 monkeypatch 成记录器后跑纯控制流。

对位真实行为 vllm_ascend/device_allocator/camem.py:L178-L227：
  sleep(offload_tags=('weights',)):
    - 命中 tag 的：torch.empty 出 CPU pin 备份 + aclrtMemcpy D2H(kind=2) 拷回；
    - 所有项（命中与否）：unmap_and_release 释放物理页（kv_cache 无备份＝丢弃）。
  wake_up():
    - 所有项：create_and_map 把物理页重映射回原虚拟地址（handle 不变）；
    - 有 CPU 备份的：aclrtMemcpy H2D(kind=1) 拷回，并清空备份。
  关键差异点（vs cudaMemcpy）：memcpy 多 destMax 上界(=区域+size*2) 和显式 kind 方向枚举。

host 无 NPU/CANN：实际虚拟内存映射与拷贝不跑，仅验状态机的 offload/discard 路由正确。
"""
from contextlib import contextmanager

import pytest

import camem
from camem import AllocationData, CaMemAllocator


class _FakeTensor:
    def __init__(self, size):
        self._size = size
        self._ptr = 0xC000_0000 + size  # 任意可区分的「CPU 备份」指针

    def data_ptr(self):
        return self._ptr

    def numel(self):
        return self._size

    def element_size(self):
        return 1


class _FakeNpu:
    empty_cache_calls = 0

    @classmethod
    def empty_cache(cls):
        cls.empty_cache_calls += 1


class _FakeTorch:
    uint8 = "uint8"
    npu = _FakeNpu

    @staticmethod
    def empty(size, dtype=None, device=None, pin_memory=False):
        assert pin_memory is True  # ascend 写死 pin_memory=True
        assert device == "cpu"
        return _FakeTensor(size)


@pytest.fixture
def patched(monkeypatch):
    """把 camem 模块级的 NPU/CANN 符号换成记录器。"""
    rec = {"memcpy": [], "unmap": [], "map": []}

    def fake_memcpy(dst, dest_max, src, count, kind):
        rec["memcpy"].append((dst, dest_max, src, count, kind))

    monkeypatch.setattr(camem, "memcpy", fake_memcpy)
    monkeypatch.setattr(camem, "unmap_and_release", lambda h: rec["unmap"].append(h))
    monkeypatch.setattr(camem, "create_and_map", lambda h: rec["map"].append(h))
    monkeypatch.setattr(camem, "torch", _FakeTorch)
    _FakeNpu.empty_cache_calls = 0
    return rec


def _alloc_with_weights_and_kv():
    a = CaMemAllocator()
    # weights @ VA 0x1000 (1024B) ; kv_cache @ VA 0x2000 (2048B)
    a.pointer_to_data = {
        0x1000: AllocationData(handle=(0, 1024, 0x1000, 0xAAA), tag="weights"),
        0x2000: AllocationData(handle=(0, 2048, 0x2000, 0xBBB), tag="kv_cache"),
    }
    return a


def test_sleep_offloads_only_matching_tag_but_unmaps_all(patched):
    a = _alloc_with_weights_and_kv()
    a.sleep(offload_tags=("weights",))

    # weights 命中 → 有 CPU 备份；kv_cache 不命中 → 直接丢（无备份）
    assert a.pointer_to_data[0x1000].cpu_backup_tensor is not None
    assert a.pointer_to_data[0x2000].cpu_backup_tensor is None

    # 仅命中项做 D2H 拷贝（kind=2），src 是设备 VA，count 是字节数，destMax=cpu_ptr+size*2
    assert len(patched["memcpy"]) == 1
    dst, dest_max, src, count, kind = patched["memcpy"][0]
    assert kind == 2  # ACL_MEMCPY_DEVICE_TO_HOST
    assert src == 0x1000
    assert count == 1024
    assert dest_max == dst + 1024 * 2

    # 所有项（命中与否）都释放物理页
    assert patched["unmap"] == [(0, 1024, 0x1000, 0xAAA), (0, 2048, 0x2000, 0xBBB)]
    # 释放后清缓存
    assert _FakeNpu.empty_cache_calls == 1


def test_level2_empty_tuple_offloads_nothing_but_unmaps_all(patched):
    a = _alloc_with_weights_and_kv()
    a.sleep(offload_tags=tuple())  # level 2：什么都不 offload
    assert a.pointer_to_data[0x1000].cpu_backup_tensor is None
    assert a.pointer_to_data[0x2000].cpu_backup_tensor is None
    assert patched["memcpy"] == []  # 没有任何拷回
    assert len(patched["unmap"]) == 2  # 但物理页全释放


def test_wake_up_remaps_all_and_restores_backed_up(patched):
    a = _alloc_with_weights_and_kv()
    a.sleep(offload_tags=("weights",))
    patched["memcpy"].clear()  # 只看 wake_up 阶段的拷贝

    a.wake_up()

    # 所有项用原 handle（含原 VA）重新 map —— 虚拟地址不变是方案精髓
    assert patched["map"] == [(0, 1024, 0x1000, 0xAAA), (0, 2048, 0x2000, 0xBBB)]

    # 仅有 CPU 备份的（weights）做 H2D 拷回（kind=1）；kv_cache 只 map 空页
    assert len(patched["memcpy"]) == 1
    dst, dest_max, src, count, kind = patched["memcpy"][0]
    assert kind == 1  # ACL_MEMCPY_HOST_TO_DEVICE
    assert dst == 0x1000  # 拷回原设备 VA
    assert count == 1024
    assert dest_max == 0x1000 + 1024 * 2

    # 拷回后备份被清空
    assert a.pointer_to_data[0x1000].cpu_backup_tensor is None


def test_sleep_none_defaults_to_default_tag(patched):
    a = CaMemAllocator()
    a.pointer_to_data = {0x3000: AllocationData(handle=(0, 512, 0x3000, 0xCCC), tag="default")}
    a.sleep()  # offload_tags=None → 默认 ('default',)
    assert a.pointer_to_data[0x3000].cpu_backup_tensor is not None
    assert len(patched["unmap"]) == 1
