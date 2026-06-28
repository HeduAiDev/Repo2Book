"""验证 use_memory_pool 的 tag 设置/恢复 —— weights/kv_cache 分流的来源。

对位真实行为 vllm_ascend/device_allocator/camem.py:L229-L263：
  进区间 current_tag←tag、存 data 引用（绕 PyTorch GC bug），出区间 current_tag 还原。
真正进 NPU MemPool 的 use_memory_pool_with_allocator 需 NPU，host 不跑——monkeypatch
成空壳上下文，只验 tag 状态机。
"""
from contextlib import contextmanager

import camem
from camem import CaMemAllocator


@contextmanager
def _fake_pool(malloc_fn, free_fn):
    # 忠实占位：真实版进 torch.npu MemPool；这里只回一个 sentinel 给 allocator_and_pools。
    yield ("fake_pool", "fake_alloc")


def test_use_memory_pool_sets_and_restores_tag(monkeypatch):
    monkeypatch.setattr(camem, "use_memory_pool_with_allocator", _fake_pool)
    a = CaMemAllocator()
    assert a.current_tag == "default"
    with a.use_memory_pool(tag="weights"):
        assert a.current_tag == "weights"
        # 区间内存了 data 引用（绕 pytorch#146431）
        assert a.allocator_and_pools["weights"] == ("fake_pool", "fake_alloc")
    # 出区间还原
    assert a.current_tag == "default"


def test_use_memory_pool_none_uses_default_tag(monkeypatch):
    monkeypatch.setattr(camem, "use_memory_pool_with_allocator", _fake_pool)
    a = CaMemAllocator()
    a.current_tag = "preexisting"
    with a.use_memory_pool():  # tag=None → default
        assert a.current_tag == "default"
        assert "default" in a.allocator_and_pools
    assert a.current_tag == "preexisting"


def test_tag_flows_into_malloc_callback(monkeypatch):
    """tag 真正的用处：区间内的 malloc 回调把分配打上 current_tag。"""
    monkeypatch.setattr(camem, "use_memory_pool_with_allocator", _fake_pool)
    a = CaMemAllocator()
    with a.use_memory_pool(tag="kv_cache"):
        a.python_malloc_callback((0, 4096, 0x9000, 0xDDD))
    assert a.pointer_to_data[0x9000].tag == "kv_cache"
