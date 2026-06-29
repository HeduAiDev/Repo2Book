"""ch12 KV 卸载精简版 —— 验证复现 vllm-ascend 真实行为（非自洽）。

覆盖纯 Python 控制流（dossier 明示 host 可跑的部分）：
  ① block 粒度换算      expand_block_ids 粗→细 + skip_count 对齐
  ② 标准路径接入        NPUOffloadingSpec.get_manager 复用基座 / get_handlers 双向同 handler
  ③ 分层搬运节拍        transfer_async d2h(dir=1)/h2d(dir=0) 方向判定 + 指针广播 + deque 在途
  ④ 发了不等           get_finished 用 end_event.query() FIFO 回收 / wait 阻塞 synchronize
  ⑤ block 视图重建      _build_block_views 单段/多段、shape/stride 取尺寸 + _flatten_kv_value 不漏 V
  ⑥ 极简注册           register_kv_caches 去重 + 重建视图 + 算 num_cpu_blocks + 起后端
  ⑦ 底层原语           build_params/copy_blocks 指针布局 + 收口 swap_blocks_batch
  ⑧ DMA 拷贝调度        NPUDmaCopyBackend launch_copy → 后台线程 → record Event 供轮询

两条路径最终都收口到 torch.ops._C_ascend.swap_blocks_batch（host 由 runtime_stub 记录、
不真搬字节）；行为以 vllm_ascend 源码为准。
"""
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import runtime_stub
from runtime_stub import CPULoadStoreSpec, GPULoadStoreSpec, SWAP_CALLS, reset_swap_calls

from cpu_npu import CpuNpuOffloadingHandler, Transfer, expand_block_ids
from npu import NPUOffloadingSpec
from npu_mem_ops import DIRECTION_D2H, DIRECTION_H2D, BatchMemcpyParams, build_params, copy_blocks
from copy_backend import NPUDmaCopyBackend
from worker import SimpleCPUOffloadNPUWorker, _flatten_kv_value


@pytest.fixture(autouse=True)
def _clean_swaps():
    reset_swap_calls()
    yield
    reset_swap_calls()


def _make_gpu_caches(num_layers=1, num_blocks=8, block_elems=16, dtype=torch.float16):
    """每层 KV cache = (key, value) 两个独立 storage 的 tuple（昇腾真实布局）。"""
    caches = {}
    for i in range(num_layers):
        k = torch.zeros(num_blocks, block_elems, dtype=dtype)
        v = torch.zeros(num_blocks, block_elems, dtype=dtype)
        caches[f"layer{i}"] = (k, v)
    return caches


# ----------------------------- ① expand_block_ids -----------------------------
def test_expand_block_ids_matches_docstring():
    out = np.empty(12, dtype=np.int64)
    expand_block_ids(np.array([0, 1, 3]), 4, out)
    assert out.tolist() == [0, 1, 2, 3, 4, 5, 6, 7, 12, 13, 14, 15]


def test_expand_block_ids_skip_count_aligns_first_block():
    # factor=4, skip first 3 sub-slots of the first CPU block (partial block).
    out = np.empty(5, dtype=np.int64)
    expand_block_ids(np.array([0, 1]), 4, out, skip_count=3)
    assert out.tolist() == [3, 4, 5, 6, 7]


# ----------------------------- ② 标准路径接入 -----------------------------
def _spec(num_cpu_blocks=100, gpu_block_size=16, factor=4):
    return SimpleNamespace(
        kv_connector_extra_config={"num_cpu_blocks": num_cpu_blocks},
        gpu_block_size=[gpu_block_size],
        block_size_factor=factor,
    )


def test_get_manager_reuses_base_with_offloaded_block_size():
    spec = NPUOffloadingSpec(_spec(num_cpu_blocks=100, gpu_block_size=16, factor=4))
    mgr = spec.get_manager()
    # CPU 侧 block = gpu_block_size * block_size_factor；num_blocks = num_cpu_blocks。
    assert mgr.block_size == 16 * 4
    assert mgr.num_blocks == 100
    assert mgr.enable_events is False
    assert spec.get_manager() is mgr  # 缓存复用


def test_missing_num_cpu_blocks_raises():
    cfg = SimpleNamespace(kv_connector_extra_config={}, gpu_block_size=[16], block_size_factor=4)
    with pytest.raises(Exception, match="num_cpu_blocks"):
        NPUOffloadingSpec(cfg)


def test_get_handlers_registers_both_directions_same_handler():
    spec = NPUOffloadingSpec(_spec())
    kv_caches = _make_gpu_caches()
    pairs = list(spec.get_handlers(kv_caches, attn_backends={}))
    assert len(pairs) == 2
    (s0, d0, h0), (s1, d1, h1) = pairs
    assert (s0, d0) == (GPULoadStoreSpec, CPULoadStoreSpec)  # store
    assert (s1, d1) == (CPULoadStoreSpec, GPULoadStoreSpec)  # load
    assert h0 is h1  # 双向收进同一个 handler
    assert isinstance(h0, CpuNpuOffloadingHandler)


# ----------------------------- ③ 分层搬运节拍 -----------------------------
def _handler(num_cpu_blocks=100, gpu_block_size=16, cpu_block_size=64, **kw):
    return CpuNpuOffloadingHandler(
        gpu_block_size=gpu_block_size,
        cpu_block_size=cpu_block_size,
        num_cpu_blocks=num_cpu_blocks,
        gpu_caches=_make_gpu_caches(**kw),
    )


def _expected_ptrs(base_ptrs, block_ids, bpb):
    bsz_col = bpb[:, None]
    return (base_ptrs[:, None] + np.asarray(block_ids)[None, :] * bsz_col).ravel()


def test_transfer_async_d2h_direction_and_pointer_layout():
    h = _handler()
    # store: GPU[0,1,2,3] -> CPU[0]（factor=4，整块）
    src = GPULoadStoreSpec(np.array([0, 1, 2, 3]))
    dst = CPULoadStoreSpec(np.array([0]))
    assert h.transfer_async(job_id=7, spec=(src, dst)) is True

    assert len(SWAP_CALLS) == 1
    batch_src, batch_dst, batch_sizes, direction = SWAP_CALLS[0]
    assert direction == 1  # d2h / store

    # 指针布局 = base + block_id * bytes_per_block，跨 (num_sub_tensors × num_pairs) 广播。
    exp_src = _expected_ptrs(h._npu_base_ptrs, [0, 1, 2, 3], h._block_size_in_bytes_arr)
    exp_dst = _expected_ptrs(h._cpu_base_ptrs, [0, 1, 2, 3], h._block_size_in_bytes_arr)
    assert batch_src.numpy().tolist() == exp_src.tolist()
    assert batch_dst.numpy().tolist() == exp_dst.tolist()
    assert batch_sizes.numpy().tolist() == np.broadcast_to(
        h._block_size_in_bytes_arr[:, None], (2, 4)
    ).ravel().tolist()

    # 在途队列记一笔 d2h Transfer。
    assert len(h._d2h_transfers) == 1
    assert h._d2h_transfers[0].job_id == 7


def test_transfer_async_h2d_expands_src_and_dir0():
    h = _handler()
    # load: CPU[0] -> GPU[0,1,2,3]（src CPU 侧按 factor 展开）
    src = CPULoadStoreSpec(np.array([0]))
    dst = GPULoadStoreSpec(np.array([0, 1, 2, 3]))
    h.transfer_async(job_id=1, spec=(src, dst))
    _, _, _, direction = SWAP_CALLS[0]
    assert direction == 0  # h2d / load
    assert len(h._h2d_transfers) == 1


def test_transfer_async_partial_block_skip_alignment():
    h = _handler()
    # store 5 个 GPU sub-block 到 2 个 CPU block：首个 CPU block 跳过前 3 个 slot。
    src = GPULoadStoreSpec(np.array([0, 1, 2, 3, 4]))
    dst = CPULoadStoreSpec(np.array([0, 1]))
    h.transfer_async(job_id=2, spec=(src, dst))
    batch_src, batch_dst, _, _ = SWAP_CALLS[0]
    # dst sub-block ids 应为 [3,4,5,6,7]（首块跳 3），src 为 [0,1,2,3,4]。
    exp_dst = _expected_ptrs(h._cpu_base_ptrs, [3, 4, 5, 6, 7], h._block_size_in_bytes_arr)
    exp_src = _expected_ptrs(h._npu_base_ptrs, [0, 1, 2, 3, 4], h._block_size_in_bytes_arr)
    assert batch_dst.numpy().tolist() == exp_dst.tolist()
    assert batch_src.numpy().tolist() == exp_src.tolist()


def test_same_direction_transfers_queue_in_order():
    h = _handler()
    h.transfer_async(1, (GPULoadStoreSpec(np.array([0, 1, 2, 3])), CPULoadStoreSpec(np.array([0]))))
    h.transfer_async(2, (GPULoadStoreSpec(np.array([4, 5, 6, 7])), CPULoadStoreSpec(np.array([1]))))
    assert [t.job_id for t in h._d2h_transfers] == [1, 2]


# ----------------------------- ④ 发了不等：轮询/阻塞 -----------------------------
def test_get_finished_fifo_polls_query_and_recycles_events():
    h = _handler()
    h.transfer_async(11, (GPULoadStoreSpec(np.array([0, 1, 2, 3])), CPULoadStoreSpec(np.array([0]))))
    h.transfer_async(12, (CPULoadStoreSpec(np.array([0])), GPULoadStoreSpec(np.array([0, 1, 2, 3]))))
    pool_before = len(h._event_pool)
    results = h.get_finished()
    jobs = {r.job_id: r for r in results}
    assert set(jobs) == {11, 12}
    assert jobs[11].success and jobs[11].transfer_type == ("NPU", "CPU")
    assert jobs[12].transfer_type == ("CPU", "NPU")
    # 完成后队列清空，Event 回收进池（每 Transfer 2 个）。
    assert not h._d2h_transfers and not h._h2d_transfers
    assert len(h._event_pool) == pool_before + 4


def test_wait_blocks_on_matching_job_ids():
    h = _handler()
    h.transfer_async(99, (GPULoadStoreSpec(np.array([0, 1, 2, 3])), CPULoadStoreSpec(np.array([0]))))
    # 不抛即视为对命中 job 的 end_event.synchronize() 走通。
    h.wait({99})


# ----------------------------- ⑤ block 视图重建 -----------------------------
def test_flatten_kv_value_keeps_both_k_and_v():
    k = torch.zeros(4, 8)
    v = torch.zeros(4, 8)
    assert _flatten_kv_value((k, v)) == [k, v]
    assert _flatten_kv_value(k) == [k]
    assert _flatten_kv_value([k, v]) == [k, v]


def test_build_block_views_single_segment():
    t = torch.zeros(8, 16, dtype=torch.float16)  # shape[0]=8 >= num_blocks
    views = SimpleCPUOffloadNPUWorker._build_block_views("L0", t, num_blocks=4)
    assert set(views) == {"L0"}
    view = views["L0"]
    assert view.dtype == torch.int8
    assert view.shape[0] == 4
    # 每块字节 = stride(0)*element_size = 16 * 2。
    assert view.shape[1] == 16 * 2


def test_build_block_views_multi_segment_splits_k_v():
    # (N=2, num_blocks=4, ...) 堆叠布局：拆成 2 个 keyed 视图。
    t = torch.zeros(2, 4, 16, dtype=torch.float16)
    views = SimpleCPUOffloadNPUWorker._build_block_views("L0", t, num_blocks=4)
    assert set(views) == {"L0.0", "L0.1"}
    for v in views.values():
        assert v.shape[0] == 4 and v.dtype == torch.int8


def test_build_block_views_rejects_unlocatable_blocks_dim():
    t = torch.zeros(2, 2, 16)  # 既非 shape[0]>=4 也非 shape[1]>=4
    with pytest.raises(RuntimeError, match="cannot locate blocks dim"):
        SimpleCPUOffloadNPUWorker._build_block_views("L0", t, num_blocks=4)


# ----------------------------- ⑥ 极简注册 -----------------------------
def _worker(cpu_capacity_bytes=4 * 1024 * 1024, num_blocks=4):
    cfg = SimpleNamespace()
    kvc = SimpleNamespace(num_blocks=num_blocks)
    return SimpleCPUOffloadNPUWorker(cfg, kvc, cpu_capacity_bytes)


def test_init_swaps_backend_to_npu():
    w = _worker()
    assert isinstance(w._backend, NPUDmaCopyBackend)


def test_register_kv_caches_dedups_and_builds_mirrors():
    w = _worker(num_blocks=4)
    kv = {"layer0": (torch.zeros(8, 16, dtype=torch.float16), torch.zeros(8, 16, dtype=torch.float16))}
    w.register_kv_caches(kv)
    # K/V 分开分配 → 两个 keyed 视图（不漏 V）。
    assert set(w.gpu_kv_caches) == {"layer0", "layer0.1"}
    assert set(w.cpu_kv_caches) == {"layer0", "layer0.1"}
    # num_cpu_blocks = cpu_capacity_bytes // total_bytes_per_block。
    bpb = 16 * 2  # per sub-tensor
    assert w.num_cpu_blocks == (4 * 1024 * 1024) // (bpb * 2)
    for name, cpu_t in w.cpu_kv_caches.items():
        assert cpu_t.shape[0] == w.num_cpu_blocks
        assert cpu_t.device.type == "cpu"
    w._backend.shutdown()


def test_register_empty_is_noop():
    w = _worker()
    w.register_kv_caches({})  # 不抛、不建后端流


# ----------------------------- ⑦ 底层原语 + 收口 -----------------------------
def _named_views(num_blocks=4, block_bytes=32, n=2):
    return {f"t{i}": torch.zeros(num_blocks, block_bytes, dtype=torch.int8) for i in range(n)}


def test_build_params_and_copy_blocks_pointer_layout():
    src = _named_views()
    dst = _named_views()
    params = build_params(src, dst, DIRECTION_D2H)
    assert isinstance(params, BatchMemcpyParams)
    assert params.direction == DIRECTION_D2H
    assert params.num_sub_tensors == 2

    copy_blocks([0, 1], [2, 3], params)
    assert len(SWAP_CALLS) == 1
    batch_src, batch_dst, batch_sizes, direction = SWAP_CALLS[0]
    assert direction == DIRECTION_D2H
    bpb_col = params.bpb[:, None]
    exp_src = (params.src_bases[:, None] + np.array([0, 1])[None, :] * bpb_col).ravel()
    exp_dst = (params.dst_bases[:, None] + np.array([2, 3])[None, :] * bpb_col).ravel()
    assert batch_src.numpy().tolist() == exp_src.tolist()
    assert batch_dst.numpy().tolist() == exp_dst.tolist()


def test_copy_blocks_empty_is_noop():
    params = build_params(_named_views(), _named_views(), DIRECTION_H2D)
    copy_blocks([], [], params)
    assert SWAP_CALLS == []


def test_build_params_key_order_mismatch_raises():
    a = {"x": torch.zeros(4, 8, dtype=torch.int8)}
    b = {"y": torch.zeros(4, 8, dtype=torch.int8)}
    with pytest.raises(AssertionError, match="key order"):
        build_params(a, b, DIRECTION_H2D)


# ----------------------------- ⑧ DMA 拷贝调度 -----------------------------
def test_dma_backend_dispatches_on_worker_thread():
    backend = NPUDmaCopyBackend()
    npu = _named_views()
    cpu = _named_views()
    backend.init(npu, cpu, torch.device("cpu"), torch.npu.Stream(), torch.npu.Stream())

    events_list = []
    backend.launch_copy([0, 1], [0, 1], is_store=True, event_idx=5, events_list=events_list)

    # 主线程无阻塞轮询：等后台线程 record Event 进 events_list。
    deadline = time.time() + 5.0
    while not events_list and time.time() < deadline:
        time.sleep(0.01)
    assert len(events_list) == 1
    idx, event = events_list[0]
    assert idx == 5 and event.query() is True
    # store → D2H 收口 swap_blocks_batch。
    assert SWAP_CALLS and SWAP_CALLS[-1][3] == DIRECTION_D2H
    backend.shutdown()
