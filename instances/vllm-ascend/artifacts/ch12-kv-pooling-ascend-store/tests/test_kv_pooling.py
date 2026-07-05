"""ch11 KV 池化精简版 —— 验证复现 vllm-ascend 真实行为（非自洽）。

覆盖纯 Python 控制流（dossier 明示 host 可跑的部分）：
  ① 内容寻址命名     PoolKey.to_string / process_tokens 切 chunk 出 key
  ② 地址算术         prepare_value 由 block_id 算 (addr,size)
  ③ 调度器命中节拍   get_num_new_matched_tokens 命中算术 / update_state_after_alloc 的 can_load 校验
  ④ 两端队列解耦     KVTransferThread.add_request 入队 → 后台线程消费 → request_queue.join() 屏障
  ⑤ 存前 lookup 去重 KVCacheStoreSendingThread 只 put missing 块（跨请求复用）
  ⑥ 取失败记账       KVCacheStoreRecvingThread.get 失败块写入 _invalid_block_ids
  ⑦ 可插拔后端契约   Backend ABC 6 方法

实际池存取 / RDMA 搬运在 host 不可发车 —— 用纯内存 FakeBackend 替身验「契约调用顺序」，
行为以 vllm_ascend v0.21.0rc1 源码为准。
"""
import threading
from types import SimpleNamespace

import pytest
import torch

import pool_worker
from ascend_store_connector import AscendStoreConnector, LookupKeyServer
from backend.backend import Backend
from config_data import (
    AscendConnectorMetadata,
    ChunkedTokenDatabase,
    KeyMetadata,
    LoadSpec,
    PoolKey,
    ReqMeta,
)
from kv_transfer import (
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
    KVTransferThread,
    record_failed_blocks,
)
from pool_scheduler import KVPoolScheduler
from pool_worker import KVPoolWorker, backend_map
from runtime_stub import BlockHash, KVConnectorRole


# ----------------------------- 后端契约替身 -----------------------------
class FakeBackend(Backend):
    """纯内存 Backend 替身：把 6 方法契约落到一个 dict（host 无 mooncake/RDMA）。"""

    def __init__(self, existing=None):
        self.store = dict.fromkeys(existing or [], True)
        self.puts = []   # 记录每次 put 的 keys
        self.gets = []   # 记录每次 get 的 keys
        self.fail_keys = set()
        self.device_set = False
        self.registered = []

    def set_device(self):
        self.device_set = True

    def register_buffer(self, ptrs, lengths):
        self.registered.append((list(ptrs), list(lengths)))

    def exists(self, keys):
        return [1 if k in self.store else 0 for k in keys]

    def put(self, keys, addrs, sizes):
        self.puts.append(list(keys))
        for k in keys:
            self.store[k] = True

    def get(self, keys, addrs, sizes):
        self.gets.append(list(keys))
        # 约定：0=成功，非 0=失败块（recving 线程据此记 _invalid_block_ids）
        return [1 if k in self.fail_keys else 0 for k in keys]


# ----------------------------- fixtures -----------------------------
def make_token_db(block_size=16):
    meta = [KeyMetadata(model_name="m", head_or_tp_rank=0, pcp_rank=0, dcp_rank=0, pp_rank=0)]
    db = ChunkedTokenDatabase(metadata=meta, block_size=[block_size])
    # 每层一个 base_addr，block_len=1024 字节，stride 缺省 = block_len
    db.set_group_buffers(
        group_kv_caches_base_addr={0: [0]},
        group_block_len={0: [1024]},
    )
    return db


def make_block_hashes(n):
    return [BlockHash(bytes([i + 1]) * 32) for i in range(n)]


# ----------------------------- ① 内容寻址命名 -----------------------------
def test_poolkey_to_string_format():
    """PoolKey.to_string == 真实源码逐字格式（复用成立的根基：内容寻址 key）。"""
    km = KeyMetadata(model_name="Qwen", head_or_tp_rank=3, pcp_rank=1, dcp_rank=2, pp_rank=0)
    key = PoolKey(km, "deadbeef")
    assert key.to_string() == (
        "Qwen@pcp1@dcp2@head_or_tp_rank:3@pp_rank:0"
        "@group:0@cache_role:kv@cache_family:default@deadbeef"
    )


def test_same_prefix_same_key_reuse():
    """相同前缀 + 相同并行布局 → 同一 key（跨请求复用）；不同 hash → 不同 key。"""
    db = make_token_db()
    bh = make_block_hashes(2)
    keys_a = [k.to_string() for _, _, k in db.process_tokens(32, bh)]
    keys_b = [k.to_string() for _, _, k in db.process_tokens(32, bh)]
    assert keys_a == keys_b               # 同前缀复用
    assert keys_a[0] != keys_a[1]         # 不同 chunk → 不同 key


def test_process_tokens_chunking():
    """process_tokens 用 block_hashes 把 token 序列切 chunk，每 chunk 一个 (start,end,key)。"""
    db = make_token_db(block_size=16)
    bh = make_block_hashes(3)
    spans = [(s, e) for s, e, _ in db.process_tokens(40, bh)]
    # 40 token / 16：chunk0=(0,16), chunk1=(16,32), chunk2=(32,40 截断)
    assert spans == [(0, 16), (16, 32), (32, 40)]


# ----------------------------- ② 地址算术 -----------------------------
def test_prepare_value_addr_size():
    """prepare_value：addr = base + block_id*stride；size = block_len/block_size*(end-start)。"""
    db = make_token_db(block_size=16)
    addrs, sizes, block_id = db.prepare_value(16, 32, block_ids=[100, 101], kv_cache_group_id=0)
    assert block_id == 101                      # start=16 // 16 = 索引 1
    assert addrs == [0 + 101 * 1024]            # base 0 + block_id*stride(=block_len)
    assert sizes == [int(1024 / 16 * (32 - 16))]  # 1024


# ----------------------------- ③ 调度器命中节拍 -----------------------------
def make_scheduler(kv_role="kv_both", extra=None, block_size=16):
    extra = extra or {}
    ktc = SimpleNamespace(
        kv_role=kv_role,
        kv_connector_extra_config=extra,
        get_from_extra_config=lambda k, d: extra.get(k, d),
    )
    vllm_config = SimpleNamespace(
        kv_transfer_config=ktc,
        cache_config=SimpleNamespace(block_size=block_size),
        parallel_config=SimpleNamespace(data_parallel_rank=0),
    )
    sched = KVPoolScheduler(vllm_config, use_layerwise=False)
    return sched


def test_get_num_new_matched_tokens_hit_arithmetic():
    """need_to_allocate = 命中 − 本地已算；登记 LoadSpec(can_load=False)。"""
    sched = make_scheduler()
    sched.client = SimpleNamespace(lookup=lambda *a, **k: 48)  # 池命中 48 token
    req = SimpleNamespace(
        request_id="r1", prompt_token_ids=list(range(64)), num_tokens=64,
        block_hashes=make_block_hashes(4),
    )
    need, async_load = sched.get_num_new_matched_tokens(req, num_computed_tokens=16)
    assert need == 48 - 16                      # 命中 48 − 本地已算 16
    spec = sched.load_specs["r1"]
    assert (spec.vllm_cached_tokens, spec.kvpool_cached_tokens, spec.can_load) == (16, 48, False)


def test_hit_equals_total_minus_one():
    """命中 == 全 prompt → 命中减 1，保证至少 1 token 跑前向。"""
    sched = make_scheduler()
    sched.client = SimpleNamespace(lookup=lambda *a, **k: 64)
    req = SimpleNamespace(
        request_id="r2", prompt_token_ids=list(range(64)), num_tokens=64,
        block_hashes=make_block_hashes(4),
    )
    need, _ = sched.get_num_new_matched_tokens(req, num_computed_tokens=0)
    assert need == 63                          # 64 命中 → 减 1


def test_consumer_without_load_returns_zero():
    """kv_consumer 且 consumer_is_to_load=False → 不查池，直接 (0, False)。"""
    sched = make_scheduler(kv_role="kv_consumer", extra={"consumer_is_to_load": False})
    called = {"n": 0}
    sched.client = SimpleNamespace(lookup=lambda *a, **k: called.__setitem__("n", 1) or 99)
    req = SimpleNamespace(
        request_id="r3", prompt_token_ids=list(range(64)), num_tokens=64,
        block_hashes=make_block_hashes(4),
    )
    assert sched.get_num_new_matched_tokens(req, 0) == (0, False)
    assert called["n"] == 0                    # 根本没问池


def test_short_prompt_below_granularity():
    """prompt 短于一个 chunk → (0, False)，不查池。"""
    sched = make_scheduler(block_size=16)
    sched.client = SimpleNamespace(lookup=lambda *a, **k: 99)
    req = SimpleNamespace(
        request_id="r4", prompt_token_ids=list(range(8)), num_tokens=8,
        block_hashes=make_block_hashes(1),
    )
    assert sched.get_num_new_matched_tokens(req, 0) == (0, False)


def test_update_state_after_alloc_sets_can_load():
    """alloc 后缺口一致 → can_load=True（节拍一环）。"""
    sched = make_scheduler()
    sched.load_specs["r5"] = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=48, can_load=False)
    req = SimpleNamespace(request_id="r5")
    blocks = SimpleNamespace(get_block_ids=lambda: [[1, 2]])
    sched.update_state_after_alloc(req, blocks, num_external_tokens=48 - 16)
    assert sched.load_specs["r5"].can_load is True


def test_update_state_after_alloc_mismatch_raises():
    """缺口与 LoadSpec 不一致 → AssertionError（校验保护）。"""
    sched = make_scheduler()
    sched.load_specs["r6"] = LoadSpec(vllm_cached_tokens=16, kvpool_cached_tokens=48, can_load=False)
    req = SimpleNamespace(request_id="r6")
    blocks = SimpleNamespace(get_block_ids=lambda: [[1, 2]])
    with pytest.raises(AssertionError):
        sched.update_state_after_alloc(req, blocks, num_external_tokens=30)


# ----------------------------- ④ 两端队列解耦 -----------------------------
def test_transfer_thread_decoupling_and_join_barrier():
    """add_request 入队即返回 → 后台线程消费 → request_queue.join() 等全部 task_done（背压屏障）。"""
    backend = FakeBackend()
    db = make_token_db()
    ready = threading.Event()
    send = KVCacheStoreSendingThread(
        backend, db, block_size=[16], tp_rank=0, dcp_size=1, put_step=1,
        kv_role="kv_both", ready_event=ready, group_uses_align_state=[False],
    )
    send.start()
    assert ready.wait(timeout=5)                 # run() 里 set_device 后置 ready
    assert backend.device_set is True            # 线程内调了后端 set_device

    req = ReqMeta(
        req_id="rq", token_len_chunk=32,
        block_ids_by_group=[[100, 101]], block_hashes=make_block_hashes(2),
    )
    send.add_stored_request("rq")
    send.add_request(req)                         # 主循环只入队
    send.request_queue.join()                     # 屏障：阻塞到 put 落地（task_done）
    assert len(backend.puts) == 1                 # 后台线程确实搬了
    assert len(backend.puts[0]) == 2              # 2 个 chunk 全 put（池中无）


def test_base_thread_handle_request_is_noop():
    """KVTransferThread 基类 _handle_request 为空，搬运逻辑由收/发子类实现。"""
    backend = FakeBackend()
    db = make_token_db()
    t = KVTransferThread(backend, db, block_size=16, tp_rank=0, dcp_size=1,
                         ready_event=threading.Event(), name="base")
    assert t._handle_request(object()) is None


# ----------------------------- ⑤ 存前 lookup 去重 -----------------------------
def test_sending_thread_dedup_only_puts_missing():
    """存前先 lookup：池里已有的 chunk 跳过，只 put missing（跨请求复用，相同前缀只存一次）。"""
    db = make_token_db()
    bh = make_block_hashes(2)
    keys = [k.to_string() for _, _, k in db.process_tokens(32, bh)]
    backend = FakeBackend(existing=[keys[0]])    # chunk0 已在池中
    send = KVCacheStoreSendingThread(
        backend, db, block_size=[16], tp_rank=0, dcp_size=1, put_step=1,
        kv_role="kv_both", ready_event=threading.Event(), group_uses_align_state=[False],
    )
    req = ReqMeta(req_id="rq", token_len_chunk=32,
                  block_ids_by_group=[[100, 101]], block_hashes=bh)
    send.add_stored_request("rq")
    send.request_queue.put(req)                   # 配平 _handle_request 末尾 task_done
    send._handle_request(req)
    assert len(backend.puts) == 1
    assert backend.puts[0] == [keys[1]]           # 只 put 缺失的 chunk1


def test_sending_thread_all_present_no_put():
    """两 chunk 都已在池中 → 完全不 put。"""
    db = make_token_db()
    bh = make_block_hashes(2)
    keys = [k.to_string() for _, _, k in db.process_tokens(32, bh)]
    backend = FakeBackend(existing=keys)
    send = KVCacheStoreSendingThread(
        backend, db, block_size=[16], tp_rank=0, dcp_size=1, put_step=1,
        kv_role="kv_both", ready_event=threading.Event(), group_uses_align_state=[False],
    )
    req = ReqMeta(req_id="rq", token_len_chunk=32,
                  block_ids_by_group=[[100, 101]], block_hashes=bh)
    send.add_stored_request("rq")
    send.request_queue.put(req)
    send._handle_request(req)
    assert backend.puts == []


def test_sending_thread_unknown_req_skips():
    """未登记 add_stored_request 的请求 → 直接跳过、不搬。"""
    db = make_token_db()
    backend = FakeBackend()
    send = KVCacheStoreSendingThread(
        backend, db, block_size=[16], tp_rank=0, dcp_size=1, put_step=1,
        kv_role="kv_both", ready_event=threading.Event(), group_uses_align_state=[False],
    )
    req = ReqMeta(req_id="ghost", token_len_chunk=32,
                  block_ids_by_group=[[100, 101]], block_hashes=make_block_hashes(2))
    send.request_queue.put(req)
    send._handle_request(req)                      # 未 add_stored_request
    assert backend.puts == []


# ----------------------------- ⑥ 取失败记账 -----------------------------
def test_recving_thread_records_failed_blocks():
    """get 返回非 0 的块 → 写入 _invalid_block_ids；并 set_finished_request。"""
    db = make_token_db()
    bh = make_block_hashes(2)
    keys = [k.to_string() for _, _, k in db.process_tokens(32, bh)]
    backend = FakeBackend()
    backend.fail_keys = {keys[1]}                  # chunk1 取失败
    invalid = set()
    lock = threading.Lock()
    recv = KVCacheStoreRecvingThread(
        backend, db, block_size=[16], tp_rank=0, dcp_size=1,
        ready_event=threading.Event(), invalid_block_ids=invalid, invalid_block_ids_lock=lock,
    )
    req = ReqMeta(
        req_id="rq", token_len_chunk=32, block_ids_by_group=[[100, 101]], block_hashes=bh,
        load_spec=LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True, token_len=32),
    )
    recv.request_queue.put(req)
    recv._handle_request(req)
    assert len(backend.gets) == 1
    assert invalid == {101}                        # chunk1 → block_id 101 记为失败
    assert "rq" in recv.get_and_clear_finished_requests()


def test_recving_thread_get_returns_none():
    """get 返回 None（后端异常）→ 全部块记为失败。"""
    db = make_token_db()
    bh = make_block_hashes(2)
    backend = FakeBackend()
    backend.get = lambda keys, addrs, sizes: None
    invalid = set()
    recv = KVCacheStoreRecvingThread(
        backend, db, block_size=[16], tp_rank=0, dcp_size=1,
        ready_event=threading.Event(), invalid_block_ids=invalid, invalid_block_ids_lock=threading.Lock(),
    )
    req = ReqMeta(
        req_id="rq", token_len_chunk=32, block_ids_by_group=[[100, 101]], block_hashes=bh,
        load_spec=LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True, token_len=32),
    )
    recv.request_queue.put(req)
    recv._handle_request(req)
    assert invalid == {100, 101}


def test_record_failed_blocks():
    """record_failed_blocks：code != 0 的块进失败集合。"""
    assert record_failed_blocks([10, 11, 12], [0, 1, 0]) == {11}
    assert record_failed_blocks([10, 11], [0, 0]) == set()


# ----------------------------- ⑦ 可插拔后端契约 -----------------------------
def test_backend_abc_contract():
    """Backend 契约恰好 6 个抽象方法；不实现全部不能实例化。"""
    assert Backend.__abstractmethods__ == frozenset(
        {"__init__", "set_device", "register_buffer", "exists", "put", "get"}
    )
    with pytest.raises(TypeError):
        Backend()  # ABC 不可实例化


def test_lookup_via_exists_contract():
    """KVTransferThread.lookup 经后端 exists 把 int 命中码归一成 bool 列表。"""
    backend = FakeBackend(existing=["k_hit"])
    t = KVTransferThread(backend, make_token_db(), block_size=16, tp_rank=0, dcp_size=1,
                         ready_event=threading.Event(), name="t")
    assert t.lookup(["k_hit", "k_miss"]) == [True, False]


# ===================== 入口 + worker 两端落地（ascend_store_connector / pool_worker）=====================
def make_vllm_config(block_size=16, kv_role="kv_both", extra=None, rank=0):
    extra = extra or {}
    ktc = SimpleNamespace(
        kv_role=kv_role,
        kv_connector_extra_config=extra,
        get_from_extra_config=lambda k, d: extra.get(k, d),
    )
    model_config = SimpleNamespace(
        model="org/Qwen",
        get_num_layers=lambda pc: 4,
        get_total_num_kv_heads=lambda: 8,
    )
    parallel_config = SimpleNamespace(pipeline_parallel_size=1, rank=rank, data_parallel_rank=0)
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        kv_transfer_config=ktc,
        cache_config=SimpleNamespace(block_size=block_size),
    )


def make_worker(monkeypatch, *, block_size=16, kv_role="kv_both", extra=None, rank=0,
                seed_store=None):
    """构造 KVPoolWorker；monkeypatch importlib 让 backend_map 选出纯内存 FakeBackend
    （绝不导入真 mooncake_backend——host 无 torch_npu/mooncake）。"""
    extra = dict(extra or {})
    extra.setdefault("backend", "mooncake")
    backend_obj = FakeBackend(existing=seed_store)
    captured = {}

    def fake_import(path):
        captured["path"] = path
        return SimpleNamespace(
            MooncakeBackend=lambda parallel_config, **kw: backend_obj,
            MemcacheBackend=lambda parallel_config, **kw: backend_obj,
            YuanrongBackend=lambda parallel_config, **kw: backend_obj,
        )

    monkeypatch.setattr(pool_worker.importlib, "import_module", fake_import)
    w = KVPoolWorker(make_vllm_config(block_size, kv_role, extra, rank), use_layerwize=False)
    w._captured_backend_path = captured.get("path")
    return w


def make_kv_caches(num_blocks=4):
    # 一层 KV cache：[num_blocks, 2, 8] 连续 CPU 张量（host 无 NPU，用 CPU 张量验地址注册控制流）。
    return {"layer0": torch.zeros(num_blocks, 2, 8, dtype=torch.float32)}


# ----- backend_map 动态选后端 -----
def test_backend_map_entries():
    """backend_map 把 backend 名映射到模块路径 + 类名（可插拔注入点）。"""
    assert set(backend_map) == {"mooncake", "memcache", "yuanrong"}
    assert backend_map["mooncake"]["name"] == "MooncakeBackend"
    assert backend_map["yuanrong"]["path"].endswith("backend.yuanrong_backend")


def test_worker_dynamic_backend_selection(monkeypatch):
    """worker 按 extra_config['backend'] importlib.import_module 出对应模块路径。"""
    w = make_worker(monkeypatch, extra={"backend": "yuanrong"})
    assert w._captured_backend_path == backend_map["yuanrong"]["path"]
    assert isinstance(w.m_store, FakeBackend)


def test_worker_unknown_backend_asserts(monkeypatch):
    """未知后端名 → backend_map.get 返回 None → assert 拦下。"""
    monkeypatch.setattr(pool_worker.importlib, "import_module", lambda p: SimpleNamespace())
    with pytest.raises(AssertionError):
        KVPoolWorker(make_vllm_config(extra={"backend": "does_not_exist"}), use_layerwize=False)


# ----- register_kv_caches：注册显存 + 起后台线程（两端解耦 worker 侧落地）-----
def test_register_kv_caches_registers_and_starts_threads(monkeypatch):
    """register_buffer 把显存区段注册进后端；kv_producer/both 起「发」后台线程。"""
    w = make_worker(monkeypatch, kv_role="kv_both")
    w.register_kv_caches(make_kv_caches())
    assert len(w.m_store.registered) == 1            # 注册了一段显存区
    assert w.kv_send_thread is not None              # 起了发送线程
    assert w.kv_send_thread.is_alive()
    assert w.kv_recv_thread is None                  # load_async=False → 不起收线程
    assert w.m_store.device_set is True              # 线程内调了后端 set_device


def test_register_kv_caches_load_async_starts_recv(monkeypatch):
    """load_async=True → 额外起「收」后台线程。"""
    w = make_worker(monkeypatch, kv_role="kv_both", extra={"load_async": True})
    w.register_kv_caches(make_kv_caches())
    assert w.kv_recv_thread is not None
    assert w.kv_recv_thread.is_alive()


# ----- start_load_kv：同步 / 异步分支 -----
def _load_req(block_size=16):
    bh = make_block_hashes(2)
    return ReqMeta(
        req_id="rq", token_len_chunk=32,
        block_ids_by_group=[[100, 101]], block_hashes=bh,
        kv_cache_group_ids=[0],
        load_spec=LoadSpec(vllm_cached_tokens=0, kvpool_cached_tokens=32, can_load=True),
    )


def test_start_load_kv_sync_calls_get(monkeypatch):
    """load_async=False → start_load_kv 当场 m_store.get（阻塞主循环取 KV）。"""
    w = make_worker(monkeypatch, kv_role="kv_both")
    w.register_kv_caches(make_kv_caches())
    meta = AscendConnectorMetadata(set(), set())
    meta.add_request(_load_req())
    w.start_load_kv(meta)
    assert len(w.m_store.gets) == 1                  # 同步当场取
    assert len(w.m_store.gets[0]) == 2               # 2 个 chunk


def test_start_load_kv_async_enqueues(monkeypatch):
    """load_async=True → 只把 request 丢进 recv 线程队列（主循环不阻塞），后台搬。"""
    w = make_worker(monkeypatch, kv_role="kv_both", extra={"load_async": True})
    w.register_kv_caches(make_kv_caches())
    meta = AscendConnectorMetadata(set(), set())
    meta.add_request(_load_req())
    w.start_load_kv(meta)
    w.kv_recv_thread.request_queue.join()            # 等后台线程搬完
    assert len(w.m_store.gets) == 1                  # 后台线程确实取了
    assert "rq" in w.kv_recv_thread.get_and_clear_finished_requests()


def test_start_load_kv_skips_when_cannot_load(monkeypatch):
    """load_spec.can_load=False → 跳过，不取。"""
    w = make_worker(monkeypatch, kv_role="kv_both")
    w.register_kv_caches(make_kv_caches())
    req = _load_req()
    req.load_spec.can_load = False
    meta = AscendConnectorMetadata(set(), set())
    meta.add_request(req)
    w.start_load_kv(meta)
    assert w.m_store.gets == []


# ----- wait_for_save：背压屏障 + 存 -----
def test_wait_for_save_join_barrier_puts(monkeypatch):
    """wait_for_save 把 can_save 请求入发送队列，末尾 request_queue.join() 等 put 落地。"""
    w = make_worker(monkeypatch, kv_role="kv_both")
    w.register_kv_caches(make_kv_caches())
    save_req = ReqMeta(
        req_id="rq", token_len_chunk=32,
        block_ids_by_group=[[100, 101]], block_hashes=make_block_hashes(2),
        kv_cache_group_ids=[0], can_save=True,
    )
    meta = AscendConnectorMetadata(set(), set())
    meta.add_request(save_req)
    w.wait_for_save(meta)                            # 内部 join() 阻塞到 put 完成
    assert len(w.m_store.puts) == 1                  # 存进池了
    assert len(w.m_store.puts[0]) == 2


# ----- get_finished：上报异步收/发完成 -----
def test_get_finished_reports_done_sending(monkeypatch):
    """finished_req_id 对应的发送任务已清零 → 计入 done_sending。"""
    w = make_worker(monkeypatch, kv_role="kv_both")
    w.register_kv_caches(make_kv_caches())
    w.kv_send_thread.stored_requests["r"] = 0        # 该请求的 put 已全部 task_done
    meta = AscendConnectorMetadata(set(), set())
    done_sending, done_recving = w.get_finished({"r"}, meta)
    assert done_sending == {"r"}
    assert done_recving == set()                     # load_async=False


# ----- lookup_scheduler：真正调 m_store.exists 算命中前缀 -----
def test_lookup_scheduler_full_hit(monkeypatch):
    """两 chunk 都在池中 → 命中整段（32）。"""
    w = make_worker(monkeypatch, kv_role="kv_both")
    bh = make_block_hashes(2)
    keys = [k.to_string() for _, _, k in w.token_database.process_tokens(32, bh)]
    w.m_store.store = dict.fromkeys(keys, True)
    assert w.lookup_scheduler(32, bh, [0]) == 32


def test_lookup_scheduler_partial_hit(monkeypatch):
    """只有 chunk0 在池中 → 命中前缀 16。"""
    w = make_worker(monkeypatch, kv_role="kv_both")
    bh = make_block_hashes(2)
    keys = [k.to_string() for _, _, k in w.token_database.process_tokens(32, bh)]
    w.m_store.store = {keys[0]: True}
    assert w.lookup_scheduler(32, bh, [0]) == 16


def test_lookup_scheduler_no_hit(monkeypatch):
    """池里没有 → 命中 0。"""
    w = make_worker(monkeypatch, kv_role="kv_both")
    bh = make_block_hashes(2)
    assert w.lookup_scheduler(32, bh, [0]) == 0


# ----- AscendStoreConnector：入口 role 分派 + 薄转发 -----
def test_connector_scheduler_role_builds_scheduler():
    """role=SCHEDULER → 实例化 KVPoolScheduler，不建 worker。"""
    conn = AscendStoreConnector(make_vllm_config(), KVConnectorRole.SCHEDULER)
    assert isinstance(conn.connector_scheduler, KVPoolScheduler)
    assert conn.connector_worker is None


def test_connector_worker_role_builds_worker_and_lookup_server(monkeypatch):
    """role=WORKER + rank0 → 实例化 KVPoolWorker 并起 LookupKeyServer（服务端持有同一 worker）。"""
    backend_obj = FakeBackend()
    monkeypatch.setattr(
        pool_worker.importlib, "import_module",
        lambda p: SimpleNamespace(MooncakeBackend=lambda parallel_config, **kw: backend_obj),
    )
    conn = AscendStoreConnector(make_vllm_config(rank=0), KVConnectorRole.WORKER)
    assert isinstance(conn.connector_worker, KVPoolWorker)
    assert isinstance(conn.lookup_server, LookupKeyServer)
    assert conn.lookup_server.pool_worker is conn.connector_worker


def test_connector_worker_role_nonzero_rank_no_server(monkeypatch):
    """非 rank0 → 不起 LookupKeyServer（每节点只 rank0 一个 REP 服务端）。"""
    backend_obj = FakeBackend()
    monkeypatch.setattr(
        pool_worker.importlib, "import_module",
        lambda p: SimpleNamespace(MooncakeBackend=lambda parallel_config, **kw: backend_obj),
    )
    conn = AscendStoreConnector(make_vllm_config(rank=1), KVConnectorRole.WORKER)
    assert isinstance(conn.connector_worker, KVPoolWorker)
    assert not hasattr(conn, "lookup_server")


def test_connector_forwards_scheduler_hooks():
    """薄分发层：调度侧钩子原样转发给 connector_scheduler。"""
    conn = AscendStoreConnector(make_vllm_config(), KVConnectorRole.SCHEDULER)
    conn.connector_scheduler = SimpleNamespace(
        get_num_new_matched_tokens=lambda r, n: (7, True),
        build_connector_meta=lambda so: "META",
    )
    req = SimpleNamespace(request_id="r")
    assert conn.get_num_new_matched_tokens(req, 3) == (7, True)
    assert conn.build_connector_meta("so") == "META"


def test_connector_wait_for_save_consumer_guard():
    """kv_consumer 且 consumer_is_to_put=False → wait_for_save 直接返回，不转发 worker。"""
    conn = AscendStoreConnector(
        make_vllm_config(kv_role="kv_consumer"), KVConnectorRole.SCHEDULER
    )
    called = {"n": 0}
    conn.connector_worker = SimpleNamespace(
        wait_for_save=lambda m: called.__setitem__("n", 1)
    )
    conn.wait_for_save()
    assert called["n"] == 0
