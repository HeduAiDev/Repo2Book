# m14 生态对照：MooncakeStore（lookup_async None=稍后再问 / 双 no-op 契约面）+
# LMCache（use_native 双路 lazy import）+ MultiConnector（首命中者获加载权）。
import sys
import threading
import time
import types

from _driver_common import TESTS, dump

sys.path.insert(0, str(TESTS))
import zmq  # noqa: E402
from _kv_harness import (  # noqa: E402
    FakePoolA,
    FakePoolB,
    fill_block_hashes,
    make_blocks,
    make_kv_config,
    make_request,
    make_vllm_config,
)

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.protocol import LOOKUP_MSG  # noqa: E402


# ── Mooncake ZMQ REP 测试服务（按脚本回 lookup 结果）──
class MooncakeServer:
    def __init__(self, delay, hits):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.port = self.sock.bind_to_random_port("tcp://127.0.0.1")
        self.delay = delay
        self.hits = hits
        self.requests = []
        self.running = True
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        while self.running:
            try:
                frames = self.sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.005)
                continue
            assert bytes(frames[0]) == LOOKUP_MSG
            num_tokens = int.from_bytes(frames[1], "big")
            self.requests.append(num_tokens)
            if self.delay:
                time.sleep(self.delay)
            self.sock.send(self.hits.get(str(num_tokens), 0).to_bytes(4, "big"))

    def close(self):
        self.running = False
        self.sock.close(linger=0)
        self.ctx.term()


import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker as wmod  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (  # noqa: E402
    MooncakeStoreScheduler,
)

# ── ① lookup_async：后台 ZMQ 线程未就绪 → None=稍后再问；就绪 → 命中差值 ──
server = MooncakeServer(delay=0.35, hits={"16": 12})
wmod.get_zmq_rpc_path_lookup = lambda vcfg: f"tcp://127.0.0.1:{server.port}"
try:
    vcfg = make_vllm_config(
        extra_config={"lookup_async": True}, connector="MooncakeStoreConnector"
    )
    sched = MooncakeStoreScheduler(vcfg, make_kv_config())
    req = make_request("r1", list(range(16)))
    fill_block_hashes(req, 4)
    first, la1 = sched.get_num_new_matched_tokens(req, 0)
    time.sleep(0.5)
    second, la2 = sched.get_num_new_matched_tokens(req, 0)
    mooncake_async = {
        "prompt_tokens": 16,
        "first_call": [str(first), la1],
        "second_call": [second, la2],
        "kvpool_cached_tokens": sched.load_specs["r1"].kvpool_cached_tokens,
        "need_to_allocate": second,
        "anchor": "store/scheduler.py:L81-L134（None=稍后再问——ch16 None 语义的第三种填法）",
        "channel": "真 ZMQ REQ（LookupKeyClient）对测试 REP 服务",
    }
    sched.client.close()
finally:
    server.close()

# ── ② 同步查询 + 本地已算对齐 ──
server2 = MooncakeServer(delay=0.0, hits={"16": 12, "8": 3})
wmod.get_zmq_rpc_path_lookup = lambda vcfg: f"tcp://127.0.0.1:{server2.port}"
try:
    vcfg2 = make_vllm_config(
        extra_config={"lookup_async": False}, connector="MooncakeStoreConnector"
    )
    sched2 = MooncakeStoreScheduler(vcfg2, make_kv_config())
    req2 = make_request("r1", list(range(16)))
    fill_block_hashes(req2, 4)
    n, la = sched2.get_num_new_matched_tokens(req2, 0)
    req3 = make_request("r2", list(range(16)))
    fill_block_hashes(req3, 4)
    n2, _ = sched2.get_num_new_matched_tokens(req3, 12)
    mooncake_sync = {
        "zero_computed": [n, la],
        "computed_12_external_12": n2,
        "formula": "need_to_allocate = max(0, num_external_hit_tokens - num_computed_tokens)",
    }
    sched2.client.close()
finally:
    server2.close()

# ── ③ 双 no-op 契约面：I/O 全在 get_finished 发 ──
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (  # noqa: E402
    MooncakeStoreWorker,
)

w = MooncakeStoreWorker.__new__(MooncakeStoreWorker)
noop_ok = w.start_load_kv(None) is None and w.wait_for_save(None) is None
try:
    w.get_finished(set(), None)
    get_finished_impl = "implemented"
except NotImplementedError:
    get_finished_impl = "NotImplementedError（实现体须 MooncakeDistributedStore 外部包——删除项 6）"
mooncake_noop = {
    "start_load_kv_noop": noop_ok,
    "wait_for_save_noop": noop_ok,
    "get_finished_on_skeleton": get_finished_impl,
    "docstring": "All load and store I/O requests are issued here (after model compute is launched)——store/worker.py:L1538-L1562",
}

# ── ④ LMCache 双路 lazy import ──
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (  # noqa: E402
    LMCacheConnectorV1,
)

made = []


class NativeImpl:
    def __init__(self, vllm_config, role, parent):
        made.append("native")


class LatestImpl:
    def __init__(self, vllm_config, role, parent):
        made.append("latest")


native_mod = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration")
adapter = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter")
adapter.LMCacheConnectorV1Impl = NativeImpl
native_mod.vllm_v1_adapter = adapter
ext_pkg = types.ModuleType("lmcache")
ext_int = types.ModuleType("lmcache.integration")
ext_vllm = types.ModuleType("lmcache.integration.vllm")
ext_adapter = types.ModuleType("lmcache.integration.vllm.vllm_v1_adapter")
ext_adapter.LMCacheConnectorV1Impl = LatestImpl
ext_int.vllm = ext_vllm
ext_vllm.vllm_v1_adapter = ext_adapter
for name, mod in {
    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration": native_mod,
    "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter": adapter,
    "lmcache": ext_pkg,
    "lmcache.integration": ext_int,
    "lmcache.integration.vllm": ext_vllm,
    "lmcache.integration.vllm.vllm_v1_adapter": ext_adapter,
}.items():
    sys.modules.setdefault(name, mod)

c_native = LMCacheConnectorV1(
    make_vllm_config(extra_config={"use_native": True}, connector="LMCacheConnectorV1"),
    KVConnectorRole.WORKER,
    make_kv_config(),
)
c_latest = LMCacheConnectorV1(
    make_vllm_config(extra_config={"use_native": False}, connector="LMCacheConnectorV1"),
    KVConnectorRole.WORKER,
    make_kv_config(),
)
lmcache_case = {
    "use_native_true_loads": made[0],
    "use_native_false_loads": made[1],
    "distinct_engines": c_native._lmcache_engine is not c_latest._lmcache_engine,
    "paths": {
        "native": "vllm 内置 lmcache_integration 适配器（引擎内置形态）",
        "latest": "外部 lmcache 包 vllm_v1_adapter（独立项目嵌入式，latest dev）",
    },
    "anchor": "lmcache_connector.py:L83-L113（两边都 lazy import——不装包不付出导入成本）",
}

# ── ⑤ MultiConnector：列表顺序首个命中数>0 者获加载权 ──
for name in ("FakePoolA", "FakePoolB"):
    try:
        KVConnectorFactory.register_connector(name, "_kv_harness", name)
    except ValueError:
        pass
FakePoolA.hits = 0
FakePoolB.hits = 5
FakePoolA.alloc_calls = []
FakePoolB.alloc_calls = []
vcfg3 = make_vllm_config(
    extra_config={
        "connectors": [
            {"kv_connector": "FakePoolA", "kv_role": "kv_both"},
            {"kv_connector": "FakePoolB", "kv_role": "kv_both"},
        ]
    },
    connector="MultiConnector",
)
vcfg3.scheduler_config.disable_hybrid_kv_cache_manager = True
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import MultiConnector  # noqa: E402

mc = MultiConnector(vcfg3, KVConnectorRole.SCHEDULER, make_kv_config())
req4 = make_request("r1", list(range(8)))
fill_block_hashes(req4, 2)
n4, la4 = mc.get_num_new_matched_tokens(req4, 0)
FakePoolA.hits = 20  # 后续更长命中不覆盖首个非零者？—— 判据只首次赋值（本轮 A 先非零 → A 获胜）
req5 = make_request("r2", list(range(8)))
fill_block_hashes(req5, 2)
n5, _ = mc.get_num_new_matched_tokens(req5, 0)
mc.update_state_after_alloc(req4, make_blocks([[1, 2]]), 5)
multi_case = {
    "round1_A0_B5": [n4, la4],
    "round1_winner": "B（A=0 → 列表顺序下一个命中数>0 者）",
    "round2_A20": n5,
    "round2_winner": "A（本轮 A 先非零——每请求独立判定，判据只首次赋值）",
    "alloc_routed_A": list(FakePoolA.alloc_calls[-1]),
    "alloc_routed_B": list(FakePoolB.alloc_calls[-1]),
    "store_policy": "全部子 connector 都收 store",
    "anchor": "multi_connector.py:L399-L404（The first connector that has new matched tokens will be assigned）",
    "compose": "P/D（NIXL/MooncakeConnector）与池化（Offloading/MooncakeStore）同引擎并存的官方形态",
}

dump(
    "m14.json",
    {
        "mooncake_lookup_async": mooncake_async,
        "mooncake_lookup_sync": mooncake_sync,
        "mooncake_noop_contract": mooncake_noop,
        "lmcache_dual_path": lmcache_case,
        "multiconnector_first_hit": multi_case,
    },
)
