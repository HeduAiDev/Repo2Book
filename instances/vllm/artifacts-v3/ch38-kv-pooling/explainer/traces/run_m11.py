# m11 分层池五原则：cascade 全层下推 + promotion 两跳 + RETRY 语义（真 fs 副层真文件 I/O）。
import os
import tempfile
import time

from _driver_common import TESTS, ctx, dump, key

import sys

sys.path.insert(0, str(TESTS))
import numpy as np  # noqa: E402
from _kv_harness import make_kv_config, make_vllm_config  # noqa: E402

import vllm.v1.kv_offload.tiering.spec as tiering_spec_mod  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (  # noqa: E402
    build_offloading_config,
)
from vllm.v1.kv_offload.base import LookupResult, ScheduleEndContext  # noqa: E402
from vllm.v1.kv_offload.factory import OffloadingSpecFactory  # noqa: E402


class FakeShmRegion:
    """SharedOffloadRegion 的 host 替身：numpy 池 + memoryview（真部署 /dev/shm mmap）。"""

    def __init__(self, num_blocks, chunk_bytes):
        self.pool = np.zeros((num_blocks, chunk_bytes), dtype=np.uint8)
        self.total_size_bytes = num_blocks * chunk_bytes

    def create_kv_memoryview(self):
        return memoryview(self.pool)


tmp = tempfile.mkdtemp(prefix="ch38_m11_", dir=str(TESTS.parent))  # 项目树内临时目录
root = os.path.join(tmp, "kvfs")

_orig = tiering_spec_mod.SharedOffloadRegion


def _fake_region(engine_id, num_blocks, rank, kv_bytes_per_block, cpu_page_size):
    return FakeShmRegion(num_blocks, kv_bytes_per_block)


tiering_spec_mod.SharedOffloadRegion = _fake_region
try:
    cfg = make_kv_config(num_blocks=8)
    vcfg = make_vllm_config(
        extra_config={
            "cpu_bytes_to_use": 1 << 20,
            "spec_name": "TieringOffloadingSpec",
            "secondary_tiers": [
                {
                    "type": "fs",
                    "root_dir": root,
                    "n_read_threads": 2,
                    "n_write_threads": 2,
                    "locality": "LOCAL",
                }
            ],
        }
    )
    spec = OffloadingSpecFactory.create_spec(build_offloading_config(vcfg, cfg))
    manager = spec.get_manager()
    region = spec._scheduler_mmap  # 调度器侧 mmap（rank=None）——host 上是 numpy 池替身
finally:
    tiering_spec_mod.SharedOffloadRegion = _orig

pool = region.pool
ctx1 = ctx("r1")
k = key(42)

# ── ① store：primary 准入 → complete_store → cascade 到 fs 层 ──
manager.on_new_request(ctx1)
out = manager.prepare_store([k], ctx1)
payload = np.arange(16, dtype=np.uint8)
slot0 = int(out.store_spec.block_ids[0])
pool[slot0][:16] = payload
manager.complete_store([k], ctx1)
hit_after_store = manager.lookup(k, ctx1)

def _walk(root):
    for dirpath, _dirnames, filenames in os.walk(root):
        for fn in filenames:
            yield os.path.join(dirpath, fn)


deadline = time.monotonic() + 5
files = []
while time.monotonic() < deadline:
    all_files = [str(p) for p in _walk(root)] if os.path.isdir(root) else []
    files = [p for p in all_files if p.endswith(".bin")]  # config.json 先落盘——只数块文件
    if files:
        break
    manager._maybe_process_finished_jobs()
    time.sleep(0.01)

store_cascade = {
    "prepare_store_keys": 1,
    "primary_slot": slot0,
    "payload_first16": [int(x) for x in payload[:4]] + ["...(16 bytes 0..15)"],
    "lookup_after_complete_store": hit_after_store.name,
    "cascade_file_count": len(files),
    "cascade_file_relpath": os.path.relpath(files[0], root).replace("\\", "/") if files else None,
    "cascade_anchor": "tiering/manager.py:L586-L630：complete_store 即向全部 secondary 下推（prepare_read 钉 ref_cnt）",
}

# ── ② primary 清空（reset_cache 只清 primary；fs 层刻意保留）→ 查询两跳 promotion ──
manager.reset_cache()
end_ctx = ScheduleEndContext(new_req_ids=[], preempted_req_ids=[])
rounds = []
r1 = manager.lookup(k, ctx1)  # fs 异步查询入队 → RETRY
rounds.append(("lookup#1", r1.name))
manager.on_schedule_end(end_ctx)  # flush 查询批
r2 = manager.lookup(k, ctx1)  # fs HIT → promotion 占位（ref_cnt=-1）→ RETRY
rounds.append(("lookup#2", r2.name))
manager.on_schedule_end(end_ctx)  # 批量 submit_load（fs→primary）
poll_count = 0
deadline2 = time.monotonic() + 5
while manager.lookup(k, ctx1) is not LookupResult.HIT:
    manager.on_schedule_end(end_ctx)
    poll_count += 1
    time.sleep(0.01)  # 让 fs 读线程拿到 GIL
    assert time.monotonic() < deadline2, "promotion 应完成并转 HIT"
final = manager.lookup(k, ctx1)
hit_rows = [row for row in pool if row[0] == 0]
promoted_bytes_match = any(bytes(row[:16]) == bytes(payload) for row in hit_rows)

promotion = {
    "reset_clears_primary_only": "fs 文件仍在（secondary 刻意不 reset——tiering/manager.py:L773-L787）",
    "rounds": [[a, b] for a, b in rounds],
    "final_lookup": final.name,
    "extra_schedule_end_rounds": poll_count,
    "promotion_slot_has_payload": promoted_bytes_match,
    "two_hop": "secondary(fs)→primary(CPU) 一跳 + primary→GPU 一跳；期间请求 RETRY 稍后再问",
    "anchors": {
        "lookup": "tiering/manager.py:L282-L350（primary 先查→secondary 命中即 _initiate_promotion→RETRY；primary 满→MISS）",
        "promotion_defer": "tiering/manager.py:L380-L427（prepare_write 占位 ref_cnt=-1；on_schedule_end 批量 submit_load）",
        "gateway": "tiering/manager.py:L4-L21 五原则 docstring（CPU primary 是唯一能 DMA GPU 的层）",
    },
}

# ── ③ 分层禁用 store_threshold ≥2 ──
tiering_spec_mod.SharedOffloadRegion = lambda *a, **k: FakeShmRegion(1, 4096)
try:
    vcfg2 = make_vllm_config(
        extra_config={
            "cpu_bytes_to_use": 1 << 20,
            "spec_name": "TieringOffloadingSpec",
            "store_threshold": 2,
            "secondary_tiers": [],
        }
    )
    spec2 = OffloadingSpecFactory.create_spec(build_offloading_config(vcfg2, make_kv_config()))
    try:
        spec2.get_manager()
        threshold_gate = "constructed"
    except ValueError as e:
        threshold_gate = f"ValueError: {e}"
finally:
    tiering_spec_mod.SharedOffloadRegion = _orig

# ── ④ 副层工厂注册：fs/p2p 示范两条（obj/example 已删）──
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory  # noqa: E402
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager  # noqa: E402
from vllm.v1.kv_offload.tiering.p2p.manager import P2PSecondaryTierManager  # noqa: E402

factory_case = {
    "fs_registered": SecondaryTierFactory.get_tier_class({"type": "fs"}) is FileSystemTierManager,
    "p2p_registered": SecondaryTierFactory.get_tier_class({"type": "p2p"}) is P2PSecondaryTierManager,
}
try:
    SecondaryTierFactory.get_tier_class({"type": "obj"})
    factory_case["obj"] = "resolved"
except ValueError as e:
    factory_case["obj"] = f"ValueError: {e}"

# ── ⑤ 调度器侧 mmap 同一池（rank=None）：secondary 的 I/O 全在调度器进程 ──
sched_side = {
    "anchor": "tiering/spec.py:L170-L215（rank=None mmap 同一 /dev/shm 区；primary.get_kv_memoryview 供 secondary 直读写）",
    "split": "GPU↔CPU DMA 在 worker 进程、CPU↔secondary I/O 在调度器进程——同一物理池两个进程各管一段搬运",
    "host_note": "host 无 /dev/shm：本 trace 以 numpy 池替身承载 create_kv_memoryview 位（真部署 mmap 文件）",
    "io_note": "Windows host：O_DIRECT 不支持 → fs 层自动回退 buffered I/O（源码自带回退日志）；Linux 真部署走 O_DIRECT+readv（fs/io.py 带 host SEAM：readv 换 os.read+拷贝、强制 O_BINARY）",
}

import shutil  # noqa: E402

shutil.rmtree(tmp, ignore_errors=True)

dump(
    "m11.json",
    {
        "store_then_cascade_to_fs": store_cascade,
        "promotion_two_hop_retry": promotion,
        "store_threshold_gate": threshold_gate,
        "secondary_tier_factory": factory_case,
        "scheduler_side_mmap": sched_side,
    },
)
