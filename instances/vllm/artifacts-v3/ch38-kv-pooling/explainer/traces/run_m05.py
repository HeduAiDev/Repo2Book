# m5 延迟一步提交（NOTE(orozery)）+ 一个块进出池的一生全链路（供 m5/m8/m9 叙事）。
from _driver_common import TESTS, dump

import sys

sys.path.insert(0, str(TESTS))
import torch  # noqa: E402
from _kv_harness import (  # noqa: E402
    LAYER_NAMES,
    assemble_engine,
    block_checksum,
    fill_block_hashes,
    make_blocks,
    make_kv_caches,
    make_kv_config,
    make_request,
    make_scheduler_output,
    make_vllm_config,
)

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (  # noqa: E402
    OffloadingConnectorMetadata,
)
from vllm.v1.outputs import KVConnectorOutput  # noqa: E402

kv_config = make_kv_config()
eng = assemble_engine(
    make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20}), kv_config, make_kv_caches(kv_config)
)
layer0 = eng.kv_caches[LAYER_NAMES[0]]
page = layer0.shape[1]

# ── 步 N：prompt 16 token = 4 块 → store job 0 ──
req = make_request("r1", list(range(16)))
eng.scheduler.on_new_request(req)
fill_block_hashes(req, 4)
n_hit, async_load = eng.scheduler.get_num_new_matched_tokens(req, 0)
meta = eng.scheduler.build_connector_meta(
    make_scheduler_output(
        new_reqs=[{"req_id": "r1", "block_ids": ([1, 2, 3, 4],)}],
        num_scheduled_tokens={"r1": 16},
    )
)
store_jid = next(iter(meta.store_jobs))

# 每块写入互不相同的字节（0x30..0x33 = 48..51）
fill_bytes = [48 + i for i in range(4)]
for i, bid in enumerate([1, 2, 3, 4]):
    layer0[bid] = torch.full_like(layer0[bid], fill_bytes[i])
gpu_checksums = [int(layer0[bid].sum()) for bid in [1, 2, 3, 4]]

step_n = {
    "prompt_tokens": 16,
    "gpu_blocks": [1, 2, 3, 4],
    "fill_byte_per_block": fill_bytes,
    "gpu_block_checksums": gpu_checksums,
    "page_bytes": page,
    "store_job_id": store_jid,
    "lookup_before_store": {"num_hit": n_hit, "load_async": async_load},
}

# ── 延迟一步：get_finished 只入队不提交 ──
eng.worker.prepare_store_kv(meta)
after_queue = list(eng.host_worker.submitted)
sending0, recving0 = eng.worker.get_finished(set())
after_get_finished = list(eng.host_worker.submitted)
deferred = {
    "after_prepare_store_kv_submitted": after_queue,
    "after_get_finished_submitted": after_get_finished,
    "finished_sending": sorted(sending0),
    "finished_recving": sorted(recving0),
    "queue_len": len(eng.worker._unsubmitted_store_jobs),
    "note_anchor": "offloading/worker.py:L338-L344 NOTE(orozery)：卸载排在采样相关传输之后，不拖慢 token 生成",
    "no_token_step_note": "get_finished 在无 token 步也会被走到（ch16 kv_connector_no_forward）——store 永不遗漏",
}

# ── 步 N+1：start_kv_transfers 开头才提交 → host 同步 DMA 立即完成 ──
next_meta = OffloadingConnectorMetadata(load_jobs={}, store_jobs={})
eng.worker.start_kv_transfers(next_meta)
submitted_next = list(eng.host_worker.submitted)
sending1, recving1 = eng.worker.get_finished(set())
wmeta = eng.worker.build_connector_worker_meta()
cpu_tensor = eng.host_worker.cpu_tensors[0]
cpu_checksums = [int(cpu_tensor[i].sum()) for i in range(4)]
byte_equal = [
    block_checksum(layer0, b, page) == bytes(cpu_tensor[i].numpy())
    for i, b in enumerate([1, 2, 3, 4])
]
step_n1 = {
    "submitted_at_next_start": [(j, d) for j, d in submitted_next],
    "finished_sending_after_store": sorted(sending1),
    "finished_recving_after_store": sorted(recving1),
    "completed_jobs": {str(k): v for k, v in wmeta.completed_jobs.items()},
    "cpu_slot_checksums": cpu_checksums,
    "byte_equal_gpu_vs_cpu": byte_equal,
}

# ── 调度器结算（world_size=1 → 聚满即 complete_store）──
eng.scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=wmeta))
settled = {
    "has_pending_push_work": eng.scheduler.has_pending_push_work(),
    "blocks_freed": False,
    "note": "store 不发 finished_sending：完成走 completed_jobs 计数（m9）",
}

# ── 查：同前缀新请求 → 16 token 全命中 ──
req2 = make_request("r2", list(range(16)))
eng.scheduler.on_new_request(req2)
fill_block_hashes(req2, 4, seed=0)
n_hit2, async2 = eng.scheduler.get_num_new_matched_tokens(req2, 0)
query = {"num_hit_tokens": n_hit2, "load_async": async2, "prompt_tokens": 16}

# ── 载：update_state_after_alloc → load job 1 → CPU→GPU DMA → finished_recving ──
eng.scheduler.update_state_after_alloc(req2, make_blocks([[4, 5, 6, 7]]), 16)
meta2 = eng.scheduler.build_connector_meta(
    make_scheduler_output(cached_reqs=[{"req_id": "r2"}], num_scheduled_tokens={"r2": 1})
)
load_jid = next(iter(meta2.load_jobs))
job = meta2.load_jobs[load_jid]
eng.worker.start_kv_transfers(meta2)
submitted_load = list(eng.host_worker.submitted)
_, recv = eng.worker.get_finished(set())
wmeta2 = eng.worker.build_connector_worker_meta()
eng.scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=wmeta2))
dst_equal = [
    block_checksum(layer0, b, page) == bytes(cpu_tensor[i].numpy())
    for i, b in enumerate([4, 5, 6, 7])
]
load = {
    "load_job_id": load_jid,
    "dst_gpu_blocks": [int(b) for b in job.dst_spec.block_ids],
    "dst_group_sizes": [int(g) for g in job.dst_spec.group_sizes],
    "dst_block_indices": [int(b) for b in job.dst_spec.block_indices],
    "src_cpu_slots": [int(b) for b in job.src_spec.block_ids],
    "submitted_sequence": [(j, d) for j, d in submitted_load],
    "finished_recving": sorted(recv),
    "dst_block_checksums": [int(layer0[b].sum()) for b in [4, 5, 6, 7]],
    "byte_equal_dst_vs_cpu_pool": dst_equal,
    "handoff": "finished_recving → ch16 WAITING_FOR_REMOTE_KVS 提升路径（_try_promote 补缓存+全命中退一 token）",
}

# ── 命中块正被加载 → 后续同前缀请求延后（_chunks_being_loaded 已在结算后清空，此处记录语义）──
inflight_note = {
    "chunks_being_loaded_after_settle": sorted(str(k) for k in (eng.scheduler._chunks_being_loaded or set())),
    "gate_anchor": "offloading/scheduler.py:L693-L717：命中块在 _chunks_being_loaded → None 延后",
}

dump(
    "m05.json",
    {
        "step_n_store_collected": step_n,
        "deferred_one_step": deferred,
        "step_n1_submitted": step_n1,
        "scheduler_settled": settled,
        "query_same_prefix": query,
        "load_roundtrip": load,
        "inflight_gate": inflight_note,
    },
)
