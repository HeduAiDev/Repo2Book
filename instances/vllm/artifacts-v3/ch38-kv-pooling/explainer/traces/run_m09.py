# m9 完成回收：completed_jobs {job:count} 逐 worker 计数、聚满 world_size 才结算；
# store 永不发 finished_sending；has_pending_push_work 保活。
from _driver_common import TESTS, ctx, dump, key

import sys

sys.path.insert(0, str(TESTS))
from _kv_harness import (  # noqa: E402
    assemble_engine,
    fill_block_hashes,
    make_kv_caches,
    make_kv_config,
    make_request,
    make_scheduler_output,
    make_vllm_config,
)

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (  # noqa: E402
    OffloadingWorkerMetadata,
)
from vllm.v1.outputs import KVConnectorOutput  # noqa: E402

# ── ① 跨 worker 聚合语义：aggregate 求和 ──
w1 = OffloadingWorkerMetadata()
w2 = OffloadingWorkerMetadata()
w1.mark_completed(0)
w2.mark_completed(0)
w2.mark_completed(7)
merged = w1.aggregate(w2)
aggregate_case = {
    "w1_completed": {"0": 1},
    "w2_completed": {"0": 1, "7": 1},
    "merged": {str(k): v for k, v in merged.completed_jobs.items()},
    "docstring_anchor": "common.py:L75-L83：each worker reports {job_id: 1}；aggregate() sums counts across workers",
}

# ── ② world_size=2：两个 worker 各报一次才结算 ──
kv_config = make_kv_config()
eng = assemble_engine(
    make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20}, world_size=2),
    kv_config,
    make_kv_caches(kv_config),
)
req = make_request("r1", list(range(8)))
eng.scheduler.on_new_request(req)
fill_block_hashes(req, 2)
meta = eng.scheduler.build_connector_meta(
    make_scheduler_output(
        new_reqs=[{"req_id": "r1", "block_ids": ([1, 2],)}],
        num_scheduled_tokens={"r1": 8},
    )
)
job_ids = sorted(meta.store_jobs)
st = eng.scheduler._req_status["r1"]
key0 = st.group_states[0].offload_keys[0]


def lookup_state():
    return eng.manager.lookup(key0, ctx()).name


half1 = OffloadingWorkerMetadata()
half1.mark_completed(*job_ids)
eng.scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=half1))
after_first = {
    "world_size": 2,
    "pending_count_after_first_report": eng.scheduler._jobs[job_ids[0]].pending_count,
    "lookup_state": lookup_state(),
    "has_pending_push_work": eng.scheduler.has_pending_push_work(),
}
half2 = OffloadingWorkerMetadata()
half2.mark_completed(*job_ids)
eng.scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=half2))
after_second = {
    "pending_count_after_second_report": 0 if job_ids[0] not in eng.scheduler._jobs else -1,
    "job_settled": job_ids[0] not in eng.scheduler._jobs,
    "lookup_state": lookup_state(),
    "has_pending_push_work": eng.scheduler.has_pending_push_work(),
    "req_status_cleaned": "r1" not in eng.scheduler._req_status,
}

# ── ③ world_size=1 对照：单 worker 一次报满即结算 ──
kv_config1 = make_kv_config()
eng1 = assemble_engine(
    make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20}, world_size=1),
    kv_config1,
    make_kv_caches(kv_config1),
)
req1 = make_request("r1", list(range(8)))
eng1.scheduler.on_new_request(req1)
fill_block_hashes(req1, 2)
meta1 = eng1.scheduler.build_connector_meta(
    make_scheduler_output(
        new_reqs=[{"req_id": "r1", "block_ids": ([1, 2],)}],
        num_scheduled_tokens={"r1": 8},
    )
)
jid1 = next(iter(meta1.store_jobs))
key0_1 = eng1.scheduler._req_status["r1"].group_states[0].offload_keys[0]
single = OffloadingWorkerMetadata()
single.mark_completed(jid1)
eng1.scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=single))
world1_case = {
    "world_size": 1,
    "lookup_state_after_single_report": eng1.manager.lookup(key0_1, ctx()).name,
    "has_pending_push_work": eng1.scheduler.has_pending_push_work(),
}

# ── ④ store 完成不发 finished_sending（对照 load 的 finished_recving）──
no_sending = {
    "store_completion_channel": "completed_jobs 计数（不发 finished_sending）",
    "load_completion_channel": "finished_recving → ch16 提升路径",
    "anchor": "offloading/worker.py:L346-L381 get_finished docstring",
    "stale_job_note": "reset_cache 后 _stale_job_threshold 之后的旧 job 回报被丢弃（scheduler.py:L1220-L1226）",
}

dump(
    "m09.json",
    {
        "aggregate_across_workers": aggregate_case,
        "world2_first_report": after_first,
        "world2_second_report_settles": after_second,
        "world1_single_report_settles": world1_case,
        "store_no_finished_sending": no_sending,
    },
)
