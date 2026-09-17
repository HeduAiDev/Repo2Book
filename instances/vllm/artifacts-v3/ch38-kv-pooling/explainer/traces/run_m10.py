# m10 flush 围栏：块复用/抢占/终局三触发 → jobs_to_flush → handle_preemptions 抢先提交+wait。
from _driver_common import TESTS, dump

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

from vllm.v1.request import RequestStatus  # noqa: E402


def new_engine():
    cfg = make_kv_config()
    return assemble_engine(
        make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20}), cfg, make_kv_caches(cfg)
    )


# ── ① 终局触发：请求带在飞 store 结束 → request_finished 登记 + 本步即收 flush ──
eng = new_engine()
req = make_request("r1", list(range(8)))
eng.scheduler.on_new_request(req)
fill_block_hashes(req, 2)
req.status = RequestStatus.FINISHED_STOPPED
takeover, params = eng.scheduler.request_finished(req)
meta = eng.scheduler.build_connector_meta(
    make_scheduler_output(
        new_reqs=[{"req_id": "r1", "block_ids": ([1, 2],)}],
        num_scheduled_tokens={"r1": 8},
        finished_req_ids={"r1"},
    )
)
terminal_case = {
    "request_finished_returns": [takeover, str(params)],
    "store_job_ids": sorted(meta.store_jobs),
    "jobs_to_flush_at_terminal_step": sorted(meta.jobs_to_flush or set()),
    "watched_blocks": sorted(eng.scheduler._block_id_to_pending_jobs),
    "note": "终局步：finished 请求的新块 [1,2] 同时也在本步分配集 → 命中盯防账本",
}

# 真实步序：get_finished 里只入延迟队列（未提交）
eng.worker.prepare_store_kv(meta)
queued_not_submitted = list(eng.host_worker.submitted)

# ── ② 块复用触发：新请求复用块 1 → 盯防账本命中 → flush ──
req2 = make_request("r2", [1, 2, 3, 4])
fill_block_hashes(req2, 1)
eng.scheduler.on_new_request(req2)
meta2 = eng.scheduler.build_connector_meta(
    make_scheduler_output(
        new_reqs=[{"req_id": "r2", "block_ids": ([1],)}],
        num_scheduled_tokens={"r2": 4},
    )
)
reuse_case = {
    "reused_block": 1,
    "jobs_to_flush_on_reuse": sorted(meta2.jobs_to_flush or set()),
    "submitted_before_fence": [(j, d) for j, d in queued_not_submitted],
    "watched_blocks_before_fence": sorted(eng.scheduler._block_id_to_pending_jobs),
}

# ── ③ worker 侧围栏：抢先提交 + wait 同步 ──
waits = []
orig_wait = eng.host_worker.wait
eng.host_worker.wait = lambda ids: waits.append(sorted(ids))
eng.worker.handle_preemptions(meta2)
fence = {
    "submitted_by_fence": [(j, d) for j, d in eng.host_worker.submitted],
    "wait_calls": waits,
    "deferred_queue_cleared": len(eng.worker._unsubmitted_store_jobs) == 0,
    "order": "先把 jobs_to_flush 的 store 从延迟队列弹出提交，再 wait() 同步等到完成",
    "anchor": "offloading/worker.py:L292-L317（ch16 m7 handle_preemptions 挂点的池化填实）",
}

# ── ④ 抢占触发：running 请求被抢占 → 其在飞 store 进 flush ──
eng4 = new_engine()
req4 = make_request("r1", list(range(8)))
eng4.scheduler.on_new_request(req4)
fill_block_hashes(req4, 2)
meta4 = eng4.scheduler.build_connector_meta(
    make_scheduler_output(
        new_reqs=[{"req_id": "r1", "block_ids": ([1, 2],)}],
        num_scheduled_tokens={"r1": 8},
    )
)
meta4b = eng4.scheduler.build_connector_meta(
    make_scheduler_output(
        cached_reqs=[{"req_id": "r1", "block_ids": ([],)}],
        num_scheduled_tokens={"r1": 0},
        preempted_req_ids={"r1"},
    )
)
preempt_case = {
    "store_job_ids_step1": sorted(meta4.store_jobs),
    "preempted_req": "r1",
    "jobs_to_flush_on_preemption": sorted(meta4b.jobs_to_flush or set()),
}

# ── ⑤ request_finished 不接管（对照 ch37 P/D 的 True 接管）──
eng5 = new_engine()
req5 = make_request("r1", list(range(8)))
eng5.scheduler.on_new_request(req5)
ok5, p5 = eng5.scheduler.request_finished(req5)
stray = make_request("rX", [1])
okx, _ = eng5.scheduler.request_finished(stray)
request_finished_case = {
    "tracked_request_returns": [ok5, str(p5)],
    "untracked_request_returns": [okx, "None"],
    "verdict": "返回 False 不接管块释放——只登记 non_sliding_window_block_ids 供 flush 盯防",
}

dump(
    "m10.json",
    {
        "terminal_trigger": terminal_case,
        "block_reuse_trigger": reuse_case,
        "worker_fence_submit_and_wait": fence,
        "preemption_trigger": preempt_case,
        "request_finished_no_takeover": request_finished_case,
    },
)
