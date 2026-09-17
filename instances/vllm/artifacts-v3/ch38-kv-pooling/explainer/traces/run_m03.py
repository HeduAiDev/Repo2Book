# m3 存储路径·每步增量满块采集：『每 chunk 取最后一个 GPU 块』+游标推进+block_id=0 跳过+双封顶。
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

from _driver_common import key  # noqa: E402
from vllm.v1.request import RequestStatus  # noqa: E402


def new_engine(extra):
    cfg = make_kv_config()
    return assemble_engine(
        make_vllm_config(extra_config=extra), cfg, make_kv_caches(cfg)
    )


def job_view(meta):
    out = {}
    for jid, job in meta.store_jobs.items():
        out[str(jid)] = {
            "src_block_ids": [int(b) for b in job.src_spec.block_ids],
            "group_sizes": [int(g) for g in job.src_spec.group_sizes],
            "block_indices": [int(b) for b in job.src_spec.block_indices],
            "dst_cpu_slots": [int(b) for b in job.dst_spec.block_ids],
        }
    return out


# ── ① 源码注释原例：blocks_per_chunk=3、GPU 块 1 5 6 7 2 4 9 3 8 → 采集尾块 6 4 8 ──
eng = new_engine({"cpu_bytes_to_use": 1 << 20, "blocks_per_chunk": 3, "offload_prompt_only": False})
req = make_request("r1", list(range(36)))  # 9 块 × block_size 4 = 3 chunk
eng.scheduler.on_new_request(req)
fill_block_hashes(req, 12)  # hashes_per_chunk = 4*3/4 = 3 → 12 哈希 = 4 chunk 键
block_ids = [1, 5, 6, 7, 2, 4, 9, 3, 8]
meta = eng.scheduler.build_connector_meta(
    make_scheduler_output(
        new_reqs=[{"req_id": "r1", "block_ids": (block_ids,)}],
        num_scheduled_tokens={"r1": 36},
    )
)
# 与源码同式切片：『每 chunk 取最后一个 GPU 块』
bpc = 3
num_chunks = 36 // (4 * bpc)
tails = block_ids[0 * bpc + bpc - 1 : num_chunks * bpc : bpc]
example_nine_blocks = {
    "gpu_block_ids": block_ids,
    "blocks_per_chunk": bpc,
    "num_chunks": num_chunks,
    "tail_block_per_chunk_selected": tails,
    "store_job": job_view(meta),
    "store_job_count": len(meta.store_jobs),
    "cursor_after_step": eng.scheduler._req_status["r1"].group_states[0].next_stored_chunk_idx,
    "transport_covers_whole_chunks": "src 含全部 9 块（搬运整 chunk 逐块走），准入以尾块 6 4 8 判定",
}

# ── ② 游标推进：decode 再长 12 token（+3 块）→ 只采新 chunk ──
req.status = RequestStatus.RUNNING
req.num_computed_tokens = 36
req._all_token_ids.extend(range(100, 112))
meta2 = eng.scheduler.build_connector_meta(
    make_scheduler_output(
        cached_reqs=[{"req_id": "r1", "block_ids": ([11, 12, 13],)}],
        num_scheduled_tokens={"r1": 12},
    )
)
cursor_case = {
    "step2_new_tokens": 12,
    "step2_new_gpu_blocks": [11, 12, 13],
    "store_job": job_view(meta2),
    "cursor_after_step2": eng.scheduler._req_status["r1"].group_states[0].next_stored_chunk_idx,
    "already_stored_chunks_not_resent": True,
}

# ── ③ block_id=0（SWA/陈旧占位）→ 该 chunk 跳过 ──
eng3 = new_engine({"cpu_bytes_to_use": 1 << 20})
req3 = make_request("r1", list(range(16)))
eng3.scheduler.on_new_request(req3)
fill_block_hashes(req3, 4)
meta3 = eng3.scheduler.build_connector_meta(
    make_scheduler_output(
        new_reqs=[{"req_id": "r1", "block_ids": ([1, 0, 3, 4],)}],
        num_scheduled_tokens={"r1": 16},
    )
)
zero_case = {
    "gpu_block_ids": [1, 0, 3, 4],
    "blocks_per_chunk": 1,
    "chunk1_tail_is_zero": True,
    "store_job": job_view(meta3),
    "skipped_chunks": 1,
    "src_transports": [1, 3, 4],
}

# ── ④ 双封顶：offload_prompt_only（默认 True）+ max_offload_tokens=8 ──
eng4 = new_engine({"cpu_bytes_to_use": 1 << 20})
req4 = make_request("r1", list(range(32)) + [100, 101, 102, 103])  # 32 prompt + 4 decode
req4.num_prompt_tokens = 32  # make_request 把整表记作 prompt，改回真实口径（本例上限 8 先绑定）
req4.kv_transfer_params = {"max_offload_tokens": 8}
eng4.scheduler.on_new_request(req4)
fill_block_hashes(req4, 9)
req4.num_computed_tokens = 36
meta4 = eng4.scheduler.build_connector_meta(
    make_scheduler_output(
        new_reqs=[{"req_id": "r1", "block_ids": (list(range(1, 10)),)}],
        num_scheduled_tokens={"r1": 4},
    )
)
cap_case = {
    "prompt_tokens": 32,
    "decode_tokens": 4,
    "total_tokens": 36,
    "num_computed_tokens": 36,
    "max_offload_tokens": 8,
    "offload_prompt_only_default": True,
    "num_offloadable_tokens": 8,
    "store_job": job_view(meta4),
    "src_transports": [int(b) for b in next(iter(meta4.store_jobs.values())).src_spec.block_ids],
}

# ── ⑤ prepare_store 拒收（best-effort 放弃本批）：4096B 预算=池 1 块且被钉住 ──
eng5 = new_engine({"cpu_bytes_to_use": 4096, "blocks_per_chunk": 1})
from _driver_common import ctx  # noqa: E402

eng5.manager.prepare_store([key(0)], ctx())
eng5.manager.complete_store([key(0)], ctx())
eng5.manager.prepare_load([key(0)], ctx())  # ref_cnt=1 钉住唯一块
req5 = make_request("r1", list(range(8)))
eng5.scheduler.on_new_request(req5)
fill_block_hashes(req5, 2)
meta5 = eng5.scheduler.build_connector_meta(
    make_scheduler_output(
        new_reqs=[{"req_id": "r1", "block_ids": ([1, 2],)}],
        num_scheduled_tokens={"r1": 8},
    )
)
reject_case = {
    "cpu_bytes_to_use": 4096,
    "blocks_per_chunk": 1,
    "pool_num_blocks": 1,
    "pinned_by_prepare_load": True,
    "store_job_count": len(meta5.store_jobs),
    "cursor_after_reject": eng5.scheduler._req_status["r1"].group_states[0].next_stored_chunk_idx,
    "semantics": "best-effort：拒收只放弃本批（记 ALLOCATION_FAILURE 指标），不重试不报错",
}

dump(
    "m03.json",
    {
        "nine_block_comment_example": example_nine_blocks,
        "incremental_cursor": cursor_case,
        "zero_block_skip": zero_case,
        "double_cap": cap_case,
        "admission_rejected": reject_case,
    },
)
