# m16 请求级旋钮：max_offload_tokens 只卸前 N token + kv_load_tiers 层过滤器（TierFilter）。
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

from vllm.v1.kv_offload.base import Locality, Medium, TierFilter  # noqa: E402


def new_engine():
    cfg = make_kv_config()
    return assemble_engine(
        make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20}), cfg, make_kv_caches(cfg)
    )


# ── ① max_offload_tokens=8：32 prompt + 4 decode 的请求只卸前 8 token（2 块）──
eng = new_engine()
req = make_request(
    "r1", list(range(32)) + [100, 101, 102, 103], kv_transfer_params={"max_offload_tokens": 8}
)
# make_request 把整表 36 token 都记作 prompt；手工把口径改回「prompt 32 + decode 4」
# （Request 属性可写，_calc_num_offloadable_tokens 只读它）。
req.num_prompt_tokens = 32
eng.scheduler.on_new_request(req)
fill_block_hashes(req, 9)
req.num_computed_tokens = 36
meta = eng.scheduler.build_connector_meta(
    make_scheduler_output(
        new_reqs=[{"req_id": "r1", "block_ids": (list(range(1, 10)),)}],
        num_scheduled_tokens={"r1": 4},
    )
)
job = next(iter(meta.store_jobs.values()))
st = eng.scheduler._req_status["r1"]
max_offload = {
    "prompt_tokens": 32,
    "decode_tokens": 4,
    "total_tokens": 36,
    "max_offload_tokens": 8,
    "parsed_max_offload_tokens": st.max_offload_tokens,
    "num_offloadable_tokens_capped": 8,
    "src_blocks_transported": [int(b) for b in job.src_spec.block_ids],
    "use_case": "已知前缀值得缓存、请求特有尾部不值得（docs：experimental）",
}

# ── ② 对照：无旋钮 → offload_prompt_only（默认 True）卸 prompt 32 token（8 块）──
eng2 = new_engine()
req2 = make_request("r2", list(range(32)) + [100, 101, 102, 103])
req2.num_prompt_tokens = 32  # 同①：把口径改回「prompt 32 + decode 4」
eng2.scheduler.on_new_request(req2)
fill_block_hashes(req2, 9)
req2.num_computed_tokens = 36
meta2 = eng2.scheduler.build_connector_meta(
    make_scheduler_output(
        new_reqs=[{"req_id": "r2", "block_ids": (list(range(1, 10)),)}],
        num_scheduled_tokens={"r2": 4},
    )
)
job2 = next(iter(meta2.store_jobs.values()))
st2 = eng2.scheduler._req_status["r2"]
no_knob = {
    "num_prompt_tokens": 32,
    "num_offloadable_tokens": eng2.scheduler._calc_num_offloadable_tokens(st2, 36),
    "src_blocks_transported": [int(b) for b in job2.src_spec.block_ids],
    "note": "offload_prompt_only 默认 True → prompt-only 上限 32 token 绑定（decode 尾 4 token 不卸）",
}

# ── ③ 非法值忽略 ──
eng3 = new_engine()
req3 = make_request("r3", list(range(16)), kv_transfer_params={"max_offload_tokens": "big"})
eng3.scheduler.on_new_request(req3)
invalid = {
    "raw_value": "big",
    "parsed": str(eng3.scheduler._req_status["r3"].max_offload_tokens),
    "note": "type(raw) is int 才收——字符串打 warning 忽略",
}

# ── ④ kv_load_tiers：per-request 层过滤器 ──
eng4 = new_engine()
req4 = make_request(
    "r4",
    [1] * 8,
    kv_transfer_params={
        "kv_load_tiers": [
            {"medium": "cpu"},
            {"medium": "storage", "locality": "local"},
        ]
    },
)
eng4.scheduler.on_new_request(req4)
f = eng4.scheduler._req_status["r4"].req_context.load_tier_filter
req5 = make_request("r5", [1], kv_transfer_params={"kv_load_tiers": [{"medium": "bogus"}]})
eng4.scheduler.on_new_request(req5)
f2 = eng4.scheduler._req_status["r5"].req_context.load_tier_filter
req6 = make_request("r6", [1], kv_transfer_params={"kv_load_tiers": []})
eng4.scheduler.on_new_request(req6)
f6 = eng4.scheduler._req_status["r6"].req_context.load_tier_filter
tiers = {
    "matcher_count": len(f.matchers),
    "cpu_tier_allowed": f.allows(Medium.CPU, None),
    "storage_local_allowed": f.allows(Medium.STORAGE, Locality.LOCAL),
    "storage_remote_allowed": f.allows(Medium.STORAGE, Locality.REMOTE),
    "bogus_entry_falls_back": f2 is TierFilter.ALL,
    "bogus_allows_storage": f2.allows(Medium.STORAGE, None),
    "explicit_empty_allows_cpu": f6.allows(Medium.CPU, None),
    "all_filter_allows_remote_storage": TierFilter.ALL.allows(Medium.STORAGE, Locality.REMOTE),
    "use_case": "热数据只走 CPU 层、冷批量才许下探 fs——per-request 按 medium/locality 限定参与的层",
    "anchor": "offloading/scheduler.py:L387-L442（_parse_tier_filter/_create_req_context）+ base.py:L61-L87",
}

dump(
    "m16.json",
    {
        "max_offload_tokens_cap": max_offload,
        "no_knob_control": no_knob,
        "invalid_value_ignored": invalid,
        "kv_load_tiers_filter": tiers,
    },
)
