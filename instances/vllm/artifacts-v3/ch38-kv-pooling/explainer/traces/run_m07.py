# m7 四态查询：HIT/HIT_PENDING/RETRY/MISS 前缀查找 + SWA 尾扫 + 多组收敛 + 在飞互斥 + touch。
from _driver_common import TESTS, ctx, dump, key

import sys

sys.path.insert(0, str(TESTS))
import torch  # noqa: E402
from _kv_harness import (  # noqa: E402
    assemble_engine,
    fill_block_hashes,
    make_kv_caches,
    make_kv_config,
    make_request,
    make_vllm_config,
)

from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.base import LookupResult, make_offload_key  # noqa: E402


class ScriptedManager:
    """按 key 回放四态的假账本（只驱动调度器查找算法；RETRY 真源在 tiering 层，见 m11）。"""

    def __init__(self, results):
        self.results = results
        self.seen = []
        self.touched = []

    def lookup(self, k, c):
        self.seen.append(k)
        return LookupResult[self.results.get(k, "MISS")]

    def touch(self, keys, c):
        self.touched.extend(keys)


def sched():
    cfg = make_kv_config()
    eng = assemble_engine(
        make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20}), cfg, make_kv_caches(cfg)
    )
    return eng.scheduler


s = sched()

# ── ① 前缀查找：HIT 计数、MISS break（MISS 之后必 MISS）──
keys4 = [key(i) for i in range(4)]
s.manager = ScriptedManager({keys4[0]: "HIT", keys4[1]: "HIT", keys4[3]: "HIT"})
n = s._maximal_prefix_lookup(keys4, ctx(), None, None, 0)
prefix_hit_miss = {
    "keys": [0, 1, 2, 3],
    "scripted_states": ["HIT", "HIT", "MISS", "HIT"],
    "returned_hit_count": n,
    "verdict": "k2 MISS 即 break——k3 在池也不看（链式早停）",
}

# ── ② HIT_PENDING：计数但整体 None（写入在飞）──
keys3 = [key(i) for i in range(3)]
s.manager = ScriptedManager({keys3[0]: "HIT", keys3[1]: "HIT_PENDING"})
r = s._maximal_prefix_lookup(keys3, ctx(), None, None, 0)
hit_pending_case = {
    "keys": [0, 1, 2],
    "scripted_states": ["HIT", "HIT_PENDING", "MISS"],
    "returned": str(r),
    "verdict": "HIT_PENDING 计数（hit_count 继续走）但 defer_lookup=True → 整体 None 稍后再问",
}

# ── ③ RETRY：不计数、不 break——继续扫到 MISS（让 manager 踢异步查询）──
s.manager = ScriptedManager({keys3[0]: "RETRY", keys3[1]: "HIT", keys3[2]: "MISS"})
r = s._maximal_prefix_lookup(keys3, ctx(), None, None, 0)
retry_case = {
    "keys": [0, 1, 2],
    "scripted_states": ["RETRY", "HIT", "MISS"],
    "returned": str(r),
    "seen_keys_in_order": [int.from_bytes(k[:32], "big") for k in s.manager.seen],
    "scanned_all": s.manager.seen == keys3,
    "verdict": "RETRY 不 break：扫描穿到 MISS 才停——manager 借此踢异步查询",
}

# ── ④ 入口层：任一 defer 态 → get_num_new_matched_tokens 返回 None（接 ch16 skipped 队列）──
s2 = sched()
req = make_request("r1", list(range(16)))
s2.on_new_request(req)
fill_block_hashes(req, 4)
s2.manager = ScriptedManager({key(0): "HIT_PENDING"})
n, la = s2.get_num_new_matched_tokens(req, 0)
entry_none = {
    "request_tokens": 16,
    "first_chunk_state": "HIT_PENDING",
    "returned_num": str(n),
    "returned_load_async": la,
    "verdict": "(None, False) = 稍后再问——请求进 ch16 skipped 队列下步重查",
}

# ── ⑤ 在飞 job 互斥：transfer_jobs 非空 → None 延后 ──
s3 = sched()
req3 = make_request("r1", list(range(16)))
s3.on_new_request(req3)
fill_block_hashes(req3, 4)
s3._req_status["r1"].transfer_jobs.add(99)
n5, _ = s3.get_num_new_matched_tokens(req3, 0)
inflight_mutex = {
    "in_flight_job_ids": [99],
    "returned_num": str(n5),
    "invariant_anchor": "scheduler.py:L283-L285：at any given time either a single load job, or one or more store jobs",
}

# ── ⑥ touch 覆盖 GPU 命中块（OffloadingManager.touch 的存在理由）──
s4 = sched()
req4 = make_request("r1", list(range(16)))
s4.on_new_request(req4)
hashes = fill_block_hashes(req4, 4)
s4.manager = ScriptedManager({})
s4.get_num_new_matched_tokens(req4, 8)  # GPU 已算 8 token（2 块本地前缀缓存命中）
s4.manager.touched.clear()
s4._touch(s4._req_status["r1"])
touch_case = {
    "gpu_computed_tokens": 8,
    "gpu_hit_chunks": 2,
    "touched_key_count": len(s4.manager.touched),
    "touched_covers_all_4_chunks": set(s4.manager.touched)
    == {make_offload_key(h, 0) for h in hashes},
    "verdict": "GPU 前缀缓存命中的块也要刷池内新鲜度（不读也要 touch）",
}

# ── ⑦ SWA 组从尾数连续窗口 ──
keys6 = [key(i) for i in range(6)]
s.manager = ScriptedManager({keys6[3]: "HIT", keys6[4]: "HIT", keys6[5]: "MISS"})
r7 = s._sliding_window_lookup(keys6, 2, ctx())
s.manager = ScriptedManager({keys6[3]: "HIT", keys6[5]: "HIT"})
r7b = s._sliding_window_lookup(keys6, 2, ctx())
swa_case = {
    "keys": [0, 1, 2, 3, 4, 5],
    "window_chunks": 2,
    "case_a_states": ["MISS", "MISS", "MISS", "HIT", "HIT", "MISS"],
    "case_a_returned_end_idx": r7,
    "case_b_states": ["MISS", "MISS", "MISS", "HIT", "MISS", "HIT"],
    "case_b_returned": r7b,
    "verdict": "从尾向前扫连续命中窗口；窗口中断（case_b）→ 0（无可用命中）",
}

# ── ⑧ 多组收敛：full 组长命中被 SWA 组收紧 → 12 token ──
full = FullAttentionSpec(block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float32)
swa = SlidingWindowSpec(
    block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float32, sliding_window=4
)
cfg = KVCacheConfig(
    num_blocks=8,
    kv_cache_tensors=[
        KVCacheTensor(size=full.page_size_bytes * 8, shared_by=["f.0"]),
        KVCacheTensor(size=swa.page_size_bytes * 8, shared_by=["s.0"]),
    ],
    kv_cache_groups=[
        KVCacheGroupSpec(layer_names=["f.0"], kv_cache_spec=full),
        KVCacheGroupSpec(layer_names=["s.0"], kv_cache_spec=swa),
    ],
)
eng = assemble_engine(
    make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20}), cfg, make_kv_caches(cfg)
)
s5 = eng.scheduler
req5 = make_request("r1", list(range(16)))
s5.on_new_request(req5)
fill_block_hashes(req5, 4)
full_keys = [make_offload_key((0 * 1_000_003 + i).to_bytes(32, "big"), 0) for i in range(4)]
swa_keys = [make_offload_key((0 * 1_000_003 + i).to_bytes(32, "big"), 1) for i in range(4)]
results = {k: "HIT" for k in full_keys}
results[swa_keys[2]] = "HIT"
s5.manager = ScriptedManager(results)
n8, _ = s5.get_num_new_matched_tokens(req5, 0)
convergence = {
    "request_tokens": 16,
    "full_group_chunks": 4,
    "full_group_all_hit_tokens": 16,
    "swa_window_chunks": 1,
    "swa_tail_states": ["MISS", "HIT"],
    "swa_supported_chunks": 3,
    "returned_num_hit_tokens": n8,
    "verdict": "full 组 16 token 被 SWA 组收紧到 12——多组收敛循环取交集（混合不动点）",
}

dump(
    "m07.json",
    {
        "prefix_hit_miss_break": prefix_hit_miss,
        "hit_pending_defers": hit_pending_case,
        "retry_keeps_scanning": retry_case,
        "entry_none_semantics": entry_none,
        "inflight_job_mutex": inflight_mutex,
        "touch_covers_gpu_hits": touch_case,
        "swa_tail_window": swa_case,
        "multi_group_convergence": convergence,
        "four_states_legend": {
            "HIT": "在池且可读",
            "HIT_PENDING": "在池、写入在飞——计数但整体 None",
            "RETRY": "位置未定——不计数、不 break（manager 踢异步查询）",
            "MISS": "确定不在——break（MISS 之后必 MISS）",
            "None_by_connector": "任一 defer 态或命中块正被加载 → get_num_new_matched_tokens 返回 None（稍后再问）",
        },
    },
)
