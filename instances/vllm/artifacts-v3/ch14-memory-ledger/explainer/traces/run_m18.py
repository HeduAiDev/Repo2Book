# run_m18.py — m18 逐组收益合成：把 m5（组化）/m11（回收感知门）/m15（混合容量）
# 三处既有场景的算术复算到一张可对照的表上。场景参数与公式出处：
#   - 混合门（窗 512、块 16、在途 0、池 1000、4096-token 请求）：traces/m11.json
#     hybrid_gate；SWA cap = cdiv(min(W-1+in_flight, L), bs)+1（精简版与 pin 同源，
#     kv_cache_interface.py:L587-L608）。
#   - #39734 心算场景（100000-token prompt、窗 1024、块 16、在途按一个 16-token
#     chunk 计）：chapter.md「第二道」的示意推演，此处把 cdiv 算实。
#   - 混合容量核算（窗 512、在途 8192、max_len 4096、池 1024）：traces/m15.json。
# cdiv 用精简版 math_utils（与 pin vllm.utils.math_utils 同一实现）。
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from implementation.math_utils import cdiv

OUT = {}

# 场景一：混合准入门（m11 hybrid_gate 的复算）
W, BS, INFLIGHT, L, POOL = 512, 16, 0, 4096, 1000
full_blocks = cdiv(L, BS)
swa_capped = cdiv(min(W - 1 + INFLIGHT, L), BS) + 1
swa_uncapped_full_len = cdiv(L, BS)
OUT["hybrid_gate"] = {
    "window": W, "block_size": BS, "in_flight": INFLIGHT, "max_len": L, "pool_blocks": POOL,
    "full_group_blocks": full_blocks, "swa_group_capped": swa_capped,
    "swa_cap_formula": f"cdiv(min({W - 1}+{INFLIGHT}, {L}), {BS})+1",
    "total_grouped": full_blocks + swa_capped,
    "total_if_one_table_full_semantics": full_blocks + swa_uncapped_full_len,
    "concurrency_grouped": round(POOL / (full_blocks + swa_capped), 4),
    "concurrency_one_table": round(POOL / (full_blocks + swa_uncapped_full_len), 4),
}

# 场景二：#39734 心算（100k prompt，窗 1024）
L2, W2, BS2, INFLIGHT2 = 100000, 1024, 16, 16
OUT["deadlock_scene"] = {
    "prompt_len": L2, "window": W2, "block_size": BS2, "in_flight_one_chunk": INFLIGHT2,
    "full_len_blocks": cdiv(L2, BS2),
    "recycling_formula": f"cdiv(min({W2 - 1}+{INFLIGHT2}, {L2}), {BS2})+1",
    "recycling_aware_blocks": cdiv(min(W2 - 1 + INFLIGHT2, L2), BS2) + 1,
}

# 场景三：混合容量核算（m15 混合行复算）
W3, BS3, INFLIGHT3, L3, POOL3 = 512, 16, 8192, 4096, 1024
cap3 = cdiv(min(W3 - 1 + INFLIGHT3, L3), BS3) + 1
OUT["hybrid_capacity"] = {
    "window": W3, "in_flight": INFLIGHT3, "max_len": L3, "pool_blocks": POOL3,
    "full_group_blocks": cdiv(L3, BS3), "swa_cap_inflight_dominated": cap3,
    "per_request_blocks": cdiv(L3, BS3) + cap3,
    "max_concurrency": round(POOL3 / (cdiv(L3, BS3) + cap3), 4),
    "uniform_llama_reference": {"per_request": cdiv(L3, BS3), "concurrency": round(POOL3 / cdiv(L3, BS3), 4)},
    "long_seq_window_saves": {"len": 32768, "full_blocks": cdiv(32768, BS3),
                              "swa_cap": cdiv(min(W3 - 1 + INFLIGHT3, 32768), BS3) + 1},
}

# 场景四：运行期还账（full 组恒 0 vs SWA 组逐轮收）——引用 m13 既有 trace 的对照行
OUT["reclaim_runtime"] = {
    "full_manager": "get_num_skipped_tokens 恒 0（基类默认）——从不还账，全长持有到请求结束",
    "swa_manager": "remove_skipped_blocks 每 chunk 前先跑（窗外整块归池、原位 NULL）",
    "m13_trace_ref": "traces/m13.json（推进一~三：实持 4→3→2→1、free 3→4→5→6）",
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m18.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
print(json.dumps(OUT, ensure_ascii=False, indent=1))
