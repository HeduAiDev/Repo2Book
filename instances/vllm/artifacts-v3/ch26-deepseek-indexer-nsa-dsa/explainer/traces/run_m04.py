# ch26-m04 prefill 双预算切块 —— split_indexer_prefill_chunks 逐字算法的
# 心算例（贪心累积逐步账）+ 单请求超预算 M 子切 + DSV3.2 实尺账。
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import dump, make_v32_hf_config, make_vllm_config

doc = {"mechanism": "ch26-m04 prefill 双预算切块（workspace N + logits M·N·4）",
       "source": "run_m04.py（host；split_indexer_prefill_chunks 逐字函数直跑）",
       "code_anchor": "backends/mla/indexer.py:L77-L123（算法）/ "
                      "sparse_attn_indexer.py:L449-L476（chunk 消费）"}

from vllm.v1.attention.backends.mla.indexer import (
    get_max_prefill_buffer_size, split_indexer_prefill_chunks)
import vllm.envs as envs


def chunks_repr(chunks):
    return [f"(req{c[0].start}:{c[0].stop}, q{c[1].start}:{c[1].stop})"
            for c in chunks]


# ── A. 心算例：贪心累积逐步账（logits 预算是绑定约束）───────────────────────
seq = torch.tensor([300, 200, 250])
qry = torch.tensor([100, 60, 80])
ws, budget_bytes = 500, 4 * 30000
chunks = split_indexer_prefill_chunks(seq, qry, ws, budget_bytes)
doc["worked_small"] = {
    "seq_lens": [300, 200, 250], "query_lens": [100, 60, 80],
    "workspace_N_limit": ws,
    "logits_budget_bytes": budget_bytes,
    "logits_budget_elems": budget_bytes // 4,
    "greedy_steps": [
        {"chunk": 1, "try_req": 0, "N": 300, "M": 100, "MN": 30000,
         "MN_limit": 30000, "verdict": "accept（30000 ≤ 30000 且 300 ≤ 500）"},
        {"chunk": 1, "try_req": 1, "N": 500, "M": 160, "MN": 80000,
         "MN_limit": 30000, "verdict": "reject（80000 > 30000）→ chunk1 封口=req0"},
        {"chunk": 2, "try_req": 1, "N": 200, "M": 60, "MN": 12000,
         "MN_limit": 30000, "verdict": "accept"},
        {"chunk": 2, "try_req": 2, "N": 450, "M": 140, "MN": 63000,
         "MN_limit": 30000, "verdict": "reject（63000 > 30000）→ chunk2 封口=req1"},
        {"chunk": 3, "try_req": 2, "N": 250, "M": 80, "MN": 20000,
         "MN_limit": 30000, "verdict": "accept（末请求自然收尾）"},
    ],
    "chunks_out": chunks_repr(chunks),
    "num_chunks": len(chunks),
    "note": "每步两个约束同时查：new_n ≤ workspace 与 new_m*new_n ≤ 预算元素数"
            "（字节//4）；先到先贪，装不下就封口开新块",
}

# ── B. N 约束绑定例：workspace 先超 ────────────────────────────────────────
seq_b = torch.tensor([100, 100, 100])
qry_b = torch.tensor([10, 10, 10])
chunks_b = split_indexer_prefill_chunks(seq_b, qry_b, 250, 10 ** 9)
doc["n_constraint_split"] = {
    "seq_lens": [100, 100, 100], "query_lens": [10, 10, 10],
    "workspace_N_limit": 250,
    "step_account": "req0+req1 → N=200 ≤ 250 收；req2 并入 → N=300 > 250 拒",
    "chunks_out": chunks_repr(chunks_b),
    "chunk1_M_N": [20, 200], "chunk2_M_N": [10, 100],
}

# ── C. 单请求超预算：按 M（query 维）子切 ──────────────────────────────────
seq_c = torch.tensor([1000])
qry_c = torch.tensor([40])
chunks_c = split_indexer_prefill_chunks(seq_c, qry_c, 100, 4 * 2500)
doc["single_request_subchunk"] = {
    "seq_lens": [1000], "query_lens": [40],
    "workspace_N_limit": 100, "logits_budget_elems": 2500,
    "max_q_per_piece": 2500 // 1000,
    "num_pieces": len(chunks_c),
    "first_three_query_slices": [f"{c[1].start}:{c[1].stop}"
                                 for c in chunks_c[:3]],
    "note": "end==start（单请求也装不下）→ 不封口、转入 query 维子切："
            "max_q = 预算元素//chunk_n = 2，40 行切成 20 片——O(L²) 打分的"
            "峰值显存就是这里被压住的",
}

# ── D. DSV3.2 实尺账 ────────────────────────────────────────────────────────
hf32 = make_v32_hf_config()
vllm32 = make_vllm_config(hf32, max_model_len=163840)
ws32 = get_max_prefill_buffer_size(vllm32)
budget32 = envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB * 1024 * 1024
elems32 = budget32 // 4
M, N = 16384, 163840  # 一次 16k-token prefill 段打满 163840 历史
max_q = elems32 // N
pieces = (M + max_q - 1) // max_q
last_rows = M - (pieces - 1) * max_q
peak_bytes = max_q * N * 4
unbounded_bytes = M * N * 4
chunks_d = split_indexer_prefill_chunks(torch.tensor([N]), torch.tensor([M]),
                                        ws32, budget32)
doc["dsv32_full_prefill"] = {
    "max_model_len": 163840,
    "workspace_entries": int(ws32),
    "env_logits_budget_mb": int(envs.VLLM_SPARSE_INDEXER_MAX_LOGITS_MB),
    "logits_budget_bytes": int(budget32),
    "logits_budget_elems": int(elems32),
    "M_query_tokens": M, "N_history": N,
    "unbounded_logits_bytes": int(unbounded_bytes),
    "unbounded_logits_mib": f"{unbounded_bytes / 1048576:g}",
    "max_q_per_piece": int(max_q),
    "num_pieces": int(pieces),
    "last_piece_rows": int(last_rows),
    "peak_bytes_per_piece": int(peak_bytes),
    "peak_mib_per_piece": f"{peak_bytes / 1048576:.2f}",
    "pieces_from_function": len(chunks_d),
    "note": "不切块的 logits 张量 16384×163840×4 B = 10240 MiB；按 M 子切成 "
            f"{pieces} 片后每片峰值 {peak_bytes / 1048576:.2f} MiB——解析式 "
            "M·N·4 直接写成切块条件",
}

doc["table_rows_echo"] = [
    ["贪心-1", "req0 试装入 chunk1", "N=300, M=100, M·N=30000",
     "30000 ≤ 30000 且 300 ≤ 500 → 收"],
    ["贪心-2", "req1 试并入 chunk1", "N=500, M=160, M·N=80000",
     "80000 > 30000 → 拒，chunk1 封口"],
    ["贪心-3", "req1 开 chunk2", "N=200, M=60, M·N=12000", "收"],
    ["贪心-4", "req2 试并入 chunk2", "N=450, M=140, M·N=63000",
     "63000 > 30000 → 拒，chunk2 封口"],
    ["贪心-5", "req2 开 chunk3", "N=250, M=80, M·N=20000", "收（末请求收尾）"],
    ["N约束", "workspace 先超：[100,100,100]/[10,10,10]，上限 250",
     "N: 200 → 300", "300 > 250 → req2 另开一块（2 块收尾）"],
    ["M子切", "单请求 [1000]/[40]，预算 2500 元素", "max_q = 2500//1000 = 2",
     "40 行 → 20 片，首三片 0:2 / 2:4 / 4:6"],
    ["实尺", "16384 query × 163840 历史", f"不切 {unbounded_bytes} B = "
     f"{unbounded_bytes / 1048576:g} MiB",
     f"max_q={max_q} → {pieces} 片（末片 {last_rows} 行），片峰 "
     f"{peak_bytes} B（{peak_bytes / 1048576:.2f} MiB）"],
]

dump("m04.json", doc)
print("m04.json written | small:", chunks_repr(chunks), "| sub:",
      len(chunks_c), "pieces | real:", pieces, "pieces, max_q =", max_q)
