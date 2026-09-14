# ch26-m12 V4 短上下文全选快路径 —— _fill_short_context_topk_indices 的
# 逐行账（含 =topk 边界）+ 『省打分不省建缓存』的双产物证据。
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from trace_common import _cos_sin_cache, dump

doc = {"mechanism": "ch26-m12 V4 短上下文全选快路径（_fill_short_context_topk_indices）",
       "source": "run_m12.py（host；kernel 垫片=HOST SEAM 精确数学）",
       "code_anchor": "models/deepseek_v4/attention.py:L72-L87（kernel）/ "
                      "L843-L863（快路径分支）"}

import importlib

att = importlib.import_module("vllm.models.deepseek_v4.attention")

# ── A. 逐行账：candidates = (pos+1)//4 ≤ TOP_K → 直填 0..n-1 ───────────────
positions = torch.tensor([7, 15, 20, 31, 32])
buf = torch.full((5, 8), -7, dtype=torch.int32)
att._fill_short_context_topk_indices[(5,)](
    buf, positions, TOP_K=8, COMPRESS_RATIO=4,
    PADDED_TOP_K=8, num_warps=1)
rows = []
for i, p in enumerate(positions.tolist()):
    cands = (p + 1) // 4
    row = buf[i].tolist()
    rows.append({
        "pos": p, "candidates": cands, "le_topk": cands <= 8,
        "selected": [v for v in row if v >= 0],
        "sentinels": sum(1 for v in row if v == -1),
        "row": row})
doc["fill_rows"] = rows
doc["boundary_account"] = {
    "pos31_candidates": 8, "pos31_equal_topk": True,
    "pos31_row_all_selected": sum(1 for v in buf[3].tolist() if v >= 0) == 8,
    "pos31_note": "candidates=8 恰等于 topk=8：≤ 判定含等号——数学上全选即最优"
                  "（8 个候选选 8 个），一个 -1 都不出",
    "pos32_candidates": 8,
    "pos32_note": "(32+1)//4=8：pos 32 仍全选（33//4 商 8）",
    "batch_condition": "快路径按批判：max_seq_len//compress_ratio ≤ topk_tokens"
                       "（attention.py:L843-L845）——批内 max pos ≤ 31 时必走",
    "batch_examples": [
        {"max_seq_len": 32, "max_seq_len_div4": 8, "fast_path": True},
        {"max_seq_len": 36, "max_seq_len_div4": 9, "fast_path": False}],
    "dsv32_scale_note": "topk=2048 时阈值 = 4×2048 = 8192：上下文 ≤ 8192 的"
                        "请求整批判定全选（短请求 prefill/早期 decode 全走快路径）",
}

# ── B. 省打分不省建缓存：同一拍的双产物 ────────────────────────────────────
from vllm.models.deepseek_v4.common.ops import (compress_norm_rope_store_triton,
                                                save_partial_states)

torch.manual_seed(7)
state_cache = torch.zeros(2, 4, 512)
kv = torch.randn(4, 256); score = torch.randn(4, 256)
ape = torch.randn(4, 256)
save_partial_states(kv=kv, score=score, ape=ape,
                    positions=torch.tensor([0, 1, 2, 3]),
                    state_cache=state_cache,
                    slot_mapping=torch.tensor([0, 1, 2, 3], dtype=torch.int64),
                    block_size=4, state_width=256, compress_ratio=4,
                    pdl_kwargs={})
kv_cache = torch.zeros(2, 4, 132, dtype=torch.uint8)


class _KMeta:
    def __init__(self, slot_mapping):
        self.slot_mapping = slot_mapping


compress_norm_rope_store_triton(
    state_cache=state_cache, num_actual=1,
    token_to_req_indices=torch.tensor([0], dtype=torch.int32),
    positions=torch.tensor([3]),
    slot_mapping=torch.tensor([0], dtype=torch.int64),
    block_table=torch.tensor([[0, 1]], dtype=torch.int32), block_size=4,
    state_width=256, cos_sin_cache=_cos_sin_cache(64, 64),
    kv_cache=kv_cache, k_cache_metadata=_KMeta(
        torch.tensor([0], dtype=torch.int64)), pdl_kwargs={},
    head_dim=128, rope_head_dim=64, compress_ratio=4,
    overlap=True, use_fp4_cache=False,
    rms_norm_weight=torch.ones(128), rms_norm_eps=1e-6,
    quant_block=128, token_stride=128, scale_dim=4)
buf_b = torch.full((1, 8), -7, dtype=torch.int32)
att._fill_short_context_topk_indices[(1,)](
    buf_b, torch.tensor([3]), TOP_K=8, COMPRESS_RATIO=4,
    PADDED_TOP_K=8, num_warps=1)
written = kv_cache.reshape(-1, 132)[0]
doc["fast_path_still_builds_cache"] = {
    "note": "快路径分支里 compressor(compressed_kv_score, ...) 仍先执行"
            "（attention.py:L843-L845：『candidates num smaller than topk, "
            "every candidate is selected but we still need to build k cache』"
            "注释原话）——后续拍要用",
    "cache_row_written": bool(written[128:].view(torch.float32).item() > 0),
    "cache_scale_written": float(written[128:].view(torch.float32).item()),
    "buffer_row_filled": buf_b[0].tolist(),
    "pos3_candidates": 1,
    "dual_artifact": "同一拍：k_cache slot 0 有量化压缩键 + buffer 行 [0, -1, …]"
                     "（1 个候选选 1 个）",
}

doc["table_rows_echo"] = [
    [f"pos {r['pos']}", f"candidates = ({r['pos']}+1)//4 = {r['candidates']}",
     f"选中 {r['selected']}" if r["selected"] else "全 -1",
     f"-1 ×{r['sentinels']}" if r["sentinels"] else "无哨兵"] for r in rows
] + [
    ["pos 31 边界", "candidates = 32//4 = 8 = topk", "整行 [0..7] 无 -1",
     "≤ 含等号：全选即最优"],
    ["批判定", "max_seq_len=32 → 32//4=8 ≤ 8 快路径",
     "max_seq_len=36 → 9 > 8 正常打分", "批内 max pos ≤ 31 必走快路径"],
    ["实尺阈值", "topk=2048", "阈值 = 4×2048 = 8192",
     "上下文 ≤ 8192 整批判全选"],
    ["不省建缓存", "pos 3 一拍双产物", "k_cache slot 0 写入量化压缩键",
     "buffer 行 [0, -1, -1, -1, -1, -1, -1, -1]"],
]

dump("m12.json", doc)
print("m12.json written | pos31 row:", buf[3].tolist(),
      "| cache written:", doc["fast_path_still_builds_cache"]["cache_row_written"],
      "| buf_b:", buf_b[0].tolist())
