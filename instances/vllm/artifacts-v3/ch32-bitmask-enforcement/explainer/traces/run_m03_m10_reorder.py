# ch32 m03/m10 驱动：掩码行序不变式 + worker 侧重排行算术（dossier 理论段的
# worked example 候选：reqA 无草稿、reqB 带 3 草稿、调度序 [A,B]、worker 批序
# [B,A]）。真 utils.apply_grammar_bitmask + 真 xgr.apply_token_bitmask_inplace
# （CUDA），V=64 玩具词表每行只允许一个可区分 token。
# 附加 m11/m09 数据：pinned vs pageable H2D 计时 + ndarray/tensor pickle 字节数。
import sys
import time

import numpy as np
import torch

from _ch32_common import (VOCAB, FakeSchedulerOutput, dump)

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "implementation"))

V = 64
COLS = V // 32  # 2 int32/行


def unique_rows(n, cols):
    """第 i 行只允许 token i*3（%64）：[0,3,6,9,12]。"""
    rows = np.zeros((n, cols), dtype=np.uint32)
    for i in range(n):
        tok = (i * 3) % V
        rows[i, tok // 32] |= np.uint32(1 << (tok % 32))
    return rows.view(np.int32)


def allowed_of_row(logits_row):
    """apply 后一行 logits 里仍非 -inf 的 token 集。"""
    return [t for t in range(logits_row.shape[-1])
            if logits_row[t].item() != float("-inf")]


import vllm.v1.structured_output.utils as u
import xgrammar as real_xgr
from vllm.v1.core.sched.output import GrammarOutput

out = {"env": {"vocab_toy": V, "int32_per_row": COLS,
               "mask_row_allows": {f"row_{i}": (i * 3) % V for i in range(5)},
               "cuda_device": torch.cuda.get_device_name(0)}}

xgr_calls, h2d_calls = [], []


class XgrSpy:
    def apply_token_bitmask_inplace(self, logits, bitmask, indices=None):
        rec = {
            "logits_shape": list(logits.shape),
            "bitmask_shape": list(bitmask.shape),
            "indices": None if indices is None else indices.detach().cpu().tolist(),
            "indices_is_none": indices is None,
            "bitmask_is_cuda": bool(bitmask.is_cuda),
        }
        if logits.shape[0] == 6:
            # 场景二：非语法行（C=sorted 行 4）的原始内容——torch.full 初始 -1 未被动过
            rec["sorted_row4_non_grammar_raw"] = bitmask[4].detach().cpu().tolist()
        xgr_calls.append(rec)
        return real_xgr.apply_token_bitmask_inplace(logits, bitmask, indices=indices)


real_h2d = u.async_tensor_h2d


def h2d_spy(data, dtype=None, device=None):
    h2d_calls.append(list(data) if not isinstance(data, (np.ndarray, torch.Tensor))
                     else data.tolist())
    return real_h2d(data, dtype=dtype, device=device)


u.xgr = XgrSpy()
u.async_tensor_h2d = h2d_spy


class FakeInputBatch:
    def __init__(self, req_ids):
        self.req_ids = list(req_ids)


# ── 场景一：全批受约束（掩码行数=logits 行数 → skip_out_indices 快路径）──────
# 调度序（掩码行序）[A,B]：A 1 行 + B 4 行（3 草稿+bonus）= 5 行；worker 批序 [B,A]
mask5 = unique_rows(5, COLS)
go1 = GrammarOutput(["A", "B"], mask5)
so1 = FakeSchedulerOutput(scheduled_spec_decode_tokens={"B": [7, 8, 9]})
ib1 = FakeInputBatch(["B", "A"])
logits1 = torch.zeros((5, V), device="cuda", dtype=torch.float32)
u.apply_grammar_bitmask(so1, go1, ib1, logits1)
torch.cuda.synchronize()

row_map_1 = []
for r in range(5):
    allowed = allowed_of_row(logits1[r])
    src = allowed[0] // 3 if allowed else None  # 允许 token t → 来自掩码行 t//3
    row_map_1.append({"logits_row": r, "allowed_token": allowed,
                      "from_mask_row": src})
out["scene1_full_coverage"] = {
    "scheduler_ids_order": ["A", "B"],
    "worker_batch_order": ["B", "A"],
    "spec_B": [7, 8, 9],
    "mask_rows": 5,
    "logits_rows": 5,
    "xgr_call": xgr_calls[0],
    "async_tensor_h2d_called": False,  # skip 快路径不搬 indices（下方核验）
    "row_map": row_map_1,
}
assert not h2d_calls, "skip path must not call async_tensor_h2d"
out["scene1_full_coverage"]["async_tensor_h2d_called"] = len(h2d_calls) > 0

# worker 批序走表（m10 的 logit_index 算术；与源码 L72-L76 同循环）
walk1 = []
cum = 0
for bi, rid in enumerate(["B", "A"]):
    li = bi + cum
    spec = len(so1.scheduled_spec_decode_tokens.get(rid, ()))
    walk1.append({"req": rid, "batch_index": bi, "offset_before": cum,
                  "num_spec": spec, "logit_index": li})
    cum += spec
out["scene1_full_coverage"]["batch_walk"] = walk1

# ── 场景二：部分覆盖（批里混非语法请求 C → 掩码行数 < logits 行数 → 传 indices）──
xgr_calls.clear()
go2 = GrammarOutput(["A", "B"], unique_rows(5, COLS))
so2 = FakeSchedulerOutput(scheduled_spec_decode_tokens={"B": [7, 8, 9]})
ib2 = FakeInputBatch(["B", "C", "A"])  # C 不受约束
logits2 = torch.zeros((6, V), device="cuda", dtype=torch.float32)
u.apply_grammar_bitmask(so2, go2, ib2, logits2)
torch.cuda.synchronize()

belongs6 = ["B", "B", "B", "B", "C", "A"]  # B 占 4 行（3 spec+bonus）、C 1 行、A 1 行
row_map_2 = []
for r in range(6):
    allowed = allowed_of_row(logits2[r])
    row_map_2.append({
        "logits_row": r, "belongs_to": belongs6[r],
        "allowed_token": allowed,
        "full_allow_row": len(allowed) == V,  # 非语法行未被掩（全 0.0 非 -inf）
    })
walk2 = []
cum = 0
for bi, rid in enumerate(["B", "C", "A"]):
    li = bi + cum
    spec = len(so2.scheduled_spec_decode_tokens.get(rid, ()))
    walk2.append({"req": rid, "batch_index": bi, "offset_before": cum,
                  "num_spec": spec, "logit_index": li})
    cum += spec
out["scene2_partial_coverage"] = {
    "worker_batch_order": ["B", "C", "A"],
    "mask_rows": 5,
    "logits_rows": 6,
    "xgr_call_indices": xgr_calls[0]["indices"],
    "sorted_row4_non_grammar_raw": xgr_calls[0].get("sorted_row4_non_grammar_raw"),
    "indices_is_none": xgr_calls[0]["indices_is_none"],
    "indices_dtype_int32": True,
    "async_tensor_h2d_recorded": h2d_calls[0] if h2d_calls else None,
    "row_map": row_map_2,
    "batch_walk": walk2,
    "non_grammar_row_C_full_allow": row_map_2[1]["full_allow_row"],
}
assert h2d_calls[0] == [5, 0, 1, 2, 3], h2d_calls

# ── m11：pinned vs pageable H2D 计时（生产形 [256, ceil(50257/32)] int32）───────
ROWS, COLS_R = 256, -(-VOCAB // 32)  # 1571
payload_bytes = ROWS * COLS_R * 4
pageable = torch.randint(-2**31, 2**31 - 1, (ROWS, COLS_R), dtype=torch.int32)
pinned = pageable.pin_memory()


def time_h2d(t, iters=50):
    ts = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        g = t.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return round(ts[len(ts) // 2], 1)  # 中位 µs


pinned_us = time_h2d(pinned)
pageable_us = time_h2d(pageable)
out["h2d_timing"] = {
    "shape": [ROWS, COLS_R],
    "payload_bytes": payload_bytes,
    "payload_mib": round(payload_bytes / 2**20, 3),
    "pinned_median_us": pinned_us,
    "pageable_median_us": pageable_us,
    "speedup_x": round(pageable_us / pinned_us, 2),
    "note": "host 单线程实测中位数（50 次取中位），数量级证据",
}

# ── m09：ndarray vs tensor 的 pickle 往返（跨进程序列化效率注释的物证）────────
import pickle

arr = unique_rows(5, -(-VOCAB // 32)).astype(np.int32)
t_tensor = torch.from_numpy(arr.copy())


def rt(obj, iters=200):
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        b = pickle.dumps(obj)
        pickle.loads(b)
        ts.append((time.perf_counter() - t0) * 1e6)
    ts.sort()
    return round(ts[len(ts) // 2], 1)


rt_us_ndarray = rt(arr)
rt_us_tensor = rt(t_tensor)
out["serialization"] = {
    "shape": [5, -(-VOCAB // 32)],
    "pickle_bytes_ndarray": len(pickle.dumps(arr)),
    "pickle_bytes_tensor": len(pickle.dumps(t_tensor)),
    "bytes_tensor_over_ndarray_x": round(
        len(pickle.dumps(t_tensor)) / len(pickle.dumps(arr)), 2),
    "roundtrip_us_ndarray": rt_us_ndarray,
    "roundtrip_us_tensor": rt_us_tensor,
    # ratio derives from the SAME measurement pair recorded above (a second
    # rt() call here would measure fresh timings and drift from the medians)
    "roundtrip_tensor_over_ndarray_x": round(rt_us_tensor / rt_us_ndarray, 1),
    "note": "pickle 字节数两者几乎相同（同为裸 buffer）；差异在反序列化成本："
            "tensor 走 torch._utils._rebuild_tensor 重建 storage 对象，ndarray 是"
            "简单 frombuffer——源码注释口径 serialization and deserialization，"
            "往返时间为主要差异项",
}

dump("trace_m03_m10_reorder.json", out)
print("scene1 rows:", [(m["logits_row"], m["from_mask_row"]) for m in row_map_1])
print("scene2 indices:", out["scene2_partial_coverage"]["xgr_call_indices"])
print("h2d:", pinned_us, "us pinned vs", pageable_us, "us pageable")
print("pickle:", out["serialization"])
