"""ch32 explainer driver t04 - m09 行映射(投机:一个请求占多行) + m10 Triton kernel 位解包与 -inf.

真机 GPU(RTX PRO 6000 Blackwell)+ 真 @triton.jit kernel:
  batch 顺序 [rB, rA, rC];rA 带 2 个草稿 -> 占 3 行 logits;rB 非结构化 1 行;rC 结构化 1 行
  cu_num_logits = [0, 1, 4, 5];grammar_req_ids = [rA, rC](调度顺序)
  -> mapping = [1,2,3] ++ [4] = [1,2,3,4],num_masks=4,assert 通过
另记录 kernel 分块几何:BLOCK_SIZE=8192,grid=(num_masks, cdiv(V, 8192)),
以及按源码常量对 Qwen 词表 |V|=152064 的推算(非实测:cols/每行字节数/grid 第二维)。
输出 JSON 存 t04_kernel_apply.json。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(CH, "implementation"))
sys.path.insert(0, os.path.join(CH, "tests"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import triton  # noqa: E402

from input_batch import InputBatch  # noqa: E402
from structured_outputs import StructuredOutputsWorker  # noqa: E402

VOCAB = 96
COLS = -(-VOCAB // 32)
BLOCK_SIZE = 8192
device = torch.device("cuda")

out = {"env": {
    "gpu": torch.cuda.get_device_name(0),
    "torch": torch.__version__,
    "triton": triton.__version__,
}}


def pack(allowed, cols=COLS):
    row = np.zeros(cols, dtype=np.int32)
    for t in allowed:
        row[t // 32] |= np.int32(1 << (t % 32))
    return row


# 掩码 4 行:rA 的 3 个位置 + rC 的 1 行
allowed_rows = [{5, 7}, {9}, {11, 13}, {40, 41, 42}]
bm = np.stack([pack(a) for a in allowed_rows])

req_ids_batch = ["rB", "rA", "rC"]
cu = np.array([0, 1, 4, 5], dtype=np.int32)
ib = InputBatch(req_ids=req_ids_batch,
                logits_indices=torch.arange(5, dtype=torch.int32, device=device),
                cu_num_logits=torch.from_numpy(cu).to(device),
                cu_num_logits_np=cu,
                has_structured_output_reqs=True)
grammar_req_ids = ["rA", "rC"]

# 复算 mapping(与 worker 内部同一算法,便于观测)
req_id_to_idx = {r: i for i, r in enumerate(req_ids_batch)}
mapping = []
for rid in grammar_req_ids:
    i = req_id_to_idx[rid]
    mapping.extend(range(int(cu[i]), int(cu[i + 1])))

worker = StructuredOutputsWorker(max_num_logits=16, vocab_size=VOCAB, device=device)
logits = torch.zeros((5, VOCAB), dtype=torch.float32, device=device)
worker.apply_grammar_bitmask(logits, ib, grammar_req_ids, bm)
torch.cuda.synchronize()

rows = []
owner = {0: "rB(非结构化)", 1: "rA 位置0", 2: "rA 位置1", 3: "rA 位置2", 4: "rC 位置0"}
for i in range(5):
    finite = torch.isfinite(logits[i]).nonzero().flatten().tolist()
    rows.append({
        "logits_row": i,
        "owner": owner[i],
        "bitmask_row_mapped_here": mapping.index(i) if i in mapping else None,
        "num_surviving_tokens": len(finite),
        "surviving_token_ids": finite,
        "num_neg_inf": int(VOCAB - len(finite)),
        "all_masked_are_neg_inf": bool(
            torch.isneginf(logits[i][~torch.isfinite(logits[i])]).all().item())
        if len(finite) < VOCAB else True,
    })

out["mapping"] = {
    "batch_req_ids": req_ids_batch,
    "cu_num_logits": cu.tolist(),
    "grammar_req_ids_schedule_order": grammar_req_ids,
    "req_id_to_idx": req_id_to_idx,
    "mapping_bitmask_row_to_logits_row": mapping,
    "num_masks": int(bm.shape[0]),
    "len_mapping": len(mapping),
    "assert_num_masks_eq_len_mapping": bool(bm.shape[0] == len(mapping)),
}
out["kernel_geometry_toy"] = {
    "vocab_size": VOCAB,
    "bitmask_stride_cols": COLS,
    "bytes_per_row": COLS * 4,
    "BLOCK_SIZE": BLOCK_SIZE,
    "int32_per_block": BLOCK_SIZE // 32,
    "grid_dim0_num_masks": int(bm.shape[0]),
    "grid_dim1_cdiv_V_BLOCK": triton.cdiv(VOCAB, BLOCK_SIZE),
    "total_programs": int(bm.shape[0]) * triton.cdiv(VOCAB, BLOCK_SIZE),
}
V_REAL = 152064
out["kernel_geometry_real_vocab_derived_not_measured"] = {
    "vocab_size": V_REAL,
    "bitmask_cols_int32": -(-V_REAL // 32),
    "bytes_per_row": (-(-V_REAL // 32)) * 4,
    "kib_per_row": round((-(-V_REAL // 32)) * 4 / 1024, 4),
    "grid_dim1_cdiv_V_BLOCK": triton.cdiv(V_REAL, BLOCK_SIZE),
    "int32_per_block": BLOCK_SIZE // 32,
    "batch256_no_spec_total_mib": round(
        256 * (-(-V_REAL // 32)) * 4 / 1024 / 1024, 4),
    "note": "按源码常量推算(|V|=152064, BLOCK_SIZE=8192),非实测耗时",
}
out["logits_rows"] = rows

# 位解包的逐位对照:取 bitmask 第 0 行第 0 列,列出 32 个 bit 与最终 logits 的对应
col0 = int(bm[0][0])
bits = [(col0 >> b) & 1 for b in range(32)]
out["bit_unpack_row0_col0"] = {
    "packed_int32_value": col0,
    "bits_lsb_first": bits,
    "bit_eq_1_token_ids": [b for b in range(32) if bits[b]],
    "kernel_masks_when_bit_eq_0": True,
    "logits_row1_finite_tokens_below_32": [
        t for t in torch.isfinite(logits[1]).nonzero().flatten().tolist() if t < 32],
}

path = os.path.join(HERE, "t04_kernel_apply.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
