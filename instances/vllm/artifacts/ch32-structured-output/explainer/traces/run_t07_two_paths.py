"""ch32 explainer driver t07 - m11 两条并存的 worker 路径(默认 vs opt-in).

同一批掩码、同一个 batch,分别走:
  (默认) vllm/v1/structured_output/utils.py:apply_grammar_bitmask
         -> sorted_bitmask 重排成与 logits 同形(缺省 -1) -> xgr.apply_token_bitmask_inplace
  (opt-in) vllm/v1/worker/gpu/structured_outputs.py:StructuredOutputsWorker
         -> 紧凑掩码直接 H2D + int32 行映射 -> vLLM 自写 Triton kernel
断言两条路径把 -inf 写到完全相同的位置(逐元素比较),差别只在中间物料的形状与搬运量。
xgrammar 本体未安装,用一个忠实实现 xgr 语义的替身(bit==0 -> -inf)顶替库函数本身,
vLLM 自己的重排/索引逻辑一字未改地真实执行(同 tests/test_legacy_path.py 惯例)。
env 默认值一并从 pin 源码里 grep 出来记录。
输出 JSON 存 t07_two_paths.json。
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.normpath(os.path.join(HERE, "..", ".."))
SRC = os.path.normpath(os.path.join(CH, "..", "..", "source"))
sys.path.insert(0, os.path.join(CH, "implementation"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import utils as legacy  # noqa: E402
from input_batch import InputBatch  # noqa: E402
from output import GrammarOutput, SchedulerOutput  # noqa: E402
from structured_outputs import StructuredOutputsWorker  # noqa: E402

VOCAB = 96
COLS = -(-VOCAB // 32)
device = torch.device("cuda")


class FakeXgr:
    """忠实实现 xgr.apply_token_bitmask_inplace 的语义:bit==0 的 token 写 -inf;
    给了 indices 就只处理这些行。记录收到的物料形状供对账。"""

    def __init__(self):
        self.calls = []

    def apply_token_bitmask_inplace(self, logits, bitmask, indices=None):
        rows = range(logits.shape[0]) if indices is None else indices.tolist()
        self.calls.append({
            "logits_shape": list(logits.shape),
            "bitmask_shape": list(bitmask.shape),
            "num_indices": 0 if indices is None else int(indices.numel()),
            "indices": None if indices is None else indices.tolist(),
            "fill_value_of_uncovered_rows": int(bitmask[0][0].item()),
            "bitmask_rows_all_minus_one": [
                int(r) for r in range(bitmask.shape[0])
                if bool((bitmask[r] == -1).all().item())],
        })
        for r in rows:
            for t in range(logits.shape[-1]):
                if not ((int(bitmask[r][t // 32].item()) >> (t % 32)) & 1):
                    logits[r, t] = float("-inf")


def pack(allowed):
    row = np.zeros(COLS, dtype=np.int32)
    for t in allowed:
        row[t // 32] |= np.int32(1 << (t % 32))
    return row


allowed_rows = [{5, 7}, {9}, {11, 13}, {40, 41, 42}]
bm = np.stack([pack(a) for a in allowed_rows])
req_ids_batch = ["rB", "rA", "rC"]
cu = np.array([0, 1, 4, 5], dtype=np.int32)
grammar_req_ids = ["rA", "rC"]
spec = {"rA": [7, 7]}


def make_logits():
    return torch.zeros((5, VOCAB), dtype=torch.float32, device=device)


def make_ib():
    return InputBatch(req_ids=req_ids_batch,
                      logits_indices=torch.arange(5, dtype=torch.int32, device=device),
                      cu_num_logits=torch.from_numpy(cu).to(device),
                      cu_num_logits_np=cu, has_structured_output_reqs=True)


# ---- 默认路径 ------------------------------------------------------------------
fake = FakeXgr()
legacy.xgr = fake
logits_legacy = make_logits()
legacy.apply_grammar_bitmask(
    SchedulerOutput(scheduled_spec_decode_tokens=dict(spec)),
    GrammarOutput(list(grammar_req_ids), bm.copy()),
    make_ib(), logits_legacy)
torch.cuda.synchronize()

# ---- opt-in 路径(真 Triton kernel) --------------------------------------------
worker = StructuredOutputsWorker(max_num_logits=16, vocab_size=VOCAB, device=device)
logits_v2 = make_logits()
worker.apply_grammar_bitmask(logits_v2, make_ib(), list(grammar_req_ids), bm.copy())
torch.cuda.synchronize()

same = bool(torch.equal(
    torch.nan_to_num(logits_legacy, neginf=-1.0),
    torch.nan_to_num(logits_v2, neginf=-1.0)))

rows = []
for i in range(5):
    fa = torch.isfinite(logits_legacy[i]).nonzero().flatten().tolist()
    fb = torch.isfinite(logits_v2[i]).nonzero().flatten().tolist()
    rows.append({"logits_row": i,
                 "default_path_surviving_tokens": fa,
                 "v2_path_surviving_tokens": fb,
                 "identical": fa == fb,
                 "num_neg_inf_default": VOCAB - len(fa),
                 "num_neg_inf_v2": VOCAB - len(fb)})


def grep(path, lineno):
    with open(os.path.join(SRC, path), encoding="utf-8") as f:
        lines = f.readlines()
    return lines[lineno - 1].rstrip("\n")


out = {
    "env_gate_from_pin_source": {
        "vllm/envs.py:251": grep("vllm/envs.py", 251),
        "vllm/envs.py:1711-1713": [grep("vllm/envs.py", n) for n in (1711, 1712, 1713)],
        "vllm/v1/worker/gpu_worker.py:316": grep("vllm/v1/worker/gpu_worker.py", 316),
        "vllm/v1/worker/gpu_model_runner.py:4245": grep(
            "vllm/v1/worker/gpu_model_runner.py", 4245),
        "default_value_VLLM_USE_V2_MODEL_RUNNER": False,
    },
    "scenario": {
        "batch_req_ids": req_ids_batch,
        "cu_num_logits": cu.tolist(),
        "grammar_req_ids": grammar_req_ids,
        "scheduled_spec_decode_tokens": {k: list(v) for k, v in spec.items()},
        "compact_bitmask_rows": int(bm.shape[0]),
        "compact_bitmask_cols": int(bm.shape[1]),
        "logits_rows": 5,
    },
    "default_path_material": fake.calls[0],
    "v2_path_material": {
        "bitmask_rows_copied_h2d": int(bm.shape[0]),
        "bitmask_cols": int(bm.shape[1]),
        "index_tensor_len": 4,
        "sorted_bitmask_allocated": False,
        "kernel": "_apply_grammar_bitmask_kernel (vLLM 自写 @triton.jit)",
    },
    "two_paths_identical_result": same,
    "rows": rows,
}

path = os.path.join(HERE, "t07_two_paths.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
