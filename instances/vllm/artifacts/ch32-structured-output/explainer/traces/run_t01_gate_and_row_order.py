"""ch32 explainer driver t01 - m01 调度侧门控 + m02 行序不变式 + m03 形状预算.

跑精简版真实控制流:
  Scheduler._update_after_schedule  -> has_structured_output_requests 逐请求置位
  Scheduler.get_grammar_bitmask     -> 按 num_scheduled_tokens 迭代顺序收集 req_id
  StructuredOutputManager.grammar_bitmask -> 装配紧凑掩码(行数=结构化请求数)
  StructuredOutputsWorker.apply_grammar_bitmask (GPU, 真 Triton kernel)
      -> 用 req_id->batch_idx 字典把「调度顺序的第 k 行」落到「batch 顺序的某一 logits 行」

场景刻意让两侧顺序不同(调度顺序 rA,rB,rC,rD;batch 顺序 rD,rB,rA,rC),
若掩码按行号直接对齐 logits 行就会错到别人身上 -- 不变式成立与否肉眼可辨。
输出 JSON 存 t01_gate_and_row_order.json。
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(CH, "implementation"))
sys.path.insert(0, os.path.join(CH, "tests"))

import numpy as np
import torch

from conftest import FakeBackend, FakeGrammar, make_request  # noqa: E402
from input_batch import InputBatch  # noqa: E402
from output import SchedulerOutput  # noqa: E402
from request import Request  # noqa: E402
from sampling_params import SamplingParams  # noqa: E402
from scheduler import Scheduler  # noqa: E402
from structured_output_manager import StructuredOutputManager  # noqa: E402
from structured_outputs import StructuredOutputsWorker  # noqa: E402

VOCAB = 96
MAX_NUM_SEQS = 4
MAX_NUM_SPEC = 0

out = {"params": {"vocab_size": VOCAB, "max_num_seqs": MAX_NUM_SEQS,
                  "max_num_spec_tokens": MAX_NUM_SPEC,
                  "bitmask_cols_ceil_V_over_32": -(-VOCAB // 32),
                  "bitmask_buffer_rows": MAX_NUM_SEQS * (1 + MAX_NUM_SPEC)}}

mgr = StructuredOutputManager(max_num_seqs=MAX_NUM_SEQS,
                              max_num_spec_tokens=MAX_NUM_SPEC)
mgr.backend = FakeBackend(VOCAB)
sched = Scheduler(mgr)

gA = FakeGrammar([{5, 7}])
gD = FakeGrammar([{40, 41, 42}])
rA = make_request("rA", gA)
rD = make_request("rD", gD)
rC = make_request("rC", FakeGrammar([{1}]))
rC.prompt_token_ids = list(range(8))
rC._all_token_ids = list(range(8))
rC.all_token_ids = rC._all_token_ids
rB = Request("rB", prompt_token_ids=[1, 2, 3],
             sampling_params=SamplingParams(max_tokens=16, structured_outputs=None))
for r in (rA, rB, rC, rD):
    sched.requests[r.request_id] = r

so = SchedulerOutput(num_scheduled_tokens={"rA": 3, "rB": 3, "rC": 4, "rD": 3})

# ---- 阶段 1:_update_after_schedule 的逐请求门控 (逐步骤展开以便观测) ----------
gate_rows = []
for req_id, n in so.num_scheduled_tokens.items():
    request = sched.requests[req_id]
    request.num_computed_tokens += n
    request.is_prefill_chunk = request.num_computed_tokens < (
        request.num_tokens + request.num_output_placeholders)
    so.has_structured_output_requests |= (
        request.use_structured_output and not request.is_prefill_chunk)
    gate_rows.append({
        "req_id": req_id,
        "num_scheduled_tokens": n,
        "num_computed_tokens": request.num_computed_tokens,
        "num_tokens": request.num_tokens,
        "is_prefill_chunk": bool(request.is_prefill_chunk),
        "use_structured_output": bool(request.use_structured_output),
        "contributes": bool(request.use_structured_output
                            and not request.is_prefill_chunk),
        "has_structured_output_requests_after": bool(
            so.has_structured_output_requests),
    })
out["stage1_schedule_gate"] = gate_rows

# 交叉验证:直接调精简版的 _update_after_schedule(状态已推进,故用一份新 Scheduler)
sched2 = Scheduler(StructuredOutputManager(max_num_seqs=MAX_NUM_SEQS))
sched2.structured_output_manager.backend = FakeBackend(VOCAB)
for rid, ntok, grammar in (("rA", 3, FakeGrammar([{5, 7}])), ("rC", 4, FakeGrammar([{1}])),
                           ("rD", 3, FakeGrammar([{40, 41, 42}]))):
    rr = make_request(rid, grammar)
    if rid == "rC":
        rr.prompt_token_ids = list(range(8))
        rr._all_token_ids = list(range(8))
        rr.all_token_ids = rr._all_token_ids
    sched2.requests[rid] = rr
sched2.requests["rB"] = Request(
    "rB", prompt_token_ids=[1, 2, 3],
    sampling_params=SamplingParams(max_tokens=16, structured_outputs=None))
so2 = SchedulerOutput(num_scheduled_tokens={"rA": 3, "rB": 3, "rC": 4, "rD": 3})
sched2._update_after_schedule(so2)
out["stage1_crosscheck_via_real_method"] = {
    "has_structured_output_requests": bool(so2.has_structured_output_requests),
    "is_prefill_chunk": {k: bool(v.is_prefill_chunk) for k, v in sched2.requests.items()},
}

# ---- 阶段 2:get_grammar_bitmask 装配 -------------------------------------------
grammar_output = sched.get_grammar_bitmask(so)
ids = grammar_output.structured_output_request_ids
bm = grammar_output.grammar_bitmask
out["stage2_assemble"] = {
    "structured_output_request_ids": ids,
    "bitmask_shape_rows": int(bm.shape[0]),
    "bitmask_shape_cols": int(bm.shape[1]),
    "buffer_rows_allocated": MAX_NUM_SEQS * (1 + MAX_NUM_SPEC),
    "rows_after_trim": int(bm.shape[0]),
    "row_to_req": {str(k): ids[k] for k in range(len(ids))},
    "row0_allowed_tokens": sorted({5, 7}),
    "row1_allowed_tokens": sorted({40, 41, 42}),
}

# ---- 阶段 3:worker 侧落地(batch 顺序与调度顺序不同) ---------------------------
req_ids_batch = ["rD", "rB", "rA", "rC"]
cu = np.array([0, 1, 2, 3, 3], dtype=np.int32)
device = torch.device("cuda")
ib = InputBatch(req_ids=req_ids_batch,
                logits_indices=torch.arange(3, dtype=torch.int32, device=device),
                cu_num_logits=torch.from_numpy(cu).to(device),
                cu_num_logits_np=cu,
                has_structured_output_reqs=True)
worker = StructuredOutputsWorker(max_num_logits=8, vocab_size=VOCAB, device=device)
logits = torch.zeros((3, VOCAB), dtype=torch.float32, device=device)
worker.apply_grammar_bitmask(logits, ib, ids, bm)
torch.cuda.synchronize()

rows = []
for i, rid in enumerate(req_ids_batch[:3]):
    finite = torch.isfinite(logits[i]).nonzero().flatten().tolist()
    rows.append({
        "logits_row": i,
        "batch_req_id": rid,
        "num_surviving_tokens": len(finite),
        "surviving_token_ids": finite,
        "num_neg_inf": int(VOCAB - len(finite)),
    })
out["stage3_worker_apply"] = {
    "batch_req_ids": req_ids_batch,
    "cu_num_logits": cu.tolist(),
    "mapping_bitmask_row_to_logits_row": [2, 0],
    "num_masks": int(bm.shape[0]),
    "len_mapping": 2,
    "rows": rows,
}

path = os.path.join(HERE, "t01_gate_and_row_order.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
