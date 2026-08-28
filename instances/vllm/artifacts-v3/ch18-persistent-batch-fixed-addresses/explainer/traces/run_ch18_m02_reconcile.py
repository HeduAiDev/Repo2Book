"""ch18-m02 持久批次差量调和 _update_states —— 驱动脚本（host 纯 CPU）。

五拍剧本穿真实两段式协议（execute_model → sample_tokens），每拍记录：
- wire：SchedulerOutput 差量载荷形状（m01 的数字来源）
- after_update：_update_states 之后的批次状态（slot 布局/requests 缓存/块表/token 行）
- prepare：_prepare_inputs 收出的 input_ids 与 positions（m06 交叉证据）
- sample：采样 token 与写回后的 token 行（m09 闭环证据）
- data_ptrs：6 个持久缓冲的地址（m14 固定地址证据）

剧本（max_num_reqs=4 / max_model_len=32 / block_size=16 → 每请求至多 2 块）：
  拍1  r1/r2 首次全量进批（prefill），各采样 1 token
  拍2  decode：cached_reqs 只发 diff（r1 追加新块 [3]、r2 无新块）；positions 落在
       上一拍写回的 token 上——写回→收集的闭环
  拍3  r2 finished（出缓存+出批次）+ r3 新请求 → r3 落进 r2 的洞（pop_removed=1）
  拍4  r3 被抢占未调度 → 出批次但留 requests 缓存（unscheduled 支）
  拍5  r3 resumed：new_block_ids 整体替换 [4]→[5]、num_computed=0、全量重算
       （含自己的 output token——重算换 logits 的语义）

跑法：python explainer/traces/run_ch18_m02_reconcile.py（在本章目录下）
产物：explainer/traces/ch18_m02_reconcile.json
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation._host_seams import SamplingParams  # noqa: E402
from implementation.gpu_model_runner import GPUModelRunner  # noqa: E402
from implementation.output import (  # noqa: E402
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)

VOCAB = 32
MAX_NUM_REQS = 4
MAX_MODEL_LEN = 32
MAX_NUM_BATCHED_TOKENS = 64

# 每拍脚本 logits：one-hot 行，argmax = pick（贪心采样可预测）
SCRIPTED_PICKS = [
    {"r1": 11, "r2": 21},   # 拍1 prefill
    {"r1": 12, "r2": 22},   # 拍2 decode
    {"r1": 13, "r3": 31},   # 拍3 r2 finished + r3 新
    {"r1": 14},             # 拍4 r3 被抢
    {"r1": 15, "r3": 30},   # 拍5 r3 resumed 全量重算
]


def _greedy():
    return SamplingParams(temperature=0.0)


def _vllm_config():
    from types import SimpleNamespace

    model_config = SimpleNamespace(
        max_model_len=MAX_MODEL_LEN,
        runner_type="generate",
        enable_prompt_embeds=False,
        uses_mrope=False,
        uses_xdrope_dim=0,
        is_encoder_decoder=False,
        dtype=torch.float32,
        logits_processors=None,
        get_vocab_size=lambda: VOCAB,
    )
    cache_config = SimpleNamespace(
        block_size=16,
        calculate_kv_scales=False,
        use_replayssm=False,
        mamba_cache_mode="align",
        kv_sharing_fast_prefill=False,
    )
    scheduler_config = SimpleNamespace(
        max_num_batched_tokens=MAX_NUM_BATCHED_TOKENS,
        max_num_seqs=MAX_NUM_REQS,
        async_scheduling=False,
    )
    parallel_config = SimpleNamespace(
        decode_context_parallel_size=1,
        cp_kv_cache_interleave_size=1,
    )
    return SimpleNamespace(
        model_config=model_config,
        cache_config=cache_config,
        scheduler_config=scheduler_config,
        parallel_config=parallel_config,
        speculative_config=None,
        compilation_config=None,
        lora_config=None,
        load_config=None,
        offload_config=None,
        observability_config=None,
        reasoning_config=None,
    )


def _runner():
    r = GPUModelRunner(_vllm_config(), torch.device("cpu"))
    from types import SimpleNamespace
    # 单一 full-attention KV group（_may_reorder_batch 读 len(kv_cache_groups)）
    r.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
    return r


def _new_req(req_id, prompt, block_ids, num_computed=0):
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=list(prompt),
        mm_features=[],
        sampling_params=_greedy(),
        pooling_params=None,
        block_ids=(list(block_ids),),
        num_computed_tokens=num_computed,
        lora_request=None,
    )


def _cached(req_ids=(), resumed=(), new_blocks=(), computed=(), outputs=()):
    return CachedRequestData(
        req_ids=list(req_ids),
        resumed_req_ids=set(resumed),
        new_token_ids=[[] for _ in req_ids],
        all_token_ids={},
        new_block_ids=[(list(b),) if b is not None else None for b in new_blocks],
        num_computed_tokens=list(computed),
        num_output_tokens=list(outputs),
    )


def _sched_output(new_reqs=(), cached=None, num_scheduled=None,
                  total=None, finished=()):
    num_scheduled = num_scheduled or {}
    if total is None:
        total = sum(num_scheduled.values())
    return SchedulerOutput(
        scheduled_new_reqs=list(new_reqs),
        scheduled_cached_reqs=cached or _cached(),
        num_scheduled_tokens=num_scheduled,
        total_num_scheduled_tokens=total,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(finished),
        free_encoder_mm_hashes=[],
    )


def _logits_row(pick):
    row = [0.0] * VOCAB
    row[pick % VOCAB] = 1.0
    return row


def _snapshot_batch(r):
    """批次状态快照（_update_states 之后）。"""
    ib = r.input_batch
    rows = {}
    for rid, idx in sorted(ib.req_id_to_index.items(), key=lambda kv: kv[1]):
        n = int(ib.num_tokens_no_spec[idx])
        rows[f"{rid}@{idx}"] = ib.token_ids_cpu[idx, : n + 3].tolist()
    return {
        "req_ids": list(ib.req_ids),
        "req_id_to_index": dict(ib.req_id_to_index),
        "num_reqs": int(ib.num_reqs),
        "num_computed_tokens_cpu": ib.num_computed_tokens_cpu.tolist()[: ib.num_reqs],
        "num_tokens_no_spec": ib.num_tokens_no_spec.tolist()[: ib.num_reqs],
        "block_rows": {
            f"{rid}@{idx}": ib.block_table[0].block_table.np[idx, :2].tolist()
            for rid, idx in sorted(ib.req_id_to_index.items(), key=lambda kv: kv[1])
        },
        "requests_cache_keys": sorted(r.requests.keys()),
        "token_rows_active_prefix_plus3": rows,
    }


def _data_ptrs(r):
    return {
        "input_ids_cpu": r.input_ids.cpu.data_ptr(),
        "input_ids_gpu": r.input_ids.gpu.data_ptr(),
        "positions": r.positions.data_ptr(),
        "query_start_loc_cpu": r.query_start_loc.cpu.data_ptr(),
        "seq_lens": r.seq_lens.data_ptr(),
        "token_ids_cpu_tensor": r.input_batch.token_ids_cpu_tensor.data_ptr(),
    }


def main():
    r = _runner()
    r.enqueue_logits([{rid: _logits_row(pick) for rid, pick in step.items()}
                      for step in SCRIPTED_PICKS])

    beats = []

    # ---------------- 拍 1：r1/r2 首次全量（prefill） ----------------
    so1 = _sched_output(
        new_reqs=[
            _new_req("r1", [101, 102], [1]),
            _new_req("r2", [201, 202, 203], [2]),
        ],
        num_scheduled={"r1": 2, "r2": 3},
    )
    assert r.execute_model(so1) is None
    after_update_1 = _snapshot_batch(r)
    gathered_1 = r.input_ids.cpu[: so1.total_num_scheduled_tokens].tolist()
    positions_1 = r.positions[: so1.total_num_scheduled_tokens].tolist()
    ptrs_1 = _data_ptrs(r)
    out1 = r.sample_tokens(None)
    sampled_1 = {rid: out1.sampled_token_ids[i][0]
                 for i, rid in enumerate(out1.req_ids)}
    beats.append({
        "beat": 1,
        "note": "r1/r2 首次调度：scheduled_new_reqs 全量（prompt+块+采样参数），建 CachedRequestState 快照入 requests 缓存",
        "wire": {
            "new_reqs": [{"req_id": "r1", "prompt_len": 2, "block_ids_len": 1},
                         {"req_id": "r2", "prompt_len": 3, "block_ids_len": 1}],
            "cached_reqs": {"req_ids": [], "new_block_ids_lens": [],
                            "num_computed": [], "resumed": []},
            "finished": [],
            "num_scheduled_tokens": dict(so1.num_scheduled_tokens),
            "total": so1.total_num_scheduled_tokens,
        },
        "after_update": after_update_1,
        "gathered_input_ids": gathered_1,
        "positions": positions_1,
        "sampled": sampled_1,
        "after_sample": _snapshot_batch(r),
        "data_ptrs": ptrs_1,
    })

    # ---------------- 拍 2：decode，只发 diff ----------------
    so2 = _sched_output(
        cached=_cached(
            req_ids=["r1", "r2"],
            new_blocks=[[3], None],
            computed=[2, 3],
            outputs=[1, 1],
        ),
        num_scheduled={"r1": 1, "r2": 1},
    )
    assert r.execute_model(so2) is None
    after_update_2 = _snapshot_batch(r)
    gathered_2 = r.input_ids.cpu[: so2.total_num_scheduled_tokens].tolist()
    positions_2 = r.positions[: so2.total_num_scheduled_tokens].tolist()
    ptrs_2 = _data_ptrs(r)
    out2 = r.sample_tokens(None)
    sampled_2 = {rid: out2.sampled_token_ids[i][0]
                 for i, rid in enumerate(out2.req_ids)}
    beats.append({
        "beat": 2,
        "note": "老请求只发 diff：r1 追加新块 [3]（block_ids 1→[1,3]）、r2 无新块（None）；num_computed 覆盖为 [2,3]；positions=[2,3] 恰落在上拍写回的 token 上",
        "wire": {
            "new_reqs": [],
            "cached_reqs": {"req_ids": ["r1", "r2"], "new_block_ids_lens": [1, 0],
                            "num_computed": [2, 3], "resumed": []},
            "finished": [],
            "num_scheduled_tokens": dict(so2.num_scheduled_tokens),
            "total": so2.total_num_scheduled_tokens,
        },
        "after_update": after_update_2,
        "gathered_input_ids": gathered_2,
        "positions": positions_2,
        "sampled": sampled_2,
        "after_sample": _snapshot_batch(r),
        "data_ptrs": ptrs_2,
    })

    # ---------------- 拍 3：r2 finished + r3 新请求 ----------------
    so3 = _sched_output(
        new_reqs=[_new_req("r3", [301, 302], [4])],
        cached=_cached(req_ids=["r1"], new_blocks=[None], computed=[3],
                       outputs=[2]),
        num_scheduled={"r1": 1, "r3": 2},
        finished=["r2"],
    )
    assert r.execute_model(so3) is None
    after_update_3 = _snapshot_batch(r)
    gathered_3 = r.input_ids.cpu[: so3.total_num_scheduled_tokens].tolist()
    positions_3 = r.positions[: so3.total_num_scheduled_tokens].tolist()
    ptrs_3 = _data_ptrs(r)
    out3 = r.sample_tokens(None)
    sampled_3 = {rid: out3.sampled_token_ids[i][0]
                 for i, rid in enumerate(out3.req_ids)}
    beats.append({
        "beat": 3,
        "note": "r2 finished → 出 requests 缓存 + 出批次（洞@1）；r3 新请求落位时 pop_removed 复用最小空 slot=1——同拍『删→增』的洞复用",
        "wire": {
            "new_reqs": [{"req_id": "r3", "prompt_len": 2, "block_ids_len": 1}],
            "cached_reqs": {"req_ids": ["r1"], "new_block_ids_lens": [0],
                            "num_computed": [3], "resumed": []},
            "finished": ["r2"],
            "num_scheduled_tokens": dict(so3.num_scheduled_tokens),
            "total": so3.total_num_scheduled_tokens,
        },
        "after_update": after_update_3,
        "gathered_input_ids": gathered_3,
        "positions": positions_3,
        "sampled": sampled_3,
        "after_sample": _snapshot_batch(r),
        "data_ptrs": ptrs_3,
    })

    # ---------------- 拍 4：r3 被抢占（unscheduled） ----------------
    so4 = _sched_output(
        cached=_cached(req_ids=["r1"], new_blocks=[None], computed=[4],
                       outputs=[3]),
        num_scheduled={"r1": 1},
    )
    assert r.execute_model(so4) is None
    after_update_4 = _snapshot_batch(r)
    gathered_4 = r.input_ids.cpu[: so4.total_num_scheduled_tokens].tolist()
    positions_4 = r.positions[: so4.total_num_scheduled_tokens].tolist()
    ptrs_4 = _data_ptrs(r)
    out4 = r.sample_tokens(None)
    sampled_4 = {rid: out4.sampled_token_ids[i][0]
                 for i, rid in enumerate(out4.req_ids)}
    beats.append({
        "beat": 4,
        "note": "r3 本拍未调度（抢占）：unscheduled = cached∩批次 - (scheduled - resumed) = {r3} → 出批次但留 requests 缓存（快照不删，恢复靠它）",
        "wire": {
            "new_reqs": [],
            "cached_reqs": {"req_ids": ["r1"], "new_block_ids_lens": [0],
                            "num_computed": [4], "resumed": []},
            "finished": [],
            "num_scheduled_tokens": dict(so4.num_scheduled_tokens),
            "total": so4.total_num_scheduled_tokens,
        },
        "after_update": after_update_4,
        "gathered_input_ids": gathered_4,
        "positions": positions_4,
        "sampled": sampled_4,
        "after_sample": _snapshot_batch(r),
        "data_ptrs": ptrs_4,
    })

    # ---------------- 拍 5：r3 resumed（抢占恢复） ----------------
    so5 = _sched_output(
        cached=_cached(
            req_ids=["r1", "r3"],
            resumed=["r3"],
            new_blocks=[None, [5]],
            computed=[5, 0],
            outputs=[4, 1],
        ),
        num_scheduled={"r1": 1, "r3": 3},
    )
    assert r.execute_model(so5) is None
    after_update_5 = _snapshot_batch(r)
    gathered_5 = r.input_ids.cpu[: so5.total_num_scheduled_tokens].tolist()
    positions_5 = r.positions[: so5.total_num_scheduled_tokens].tolist()
    ptrs_5 = _data_ptrs(r)
    out5 = r.sample_tokens(None)
    sampled_5 = {rid: out5.sampled_token_ids[i][0]
                 for i, rid in enumerate(out5.req_ids)}
    beats.append({
        "beat": 5,
        "note": "r3 resumed：new_block_ids=[5] 整体替换（非 append，was [4]）+ num_computed=0 + 全量重算 3 token（含自己的 output 31——重算最后 token 以产 logits）",
        "wire": {
            "new_reqs": [],
            "cached_reqs": {"req_ids": ["r1", "r3"], "new_block_ids_lens": [0, 1],
                            "num_computed": [5, 0], "resumed": ["r3"]},
            "finished": [],
            "num_scheduled_tokens": dict(so5.num_scheduled_tokens),
            "total": so5.total_num_scheduled_tokens,
        },
        "after_update": after_update_5,
        "gathered_input_ids": gathered_5,
        "positions": positions_5,
        "sampled": sampled_5,
        "after_sample": _snapshot_batch(r),
        "data_ptrs": ptrs_5,
    })

    # ---------------- m14：固定地址断言（六缓冲×五拍） ----------------
    keys = list(ptrs_1.keys())
    data_ptr_stable = all(
        b["data_ptrs"][k] == ptrs_1[k] for b in beats for k in keys
    )

    trace = {
        "driver": "run_ch18_m02_reconcile.py",
        "mechanism": "ch18-m02 持久批次差量调和 _update_states（gpu_model_runner.py:L1192-L1566）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch18 implementation/ 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "trace_environment": "Windows host 纯 CPU（torch CPU + numpy；worker 侧 CUDA 面以 HOST SEAM 承载：HostEvent/HostCopyStream/脚本化 logits 行——impl-notes §Seam 清单）",
        "config": {
            "max_num_reqs": MAX_NUM_REQS,
            "max_model_len": MAX_MODEL_LEN,
            "max_num_batched_tokens": MAX_NUM_BATCHED_TOKENS,
            "block_size": 16,
            "max_blocks_per_req": MAX_MODEL_LEN // 16,
            "vocab": VOCAB,
            "async_scheduling": False,
            "sampling": "greedy（temperature=0，argmax=脚本 pick）",
        },
        "requests": {
            "r1": {"prompt": [101, 102], "admission_blocks": [1]},
            "r2": {"prompt": [201, 202, 203], "admission_blocks": [2],
                   "finish_at_beat": 3},
            "r3": {"prompt": [301, 302], "admission_blocks": [4],
                   "preempted_before_beat": 4, "resumed_at_beat": 5},
        },
        "scripted_picks": SCRIPTED_PICKS,
        "beats": beats,
        "data_ptr_keys": keys,
        "data_ptr_stable_across_5_beats": data_ptr_stable,
    }

    out_path = os.path.join(os.path.dirname(__file__), "ch18_m02_reconcile.json")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(trace, f, ensure_ascii=False, indent=1)
    print(f"trace -> {out_path}")
    # 摘要打印（人眼速核）
    for b in beats:
        print(f"beat {b['beat']}: batch={b['after_update']['req_ids']} "
              f"cache={b['after_update']['requests_cache_keys']} "
              f"sampled={b['sampled']}")
    print("data_ptr_stable_across_5_beats =", data_ptr_stable)


if __name__ == "__main__":
    main()
