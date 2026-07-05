#!/usr/bin/env python3
"""驱动精简版 ParentRequest.fan-out，取真实数值轨迹（trace_source=run）。

只做减法的精简版位于本章 implementation/：
  parallel_sampling.py  —— ParentRequest（子 id/子参数派生 + 输出聚合）
  messages.py           —— EngineCoreRequest / SamplingParams / RequestOutputKind / CompletionOutput

本脚本以 n=4 的父请求为例，分别在 seed=None 与 seed=42 两种情形下 fan-out，
逐子记录 (idx, child_req_id, child.n, child.seed, 复用/拷贝, 缓存命中)；再跑一遍
FINAL_ONLY 输出聚合。所有打印的数字即 explainer 表格 / figure-spec 的溯源真相源。
"""
import json
import sys
from copy import copy
from pathlib import Path

HERE = Path(__file__).resolve()
IMPL = HERE.parents[2] / "implementation"
sys.path.insert(0, str(IMPL))

from messages import (  # noqa: E402
    CompletionOutput,
    EngineCoreRequest,
    RequestOutputKind,
    SamplingParams,
)
from parallel_sampling import ParentRequest  # noqa: E402

PARENT_ID = "req-abc-3f9a2b1c"  # assign_request_id 注入 8 字符后缀后的父内部 id


def make_parent(n, seed, output_kind=RequestOutputKind.CUMULATIVE):
    sp = SamplingParams(n=n, seed=seed, output_kind=output_kind)
    return EngineCoreRequest(
        request_id=PARENT_ID,
        prompt_token_ids=[1, 2, 3],
        mm_features=None,
        sampling_params=sp,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        external_req_id=PARENT_ID,
    )


def fanout(n, seed):
    """复刻 async_llm.add_request 的 n>1 fan-out 循环，逐子记录派生结果。"""
    request = make_parent(n, seed)
    parent = ParentRequest(request)
    rows = []
    for idx in range(n):
        cid, child_params = parent.get_child_info(idx)
        cache_hit = parent.cached_child_sampling_params is child_params
        last = idx == n - 1
        child_request = request if last else copy(request)
        child_request.request_id = cid
        child_request.sampling_params = child_params
        rows.append({
            "idx": idx,
            "child_req_id": cid,
            "child_n": child_params.n,
            "child_seed": child_params.seed,
            "obj": "reuse-parent" if last else "copy(request)",
            "cache": "hit" if (cache_hit and idx > 0) else "build",
        })
    return rows, parent


def aggregate_final_only(n):
    """FINAL_ONLY：逐子 completion 完成，攒齐 n 个再整批返回。"""
    request = make_parent(n, seed=None, output_kind=RequestOutputKind.FINAL_ONLY)
    parent = ParentRequest(request)
    child_ids = [parent.get_child_info(i)[0] for i in range(n)]
    steps = []
    for i, cid in enumerate(child_ids):
        co = CompletionOutput(index=i, _finished=True)
        outputs, finished = parent.get_outputs(cid, co)
        steps.append({
            "arrive_index": i,
            "child_req_id": cid,
            "remaining": len(parent.child_requests),
            "batch_len": len(outputs),
            "finished": finished,
        })
    return steps


def main():
    out = {}
    out["fanout_seed_none"] = fanout(4, seed=None)[0]
    out["fanout_seed_42"] = fanout(4, seed=42)[0]
    out["aggregate_final_only_n4"] = aggregate_final_only(4)

    # n==1 快路径：async_llm 里 is_pooling or n==1 → 不 fan-out
    req1 = make_parent(1, seed=None)
    out["fast_path_n1"] = {"n": req1.params.n, "fanout": False}

    print(json.dumps(out, ensure_ascii=False, indent=2))
    dest = HERE.parent / "fanout.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
