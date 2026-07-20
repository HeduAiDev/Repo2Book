"""ch32 explainer driver t03 - m04 并行填充的结构性前提与分块.

三个配置跑真实 StructuredOutputManager:
  cfg1 max_num_seqs=128 -> `128 < 128` 为假 -> executor_for_fillmask 根本不被构造
                            (并行分支是结构性死代码:参与装配的请求数 <= max_num_seqs)
  cfg2 max_num_seqs=256, 本步 128 个结构化请求 -> `len(ids) > 128` 为假 -> 仍走串行
  cfg3 max_num_seqs=256, 本步 256 个结构化请求, 无投机 -> 并行:ceil(256/16)=16 个任务
另记录本机 cpu_count 推出的 max_workers(与 pin 逻辑一致,但取值随机器变化)。
输出 JSON 存 t03_parallel_threshold.json。
"""
import json
import multiprocessing
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
CH = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(CH, "implementation"))
sys.path.insert(0, os.path.join(CH, "tests"))

from conftest import FakeBackend, FakeGrammar, make_request  # noqa: E402
from structured_output_manager import StructuredOutputManager  # noqa: E402

VOCAB = 96
out = {"host_env": {
    "cpu_count": multiprocessing.cpu_count(),
    "max_workers_formula_value": max(1, min(multiprocessing.cpu_count() // 2, 8)),
    "note": "max_workers = max(1, min(cpu_count//2, 8)) -- 本机 cpu_count 偏小,"
            " pin 逻辑不变但取值随部署机变化",
}}


def run_cfg(name, max_num_seqs, num_reqs, num_spec):
    mgr = StructuredOutputManager(max_num_seqs=max_num_seqs,
                                  max_num_spec_tokens=num_spec)
    mgr.backend = FakeBackend(VOCAB)
    has_executor = hasattr(mgr, "executor_for_fillmask")

    submits = []
    threads = set()
    if has_executor:
        orig_submit = mgr._async_submit_fill_bitmask
        orig_fill = mgr._fill_bitmasks

        def counting_submit(batch, _o=orig_submit):
            submits.append(len(batch))
            return _o(batch)

        def counting_fill(batch, _o=orig_fill):
            threads.add(threading.current_thread().name)
            return _o(batch)

        mgr._async_submit_fill_bitmask = counting_submit
        mgr._fill_bitmasks = counting_fill

    reqs = {}
    ids = []
    for i in range(num_reqs):
        rid = "r%03d" % i
        reqs[rid] = make_request(rid, FakeGrammar([{5, 7}]))
        ids.append(rid)
    spec = {rid: [5] * num_spec for rid in ids} if num_spec else {}
    bm = mgr.grammar_bitmask(reqs, ids, spec)

    took_parallel = bool(submits)
    return {
        "config": name,
        "max_num_seqs": max_num_seqs,
        "fill_bitmask_parallel_threshold": mgr.fill_bitmask_parallel_threshold,
        "executor_for_fillmask_constructed": has_executor,
        "num_structured_reqs_this_step": num_reqs,
        "max_num_spec_tokens": num_spec,
        "branch_taken": "parallel" if took_parallel else "serial",
        "num_submitted_tasks": len(submits),
        "task_batch_sizes": sorted(set(submits)) if submits else [],
        "num_worker_threads_observed": len(threads),
        "bitmask_rows": int(bm.shape[0]),
        "bitmask_cols": int(bm.shape[1]),
    }


out["configs"] = [
    run_cfg("cfg1_max_num_seqs_128", 128, 128, 0),
    run_cfg("cfg2_max_num_seqs_256_batch_128", 256, 128, 0),
    run_cfg("cfg3_max_num_seqs_256_batch_256", 256, 256, 0),
    run_cfg("cfg4_max_num_seqs_256_batch_256_spec2", 256, 256, 2),
]

path = os.path.join(HERE, "t03_parallel_threshold.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
