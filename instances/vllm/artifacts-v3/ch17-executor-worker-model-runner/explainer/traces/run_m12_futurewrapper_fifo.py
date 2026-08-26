"""Driver for m12 (FutureWrapper FIFO 排空) — host run against the ch17
subtract-only companion (pin vLLM v0.27.1 / 6e448d0ea).

Part A — the queue mechanics on the REAL FutureWrapper + REAL deque
(multiproc_executor.py:L70-L100), with scripted response providers that
record the drain order; plus the exception-forwarding and the
"timeout not implemented" edges.

Part B — the same mechanism over a REAL MultiprocExecutor with REAL spawned
WorkerProc children (TwoPhaseProbeWorker), in the exact shape the real
engine's async-scheduling loop uses (core.py step_with_batch_queue
L661-L672): execute_model(non_block=True) and sample_tokens(non_block=True)
issued BACK-TO-BACK — several futures in flight before any is consumed,
then collected out of order; the kth response-MQ dequeue must pair with the
kth RPC. Distinct per-round token totals make the pairing visible.

Writes m12_futurewrapper_fifo.json; table rows come from these values only.
"""
import json
import sys
import time
from collections import deque
from pathlib import Path

_CHAPTER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CHAPTER))            # import implementation
sys.path.insert(0, str(_CHAPTER / "tests"))  # import _worker_double
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from implementation._host_seams import (  # noqa: E402
    GrammarOutput,
    ParallelConfig,
    SchedulerOutput,
    VllmConfig,
)
from implementation.executor.multiproc_executor import (  # noqa: E402
    FutureWrapper,
    MultiprocExecutor,
)


def spawn(worker_cls="_worker_double.TwoPhaseProbeWorker"):
    cfg = VllmConfig(
        parallel_config=ParallelConfig(
            world_size=2,
            local_world_size=2,
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            prefill_context_parallel_size=1,
            nnodes_within_dp=1,
            node_rank_within_dp=0,
            node_rank=0,
            distributed_executor_backend="mp",
            worker_cls=worker_cls,
        )
    )
    return MultiprocExecutor(cfg)


def so(total, rid="g"):
    return SchedulerOutput(
        num_scheduled_tokens={rid: total}, total_num_scheduled_tokens=total
    )


def part_a_queue_mechanics():
    """Three FutureWrappers over one shared deque; scripted responses."""
    fq: deque = deque()
    drain_order = []

    def resp_for(k):
        def get():
            drain_order.append(k)
            return f"resp-{k}"

        return get

    queue_after_each_append = []
    futs = {}
    for k in (1, 2, 3):
        futs[k] = FutureWrapper(fq, get_response=resp_for(k))
        queue_after_each_append.append([f"f{x}" for x in reversed(range(1, k + 1))])

    newest_result = futs[3].result()  # newest collected first
    done_after_collect = {f"f{k}": futs[k].done() for k in (1, 2, 3)}
    results = {f"f{k}": futs[k].result() for k in (1, 2, 3)}
    queue_len_after = len(fq)

    # exception forwarding edge
    def boom():
        raise RuntimeError("mq says no")

    f_exc = FutureWrapper(deque(), get_response=boom)
    exc_type = exc_msg = None
    try:
        f_exc.result()
    except RuntimeError as e:
        exc_type, exc_msg = type(e).__name__, str(e)

    # timeout edge (real code raises on the way in)
    f_to = FutureWrapper(deque(), get_response=lambda: None)
    to_msg = None
    try:
        f_to.result(timeout=1)
    except RuntimeError as e:
        to_msg = str(e)

    return {
        "queue_after_each_appendleft": queue_after_each_append,
        "note_append_semantics": (
            "appendleft enters at the LEFT, result() pops from the RIGHT -> "
            "oldest first: the deque is FIFO"
        ),
        "collect_call": "f3.result() (newest first)",
        "newest_result": newest_result,
        "drain_order": drain_order,
        "done_after_collect": done_after_collect,
        "results": results,
        "queue_len_after": queue_len_after,
        "exception_probe": {"type": exc_type, "message": exc_msg},
        "timeout_probe": {"message": to_msg},
    }


def main():
    part_a = part_a_queue_mechanics()

    ex = spawn()
    fq = ex.futures_queue
    e2e = {"world_size": ex.world_size, "output_rank": ex.output_rank}

    # --- round 1: the async-scheduling shape — 2 RPCs in flight, newest
    # collected first. core.py:L661-L672 issues exec + sample back-to-back.
    fut_exec = ex.execute_model(so(3), non_block=True)
    e2e["round1"] = {"q_len_after_exec": len(fq),
                     "exec_done_before_sample": fut_exec.done()}
    fut_samp = ex.sample_tokens(None, non_block=True)
    e2e["round1"]["q_len_after_sample"] = len(fq)
    e2e["round1"]["exec_done_still"] = fut_exec.done()
    t0 = time.perf_counter()
    out_samp = fut_samp.result()  # drains fut_exec on the way
    e2e["round1"]["sample_result_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
    e2e["round1"]["sample_result"] = out_samp
    e2e["round1"]["exec_drained_by_sample_result"] = {
        "done": fut_exec.done(), "result": fut_exec.result()}
    e2e["round1"]["q_len_after_collect"] = len(fq)

    # --- round 2: engine order — consume the OLDEST first (batch_queue.pop),
    # None triggers the second harvest (core.py:L602-L604 shape).
    import numpy as np

    grammar = GrammarOutput(
        structured_output_request_ids=["g"],
        grammar_bitmask=np.array([[0b00010010]], dtype=np.int32),
    )
    fut_e2 = ex.execute_model(so(7), non_block=True)
    fut_s2 = ex.sample_tokens(grammar, non_block=True)
    e2e["round2"] = {
        "q_len_after_pair": len(fq),
        "exec_result": fut_e2.result(),          # None -> oldest first
        "sample_result": fut_s2.result(),        # dict scheduled=7
        "pairing_note": "scheduled == 7 (round-2 RPC), not 3: kth dequeue "
                        "pairs with kth RPC",
    }

    # --- round 3: TWO pairs in flight (the async steady state:
    # max_concurrent_batches == 2, config/vllm.py:L539-L548) -> 4 futures in
    # the queue; collecting only the newest drains all four in issue order.
    t_issue = time.perf_counter()
    fut_e3 = ex.execute_model(so(5), non_block=True)
    fut_s3 = ex.sample_tokens(None, non_block=True)
    fut_e4 = ex.execute_model(so(9), non_block=True)
    fut_s4 = ex.sample_tokens(None, non_block=True)
    issue_ms = (time.perf_counter() - t_issue) * 1e3
    e2e["round3"] = {
        "issued": ["exec(total=5)", "sample(None)", "exec(total=9)",
                   "sample(None)"],
        "q_len_after_4_issues": len(fq),
        "issue_ms_nonblocking": round(issue_ms, 2),
        "any_done_before_collect": [fut_e3.done(), fut_s3.done(),
                                    fut_e4.done(), fut_s4.done()],
    }
    t0 = time.perf_counter()
    newest = fut_s4.result()
    drain_ms = (time.perf_counter() - t0) * 1e3
    e2e["round3"]["newest_result"] = newest
    e2e["round3"]["drain_ms"] = round(drain_ms, 2)
    e2e["round3"]["all_done_after_drain"] = [fut_e3.done(), fut_s3.done(),
                                             fut_e4.done(), fut_s4.done()]
    e2e["round3"]["results_after_drain"] = {
        "e3": fut_e3.result(), "s3": fut_s3.result(),
        "e4": fut_e4.result(), "s4": fut_s4.result()}
    e2e["round3"]["q_len_after_drain"] = len(fq)

    procs = [h.proc for h in ex.workers]
    ex.shutdown()
    e2e["shutdown_all_exited"] = all(not p.is_alive() for p in procs)

    doc = {
        "mechanism": "m12",
        "part_a_queue_mechanics": part_a,
        "part_b_real_executor_e2e": e2e,
        "scenario_note": (
            "Part B issues execute_model + sample_tokens back-to-back with "
            "non_block=True — exactly the real step_with_batch_queue shape "
            "(vllm/v1/engine/core.py:L661-L672); under async scheduling "
            "max_concurrent_batches == 2 (vllm/config/vllm.py:L539-L548), so "
            "two pairs / four futures in flight is the designed steady state"
        ),
    }
    out = Path(__file__).resolve().parent / "m12_futurewrapper_fifo.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f"wrote {out}")
    print(json.dumps(doc, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
