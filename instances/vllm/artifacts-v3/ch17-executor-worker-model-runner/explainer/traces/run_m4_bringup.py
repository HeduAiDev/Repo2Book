"""Driver for m4 (MultiprocExecutor 拉起：星形装配 + READY 握手) — host run
against the ch17 subtract-only companion (pin vLLM v0.27.1 / 6e448d0ea).

Brings up a REAL MultiprocExecutor over 2 REAL spawned WorkerProc children
(driven through the real make_worker_process / worker_main / READY handshake /
worker_busy_loop machinery; worker side is the tests/_worker_double.ProbeWorker
resolved through the real qualname mechanism). Records only observable facts:

  * bring-up shape: 2 children, distinct pids, READY implies load_model ran
  * topology: 1 broadcast MQ + 2 response MQs + empty futures_queue + monitor
  * broadcast probe: ONE collective_rpc -> BOTH workers answer (1 enqueue,
    N readers — the control-plane/data-plane split in miniature)
  * callable branch: a cloudpickled lambda executes inside each child pid
  * shutdown: graceful exit of both children, MQs torn down

Table/figure numbers come from the JSON this writes (m4_bringup.json).
"""
import json
import os
import sys
import time
from pathlib import Path

_CHAPTER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CHAPTER))          # import implementation
sys.path.insert(0, str(_CHAPTER / "tests"))  # import _worker_double
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import _worker_double  # noqa: E402
from implementation._host_seams import VllmConfig  # noqa: E402
from implementation.executor.multiproc_executor import MultiprocExecutor  # noqa: E402


def mk_config(world_size=2, **par_over):
    from implementation._host_seams import ParallelConfig

    parallel = dict(
        world_size=world_size,
        local_world_size=world_size,
        tensor_parallel_size=world_size,
        pipeline_parallel_size=1,
        prefill_context_parallel_size=1,
        nnodes_within_dp=1,
        node_rank_within_dp=0,
        node_rank=0,
        distributed_executor_backend="mp",
        worker_cls="_worker_double.ProbeWorker",
    )
    parallel.update(par_over)
    return VllmConfig(parallel_config=ParallelConfig(**parallel))


def main():
    t0 = time.perf_counter()
    ex = MultiprocExecutor(mk_config())
    bringup_ms = (time.perf_counter() - t0) * 1e3

    parent_pid = os.getpid()
    workers = [
        {"rank": h.rank, "name": h.proc.name, "pid": h.proc.pid,
         "alive": h.proc.is_alive()}
        for h in ex.workers
    ]
    monitor = next(
        (t.name for t in __import__("threading").enumerate()
         if t.name == "MultiprocWorkerMonitor"),
        None,
    )

    # READY handshake evidence: load_model ran in-child BEFORE the READY send
    # (worker_main sends READY only after WorkerProc.__init__ which calls
    # load_model), so every worker must answer loaded=True from its own pid.
    identities = ex.collective_rpc("get_identity")
    loaded_before_ready = all(i["loaded"] for i in identities)

    # Broadcast probe: one collective_rpc (ONE enqueue into rpc_broadcast_mq,
    # multiproc_executor.py:L388) -> both workers receive and answer.
    echo_replies = ex.collective_rpc("echo", args=(7,))

    # Callable branch: the lambda is cloudpickle-serialised and executed
    # INSIDE each child process (bytes branch of worker_busy_loop).
    def child_pid(worker, salt):
        return [worker.rank, salt, os.getpid()]

    callable_replies = ex.collective_rpc(child_pid, args=("s",))

    topology = {
        "rpc_broadcast_mq_present": ex.rpc_broadcast_mq is not None,
        "response_mq_count": len(ex.response_mqs),
        "futures_queue_type": type(ex.futures_queue).__name__,
        "futures_queue_len": len(ex.futures_queue),
        "monitor_thread": monitor,
        "is_failed": ex.is_failed,
        "output_rank": ex.output_rank,
        "driver_ranks": [r for r in range(ex.world_size)
                         if ex._is_driver_worker(r)],
    }

    t1 = time.perf_counter()
    procs = [h.proc for h in ex.workers]
    ex.shutdown()
    shutdown_ms = (time.perf_counter() - t1) * 1e3
    after_shutdown = {
        "all_exited": all(not p.is_alive() for p in procs),
        "rpc_broadcast_mq": ex.rpc_broadcast_mq,  # None after teardown
        "response_mqs": len(ex.response_mqs),     # [] after teardown
    }

    doc = {
        "mechanism": "m4",
        "scenario": (
            "real MultiprocExecutor bring-up over 2 spawned WorkerProc "
            "children (ProbeWorker doubles via the real qualname mechanism); "
            "win32 spawn context (HOST SEAM; unix real default is fork/spawn "
            "per VLLM_WORKER_MULTIPROC_METHOD)"
        ),
        "config": {"backend": "mp", "world_size": 2,
                   "tensor_parallel_size": 2,
                   "worker_cls": "_worker_double.ProbeWorker"},
        "parent_pid": parent_pid,
        "bringup_ms": round(bringup_ms, 1),
        "workers": workers,
        "pids_distinct": len({w["pid"] for w in workers}) == 2,
        "pids_differ_from_parent": all(w["pid"] != parent_pid for w in workers),
        "ready_probe": {
            "identities": identities,
            "loaded_before_ready": loaded_before_ready,
            "child_pids_match_workers": (
                {i["pid"] for i in identities} == {w["pid"] for w in workers}
            ),
        },
        "broadcast_probe": {
            "rpc_calls": 1,
            "reply_count": len(echo_replies),
            "reply_ranks": sorted(r["rank"] for r in echo_replies),
            "reply_args_all_7": all(r["args"][0] == 7 for r in echo_replies),
            "note": (
                "one enqueue into rpc_broadcast_mq (multiproc_executor.py:L388) "
                "-> every reader receives it: reply_count == world_size"
            ),
        },
        "callable_probe": {
            "replies": callable_replies,
            "executed_in_child_pids": all(
                r[2] != parent_pid and r[1] == "s" for r in callable_replies
            ),
        },
        "topology": topology,
        "shutdown": {
            "graceful_ms": round(shutdown_ms, 1),
            **after_shutdown,
        },
    }
    out = Path(__file__).resolve().parent / "m4_bringup.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f"wrote {out}")
    print(json.dumps(doc, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
