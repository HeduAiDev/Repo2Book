# ch34 explainer 取证脚手架：spawn gloo 进程池。
# 协议与 tests/_pool.py 同构（(job, payload) 广播 → 各 rank 回传），但持有独立
# 注册表 _trace_jobs.JOBS——本文件是 explainer 取证件，不改 tests/ 的任何东西；
# 查不到的 job 回落 tests/_pool.JOBS（同一批标准作业可直接复用）。
from __future__ import annotations

import multiprocessing as mp
import sys
import traceback
from pathlib import Path

_TRACES = Path(__file__).resolve().parent
_TESTS = _TRACES.parents[1] / "tests"
for _p in (str(_TRACES), str(_TESTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

_TMP_DIR = _TRACES / ".tmp"


def _worker_main(rank: int, world: int, conn):
    import _pool  # tests/ 脚手架：JOBS 注册表 + _init_model_parallel 等前奏
    import _trace_jobs

    try:
        while True:
            msg = conn.recv()
            if msg[0] == "STOP":
                break
            job, payload = msg
            fn = _trace_jobs.JOBS[world].get(job)
            if fn is None:
                fn = _pool.JOBS[world][job]
            result = fn(rank, world, payload)
            conn.send(("OK", result))
    except Exception as e:  # noqa: BLE001
        conn.send(("ERR", f"{e}\n{traceback.format_exc()}"))
    finally:
        import torch.distributed as dist

        try:
            from vllm.distributed.parallel_state import (
                destroy_model_parallel,
            )

            destroy_model_parallel()
        except Exception:
            pass
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass
        conn.close()


class TracePool:
    """N 个 spawn 子进程；map(job, payload) 广播给全部 rank、收集各 rank 返回值。"""

    def __init__(self, world: int):
        _TMP_DIR.mkdir(exist_ok=True)
        self.world = world
        ctx = mp.get_context("spawn")
        self.conns = []
        self.procs = []
        for rank in range(world):
            parent, child = ctx.Pipe()
            p = ctx.Process(target=_worker_main, args=(rank, world, child))
            p.start()
            self.conns.append(parent)
            self.procs.append(p)

    def map(self, job: str, payload=None):
        for c in self.conns:
            c.send((job, payload))
        results = []
        for c in self.conns:
            tag, val = c.recv()
            if tag == "ERR":
                raise RuntimeError(f"worker error:\n{val}")
            results.append(val)
        return results

    def close(self):
        for c in self.conns:
            try:
                c.send(("STOP", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(30)
            if p.exitcode not in (0, None):
                raise RuntimeError(f"worker exited with {p.exitcode}")
        for p in self.procs:
            if p.is_alive():
                p.kill()
        for c in self.conns:
            c.close()
