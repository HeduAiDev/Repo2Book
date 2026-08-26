# Test-side worker doubles for the ch17 companion tests.
#
# These are NOT part of the subtract-only implementation: the companion's own
# GPU Worker needs a CUDA device, so the multiprocessing end-to-end tests drive
# the REAL executor / WorkerWrapperBase / WorkerProc machinery against these
# lightweight worker classes, resolved through the very same qualname mechanism
# vLLM uses (parallel_config.worker_cls -> resolve_obj_by_qualname inside
# WorkerWrapperBase.init_worker). The doubles implement the WorkerBase
# two-phase contract (worker_base.py:L142-L157) themselves so the contract is
# observable at every layer.

import os
import time

from implementation._host_seams import AsyncModelRunnerOutput


# SOURCE-anchored: subclass of the AsyncModelRunnerOutput ABC
# (vllm/v1/outputs.py:L325-L334 — get_output is a blocking call that should
# only be called once per output)
class ScriptedAsyncOutput(AsyncModelRunnerOutput):
    def __init__(self, payload, ready_after=0.0):
        import threading

        self.payload = payload
        self.ready_after = ready_after
        self.event = threading.Event()
        self.get_output_calls = 0

    def get_output(self):
        self.get_output_calls += 1
        self.event.wait(self.ready_after)
        return self.payload


# SOURCE-inspired: WorkerBase two-phase contract (vllm/v1/worker/worker_base.py:L142-L157)
class ProbeWorker:
    """Echo worker: records rank identity, answers RPCs."""

    def __init__(
        self,
        vllm_config,
        local_rank,
        rank,
        distributed_init_method,
        is_driver_worker=False,
        shared_worker_lock=None,
        **kwargs,
    ):
        self.vllm_config = vllm_config
        self.local_rank = local_rank
        self.rank = rank
        self.is_driver_worker = is_driver_worker
        self.loaded = False
        self.kv_blocks = None
        self.shutdown_called = False
        # async_output_busy_loop touches `self.worker.device` via hasattr
        self.device = f"cpu:{local_rank}"

    def init_device(self) -> None:
        pass

    def load_model(self, *, load_dummy_weights: bool = False) -> None:
        time.sleep(0.05)  # keeps the READY handshake visibly staggered
        self.loaded = True

    def get_identity(self):
        return {"rank": self.rank, "pid": os.getpid(), "loaded": self.loaded}

    def echo(self, *args, **kwargs):
        return {"rank": self.rank, "args": args, "kwargs": kwargs}

    def collective_check(self):
        return self.rank

    def boom(self):
        raise RuntimeError("boom in worker")

    def slowpoke(self):
        time.sleep(3.0)
        return "late"

    def initialize_from_config(self, kv_cache_config):
        self.kv_blocks = kv_cache_config.num_blocks

    def determine_available_memory(self) -> int:
        return 12345

    def compile_or_warm_up_model(self):
        from implementation.worker.worker_base import CompilationTimes

        return CompilationTimes(language_model=float(self.rank + 1), encoder=0.5)

    def execute_model(self, scheduler_output):
        return {
            "worker_rank": self.rank,
            "scheduled": scheduler_output.total_num_scheduled_tokens,
        }

    def sample_tokens(self, grammar_output):
        return {"worker_rank": self.rank, "grammar": grammar_output is not None}

    def check_health(self) -> None:
        return

    def shutdown(self) -> None:
        self.shutdown_called = True


class TwoPhaseProbeWorker(ProbeWorker):
    """Implements the two-phase contract like the real runner does:
    execute_model returns None + stashes; sample_tokens un-stashes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pending = None

    def execute_model(self, scheduler_output):
        if self._pending is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )
        self._pending = scheduler_output
        return None

    def sample_tokens(self, grammar_output):
        stashed = self._pending
        self._pending = None
        return {
            "worker_rank": self.rank,
            "scheduled": stashed.total_num_scheduled_tokens,
            "grammar": grammar_output is not None,
        }


class AsyncProbeWorker(ProbeWorker):
    """execute_model returns a scripted AsyncModelRunnerOutput so the
    enqueue_output async branch / async_output_copy_thread get exercised."""

    def execute_model(self, scheduler_output):
        return ScriptedAsyncOutput(
            payload={"worker_rank": self.rank, "async": True}, ready_after=0.2
        )
