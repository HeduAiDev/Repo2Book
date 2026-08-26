"""Driver for m15 (两段式契约：execute_model → None → sample_tokens) — host
run against the ch17 subtract-only companion (pin vLLM v0.27.1 / 6e448d0ea).

Part A — the runner state machine on the REAL GPUModelRunner protocol face
(assert / pack / return None / unpack-and-clear / bitmask / _sample call
site), with a REAL torch logits row and the REAL apply_grammar_bitmask +
GrammarOutput. The forward deep-water is subtraction #6 (ch18/19 domain): the
companion binds logits=None and this driver injects the scripted row via
ExecuteModelState._replace — same convention as the m15 tests. Everything
else (entry assertion, 10-field pack, exit unpack-then-clear, bitmask
application, _sample/_update_states call sites) is companion real code.

Part B — the Worker layer delegating to the runner (real Worker.execute_model
/ sample_tokens, double runner records the delegation order).

Part C — the two-phase pair over a REAL MultiprocExecutor (two spawned
children): every beat costs TWO full broadcast/harvest round trips (mp cost
of the contract, why_chains WC4), timed per beat.

Writes m15_two_phase.json; table rows come from these values only.
"""
import json
import sys
import time
from pathlib import Path

_CHAPTER = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_CHAPTER))            # import implementation
sys.path.insert(0, str(_CHAPTER / "tests"))  # import _worker_double
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from implementation._host_seams import (  # noqa: E402
    GrammarOutput,
    ModelRunnerOutput,
    ParallelConfig,
    SchedulerOutput,
    VllmConfig,
)
from implementation.executor.multiproc_executor import MultiprocExecutor  # noqa: E402
from implementation.worker.gpu_model_runner import (  # noqa: E402
    ExecuteModelState,
    GPUModelRunner,
)
from implementation.worker.gpu_worker import Worker  # noqa: E402

LOGITS_ROW = [0.0, 3.0, 0.0, 0.0, 7.0, 9.0, 0.0, 0.0]  # favorite = token 5
MASK_WORD = 0b00010010                                  # allowed = {1, 4}


def so(total, rid="g"):
    return SchedulerOutput(
        num_scheduled_tokens={rid: total}, total_num_scheduled_tokens=total
    )


class _SimpleBatch:
    def __init__(self, req_ids):
        self.req_ids = list(req_ids)


def fresh_runner():
    """The companion's __init__ deep water is ch18's; the protocol face only
    needs the single-slot field (gpu_model_runner.py:L941-L942)."""
    r = GPUModelRunner.__new__(GPUModelRunner)
    r.execute_model_state = None
    r.kv_connector_output = None
    r.input_batch = _SimpleBatch(["g"])
    r.use_async_scheduling = False
    return r


def part_a_runner_protocol():
    out = {"logits_row": LOGITS_ROW, "vocab": len(LOGITS_ROW),
           "argmax_before": int(torch.tensor(LOGITS_ROW).argmax()),
           "mask_word_binary": bin(MASK_WORD),
           "allowed_set": [i for i in range(8) if (MASK_WORD >> i) & 1]}

    r = fresh_runner()

    # ② beat 1: pack and return None
    ret1 = r.execute_model(so(3))
    st = r.execute_model_state
    out["beat1"] = {
        "call": "execute_model(批 {'g': 3})",
        "returns": ret1,
        "state_type": type(st).__name__,
        "state_field_count": len(ExecuteModelState._fields),
        "state_fields": list(ExecuteModelState._fields),
        "logits_placeholder_none": st.logits is None,  # subtraction #6
        "call_order": [],
    }

    # inject the scripted forward product (tests' m15 convention)
    logits = torch.tensor([LOGITS_ROW])
    r.execute_model_state = r.execute_model_state._replace(logits=logits)

    # misuse guard: a second execute_model while the slot holds beat 1
    err = None
    try:
        r.execute_model(so(1))
    except RuntimeError as e:
        err = str(e)
    out["misuse_guard"] = {"error_type": "RuntimeError", "message": err}

    # ④ beat 1: unpack -> clear -> bitmask -> _sample (call sites recorded
    # via instance attributes, the same probe the tests use)
    seq = []
    seen = {}

    class _SpySamplerOutput:
        sampled_token_ids = [[4]]

    def _spy_sample(logits_arg, spec_arg):
        seq.append("_sample")
        seen["sampler_argmax"] = int(logits_arg.argmax())
        seen["sampler_saw_masked"] = bool(
            logits_arg[0, 5].item() == float("-inf")
        )
        return _SpySamplerOutput()

    r.__dict__["_sample"] = _spy_sample
    r.__dict__["_update_states_after_model_execute"] = (
        lambda sampled, sched: seq.append("_update_states_after_model_execute")
    )
    grammar = GrammarOutput(
        structured_output_request_ids=["g"],
        grammar_bitmask=np.array([[MASK_WORD]], dtype=np.int32),
    )
    ret4 = r.sample_tokens(grammar)
    beat1_call_order = list(seq)  # snapshot before beat 2 appends
    masked_row = [("-inf" if v == float("-inf") else float(v))
                  for v in logits[0].tolist()]
    out["beat1"].update({
        "sample_returns": ret4,
        "state_after_sample": r.execute_model_state,  # None = unpack-then-clear
        "call_order": beat1_call_order,
        "sampler_argmax_seen": seen.get("sampler_argmax"),
        "sampler_saw_masked_logits": seen.get("sampler_saw_masked"),
        "logits_after_mask": masked_row,
        "argmax_after": int(logits.argmax()),
        "favorite_logit_now": "-inf (token 5)",
    })

    # ② beat 2: the cleared slot accepts the next beat — alternating reuse
    ret1b = r.execute_model(so(1))
    st2 = r.execute_model_state
    r.__dict__["_sample"] = lambda lg, sp: seq.append("_sample") or _SpySamplerOutput()
    ret4b = r.sample_tokens(None)
    out["beat2"] = {
        "call": "execute_model(批 {'g': 1}) after the clear",
        "returns": ret1b,
        "repacked_state_present": st2 is not None,
        "sample_returns": ret4b,
        "state_after_sample": r.execute_model_state,
        "call_order_tail": seq[-2:],
        "note": "no grammar this beat: sample_tokens(None) still runs the pair",
    }

    # empty-slot early-exit branch (non-last-PP pass-through shape)
    r2 = fresh_runner()
    early = r2.sample_tokens(None)
    out["empty_slot_early_exit"] = {
        "state_was": None,
        "returns_type": type(early).__name__,
        "req_ids": list(early.req_ids),
        "note": "with_kv_conn_output_only(None) -> EMPTY_MODEL_RUNNER_OUTPUT "
                "(outputs.py:L311-L323)",
    }
    return out


def part_b_worker_delegation():
    cfg = VllmConfig(parallel_config=ParallelConfig(world_size=1))
    w = Worker(
        vllm_config=cfg,
        local_rank=0,
        rank=0,
        distributed_init_method="tcp://127.0.0.1:1",
        is_driver_worker=True,
    )
    seq = []

    class _Runner:
        def execute_model(self, so, intermediate_tensors=None):
            seq.append("runner.execute_model")
            return None

        def sample_tokens(self, grammar_output):
            seq.append("runner.sample_tokens")
            return "MRO"

    w.model_runner = _Runner()
    exec_ret = w.execute_model(so(3))
    samp_ret = w.sample_tokens(None)
    return {
        "worker_execute_returns": exec_ret,
        "worker_sample_returns": samp_ret,
        "delegation_order": seq,
        "note": "Worker.execute_model / sample_tokens (gpu_worker.py:L1012-"
                "L1107) wrap the runner and carry the PP relay; TP=1/PP=1 "
                "here so the wrapper is a pure delegation",
    }


def part_c_mp_pair_timing():
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
            worker_cls="_worker_double.TwoPhaseProbeWorker",
        )
    )
    ex = MultiprocExecutor(cfg)
    beats = []
    for i, total in enumerate((3, 1, 1)):
        t0 = time.perf_counter()
        exec_ret = ex.execute_model(so(total))          # blocking: full trip 1
        t_exec = (time.perf_counter() - t0) * 1e3
        t1 = time.perf_counter()
        samp_ret = ex.sample_tokens(None)               # blocking: full trip 2
        t_samp = (time.perf_counter() - t1) * 1e3
        beats.append({
            "beat": i + 1, "total": total,
            "exec_returns": exec_ret, "sample_result": samp_ret,
            "exec_ms": round(t_exec, 2), "sample_ms": round(t_samp, 2),
            "beat_total_ms": round(t_exec + t_samp, 2),
        })
    # issue-side cost of the non-blocking face (enqueue + future wrap only)
    t0 = time.perf_counter()
    fut = ex.execute_model(so(1), non_block=True)
    issue_ms = (time.perf_counter() - t0) * 1e3
    fut.result()
    procs = [h.proc for h in ex.workers]
    ex.shutdown()
    return {
        "world_size": ex.world_size,
        "output_rank": ex.output_rank,
        "beats": beats,
        "nonblock_issue_ms": round(issue_ms, 2),
        "per_beat_note": "one beat = TWO broadcast/harvest round trips "
                         "(② execute_model + ④ sample_tokens), the mp cost "
                         "of the two-phase contract (WC4)",
        "shutdown_all_exited": all(not p.is_alive() for p in procs),
    }


def main():
    doc = {
        "mechanism": "m15",
        "part_a_runner_protocol": part_a_runner_protocol(),
        "part_b_worker_delegation": part_b_worker_delegation(),
        "part_c_mp_pair_timing": part_c_mp_pair_timing(),
        "seam_note": (
            "logits injected via ExecuteModelState._replace — the forward "
            "deep-water is subtraction #6 (ch18/19 domain) and binds None; "
            "assertion / 10-field pack / return None / unpack-then-clear / "
            "apply_grammar_bitmask / _sample call sites are companion real "
            "code; Part C workers are TwoPhaseProbeWorker doubles driven "
            "through the real spawn / broadcast / busy-loop machinery"
        ),
    }
    out = Path(__file__).resolve().parent / "m15_two_phase.json"
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print(f"wrote {out}")
    print(json.dumps(doc, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
