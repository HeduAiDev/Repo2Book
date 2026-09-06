"""27B real-engine trace for ch9 async rework — sync step() vs async
step_with_batch_queue() in the GPU-compute-dominant regime (Qwen3.8-27B-FP8).

Container run (host shell, from repo root). Image = vllm/vllm-openai:latest
(same deps as vllm/vllm-omni:latest — torch 2.11.0+cu130, transformers 5.14.1 —
but WITHOUT the omni image's vllm_omni patch layer, which hooks
Request.from_engine_core_request and reads omni-only fields
(EngineCoreRequest.additional_information) that pin v0.27.1 does not define,
crashing pin-tree request construction):

  MSYS_NO_PATHCONV=1 VLLM_USAGE_STATS=0 docker run --rm --gpus all \
    -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e VLLM_USE_V2_MODEL_RUNNER=0 \
    --entrypoint /usr/bin/python3 \
    -v E:/Laboratory/Repo2Book:/work \
    -v E:/Laboratory/models/Qwen3.8-27B-FP8:/models:ro \
    -w /work/instances/vllm/artifacts-v3/ch09-engine-core-step-loop \
    vllm/vllm-openai:latest explainer/traces/run_m1b_27b.py

Same instrumentation philosophy as run_m1_real.py (pin-first meta_path finder,
InprocClient manual stepping, perf_counter stamps), with TWO fixes that matter
for measuring overlap honestly on a big model:

  1. NO synchronize at beat end. run_m1_real.py synchronized the forward CUDA
     event inside step_fn's finally block — harmless on the launch-bound tiny
     model, fatal here: it would force the async engine to drain the GPU at
     every beat boundary and erase the very overlap we are measuring.
  2. GPU-side timeline via CUDA events with enable_timing, read ONLY after the
     whole run: per beat we record ev_start (before worker launches forward),
     ev_fwd (right after the launch call returns) and ev_samp (after sample
     launch returns), all on the default stream, plus ev0 anchored at a fully
     synchronized instant. Afterwards ev0.elapsed_time(ev_x) gives each beat's
     GPU-clock timestamps in ms-since-anchor. On the default stream an event
     fires when all PRIOR stream work completed, so:
       - sync engine (stream idle at launch): ev_start fires ~at its enqueue
         -> ev_samp[i] .. ev_start[i+1] IS the GPU starved-for-CPU window;
       - async engine (stream still busy): ev_start[i+1] fires when beat i's
         kernels finished -> the same difference collapsing to ~0 is the
         overlap itself.
     CPU clock and GPU clock are anchored at the same instant (ev0 recorded
     immediately after a synchronize, perf_counter read next line); drift over
     a <1 s scenario is far below ms display precision.

Derived steady-state metrics (beat periods, GPU busy/idle split, throughput
gain, overlap evidence) are computed in-script and written into the trace so
every number the explainer table cites is literally in this file.
"""
import gc
import importlib.machinery
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback

# Compile caches must be redirected BEFORE vllm import (envs read at import).
# Both live inside the chapter dir (dot-prefixed project-internal temp dirs,
# deleted after the取证 lands). Sharing them between the two engine boots
# equalizes compiled-kernel warmth — the second boot reuses the first's
# inductor artifacts instead of compiling cold.
CHAPTER = "/work/instances/vllm/artifacts-v3/ch09-engine-core-step-loop"
PIN = "/work/instances/vllm/source"
CACHE = os.path.join(CHAPTER, ".cache-27b")
os.environ.setdefault("VLLM_CACHE_ROOT", CACHE)
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", os.path.join(CACHE, "inductor"))
OUT = os.path.join(CHAPTER, "explainer", "traces", "m1b_27b.json")
MODEL = "/models"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# --- pin-first meta_path finder (ch13 dual-path precedent) -------------------
class PinFirstFinder:
    """vllm.* pure-python -> PIN tree; compiled ext -> image site-packages."""

    def __init__(self):
        self.sp_vllm = None
        for p in sys.path:
            cand = os.path.join(p, "vllm")
            if os.path.isdir(cand) and not cand.startswith("/work"):
                self.sp_vllm = cand
                break

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] != "vllm":
            return None
        rel = fullname[len("vllm"):].lstrip(".")
        parts = rel.split(".") if rel else []
        base = os.path.join(PIN, "vllm", *parts)
        pin_cands = [os.path.join(base, "__init__.py")] if not rel else [
            base + ".py",
            os.path.join(base, "__init__.py"),
        ]
        for cand in pin_cands:
            if os.path.exists(cand):
                return importlib.util.spec_from_file_location(fullname, cand)
        if self.sp_vllm:
            base2 = os.path.join(self.sp_vllm, *parts)
            if os.path.isdir(base2):
                stem = parts[-1] if parts else "vllm"
                for fn in os.listdir(base2):
                    if fn.startswith(stem + ".") and fn.endswith(".so"):
                        loader = importlib.machinery.ExtensionFileLoader(
                            fullname, os.path.join(base2, fn)
                        )
                        return importlib.util.spec_from_loader(fullname, loader)
            else:
                d = os.path.dirname(base2)
                stem = os.path.basename(base2)
                if os.path.isdir(d):
                    for fn in sorted(os.listdir(d)):
                        if not fn.startswith(stem + "."):
                            continue
                        if fn.endswith(".so"):
                            loader = importlib.machinery.ExtensionFileLoader(
                                fullname, os.path.join(d, fn)
                            )
                            return importlib.util.spec_from_loader(fullname, loader)
                        if fn == stem + ".py":
                            return importlib.util.spec_from_file_location(
                                fullname, os.path.join(d, fn)
                            )
        return None


sys.meta_path.insert(0, PinFirstFinder())

import torch  # noqa: E402

import vllm  # noqa: E402

assert "/work/instances/vllm/source" in vllm.__file__, vllm.__file__

# Harness adaptation (see meta.registry_note): the pin registry builds
# _ModelInfo in a fresh `python -m vllm.model_executor.models.registry`
# subprocess, which resolves vllm from the IMAGE's site-packages (v0.26.0,
# an older _ModelInfo schema than pin v0.27.1). The pickled instance comes
# back missing pin's newer fields and ModelConfig crashes on first access
# (supports_replayssm — this model is hybrid). Inspect in-process instead:
# the meta_path finder above makes THIS process resolve the pin tree, so
# from_model_cls builds the pin's own _ModelInfo. The subprocess exists to
# "avoid initializing CUDA" before forking workers — InprocClient never
# forks, so the concern does not apply to this harness.
import vllm.model_executor.models.registry as _registry_mod  # noqa: E402

_registry_mod._run_in_subprocess = lambda fn: fn()

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.v1.engine import EngineCoreRequest  # noqa: E402

NOW = time.perf_counter


def r3(x):
    return round(x * 1000.0, 3)  # seconds -> ms, 3 decimals


# --- per-beat tracer ----------------------------------------------------------
class Tracer:
    def __init__(self):
        self.beats = []
        self.cur = None
        self.t0 = 0.0
        self.t0_abs = 0.0        # perf_counter seconds (absolute) at beat start
        self.phase = "warmup"
        self.finished_ids = set()
        # GPU events, read only after the run (zero distortion):
        self.ev_anchor = None    # reference event (enable_timing) at anchor
        self.t_anchor = 0.0      # perf_counter seconds at the same instant
        self.events = []         # [(beat_dict, ev_start, ev_fwd, ev_samp)]

    def anchor(self):
        torch.cuda.synchronize()
        self.ev_anchor = torch.cuda.Event(enable_timing=True)
        self.ev_anchor.record()
        self.t_anchor = NOW()

    def start_beat(self):
        self.cur = {
            "phase": self.phase,
            "events": [],
            "batch": None,
            "total_scheduled": None,
            "finished_riding_batch": None,
            "fwd_pending_after_bitmask": None,
            "fwd_pending_at_sample_entry": None,
            "fwd_pending_at_beat_end": None,
            "t0_abs_s": NOW(),
        }
        self.beats.append(self.cur)
        self.t0 = NOW()
        self.events.append((self.cur, None, None, None))

    def ev(self, tag, **kw):
        if self.cur is None:
            return
        rec = {"tag": tag, "t_ms": r3(NOW() - self.t0)}
        rec.update(kw)
        self.cur["events"].append(rec)

    def wall(self, tag, t_start):
        self.ev(tag + "_wall_ms", dur_ms=r3(NOW() - t_start))


TRACER = None  # current tracer (patches reference this)


class TimedFuture:
    """Transparent proxy that times result() on the future handed to step()."""

    def __init__(self, fut, tag):
        self._f = fut
        self._tag = tag

    def result(self, timeout=None):
        TRACER.ev(self._tag + ".result.start")
        t = NOW()
        try:
            r = self._f.result(timeout)
        except Exception:
            TRACER.ev(self._tag + ".result.exc")
            raise
        TRACER.ev(self._tag + ".result.end", wait_ms=r3(NOW() - t))
        return r

    def done(self):
        return self._f.done()


def instrument(llm):
    ec = llm.llm_engine.engine_core.engine_core  # EngineCore
    sched = ec.scheduler
    ex = ec.model_executor
    mr = ex.driver_worker.worker.model_runner

    orig_schedule = sched.schedule

    def timed_schedule(*a, **k):
        t = NOW()
        r = orig_schedule(*a, **k)
        try:
            nsd = dict(r.num_scheduled_tokens)
            TRACER.cur["batch"] = {str(rid): int(n) for rid, n in nsd.items()}
            TRACER.cur["total_scheduled"] = int(r.total_num_scheduled_tokens)
            fr = list(getattr(r, "finished_req_ids", None) or [])
            TRACER.cur["finished_riding_batch"] = [str(x) for x in fr]
        except Exception as e:
            TRACER.ev("①.schedule.shape_error", err=repr(e)[:120])
        TRACER.wall("①.schedule", t)
        return r

    sched.schedule = timed_schedule

    orig_exec = ex.execute_model

    def timed_exec(scheduler_output, non_block=False):
        t = NOW()
        r = orig_exec(scheduler_output, non_block=non_block)
        TRACER.ev("②.execute_model", non_block=non_block, wall_ms=r3(NOW() - t))
        return TimedFuture(r, "②future") if hasattr(r, "result") else r

    ex.execute_model = timed_exec

    orig_bm = sched.get_grammar_bitmask

    def timed_bm(*a, **k):
        t = NOW()
        r = orig_bm(*a, **k)
        TRACER.wall("③.bitmask", t)
        cur = TRACER.cur
        _, _, ev_fwd, _ = TRACER.events[-1]
        if ev_fwd is not None:
            cur["fwd_pending_after_bitmask"] = not ev_fwd.query()
        return r

    sched.get_grammar_bitmask = timed_bm

    orig_sample_ex = ex.sample_tokens

    def timed_sample_ex(grammar_output, non_block=False):
        t = NOW()
        r = orig_sample_ex(grammar_output, non_block=non_block)
        TRACER.ev("④.sample_tokens", non_block=non_block, wall_ms=r3(NOW() - t))
        return TimedFuture(r, "④future") if hasattr(r, "result") else r

    ex.sample_tokens = timed_sample_ex

    orig_ufo = sched.update_from_output

    def timed_ufo(*a, **k):
        t = NOW()
        r = orig_ufo(*a, **k)
        TRACER.wall("⑤.update_from_output", t)
        try:
            for eco in (r or {}).values():
                for o in eco.outputs:
                    if o.new_token_ids:
                        TRACER.cur.setdefault("outputs", {})[o.request_id] = list(
                            o.new_token_ids
                        )
                    if o.finished:
                        TRACER.cur.setdefault("finished_here", []).append(
                            o.request_id
                        )
                        TRACER.finished_ids.add(o.request_id)
                        fr = o.finish_reason
                        TRACER.cur.setdefault("finish_reasons", {})[o.request_id] = (
                            fr.name if hasattr(fr, "name") else str(fr)
                        )
        except Exception as e:
            TRACER.ev("⑤.shape_error", err=repr(e)[:120])
        return r

    sched.update_from_output = timed_ufo

    orig_mr_exec = mr.execute_model

    def timed_mr_exec(scheduler_output, *a, **k):
        # ev_start fires when all PRIOR default-stream work is done: stream
        # idle -> ~now (launch begin); stream busy (async engine) -> when the
        # previous beat's kernels land (this beat's kernels may begin).
        ev_start = torch.cuda.Event(enable_timing=True)
        ev_start.record()
        t = NOW()
        r = orig_mr_exec(scheduler_output, *a, **k)
        wall = r3(NOW() - t)
        ev_fwd = torch.cuda.Event(enable_timing=True)
        ev_fwd.record()  # fires when THIS beat's forward kernels complete
        TRACER.ev("②w.mr_execute_model", wall_ms=wall, ret=type(r).__name__)
        cur = TRACER.cur
        b = TRACER.events[-1]
        TRACER.events[-1] = (b[0], ev_start, ev_fwd, b[3])
        return r

    mr.execute_model = timed_mr_exec

    orig_mr_sample = mr.sample_tokens

    def timed_mr_sample(grammar_output, *a, **k):
        cur = TRACER.cur
        _, _, ev_fwd, _ = TRACER.events[-1]
        if ev_fwd is not None:
            cur["fwd_pending_at_sample_entry"] = not ev_fwd.query()
        t = NOW()
        r = orig_mr_sample(grammar_output, *a, **k)
        ev_samp = torch.cuda.Event(enable_timing=True)
        ev_samp.record()  # fires after mask+sample kernels complete
        TRACER.ev("④w.mr_sample_tokens", wall_ms=r3(NOW() - t), ret=type(r).__name__)
        b = TRACER.events[-1]
        TRACER.events[-1] = (b[0], b[1], b[2], ev_samp)
        return r

    mr.sample_tokens = timed_mr_sample

    orig_step_fn = ec.step_fn

    def timed_step(*a, **k):
        TRACER.start_beat()
        t = NOW()
        try:
            return orig_step_fn(*a, **k)
        finally:
            cur = TRACER.cur
            cur["step_wall_ms"] = r3(NOW() - t)
            _, _, ev_fwd, _ = TRACER.events[-1]
            if ev_fwd is not None:
                # Non-blocking probe only — the whole point (see module
                # docstring): never synchronize inside the loop.
                cur["fwd_pending_at_beat_end"] = not ev_fwd.query()
            TRACER.cur = None

    ec.step_fn = timed_step
    return ec, sched, ex, mr


# class-level patch: AsyncGPUModelRunnerOutput.get_output (the D2H wait)
from vllm.v1.worker.gpu_model_runner import AsyncGPUModelRunnerOutput as AMRO  # noqa: E402

_orig_get_output = AMRO.get_output


def _timed_get_output(self):
    TRACER.ev("D2H.get_output.start")
    t = NOW()
    r = _orig_get_output(self)
    TRACER.ev("D2H.get_output.end", wait_ms=r3(NOW() - t))
    return r


AMRO.get_output = _timed_get_output


# --- scenario / driver --------------------------------------------------------
def core_request(rid, token_ids, max_tokens):
    return EngineCoreRequest(
        request_id=rid,
        prompt_token_ids=list(token_ids),
        mm_features=None,
        sampling_params=SamplingParams(
            temperature=0.0, max_tokens=max_tokens, ignore_eos=True
        ),
        pooling_params=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )


def drive(llm, adds, max_beats=200):
    """Manual stepping = the busy loop's engine-step branch, minus input
    polling: LLMEngine.step() -> InprocClient.get_output() -> step_fn().
    Consecutive beats are driven back-to-back on one thread, so whatever
    overlap the engine can express, it expresses here too."""
    le = llm.llm_engine
    ic = le.engine_core
    n_fin0 = len(TRACER.finished_ids)
    want = {r.request_id for _, r in adds}
    beat = 0
    while len(TRACER.finished_ids) - n_fin0 < len(want) and beat < max_beats:
        for ab, req in adds:
            if ab == beat:
                ic.add_request(req)
        le.step()
        beat += 1
    for _ in range(3):  # flush beat + idle-guard beat(s)
        le.step()


def build_llm(async_scheduling, skip_tok, eager, limit_mm):
    kwargs = dict(
        model=MODEL,
        skip_tokenizer_init=skip_tok,
        max_model_len=256,
        max_num_seqs=4,
        max_num_batched_tokens=512,
        gpu_memory_utilization=0.60,
        enforce_eager=eager,
        async_scheduling=async_scheduling,
        enable_flashinfer_autotune=False,
        disable_log_stats=True,
    )
    if limit_mm:
        # text-only取证: skip the vision tower profiling/encoder cache
        kwargs["limit_mm_per_prompt"] = {"image": 0, "video": 0}
    return LLM(**kwargs)


def scenario():
    """Long steady decode so beat-rate differences are measured, not noise:
    req-A (prompt 8, max 56) starts now; req-B (prompt 16, max 40) after beat
    1 -> ~1 mixed prefill beat, ~38 double-decode beats, ~16 solo-decode
    beats, flush, guards."""
    return [
        (0, core_request("req-A", [11, 22, 33, 44, 55, 66, 77, 88], 56)),
        (1, core_request("req-B", [101, 102, 103, 104, 105, 106, 107, 108,
                                   109, 110, 111, 112, 113, 114, 115, 116], 40)),
    ]


def one_run(async_scheduling):
    global TRACER
    tr = Tracer()
    TRACER = tr
    llm = None
    boot_path = None
    err_last = None
    t = NOW()
    variants = (
        (True, False, True),    # skip_tok + compile + text-only
        (False, False, True),   # tokenizer + compile + text-only
        (True, False, False),   # skip_tok + compile + vision allowed
        (False, True, True),    # tokenizer + eager + text-only
    )
    for skip_tok, eager, limit_mm in variants:
        try:
            llm = build_llm(async_scheduling, skip_tok, eager, limit_mm)
            boot_path = (
                f"skip_tokenizer_init={skip_tok}+"
                f"{'eager' if eager else 'compile'}+"
                f"limit_mm={'0' if limit_mm else 'default'}"
            )
            break
        except Exception as e:  # try the next, gentler configuration
            print(f"[run] boot variant (skip_tok={skip_tok}, eager={eager}, "
                  f"limit_mm={limit_mm}) FAILED:", flush=True)
            traceback.print_exc()
            err_last = repr(e)[:300]
            llm = None
            # release the half-built engine's GPU memory before next attempt
            gc.collect()
            torch.cuda.empty_cache()
    if llm is None:
        return {"error": f"all boot variants failed; last: {err_last}"}
    boot_s = NOW() - t

    ec, sched, ex, mr = instrument(llm)
    attn = None
    try:
        names = set()
        for groups in mr.attn_groups:
            for g in groups if isinstance(groups, (list, tuple)) else [groups]:
                b = getattr(g, "backend", None)
                if b is not None:
                    names.add(type(b).__name__)
        attn = sorted(names)
    except Exception as e:
        attn = "n/a:" + repr(e)[:80]

    # warmup digests compile + cudagraph capture + Triton JIT for sm_120
    tr.phase = "warmup"
    drive(llm, [(0, core_request("warm", [9, 8, 7, 6, 5, 4], 4))])

    # anchor: one fully synchronized instant shared by CPU and GPU clocks
    tr.anchor()
    tr.phase = "main"
    drive(llm, scenario())

    final = {}
    for b in tr.beats:
        if b["phase"] != "main":
            continue
        for rid, toks in (b.get("outputs") or {}).items():
            final.setdefault(rid, []).extend(toks)

    # GPU timeline read-out: only now do we synchronize (run is over).
    torch.cuda.synchronize()
    ev0 = tr.ev_anchor
    for b, ev_start, ev_fwd, ev_samp in tr.events:
        # main-phase beats only: warmup events predate the anchor event, and
        # elapsed_time(end_before_start) is undefined
        if b is None or b.get("phase") != "main":
            continue
        if ev_start is not None:
            b["gpu_launch_start_ms"] = round(ev0.elapsed_time(ev_start), 3)
        if ev_fwd is not None:
            b["gpu_fwd_done_ms"] = round(ev0.elapsed_time(ev_fwd), 3)
        if ev_samp is not None:
            b["gpu_samp_done_ms"] = round(ev0.elapsed_time(ev_samp), 3)

    result = {
        "config": {
            "async_scheduling": async_scheduling,
            "step_fn": "step()" if not async_scheduling else "step_with_batch_queue()",
            "boot_path": boot_path,
            "boot_s": round(boot_s, 1),
            "enforce_eager": "eager" in boot_path,
            "attention_backend": attn,
            "use_async_scheduling_runner": bool(mr.use_async_scheduling),
            "batch_queue_size": getattr(ec, "batch_queue_size", None),
            "model": "Qwen3.8-27B-FP8 (64 layers, hidden 5120, vocab 248320, fp8)",
        },
        "t_anchor_s": tr.t_anchor,
        "final_tokens": final,
        "beats": tr.beats,
    }

    del ec, sched, ex, mr
    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return result


# --- derived steady-state metrics ---------------------------------------------
def seg_wall(b, prefix):
    for e in b["events"]:
        if e["tag"].startswith(prefix):
            return e.get("wall_ms", e.get("dur_ms", 0.0))
    return None


def ev_wait(b, tag):
    for e in b["events"]:
        if e["tag"] == tag:
            return e.get("wait_ms")
    return None


def derived(sync_run, async_run):
    """Steady window = main-phase double-decode beats ({A:1,B:1}), interior
    (drop 2 at each end) so transitions don't pollute the beat-rate numbers."""
    out = {}

    def steady(run):
        idx = [
            i for i, b in enumerate(run["beats"])
            if b["phase"] == "main" and b.get("batch") == {"req-A": 1, "req-B": 1}
        ]
        return idx[2:-2] if len(idx) > 8 else idx

    def beat_rows(run, idx):
        rows = []
        for i in idx:
            b = run["beats"][i]
            t0 = run["beats"][i]["t0_abs_s"]
            cpu = lambda b_: (b_["t0_abs_s"] - run["t_anchor_s"]) * 1000.0
            nxt = run["beats"][i + 1] if i + 1 < len(run["beats"]) else None
            row = {
                "i": i,
                "step_wall_ms": b["step_wall_ms"],
                "cpu_beat_start_ms": round(cpu(b), 3),
                "seg1_schedule_ms": seg_wall(b, "①.schedule"),
                "seg2_launch_ms": seg_wall(b, "②.execute_model"),
                "seg3_bitmask_ms": seg_wall(b, "③.bitmask"),
                "seg4_sample_ms": seg_wall(b, "④.sample_tokens"),
                "seg5_update_ms": seg_wall(b, "⑤.update_from_output"),
                "gpu_launch_start_ms": b.get("gpu_launch_start_ms"),
                "gpu_fwd_done_ms": b.get("gpu_fwd_done_ms"),
                "gpu_samp_done_ms": b.get("gpu_samp_done_ms"),
                "fwd_pending_after_bitmask": b.get("fwd_pending_after_bitmask"),
                "fwd_pending_at_sample_entry": b.get("fwd_pending_at_sample_entry"),
                "fwd_pending_at_beat_end": b.get("fwd_pending_at_beat_end"),
            }
            if nxt is not None and b.get("gpu_samp_done_ms") is not None \
                    and nxt.get("gpu_launch_start_ms") is not None:
                row["gpu_idle_gap_ms"] = round(
                    nxt["gpu_launch_start_ms"] - b["gpu_samp_done_ms"], 3)
                row["beat_period_gpu_ms"] = round(
                    nxt["gpu_launch_start_ms"] - b["gpu_launch_start_ms"], 3)
            # sync angle: how much of ④b was waiting for the forward tail
            # (lower bound: forward still had this long left at ④b entry)
            if row["seg4_sample_ms"] is not None and None not in (
                    row["cpu_beat_start_ms"], row["gpu_fwd_done_ms"]):
                entry = None
                for e in b["events"]:
                    if e["tag"] == "④.sample_tokens":
                        entry = row["cpu_beat_start_ms"] + e["t_ms"]
                if entry is not None:
                    row["fwd_tail_wait_lb_ms"] = round(
                        max(0.0, row["gpu_fwd_done_ms"] - entry), 3)
            # async angle: this beat's CPU launch window ran under the
            # PREVIOUS beat's still-executing GPU work?
            prev = run["beats"][i - 1] if i >= 1 else None
            if prev is not None and prev.get("gpu_samp_done_ms") is not None \
                    and row["cpu_beat_start_ms"] is not None:
                launch_end = row["cpu_beat_start_ms"] + (row["seg1_schedule_ms"]
                                                         or 0) + (row["seg2_launch_ms"] or 0)
                row["cpu_launch_under_prev_gpu_ms"] = round(
                    max(0.0, min(prev["gpu_samp_done_ms"], launch_end)
                        - row["cpu_beat_start_ms"]), 3)
            # async wait-chain numbers
            w = ev_wait(b, "④future.result.end")
            if w is not None:
                row["④future_wait_ms"] = w
            d = ev_wait(b, "D2H.get_output.end")
            if d is not None:
                row["D2H_wait_ms"] = d
            rows.append(row)
        return rows

    def summarize(rows):
        keys = ["step_wall_ms", "gpu_idle_gap_ms", "beat_period_gpu_ms",
                "fwd_tail_wait_lb_ms", "seg4_sample_ms", "seg1_schedule_ms",
                "seg2_launch_ms", "seg5_update_ms", "④future_wait_ms",
                "D2H_wait_ms", "cpu_launch_under_prev_gpu_ms"]
        s = {"n": len(rows)}
        for k in keys:
            vals = [r[k] for r in rows if r.get(k) is not None]
            if vals:
                s[k] = {
                    "mean": round(statistics.mean(vals), 3),
                    "median": round(statistics.median(vals), 3),
                    "min": round(min(vals), 3),
                    "max": round(max(vals), 3),
                }
        pend = [r for r in rows if r.get("fwd_pending_at_beat_end") is True]
        s["fwd_pending_at_beat_end_count"] = len(pend)
        entry_pend = [r for r in rows if r.get("fwd_pending_at_sample_entry") is True]
        s["fwd_pending_at_sample_entry_count"] = len(entry_pend)
        return s

    for name, run in (("sync", sync_run), ("async", async_run)):
        if "error" in run:
            out[name] = {"error": run["error"]}
            continue
        idx = steady(run)
        rows = beat_rows(run, idx)
        out[name] = {
            "steady_window_beat_idx": idx,
            "summary": summarize(rows),
            "rows": rows,
        }

    if "summary" in out.get("sync", {}) and "summary" in out.get("async", {}):
        ps = out["sync"]["summary"]["step_wall_ms"]["median"]
        pa = out["async"]["summary"]["step_wall_ms"]["median"]
        pg_s = out["sync"]["summary"].get("beat_period_gpu_ms", {}).get("median")
        pg_a = out["async"]["summary"].get("beat_period_gpu_ms", {}).get("median")
        cmp_ = {
            "sync_steady_step_wall_ms": ps,
            "async_steady_step_wall_ms": pa,
            "step_wall_speedup_pct": round((ps / pa - 1.0) * 100.0, 1),
            "sync_beat_period_gpu_ms": pg_s,
            "async_beat_period_gpu_ms": pg_a,
            "beat_period_speedup_pct": (
                round((pg_s / pg_a - 1.0) * 100.0, 1)
                if pg_s and pg_a else None),
            "sync_tokens_per_s_per_seq": round(1000.0 / ps * 1.0, 1),
            "async_tokens_per_s_per_seq": round(1000.0 / pa * 1.0, 1),
            "batch_size_in_window": 2,
            "final_tokens_identical": (
                sync_run.get("final_tokens") == async_run.get("final_tokens")),
        }
        out["comparison"] = cmp_
    return out


def main():
    import subprocess

    try:
        head = subprocess.run(
            ["git", "-C", PIN, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        head = "unavailable"
    meta = {
        "kind": "real_engine_trace_27b",
        "pin": "vLLM v0.27.1",
        "git_head": head,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": "%d,%d" % torch.cuda.get_device_capability(0),
        "docker_image": (
            "vllm/vllm-openai:latest (deps only, v0.26.0-era; code-under-test "
            "= pin source tree; the vllm-omni image was rejected: its "
            "vllm_omni patch layer hooks Request.from_engine_core_request and "
            "reads EngineCoreRequest.additional_information, a field pin "
            "v0.27.1 does not define)"
        ),
        "model_path": "/models (host E:/Laboratory/models/Qwen3.8-27B-FP8, mounted ro)",
        "timing_note": (
            "CPU stamps time.perf_counter; GPU stamps = CUDA events "
            "(enable_timing) on the default stream, read via "
            "ev_anchor.elapsed_time(ev_x) AFTER the run — no synchronize ever "
            "happens inside the stepping loop (the m1_real tiny harness did "
            "synchronize at beat end, which would erase async overlap). "
            "gpu_launch_start_ms: event recorded BEFORE the worker launches "
            "the forward — fires when all prior default-stream work is done "
            "(stream idle: ~its enqueue; stream busy: when the previous beat's "
            "kernels land). gpu_fwd_done_ms / gpu_samp_done_ms: events after "
            "the launch / sample calls return. cpu_*_ms are ms since t_anchor "
            "(perf_counter), GPU ms are ms since ev_anchor fired; both anchors "
            "were taken at the same fully-synchronized instant."
        ),
        "harness_note": (
            "InprocClient (VLLM_ENABLE_V1_MULTIPROCESSING=0): LLMEngine.step() "
            "drives engine_core.step_fn() back-to-back on one thread = the "
            "busy loop's engine-step branch minus input polling; V1 "
            "GPUModelRunner forced (VLLM_USE_V2_MODEL_RUNNER=0) = the file ch9 "
            "anchors. Order: sync engine first, async second (same process; "
            "compile caches shared via .cache-27b)."
        ),
        "registry_note": (
            "harness adaptation: vllm.model_executor.models.registry."
            "_run_in_subprocess monkeypatched to run in-process — the stock "
            "subprocess (`python -m vllm...`) resolves vllm from the image "
            "site-packages (v0.26.0, older _ModelInfo schema than pin "
            "v0.27.1), and the pickled-back instance lacks pin-only fields "
            "(supports_replayssm — hit by this hybrid model; the tiny-Llama "
            "m1_real run never accessed them). In-process inspection builds "
            "the pin's own _ModelInfo; step-loop code under test is "
            "untouched."
        ),
    }
    print("[run] === sync run (async_scheduling=False, EngineCore.step) ===",
          flush=True)
    sync_run = one_run(async_scheduling=False)
    if "error" in sync_run:
        print("[run] sync boot FAILED:", sync_run["error"], flush=True)
    print("[run] === async run (async_scheduling=True, "
          "step_with_batch_queue) ===", flush=True)
    async_run = one_run(async_scheduling=True)
    if "error" in async_run:
        print("[run] async boot FAILED:", async_run["error"], flush=True)

    doc = {
        "meta": meta,
        "sync_run": sync_run,
        "async_run": async_run,
        "derived": derived(sync_run, async_run),
    }
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print("[run] wrote", OUT, flush=True)

    def nb(r):
        return sum(1 for b in r.get("beats", []) if b.get("phase") == "main")

    print("[run] beats: sync main =", nb(sync_run),
          ", async main =", nb(async_run), flush=True)
    print("[run] sync final:", sync_run.get("final_tokens"), flush=True)
    print("[run] async final:", async_run.get("final_tokens"), flush=True)
    if "comparison" in doc["derived"]:
        print("[run] comparison:", json.dumps(doc["derived"]["comparison"],
                                               ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
