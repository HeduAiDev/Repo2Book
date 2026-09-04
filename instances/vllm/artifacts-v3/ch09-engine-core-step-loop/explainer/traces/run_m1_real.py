"""Real-engine trace for ch9 m1/m3 — five beats of EngineCore.step() on a REAL
pin-v0.27.1 vLLM + real GPU, replacing the host-modeled timing numbers.

Container run (host shell, from repo root):

  MSYS_NO_PATHCONV=1 VLLM_USAGE_STATS=0 docker run --rm --gpus all \
    -e VLLM_ENABLE_V1_MULTIPROCESSING=0 -e VLLM_USE_V2_MODEL_RUNNER=0 \
    --entrypoint /usr/bin/python3 -v E:/Laboratory/Repo2Book:/work \
    -w /work/instances/vllm/artifacts-v3/ch09-engine-core-step-loop \
    vllm/vllm-openai:latest explainer/traces/run_m1_real.py

What it records (two runs, one process):

  run A "sync"  : async_scheduling=False -> EngineCore.step() — the five
                  segments ①schedule ②execute_model(non_block) ③bitmask
                  ④a future.result ④b sample_tokens ⑤update_from_output,
                  wall-clock stamped with time.perf_counter, plus a CUDA
                  event recorded right after the worker launches the forward
                  (proof bitmask ran WHILE the GPU was still executing).
  run B "async" : async_scheduling=True (v0.27.1 default) ->
                  step_with_batch_queue — ② launch wall vs full forward
                  window (launch-to-D2H-done), and the AsyncOutputFuture /
                  AsyncGPUModelRunnerOutput.get_output wait chain.

Model: tiny random-weight Llama (generated once into .tiny-model/, no
network, skip_tokenizer_init + msgspec EngineCoreRequest direct injection).
InprocClient keeps EngineCore in-process (LLMEngine.step() drives the real
step_fn each beat); V1 GPUModelRunner forced (VLLM_USE_V2_MODEL_RUNNER=0)
which is exactly the file ch9 anchors (vllm/v1/worker/gpu_model_runner.py).
"""
import concurrent.futures as cf
import gc
import importlib.machinery
import importlib.util
import json
import os
import sys
import time

import torch

CHAPTER = "/work/instances/vllm/artifacts-v3/ch09-engine-core-step-loop"
PIN = "/work/instances/vllm/source"
MODEL_DIR = os.path.join(CHAPTER, ".tiny-model")
OUT = os.path.join(CHAPTER, "explainer", "traces", "m1_real.json")

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
        rel = fullname[len("vllm") :].lstrip(".")
        parts = rel.split(".") if rel else []
        base = os.path.join(PIN, "vllm", *parts)
        pin_cands = [os.path.join(base, "__init__.py")] if not rel else [
            base + ".py",
            os.path.join(base, "__init__.py"),
        ]
        for cand in pin_cands:
            if os.path.exists(cand):
                return importlib.util.spec_from_file_location(fullname, cand)
        # fall back to the image's built vllm (compiled .so / missing module)
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

import vllm  # noqa: E402

assert "/work/instances/vllm/source" in vllm.__file__, vllm.__file__

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
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
        self.phase = "warmup"
        self.gpu_ev = None  # CUDA event recorded after forward launch
        self.finished_ids = set()  # engine-side finish events (from ⑤)

    def start_beat(self):
        self.cur = {
            "phase": self.phase,
            "events": [],
            "batch": None,
            "total_scheduled": None,
            "finished_riding_batch": None,
            "fwd_pending_after_bitmask": None,
            "fwd_pending_at_sample_entry": None,
            "fwd_done_rel_ms": None,
        }
        self.beats.append(self.cur)
        self.t0 = NOW()
        self.gpu_ev = None

    def ev(self, tag, **kw):
        if self.cur is None:
            return
        rec = {"tag": tag, "t_ms": r3(NOW() - self.t0)}
        rec.update(kw)
        self.cur["events"].append(rec)

    def wall(self, tag, t_start):
        self.ev(tag + "_wall_ms", dur_ms=r3(NOW() - t_start))


TRACER = None  # current tracer (class-level patches reference this)


class TimedFuture(cf.Future):
    """Transparent proxy that times result() on the future handed to step()."""

    def __init__(self, fut, tag):
        super().__init__()
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


def instrument(llm, eager_fallback=None):
    """Patch the real objects (bound methods on instances) with timers."""
    ec = llm.llm_engine.engine_core.engine_core  # EngineCore
    sched = ec.scheduler
    ex = ec.model_executor
    mr = ex.driver_worker.worker.model_runner

    # ① scheduler.schedule — also captures the real batch shape per beat
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
        except Exception as e:  # keep tracing alive on shape surprises
            TRACER.ev("①.schedule.shape_error", err=repr(e)[:120])
        TRACER.wall("①.schedule", t)
        return r

    sched.schedule = timed_schedule

    # ② executor.execute_model — launch wall + non_block spy
    orig_exec = ex.execute_model

    def timed_exec(scheduler_output, non_block=False):
        t = NOW()
        r = orig_exec(scheduler_output, non_block=non_block)
        TRACER.ev("②.execute_model", non_block=non_block, wall_ms=r3(NOW() - t))
        return TimedFuture(r, "②future") if hasattr(r, "result") else r

    ex.execute_model = timed_exec

    # ③ scheduler.get_grammar_bitmask + CUDA-pending proof
    orig_bm = sched.get_grammar_bitmask

    def timed_bm(*a, **k):
        t = NOW()
        r = orig_bm(*a, **k)
        TRACER.wall("③.bitmask", t)
        if TRACER.gpu_ev is not None:
            TRACER.cur["fwd_pending_after_bitmask"] = not TRACER.gpu_ev.query()
        return r

    sched.get_grammar_bitmask = timed_bm

    # ④b executor.sample_tokens — wall + non_block spy (returns proxy future)
    orig_sample_ex = ex.sample_tokens

    def timed_sample_ex(grammar_output, non_block=False):
        t = NOW()
        r = orig_sample_ex(grammar_output, non_block=non_block)
        TRACER.ev("④.sample_tokens", non_block=non_block, wall_ms=r3(NOW() - t))
        return TimedFuture(r, "④future") if hasattr(r, "result") else r

    ex.sample_tokens = timed_sample_ex

    # ⑤ scheduler.update_from_output — wall + engine-side outputs (this is the
    # authoritative per-request output stream; no LLMEngine collector needed)
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

    # worker-side ②: model_runner.execute_model — the real launch segment;
    # records a CUDA event so "③ ran while GPU busy" is a query, not a claim.
    orig_mr_exec = mr.execute_model

    def timed_mr_exec(scheduler_output, *a, **k):
        t = NOW()
        r = orig_mr_exec(scheduler_output, *a, **k)
        wall = r3(NOW() - t)
        TRACER.ev("②w.mr_execute_model", wall_ms=wall, ret=type(r).__name__)
        ev = torch.cuda.Event()
        ev.record()  # default stream: fires after this beat's forward kernels
        TRACER.gpu_ev = ev
        TRACER.t_launch_done = NOW()
        return r

    mr.execute_model = timed_mr_exec

    # worker-side ④: model_runner.sample_tokens — mask+sample+(sync D2H | async
    # D2H launch). Pending check at entry proves it waited for the forward.
    orig_mr_sample = mr.sample_tokens

    def timed_mr_sample(grammar_output, *a, **k):
        if TRACER.gpu_ev is not None:
            TRACER.cur["fwd_pending_at_sample_entry"] = not TRACER.gpu_ev.query()
        t = NOW()
        r = orig_mr_sample(grammar_output, *a, **k)
        TRACER.ev("④w.mr_sample_tokens", wall_ms=r3(NOW() - t), ret=type(r).__name__)
        return r

    mr.sample_tokens = timed_mr_sample

    # step_fn boundary (what InprocClient.get_output() actually calls)
    orig_step_fn = ec.step_fn

    def timed_step(*a, **k):
        TRACER.start_beat()
        t = NOW()
        try:
            return orig_step_fn(*a, **k)
        finally:
            wall = NOW() - t
            cur = TRACER.cur
            cur["step_wall_ms"] = r3(wall)
            ev = TRACER.gpu_ev
            if ev is not None:
                t_s = NOW()
                ev.synchronize()  # GPU forward kernels all landed by now
                cur["fwd_done_rel_ms"] = r3(NOW() - TRACER.t0)
                cur["fwd_sync_extra_ms"] = r3(NOW() - t_s)
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


def drive(llm, adds, max_beats=40):
    """adds: list of (inject_after_beat, EngineCoreRequest). Manually stepping
    the REAL loop: LLMEngine.step() -> InprocClient.get_output() -> step_fn().
    Completion is detected engine-side (⑤'s finish markers); after that we
    keep stepping a few more beats so the flush beat and the idle-guard beat
    (empty scheduler, executor untouched) are recorded for real too."""
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


def build_llm(async_scheduling, eager=False):
    return LLM(
        model=MODEL_DIR,
        skip_tokenizer_init=True,
        dtype="float16",
        gpu_memory_utilization=0.10,
        max_model_len=64,
        max_num_seqs=4,
        enforce_eager=eager,
        async_scheduling=async_scheduling,
        enable_flashinfer_autotune=False,  # image flashinfer lacks the API
    )


def ensure_tiny_model():
    if os.path.exists(os.path.join(MODEL_DIR, "config.json")):
        return
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=8192,
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=64,
        rms_norm_eps=1e-5,
        attention_bias=False,
        mlp_bias=False,
        tie_word_embeddings=False,
    )
    torch.manual_seed(20260905)
    LlamaForCausalLM(cfg).to(torch.float16).save_pretrained(MODEL_DIR)


def scenario():
    """One request's life + a late arrival — same shape as the host m1 trace:
    req-A (prompt 3, max 3) starts now; req-B (prompt 4, max 2) injected after
    beat 1."""
    return [
        (0, core_request("req-A", [1, 2, 3], 3)),
        (1, core_request("req-B", [4, 5, 6, 7], 2)),
    ]


def one_run(meta, async_scheduling):
    global TRACER
    tr = Tracer()
    TRACER = tr
    llm = None
    eager_used = False
    eager_reason = None
    t = NOW()
    try:
        llm = build_llm(async_scheduling, eager=False)
    except Exception as e:  # compile/capture incompat -> fall back to eager
        eager_reason = repr(e)[:400]
        print("[run] default (compile) failed, falling back to eager:", eager_reason)
        llm = build_llm(async_scheduling, eager=True)
        eager_used = True
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

    # warmup request digests kernel JIT (first forward compiles Triton kernels
    # for sm_120); recorded under phase="warmup" and excluded from main beats.
    tr.phase = "warmup"
    drive(llm, [(0, core_request("warm", [9, 8, 7], 2))])

    tr.phase = "main"
    drive(llm, scenario())

    final = {}
    for b in tr.beats:
        if b["phase"] != "main":
            continue
        for rid, toks in (b.get("outputs") or {}).items():
            final.setdefault(rid, []).extend(toks)

    result = {
        "config": {
            "async_scheduling": async_scheduling,
            "step_fn": "step()" if not async_scheduling else "step_with_batch_queue()",
            "enforce_eager": eager_used,
            "eager_fallback_reason": eager_reason,
            "boot_s": round(boot_s, 1),
            "attention_backend": attn,
            "use_async_scheduling_runner": bool(mr.use_async_scheduling),
            "batch_queue_size": getattr(
                llm.llm_engine.engine_core.engine_core, "batch_queue_size", None
            ),
        },
        "final_tokens": final,
        "beats": tr.beats,
    }
    # release GPU before the second engine
    del ec, sched, ex, mr
    llm.llm_engine.engine_core.shutdown()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    ensure_tiny_model()
    import subprocess

    try:
        head = subprocess.run(
            ["git", "-C", PIN, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except Exception:
        head = "unavailable"
    meta = {
        "kind": "real_engine_trace",
        "pin": "vLLM v0.27.1",
        "git_head": head,
        "vllm_file": vllm.__file__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_capability": "%d,%d" % torch.cuda.get_device_capability(0),
        "docker_image": "vllm/vllm-openai:latest (deps only; code-under-test = pin source tree)",
        "container_python": sys.version.split()[0],
        "inproc_note": (
            "VLLM_ENABLE_V1_MULTIPROCESSING=0 -> InprocClient: EngineCore in this "
            "process, LLMEngine.step() drives engine_core.step_fn() directly "
            "(no busy loop / no ZMQ). Manual stepping = the real per-beat calls."
        ),
        "runner_note": (
            "VLLM_USE_V2_MODEL_RUNNER=0 -> V1 GPUModelRunner "
            "(vllm/v1/worker/gpu_model_runner.py, the file ch9 anchors). "
            "V2 runner (new default for LlamaForCausalLM in v0.27.1) needs CUDA "
            "UVA which this container runtime does not expose."
        ),
        "model": (
            "tiny random-weight LlamaForCausalLM (16 layers, hidden 1024, 16 q / 4 kv "
            "heads, vocab 8192, fp16, seed 20260905), skip_tokenizer_init=True, "
            "greedy sampling, tokens injected via EngineCoreRequest"
        ),
        "timing_note": (
            "all wall stamps time.perf_counter, ms relative to beat start; "
            "fwd_done_rel_ms = CUDA event (recorded on default stream right after "
            "worker.execute_model returns) synchronized at beat end — i.e. the "
            "forward kernels' GPU-completion instant; fwd_pending_after_bitmask / "
            "fwd_pending_at_sample_entry = non-blocking event.query() at those points."
        ),
        "flashinfer_autotune_off": (
            "enable_flashinfer_autotune=False: image flashinfer lacks "
            "set_autotune_process_group (pin kernel_warmup.py imports it); "
            "autotune only picks kernel variants, does not affect this chapter's "
            "observations"
        ),
        "sm_note": (
            "host GPU is Blackwell sm_120; pin capability-dependent branches were "
            "not modified — differences, if any, are noted per-number in explainer"
        ),
    }
    print("[run] === sync run (async_scheduling=False, EngineCore.step) ===")
    sync_run = one_run(meta, async_scheduling=False)
    print("[run] === async run (async_scheduling=True, step_with_batch_queue) ===")
    async_run = one_run(meta, async_scheduling=True)

    doc = {"meta": meta, "sync_run": sync_run, "async_run": async_run}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    print("[run] wrote", OUT)
    nb = lambda r: sum(1 for b in r["beats"] if b["phase"] == "main")
    print("[run] beats: sync main =", nb(sync_run), ", async main =", nb(async_run))
    print("[run] sync final tokens:", sync_run["final_tokens"])
    print("[run] async final tokens:", async_run["final_tokens"])


if __name__ == "__main__":
    main()
