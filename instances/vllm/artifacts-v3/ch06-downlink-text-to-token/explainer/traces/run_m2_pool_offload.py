"""Driver for m2 (thread-pool offload + add_request dispatch) — host run
against the ch06 subtract-only companion (pin vLLM v0.27.1).

Records the observable facts the m2 figure needs:
- raw prompt: tokenize (0.25s blocking seam tokenizer) runs on the renderer's
  ThreadPoolExecutor thread, NOT the event-loop thread — a heartbeat coroutine
  keeps ticking the whole time (the loop is never blocked);
- rendered EngineInput: the sync fast path runs process_inputs directly on the
  event-loop thread and makes ZERO tokenizer calls
  ("Rendered EngineInput; no blocking preprocessing needed");
- dual pools: renderer_num_workers=4 -> tokenize pool has 4 workers; the mm
  pool always has exactly 1 ("must stay single-worker per #38418"); default
  renderer_num_workers is 1;
- mm preprocessing runs on the single-worker mm pool thread;
- the sync offline face runs the same process_inputs on the caller's thread
  (no pool involvement).

Test-side seams from tests/test_downlink.py (documented HOST SEAMs): the
delayed SeamWordTokenizer stands in for the ~100ms-scale HF tokenizer.
"""
import asyncio
import importlib
import json
import sys
import threading
import time
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH / "tests"))
td = importlib.import_module("test_downlink")
downlink = td.downlink


async def heartbeat(seconds: float, interval: float = 0.01) -> int:
    ticks = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        await asyncio.sleep(interval)
        ticks += 1
    return ticks


class MMThreadTracer(td.ChatRendererSeam):
    """Records which thread each mm preprocessing job ran on."""

    def __init__(self, config, tokenizer):
        super().__init__(config, tokenizer)
        self.mm_threads: list[int] = []

    def _process_multimodal(self, *args, **kwargs):
        self.mm_threads.append(threading.get_ident())
        return super()._process_multimodal(*args, **kwargs)


MM_MESSAGES = [{"role": "user", "content": [
    {"type": "text", "text": "see"},
    {"type": "image_url", "image_url": "IMG-1"},
    {"type": "text", "text": "here"},
]}]


async def main():
    out = {
        "driver": "run_m2_pool_offload.py",
        "mechanism": "m2 线程池卸载 + add_request 按输入形态分流",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "tokenizer_block_seconds": 0.25,
        "heartbeat_interval_seconds": 0.01,
    }

    loop = asyncio.get_running_loop()
    loop_thread = threading.get_ident()

    # -- A. raw prompt: blocking tokenize goes to the renderer pool ----------
    slow_tok = td.SeamWordTokenizer(delay=0.25)
    llm, renderer, _ = td.mk_engine(slow_tok)
    pool_probe = await loop.run_in_executor(
        renderer._executor, lambda: threading.get_ident()
    )
    t0 = time.perf_counter()
    queue, ticks = await asyncio.gather(
        llm.add_request(
            "raw-1", "alpha beta gamma delta", downlink.SamplingParams(max_tokens=4)
        ),
        heartbeat(0.4),
    )
    elapsed = time.perf_counter() - t0
    tok_thread, tok_text = slow_tok.calls[0]
    frames = [e for e in llm.events if e[0] == "send_input"]
    (frame_req,) = [e[2] for e in frames]
    out["A_raw_prompt_offloaded"] = {
        "loop_thread_id": loop_thread,
        "renderer_pool_probe_thread_id": pool_probe,
        "tokenizer_thread_id": tok_thread,
        "tokenizer_thread_is_renderer_pool_thread": tok_thread == pool_probe,
        "tokenizer_thread_is_loop_thread": tok_thread == loop_thread,
        "tokenizer_saw_text": tok_text,
        "heartbeat_ticks_during_0p4s": ticks,
        "heartbeat_minimum_expected_if_loop_never_blocked": 20,
        "loop_never_blocked": ticks >= 20,
        "add_request_wall_seconds": round(elapsed, 3),
        "crossed_request_token_ids": frame_req.prompt_token_ids,
        "internal_request_id": frame_req.request_id,
    }

    # -- B. rendered EngineInput: sync fast path, zero tokenizer calls -------
    fast_tok = td.SeamWordTokenizer()
    llm2, renderer2, _ = td.mk_engine(fast_tok)
    calls_before = len(fast_tok.calls)
    proc_threads: list[int] = []
    orig_proc = llm2.input_processor.process_inputs

    def recording_proc(*args, **kwargs):
        proc_threads.append(threading.get_ident())
        return orig_proc(*args, **kwargs)

    llm2.input_processor.process_inputs = recording_proc
    engine_input = downlink.tokens_input([3, 4, 5, 6])
    await llm2.add_request(
        "fast-1", engine_input, downlink.SamplingParams(max_tokens=4)
    )
    out["B_rendered_fast_path"] = {
        "tokenizer_calls_before": calls_before,
        "tokenizer_calls_after": len(fast_tok.calls),
        "tokenizer_calls_made": len(fast_tok.calls) - calls_before,
        "process_inputs_thread_id": proc_threads[0],
        "process_inputs_ran_on_loop_thread": proc_threads[0] == loop_thread,
        "took_sync_fast_path": proc_threads[0] == loop_thread
        and len(fast_tok.calls) == calls_before,
    }

    # -- C. dual pools: 4-worker tokenize pool vs 1-worker mm pool ----------
    cfg4 = td.mk_config(model_config={"renderer_num_workers": 4})
    ex4 = downlink.BaseRenderer(cfg4, td.SeamWordTokenizer())
    mm_tracer = MMThreadTracer(td.mk_config(multimodal_config={}), td.SeamWordTokenizer())
    mm_pool_probe = await loop.run_in_executor(
        mm_tracer._mm_executor, lambda: threading.get_ident()
    )
    out["C_dual_pools"] = {
        "default_renderer_num_workers": td.mk_config().model_config.renderer_num_workers,
        "configured_renderer_num_workers": 4,
        "tokenize_pool_actual_max_workers": ex4._executor._max_workers,
        "mm_pool_actual_max_workers_with_4_workers": ex4._mm_executor._max_workers,
        "mm_pool_actual_max_workers_with_1_worker": (
            mm_tracer._mm_executor._max_workers
        ),
        "note": "ThreadPoolExecutor._max_workers = 装配工数；mm 池恒 1（#38418 P0/P1 顺序）",
    }
    ex4.shutdown()

    # -- D. mm preprocessing runs on the single-worker mm pool thread -------
    mm_tok = td.SeamWordTokenizer()
    llm3, renderer3, _ = td.mk_engine(
        mm_tok, config=td.mk_config(multimodal_config={}), renderer=mm_tracer
    )
    online3 = downlink.OnlineRenderer(
        llm3.model_config, renderer3,
        request_logger=None, chat_template=None, chat_template_content_format="string",
    )
    req3 = td.FakeChatRequest(messages=MM_MESSAGES)
    _, engine_inputs = await online3.render_chat(req3)
    await llm3.add_request(
        "mm-1", engine_inputs[0], downlink.SamplingParams(max_tokens=4)
    )
    out["D_mm_on_single_pool"] = {
        "mm_thread_id": mm_tracer.mm_threads[0],
        "mm_pool_probe_thread_id": mm_pool_probe,
        "mm_ran_on_mm_pool_thread": mm_tracer.mm_threads[0] == mm_pool_probe,
        "mm_thread_is_loop_thread": mm_tracer.mm_threads[0] == loop_thread,
        "mm_jobs_recorded": len(mm_tracer.mm_threads),
    }

    # -- E. sync offline face: same swimlane, caller's thread, no pool -------
    sync_tok = td.SeamWordTokenizer()
    llm4, renderer4, _ = td.mk_engine(sync_tok)
    main_thread = threading.get_ident()
    proc_threads4: list[int] = []
    orig4 = llm4.input_processor.process_inputs

    def recording_proc4(*args, **kwargs):
        proc_threads4.append(threading.get_ident())
        return orig4(*args, **kwargs)

    llm4.input_processor.process_inputs = recording_proc4
    llm4.input_processor.process_inputs(  # LLMEngine.add_request 同步泳道 (llm_engine.py:L250-L262)
        "sync-1",
        downlink.tokens_input([7, 8]),
        downlink.SamplingParams(max_tokens=4),
        supported_tasks=("generate",),
    )
    out["E_sync_offline_face"] = {
        "caller_thread_id": main_thread,
        "process_inputs_thread_id": proc_threads4[0],
        "process_inputs_ran_on_caller_thread": proc_threads4[0] == main_thread,
        "note": "同步离线面同一条 process_inputs 泳道，跑在调用方线程、不经池",
    }

    dest = Path(__file__).resolve().parent / "m2_pool_offload.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
