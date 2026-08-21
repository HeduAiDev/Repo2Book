"""Driver for m2 (output_handler chunked pull: chunk=128 + sleep(0)) — host
run against the ch07 companion (pin vLLM v0.27.1).

One EngineCoreOutputs batch carrying 300 per-request outputs (the shape a
6000-request load produces over a few steps on one front-end). The resident
output_handler splits it into slices of VLLM_V1_OUTPUT_PROC_CHUNK_SIZE=128
and awaits asyncio.sleep(0) BETWEEN slices — the only places the event loop
can breathe while the batch is being processed. Evidence:
- slice lengths recorded by a wrapper around process_outputs: 128, 128, 44
  (ceil(300/128) = 3 slices);
- a heartbeat task (asyncio.sleep(0) spinner) DOES get scheduled during the
  batch — it can only run at the sleep(0) yield points, proving the loop is
  not monopolized for the whole batch;
- the async face asserts the returned list is empty (outputs all went to
  collectors).
"""
import asyncio
import importlib
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH / "tests"))
td = importlib.import_module("test_uplink")
uplink = td.uplink


async def main():
    out = {
        "driver": "run_m2_chunking.py",
        "mechanism": "m2 output_handler 常驻单任务拉批分块：chunk=128 + sleep(0) 让出事件循环",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "chunk_size_default": uplink.envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE,
        "batch_outputs": 300,
    }

    h = td.Harness(tokenizer=td.fast_backend())
    # 300 registered requests, FINAL_ONLY (intermediate steps construct
    # nothing — the demux+update loop still runs for all 300 per slice)
    params = td.sp(output_kind=uplink.RequestOutputKind.FINAL_ONLY, max_tokens=32)
    for i in range(300):
        await h.add(f"ext-{i}", td.b("Hi"), params)
    await td.wait_for_add(h, 300)

    # instrument: record each process_outputs call's slice length; stop the
    # heartbeat the moment the LAST slice of the batch is processed so ticks
    # count ONLY during batch processing
    slice_lengths = []
    ticks = {"n": 0}
    done = asyncio.Event()
    expected_slices = -(-300 // 128)
    real_process = h.llm.output_processor.process_outputs

    def wrapped(engine_core_outputs, *a, **kw):
        res = real_process(engine_core_outputs, *a, **kw)
        slice_lengths.append((len(engine_core_outputs), len(res.request_outputs), len(res.reqs_to_abort)))
        if len(slice_lengths) == expected_slices:
            done.set()
        return res

    h.llm.output_processor.process_outputs = wrapped

    # heartbeat: only runs when the loop yields (i.e. at sleep(0) points)
    async def heartbeat():
        while not done.is_set():
            ticks["n"] += 1
            await asyncio.sleep(0)

    hb = asyncio.create_task(heartbeat())

    req_ids = [r.request_id for r in h.add_frames()]
    ticks_before = ticks["n"]
    h.feed([td.eco(rid, [65]) for rid in req_ids])  # one token for each of 300
    await done.wait()
    ticks_during = ticks["n"] - ticks_before
    await asyncio.sleep(0.05)
    await hb

    out["results"] = {
        "slice_lengths": [s[0] for s in slice_lengths],
        "slice_count": len(slice_lengths),
        "ceil_300_over_128": -(-300 // 128),
        "async_face_list_empty_all_slices": all(s[1] == 0 for s in slice_lengths),
        "heartbeat_ticks_during_batch_only": ticks_during,
        "heartbeat_ran_during_batch": ticks_during > 0,
        "requests_processed": sum(s[0] for s in slice_lengths),
        "yield_points_between_slices": 2,
        "note": "heartbeat 在最后一片处理完的瞬间停表——计到的每一拍都发生在批处理期间；它只能在 sleep(0) 让出点被调度，证明片间确实让出了事件循环；返回列表断言全空（输出全走 collector）",
    }
    await h.close()

    dest = Path(__file__).resolve().parent / "m2_chunking.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    asyncio.run(main())
