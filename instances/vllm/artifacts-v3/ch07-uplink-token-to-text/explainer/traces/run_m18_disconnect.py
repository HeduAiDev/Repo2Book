"""Driver for m18 (disconnect -> reverse abort, three relays) — host run
against the ch07 companion (pin vLLM v0.27.1).

The HTTP first relay (with_cancellation racing listen_for_disconnect) is the
serving domain (ch38) — here the SAME cancellation generate() sees in
production is produced by cancelling the generate task directly. Records the
timeline with a monotonic clock on both abort hops (monkeypatched recorders
around the real calls):
- relay 2: CancelledError inside generate() -> await self.abort(q.request_id,
  internal=True);
- hop 1 (this process): OutputProcessor.abort_requests removes the state and
  puts a finish_reason=ABORT terminal output into the collector (so any
  waiter wakes with finished=True instead of hanging);
- hop 2 (cross-process): engine_core.abort_requests_async sends the ABORT
  frame with the internal id — strictly AFTER hop 1 returned.
Also: the ABORT terminal output still carries the EXTERNAL id.
"""
import asyncio
import importlib
import json
import sys
import time
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH / "tests"))
td = importlib.import_module("test_uplink")
uplink = td.uplink


async def main():
    out = {
        "driver": "run_m18_disconnect.py",
        "mechanism": "m18 断连反向 abort 三层接力（F5 埋）：CancelledError → abort(internal=True) 两跳",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "relay1_note": "HTTP 第一跳 with_cancellation/listen_for_disconnect（api_utils.py）是服务面域——ch38 展开；本驱动以任务取消产生同款 CancelledError",
    }

    h = td.Harness(tokenizer=td.fast_backend())
    timeline = []
    put_log = []
    real_abort_local = uplink.OutputProcessor.abort_requests
    real_abort_remote = uplink.AsyncMPClient.abort_requests_async
    real_put = uplink.RequestOutputCollector.put

    def traced_put(self, output):
        real_put(self, output)
        if isinstance(output, uplink.RequestOutput):
            put_log.append({
                "seq": len(put_log),
                "t_abs": time.monotonic(),
                "request_id_on_output": output.request_id,
                "finished": output.finished,
                "finish_reason": output.outputs[0].finish_reason if output.outputs else None,
                "text": output.outputs[0].text if output.outputs else None,
            })

    def traced_abort_local(self, request_ids, internal):
        t0 = time.monotonic()
        res = real_abort_local(self, request_ids, internal)
        timeline.append({
            "hop": 1,
            "t_abs": time.monotonic(),
            "t_start": t0,
            "seq": len(timeline),
            "what": "OutputProcessor.abort_requests（本进程：移状态 + 投 ABORT 终态）",
            "request_ids_arg_type": "internal" if internal else "external",
            "returned_ids": list(res),
        })
        return res

    async def traced_abort_remote(self, request_ids):
        t0 = time.monotonic()
        await real_abort_remote(self, request_ids)
        timeline.append({
            "hop": 2,
            "t_abs": time.monotonic(),
            "t_start": t0,
            "seq": len(timeline),
            "what": "engine_core.abort_requests_async（跨进程：ABORT 帧过线停算）",
            "frame_ids": list(request_ids),
        })

    uplink.OutputProcessor.abort_requests = traced_abort_local
    uplink.AsyncMPClient.abort_requests_async = traced_abort_remote
    uplink.RequestOutputCollector.put = traced_put

    gen = h.llm.generate(
        td.make_request("chatcmpl-dis", td.b("Hi"), td.sp()),
        td.sp(),
        "chatcmpl-dis",
    )
    got = []
    task = asyncio.ensure_future(_collect(gen, got))
    await td.wait_for_add(h, 1)
    h.feed([td.eco(h.add_frames()[0].request_id, [65])])
    await asyncio.sleep(0.05)

    out["before_disconnect"] = {
        "outputs_yielded_before_disconnect": len(got),
        "texts": [o.outputs[0].text for o in got],
        "request_states_size": len(h.llm.output_processor.request_states),
        "external_map_size": len(h.llm.output_processor.external_req_ids),
    }

    # relay 2: client disconnect == the generate task is cancelled
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    out["timeline"] = timeline
    out["after_disconnect"] = {
        "request_states_size": len(h.llm.output_processor.request_states),
        "external_map_size": len(h.llm.output_processor.external_req_ids),
        "abort_frames": [r for t, r in h.client.sent if t is uplink.EngineCoreRequestType.ABORT],
        "hop_order_local_before_remote": [e["hop"] for e in timeline] == [1, 2],
        "hop2_started_after_hop1_finished": timeline[1]["t_start"] >= timeline[0]["t_abs"],
    }
    # every collector put, in order — the LAST one is hop 1's ABORT terminal
    terminal = put_log[-1]
    out["collector_put_log"] = put_log
    out["abort_terminal_output"] = {
        "put_seq": terminal["seq"],
        "request_id_on_output": terminal["request_id_on_output"],
        "finished": terminal["finished"],
        "finish_reason": terminal["finish_reason"],
        "text": terminal["text"],
        "put_between_hops": timeline[0]["t_abs"] <= terminal["t_abs"] <= timeline[1]["t_start"],
        "note": "hop1 在两跳之间给 collector 投 finish_reason=abort 的终态（request_id 写回外部 id）——还在 await q.get() 的消费者立即解阻塞拿到 finished=True，而非挂死",
    }

    uplink.OutputProcessor.abort_requests = real_abort_local
    uplink.AsyncMPClient.abort_requests_async = real_abort_remote
    uplink.RequestOutputCollector.put = real_put
    await h.close()

    dest = Path(__file__).resolve().parent / "m18_disconnect.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")


async def _collect(gen, sink):
    async for o in gen:
        sink.append(o)


if __name__ == "__main__":
    asyncio.run(main())
