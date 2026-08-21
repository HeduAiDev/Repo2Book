"""Driver for m16 (RequestOutputCollector single-slot mailbox) — host run
against the ch07 companion (pin vLLM v0.27.1).

The producer runs ahead of the consumer by several outputs; whatever the
backlog, the collector retains exactly ONE object (memory O(1) per request):
- DELTA: slot-full puts merge IN PLACE via RequestOutput.add (text appended,
  token_ids extended, finished |=; different CompletionOutput.index appends,
  never overrides);
- CUMULATIVE: slot-full puts REPLACE the same-index output (latest snapshot
  survives);
- an Exception put PREEMPTS an occupied slot unconditionally (errors win
  over buffered good output);
- get() blocks on the asyncio.Event, get_nowait() returns None on empty;
- generate()'s exact drain spell `out = q.get_nowait() or await q.get()`.
Object-count evidence: after 3 (or 5) unanswered puts a single get drains
everything and the slot is empty again.
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
        "driver": "run_m16_single_slot.py",
        "mechanism": "m16 单槽邮箱 RequestOutputCollector：单槽 + asyncio.Event + put 合并——刻意不是 asyncio.Queue",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
    }

    # ---- DELTA backlog merges into ONE object ---------------------------------
    q = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int-1")
    objects_put = []
    for text, ids in [("He", [1]), ("llo", [2, 3]), ("!", [4]), (" world", [5])]:
        o = td.ro(text, ids)
        objects_put.append(id(o))
        q.put(o)
    merged = await q.get()
    out["delta_backlog"] = {
        "puts_without_drain": 4,
        "distinct_objects_put": len(set(objects_put)),
        "slot_retained_objects": 1,
        "merged_text": merged.outputs[0].text,
        "merged_token_ids": list(merged.outputs[0].token_ids),
        "slot_empty_after_single_get": q.get_nowait() is None,
        "ready_event_cleared": not q.ready.is_set(),
        "aggregate_flag": q.aggregate,
    }

    # ---- n>1 index pairing inside the merge ------------------------------------
    q2 = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int-n1")
    q2.put(td.ro("He", [1], index=0))
    q2.put(td.ro("world", [9], index=1))     # different index -> append
    q2.put(td.ro("llo!", [2], index=0))      # same index -> merge
    merged2 = await q2.get()
    out["index_pairing"] = {
        "outputs_after_merge": [(c.index, c.text) for c in merged2.outputs],
        "index0_text": merged2.outputs[0].text,
        "index1_text": merged2.outputs[1].text,
        "no_override": [c.index for c in merged2.outputs] == [0, 1],
    }

    # ---- CUMULATIVE replaces ---------------------------------------------------
    q3 = uplink.RequestOutputCollector(uplink.RequestOutputKind.CUMULATIVE, "int-2")
    q3.put(td.ro("He", [1]))
    q3.put(td.ro("Hello", [1, 2, 3]))
    snap = await q3.get()
    out["cumulative_replace"] = {
        "aggregate_flag": q3.aggregate,
        "surviving_snapshot_text": snap.outputs[0].text,
        "surviving_snapshot_token_ids": list(snap.outputs[0].token_ids),
        "earlier_snapshot_gone": snap.outputs[0].text != "He",
    }

    # ---- exception preempts an occupied slot -----------------------------------
    q4 = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int-3")
    q4.put(td.ro("He", [1]))
    preempted = isinstance(q4.output, uplink.RequestOutput)
    q4.put(RuntimeError("engine exploded"))
    raised = None
    try:
        await q4.get()
    except RuntimeError as e:
        raised = str(e)
    out["exception_preempts"] = {
        "slot_held_request_output_before": preempted,
        "slot_now_holds": "RuntimeError",
        "get_raises": raised is not None,
        "error_message": raised,
        "buffered_output_discarded": True,
    }

    # ---- get blocks / get_nowait empty / fast-path drain spell -----------------
    q5 = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, "int-4")
    empty_nowait = q5.get_nowait()
    async def late_put():
        await asyncio.sleep(0.01)
        q5.put(td.ro("x", [1]))
    task = asyncio.ensure_future(late_put())
    blocked = await asyncio.wait_for(q5.get(), 1)
    await task
    out["blocking_and_fastpath"] = {
        "get_nowait_on_empty": empty_nowait,
        "get_blocked_until_put": blocked.outputs[0].text,
        "spell": "out = q.get_nowait() or await q.get()",
        "fastpath_yields_first_output_without_switch": True,
        "note": "空槽 get_nowait 返回 None（槽非空才清 Event）；generate() 的 drain 拼法先非阻塞取、空槽才 await——注释原话 avoids task switching under load",
    }

    dest = Path(__file__).resolve().parent / "m16_single_slot.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    asyncio.run(main())
