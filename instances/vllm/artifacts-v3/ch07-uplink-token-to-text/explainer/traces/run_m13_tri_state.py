"""Driver for m13 (RequestOutputKind three-state contract) — host run against
the ch07 companion (pin vLLM v0.27.1).

The SAME token stream (prompt "Hi", then 'He' 'll' 'o' with LENGTH finish)
driven through three RequestStates whose only difference is output_kind:
- DELTA: every step produces one incremental RequestOutput (text delta, new
  token ids only);
- CUMULATIVE: every step produces a full-snapshot RequestOutput (text so
  far, all token ids);
- FINAL_ONLY: intermediate steps construct NOTHING — the collector slot
  stays empty (0 puts) until the finishing step delivers a single output
  with the whole text. Zero-construction, not filtering: make_request_output
  returns None before any CompletionOutput exists.

All three ride the same process_outputs single loop (one batch per step,
three EngineCoreOutputs interleaved — demux by internal id) and their own
real RequestOutputCollector.
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

KINDS = {
    "DELTA": uplink.RequestOutputKind.DELTA,
    "CUMULATIVE": uplink.RequestOutputKind.CUMULATIVE,
    "FINAL_ONLY": uplink.RequestOutputKind.FINAL_ONLY,
}


def drain(q):
    outs = []
    while (o := q.get_nowait()) is not None:
        outs.append(o)
    return outs


async def main():
    out = {
        "driver": "run_m13_tri_state.py",
        "mechanism": "m13 三态契约 RequestOutputKind（DELTA/CUMULATIVE/FINAL_ONLY）——FINAL_ONLY 是零构造不是过滤",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "tokenizer": "Fast byte 级（'H'=72 'e'=101 'l'=108 'o'=111）",
        "steps": [
            {"round": 1, "new_token_ids": [72, 101], "finish_reason": None},
            {"round": 2, "new_token_ids": [108, 108], "finish_reason": None},
            {"round": 3, "new_token_ids": [111], "finish_reason": "LENGTH"},
        ],
    }

    op = uplink.OutputProcessor(td.fast_backend(), log_stats=False, stream_interval=1)
    queues = {}
    for name, kind in KINDS.items():
        req = td.make_request(f"ext-{name}", td.b("Hi"), td.sp(output_kind=kind))
        uplink.InputProcessor.assign_request_id(req)
        q = uplink.RequestOutputCollector(kind, req.request_id)
        op.add_request(req, None, None, 0, q)
        queues[name] = q

    rounds = []
    for step in out["steps"]:
        batch = [
            td.eco(
                op.request_states and next(
                    rs.request_id for rs in op.request_states.values()
                    if rs.output_kind == kind
                ),
                step["new_token_ids"],
                finish=None if step["finish_reason"] is None else uplink.FinishReason.LENGTH,
            )
            for kind in KINDS.values()
        ]
        # final round finishes all three: after _finish_request the states are
        # gone, so collect internal ids BEFORE the call
        result = op.process_outputs(batch)
        row = {"round": step["round"], "sync_face_list_len": len(result.request_outputs)}
        for name, q in queues.items():
            got = drain(q)
            row[name] = {
                "puts_this_round": len(got),
                "texts": [o.outputs[0].text for o in got],
                "token_counts": [len(o.outputs[0].token_ids) for o in got],
                "finished_flags": [o.finished for o in got],
                "request_id_on_output": got[0].request_id if got else None,
            }
        rounds.append(row)
    out["rounds"] = rounds

    totals = {}
    for name in KINDS:
        puts = sum(r[name]["puts_this_round"] for r in rounds)
        texts = [t for r in rounds for t in r[name]["texts"]]
        tok_counts = [c for r in rounds for c in r[name]["token_counts"]]
        totals[name] = {
            "total_request_outputs_constructed": puts,
            "concatenated_text": "".join(texts),
            "last_snapshot_text": texts[-1] if texts else None,
            "last_token_count": tok_counts[-1] if tok_counts else None,
        }
    totals["FINAL_ONLY"]["puts_before_finish"] = sum(
        r["FINAL_ONLY"]["puts_this_round"] for r in rounds[:-1]
    )
    out["totals"] = totals
    out["verdict"] = {
        "same_stream_same_texts": totals["DELTA"]["concatenated_text"]
        == totals["FINAL_ONLY"]["last_snapshot_text"]
        == totals["CUMULATIVE"]["last_snapshot_text"],
        "delta_puts": totals["DELTA"]["total_request_outputs_constructed"],
        "cumulative_puts": totals["CUMULATIVE"]["total_request_outputs_constructed"],
        "final_only_puts": totals["FINAL_ONLY"]["total_request_outputs_constructed"],
        "final_only_intermediate_puts": totals["FINAL_ONLY"]["puts_before_finish"],
        "external_id_written_back": True,
        "note": "FINAL_ONLY 中间步 0 个对象——make_request_output 在构造任何 CompletionOutput 之前就 return None（省的是构造+put+Event 唤醒整条链）",
    }

    dest = Path(__file__).resolve().parent / "m13_tri_state.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    asyncio.run(main())
