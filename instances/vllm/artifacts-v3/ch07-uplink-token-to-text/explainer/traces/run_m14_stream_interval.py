"""Driver for m14 (stream_interval throttle) — host run against the ch07
companion (pin vLLM v0.27.1).

Engine-level interval 3, DELTA output, one token per step for 8 steps then
finish. Gates (output_processor.py:L292-L313): send only when
finished OR first token (sent_tokens_offset==0) OR num_output_tokens -
sent_tokens_offset >= interval; DELTA slices token ids from
sent_tokens_offset and advances it — batches are disjoint and exhaustive.
Also the per-request clamp: SamplingParams.stream_interval is max()'ed with
the ENGINE-level value (2 -> 3 stays 3; 5 -> 3 becomes 5), the engine level
is a floor, not an override (PR #49754).
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


def drain(q):
    outs = []
    while (o := q.get_nowait()) is not None:
        outs.append(o)
    return outs


def drive(op, req, q, n_steps, finish_at_last=True):
    """One token per step: ids 97..97+n_steps-1 ('a'..)."""
    rows = []
    for i in range(n_steps):
        tid = 97 + i
        finish = uplink.FinishReason.LENGTH if (finish_at_last and i == n_steps - 1) else None
        state = op.request_states.get(req.request_id)
        offset_before = state.sent_tokens_offset if state else None
        num_out = state.detokenizer.num_output_tokens() if state else None
        op.process_outputs([td.eco(req.request_id, [tid], finish=finish)])
        got = drain(q)
        rows.append({
            "round": i + 1,
            "token_id": tid,
            "char": chr(tid),
            "num_output_tokens": num_out + 1 if num_out is not None else None,
            "finished": finish is not None,
            "sent_tokens_offset_before": offset_before,
            "sent_tokens_offset_after": state.sent_tokens_offset if state else None,
            "gated_out": len(got) == 0,
            "texts": [o.outputs[0].text for o in got],
            "token_batch_sizes": [len(o.outputs[0].token_ids) for o in got],
        })
    return rows


async def main():
    out = {
        "driver": "run_m14_stream_interval.py",
        "mechanism": "m14 stream_interval 节流：完成/首 token/攒够才发；DELTA 从 sent_tokens_offset 切；clamp 取 max",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "tokenizer": "Fast byte 级（'a'=97..'h'=104）",
    }

    def setup(engine_interval, per_request=None):
        op = uplink.OutputProcessor(
            td.fast_backend(), log_stats=False, stream_interval=engine_interval
        )
        req = td.make_request(
            "ext-throttle",
            td.b("Hi"),
            td.sp(output_kind=uplink.RequestOutputKind.DELTA, stream_interval=per_request),
        )
        uplink.InputProcessor.assign_request_id(req)
        q = uplink.RequestOutputCollector(uplink.RequestOutputKind.DELTA, req.request_id)
        op.add_request(req, None, None, 0, q)
        return op, req, q

    op, req, q = setup(3)
    rows = drive(op, req, q, 8)
    emitted = [t for r in rows for t in r["texts"]]
    out["engine_interval_3"] = {
        "engine_stream_interval": 3,
        "per_request_stream_interval": None,
        "effective_interval": op.request_states[req.request_id].stream_interval if req.request_id in op.request_states else 3,
        "rounds": rows,
        "send_rounds": [r["round"] for r in rows if not r["gated_out"]],
        "gated_rounds": [r["round"] for r in rows if r["gated_out"]],
        "texts_in_order": emitted,
        "concatenated": "".join(emitted),
        "token_batches": [b for r in rows for b in r["token_batch_sizes"]],
        "disjoint_exhaustive": "".join(emitted) == "abcdefgh",
        "note": "首 token 强制发；之后攒满 interval 才发；完成强制发——DELTA 从 sent_tokens_offset 切批，无重叠无丢失",
    }

    # ---- clamp: engine floor wins over smaller per-request value -------------
    op2, req2, q2 = setup(3, per_request=2)
    s2 = op2.request_states[req2.request_id]
    op3, req3, q3 = setup(3, per_request=5)
    s3 = op3.request_states[req3.request_id]
    rows3 = drive(op3, req3, q3, 8)
    out["clamp_max"] = {
        "engine_3_request_2_effective": s2.stream_interval,
        "engine_3_request_5_effective": s3.stream_interval,
        "request_5_send_rounds": [r["round"] for r in rows3 if not r["gated_out"]],
        "request_5_token_batches": [b for r in rows3 for b in r["token_batch_sizes"]],
        "note": "from_new_request 取 max(per-request, engine)——引擎级是下限不是覆盖（PR #49754）",
    }

    dest = Path(__file__).resolve().parent / "m14_stream_interval.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    asyncio.run(main())
