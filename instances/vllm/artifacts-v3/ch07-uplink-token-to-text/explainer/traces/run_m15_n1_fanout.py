"""Driver for m15 (n>1 fan-out + ParentRequest aggregation) — host run
against the ch07 companion (pin vLLM v0.27.1).

n=3 with seed=42 (each child clones a UNIQUE seed) and n=3 without seed
(child params object cached and reused). Records:
- fan-out: child ids "{idx}_{internal_id}", the LAST child reusing the
  ORIGINAL request object (id equality), one SHARED collector returned to
  generate();
- streaming aggregation: children's outputs arriving interleaved in one
  batch; each child's CompletionOutput forwarded as it arrives, collector
  merging BY index — outputs list grows 1 -> 2 -> 3, no override; finished
  only when the last child finishes;
- FINAL_ONLY aggregation: output_aggregator preallocated [None]*3, outputs
  stay empty until all children finished, then ONE RequestOutput with all 3.

Reproducibility: the internal-id suffix (random 8-hex from random_uuid in a
real server) is pinned to a deterministic counter here, so child ids like
"0_ext-seed-00000001" are stable across re-runs and citable as provenance.
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

# --- pin random_uuid (id suffix only) to a deterministic counter -----------
_uuid_counter = {"n": 0}
_real_random_uuid = uplink.random_uuid


def _deterministic_uuid() -> str:
    _uuid_counter["n"] += 1
    return f"{_uuid_counter['n']:08x}" + "0" * 8  # first 8 chars = counter


uplink.random_uuid = _deterministic_uuid  # assign_request_id slices :.8


def drain(q):
    outs = []
    while (o := q.get_nowait()) is not None:
        outs.append(o)
    return outs


async def run_case(h, ext, n, seed, kind):
    params = td.sp(n=n, output_kind=kind, seed=seed)
    original = td.make_request(ext, td.b("Hi"), params)
    q = await h.llm.add_request(ext, original, params)
    adds = h.add_frames()
    child_ids = [r.request_id for r in adds]
    child_params = [r.sampling_params for r in adds]
    return {
        "n": n,
        "ext_id": ext,
        "child_ids": child_ids,
        "child_id_prefixes_ok": all(
            cid == f"{i}_{cid[(2 + len(str(i))):]}" or cid.startswith(f"{i}_")
            for i, cid in enumerate(child_ids)
        ),
        "add_frames_total": len(adds),
        "last_child_reuses_original_object": adds[-1] is original,
        "first_two_children_are_copies": adds[0] is not original and adds[1] is not original,
        "shared_collector": q is not None,
        "collector_request_id": q.request_id,
        "child_param_seeds": [p.seed for p in child_params],
        "child_params_n": [p.n for p in child_params],
        "child_params_object_ids_shared": id(child_params[0]) == id(child_params[1]),
    }, q


async def main():
    out = {
        "driver": "run_m15_n1_fanout.py",
        "mechanism": "m15 n>1 扇出与父聚合：idx_ 前缀子 id、末子复用原对象、seed 逐子克隆；get_outputs 流式转发 / FINAL_ONLY 攒齐；collector.add 按 index 合并",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "tokenizer": "Fast byte 级（'0'..'2' 前缀来自 idx；子流文本用 'a'/'b'/'c' 区分）",
    }

    # ---- fan-out shape: seeded vs unseeded ------------------------------------
    h1 = td.Harness(tokenizer=td.fast_backend())
    seeded, _ = await run_case(h1, "ext-seed", 3, 42, uplink.RequestOutputKind.DELTA)
    await h1.close()

    h2 = td.Harness(tokenizer=td.fast_backend())
    unseeded, _ = await run_case(h2, "ext-noseed", 3, None, uplink.RequestOutputKind.DELTA)
    await h2.close()
    out["fanout_seeded"] = seeded
    out["fanout_unseeded"] = unseeded

    # ---- streaming aggregation over interleaved children ----------------------
    h3 = td.Harness(tokenizer=td.fast_backend())
    params = td.sp(n=3, output_kind=uplink.RequestOutputKind.DELTA)
    original = td.make_request("ext-agg", td.b("Hi"), params)
    q = await h3.llm.add_request("ext-agg", original, params)
    children = [r.request_id for r in h3.add_frames()]
    text_of = {children[0]: "a", children[1]: "b", children[2]: "c"}
    rounds = []
    # round 1: all three children each produce one token, interleaved batch
    batch1 = [td.eco(cid, td.b(text_of[cid])) for cid in children]
    # round 2: children finish one per round, reversed order (2 then 1 then 0)
    finishes = [(children[2], "!"), (children[1], "@"), (children[0], "#")]
    fed = 0
    h3.feed(batch1)
    await td.wait_for_add(h3, len(children))  # ensure handler running order stable
    await asyncio.sleep(0.05)
    rows = []
    got = drain(q)
    rows.append({
        "round": 1,
        "batch_child_texts": [text_of[c] for c in children],
        "puts": len(got),
        "delivered": [
            [(c.index, c.text) for c in o.outputs] for o in got
        ],
        "finished_flags": [o.finished for o in got],
    })
    for i, (cid, ch) in enumerate(finishes):
        h3.feed([td.eco(cid, td.b(ch), finish=uplink.FinishReason.LENGTH)])
        await asyncio.sleep(0.05)
        got = drain(q)
        rows.append({
            "round": i + 2,
            "finishing_child": text_of[cid] + "-stream",
            "final_char": ch,
            "puts": len(got),
            "delivered": [[(c.index, c.text) for c in o.outputs] for o in got],
            "finished_flags": [o.finished for o in got],
        })
    all_puts = [o for r in rows for o in []]  # placeholder
    out["stream_aggregation"] = {
        "child_order_in_outputs": [
            {"child": text_of[c], "index": i} for i, c in enumerate(children)
        ],
        "rounds": rows,
        "request_id_on_output": "ext-agg",
        "note": "每个子完成即转发（流式不攒）；collector 按 CompletionOutput.index 配对——outputs 列表逐个长出、互不覆盖；最后一个子完成才 finished",
    }
    await h3.close()

    # ---- FINAL_ONLY aggregation ------------------------------------------------
    h4 = td.Harness(tokenizer=td.fast_backend())
    params_f = td.sp(n=3, output_kind=uplink.RequestOutputKind.FINAL_ONLY)
    original_f = td.make_request("ext-final", td.b("Hi"), params_f)
    qf = await h4.llm.add_request("ext-final", original_f, params_f)
    children_f = [r.request_id for r in h4.add_frames()]
    rows_f = []
    for i, cid in enumerate(reversed(children_f)):
        h4.feed([td.eco(cid, td.b("xyz"[2 - i]) * 2, finish=uplink.FinishReason.LENGTH)])
        await asyncio.sleep(0.05)
        got = drain(qf)
        rows_f.append({
            "round": i + 1,
            "finished_children": i + 1,
            "puts": len(got),
            "delivered": [[(c.index, c.text) for c in o.outputs] for o in got],
            "finished_flags": [o.finished for o in got],
        })
    out["final_only_aggregation"] = {
        "child_order_in_outputs": [
            {"child": c, "index": i} for i, c in enumerate(children_f)
        ],
        "rounds": rows_f,
        "note": "FINAL_ONLY 预分配 output_aggregator=[None]*n；子逐个完成不发（puts=0），攒齐 n 个才一次返回、finished=True",
    }
    await h4.close()

    dest = Path(__file__).resolve().parent / "m15_n1_fanout.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {dest}")


if __name__ == "__main__":
    asyncio.run(main())
