"""Driver for m12 (_update_request_with_output：逐 token append_output_token_
ids（连带增量块哈希）+ check_stop + 截断 del new_token_ids[num_new:]) —
host run, pin vLLM v0.27.1 (scheduler.py:L2094-L2111 + request.py:L249-L265).

Phase A (trim): r1 = 16-token prompt, eos_token_id=6. Beat 1 emits [1].
Beat 2 feeds a 3-token sampled row [5,6,7] (a multi-token row like spec/jump
decoding produces): token 5 appended (no stop), token 6 appended -> EOS ->
stopped -> del new_token_ids[2:] -> the row is trimmed to [5,6]; token 7 is
NEVER appended nor emitted. The stop token itself IS appended (the user sees
the trigger) — everything after it is cut.
Phase B (incremental block hashes): block_size=4, prompt 6 tokens -> 1 full
block hash at construction; appending tokens one by one extends block_hashes
exactly when a full block completes (at 8, at 12 tokens) — the F2 thread is
spun on every output.
"""
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from implementation.kv_cache_manager import get_request_block_hasher  # noqa: E402
from implementation.output import ModelRunnerOutput  # noqa: E402
from implementation.request import Request, SamplingParams  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402


def main():
    out = {
        "driver": "run_m12_token_trim.py",
        "mechanism": "m12 逐 token 收账 + 截断（scheduler.py:L2094-L2111；append 连带增量块哈希 request.py:L249-L265）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch11 implementation/（_update_request_with_output 逐字保留）",
        "phase_A": {"config": {"block_size": 16, "prompt": 16, "eos_token_id": 6},
                    "beats": []},
        "phase_B": {"config": {"block_size": 4, "prompt": 6},
                    "appends": []},
    }

    # ---- Phase A ----
    sched = Scheduler(SchedulerConfig(), max_model_len=4096,
                      num_gpu_blocks=8, block_size=16)
    r1 = Request(request_id="r1", prompt_token_ids=list(range(16)),
                 sampling_params=SamplingParams(max_tokens=64, eos_token_id=6),
                 block_hasher=get_request_block_hasher(16))
    sched.add_request(r1)

    def stepA(tokens):
        o = sched.schedule()
        req_ids = list(o.num_scheduled_tokens)
        mro = ModelRunnerOutput(
            req_ids=req_ids, req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=[tokens.get(rid, []) for rid in req_ids])
        outputs = sched.update_from_output(o, mro)
        emitted = {ec.request_id: {"new_token_ids": list(ec.new_token_ids),
                                   "finish_reason": ec.finish_reason.name if ec.finish_reason is not None else None}
                   for eco in outputs.values() for ec in eco.outputs}
        return o, emitted

    o, emitted = stepA({"r1": [1]})
    out["phase_A"]["beats"].append({
        "beat": "A-1", "sampled_row": [1],
        "emitted": emitted, "output_token_ids": list(r1.output_token_ids),
        "block_hashes_len": len(r1.block_hashes), "stopped": False,
        "note": "prefill 首拍：逐 token 循环对单 token 行只跑一轮，未停",
    })
    o, emitted = stepA({"r1": [5, 6, 7]})
    out["phase_A"]["beats"].append({
        "beat": "A-2", "sampled_row": [5, 6, 7],
        "emitted": emitted,
        "output_token_ids": list(r1.output_token_ids),
        "block_hashes_len": len(r1.block_hashes),
        "status": r1.status.name,
        "stop_reason": r1.stop_reason,
        "note": "5 入账未停 → 6 入账命中 EOS → 停止即截断 del new_token_ids[2:]：外送 [5,6]，7 既不入账也不外送",
    })
    a2 = out["phase_A"]["beats"][1]
    assert a2["emitted"]["r1"]["new_token_ids"] == [5, 6]
    assert a2["emitted"]["r1"]["finish_reason"] == "STOP"
    assert a2["output_token_ids"] == [1, 5, 6]
    assert a2["status"] == "FINISHED_STOPPED"

    # ---- Phase B: incremental hashes ----
    r2 = Request(request_id="r2", prompt_token_ids=list(range(100, 106)),
                 sampling_params=SamplingParams(max_tokens=64),
                 block_hasher=get_request_block_hasher(4))
    out["phase_B"]["appends"].append({
        "event": "构造", "num_tokens": r2.num_tokens,
        "block_hashes_len": len(r2.block_hashes),
        "note": "prompt 6 token（block_size=4）：1 满块 → 1 哈希",
    })
    for tok, note in ((7, "第 7 token：第 2 块未满"), (8, "第 8 token：第 2 块满 → 哈希+1"),
                      (9, "第 9 token：仍在第 2/3 块边界外"), (10, "第 10 token"),
                      (11, "第 11 token"), (12, "第 12 token：第 3 块满 → 哈希+1")):
        r2.append_output_token_ids(tok)
        out["phase_B"]["appends"].append({
            "event": f"append {tok}", "num_tokens": r2.num_tokens,
            "block_hashes_len": len(r2.block_hashes), "note": note,
        })
    lens = [a["block_hashes_len"] for a in out["phase_B"]["appends"]]
    assert lens == [1, 1, 2, 2, 2, 2, 3], lens

    dest = Path(__file__).with_name("m12_token_trim.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for b in out["phase_A"]["beats"]:
        print("A", b["beat"], "emitted", b["emitted"], "ledger", b["output_token_ids"])
    for a in out["phase_B"]["appends"]:
        print("B", a["event"], "tokens", a["num_tokens"], "hashes", a["block_hashes_len"])


if __name__ == "__main__":
    main()
