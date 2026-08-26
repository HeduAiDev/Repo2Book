"""Driver for m11 (update_from_output 热循环：req_id_to_index 定位采样行 · 扣
num_in_flight_tokens · mid-prefill chunk 空行不外送 · abort 期已完成请求
continue 幂等 · woosuk 瓶颈自注) — host run, pin vLLM v0.27.1
(scheduler.py:L1670-L1764).

Budget 80 tokens. r1 = 16-token prompt; r2 = 128-token prompt (chunked).
  beat 1: r1 admitted (16, sampled row [1]); r2 first chunk 64 — mid-prefill:
      the model runner returns an EMPTY row for r2 -> nothing emitted, no
      stop check, but its in_flight is still settled (64-64=0).
  beat 2: r1 decode 1; r2 final chunk 64 — chunk-final produces logits ->
      sampled row [5] emitted.
  beat 3: both decode: rows located via req_id_to_index (r1->0, r2->1).
  beat 4 (abort during execution): schedule() ran (both decode, in_flight 1
      each) when the client disconnects -> finish_requests("r1") deletes r1
      from the ledger; update_from_output then hits `request is None` for r1
      -> `continue` (idempotent skip, no crash, no output) while r2 emits
      normally — the premise that makes ch9's abort double-delivery safe.
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
from implementation.request import Request, RequestStatus, SamplingParams  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402


def make_request(req_id, prompt_len, base):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(base, base + prompt_len)),
        sampling_params=SamplingParams(max_tokens=64),
        block_hasher=get_request_block_hasher(16),
    )


def main():
    out = {
        "driver": "run_m11_hotloop.py",
        "mechanism": "m11 update_from_output 热循环（scheduler.py:L1670-L1764；woosuk 自注 L1728-L1730 'can be a performance bottleneck'）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch11 implementation/（定位/扣在途/空行跳过/abort 幂等 continue 原样保留）",
        "config": {"max_num_batched_tokens": 80, "block_size": 16,
                   "num_gpu_blocks": 16, "prompts": {"r1": 16, "r2": 128}},
        "beats": [],
    }

    sched = Scheduler(SchedulerConfig(max_num_batched_tokens=80),
                      max_model_len=4096, num_gpu_blocks=16, block_size=16)
    r1 = make_request("r1", 16, 0)
    r2 = make_request("r2", 128, 100)
    sched.add_request(r1)
    sched.add_request(r2)

    def step(tokens_by_req, aborted=None):
        outp = sched.schedule()
        req_ids = list(outp.num_scheduled_tokens)
        mro = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
            sampled_token_ids=[tokens_by_req.get(rid, []) for rid in req_ids],
        )
        inflight_before = {rid: sched.requests[rid].num_in_flight_tokens
                           for rid in req_ids if rid in sched.requests}
        abort_note = None
        if aborted:
            ret = sched.finish_requests(aborted, RequestStatus.FINISHED_ABORTED)
            abort_note = {"aborted": list(aborted),
                          "returned": [r.request_id for r in ret]}
        outputs = sched.update_from_output(outp, mro)
        emitted = {o.request_id: list(o.new_token_ids)
                   for eco in outputs.values() for o in eco.outputs}
        return outp, mro, inflight_before, emitted, abort_note

    def beat(label, note, tokens, aborted=None):
        o, mro, before, emitted, abort_note = step(tokens, aborted)
        rec = {
            "beat": label, "note": note,
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "req_id_to_index": dict(mro.req_id_to_index),
            "sampled_rows": {rid: row for rid, row in
                             zip(mro.req_ids, mro.sampled_token_ids)},
            "in_flight_before_update": before,
            "in_flight_after": {rid: (sched.requests[rid].num_in_flight_tokens
                                      if rid in sched.requests else "deleted")
                                for rid in o.num_scheduled_tokens},
            "emitted": emitted,
        }
        if abort_note:
            rec["finish_requests_during_execution"] = abort_note
            rec["r1_in_requests"] = "r1" in sched.requests
        out["beats"].append(rec)

    beat("1", "r1 全量 16（有采样行 [1]）；r2 首 chunk 64 = mid-prefill：采样行空 → 不外送不判停，但在途照样核销（64-64=0）",
         {"r1": [1], "r2": []})
    beat("2", "r1 decode 1；r2 尾 chunk 64 = chunk 末有 logits → 采样行 [5] 外送",
         {"r1": [2], "r2": [5]})
    beat("3", "双 decode：req_id_to_index 定位各自采样行（r1→0、r2→1），在途 1→0 逐请求核销",
         {"r1": [3], "r2": [6]})
    beat("4", "执行期 abort：本拍已调度（在途 1/1）时客户端断连 → finish_requests 删 r1 → update 遇 request is None → continue 跳过（无输出不报错）；r2 正常外送",
         {"r1": [42], "r2": [7]}, aborted="r1")

    b1, b2, b3, b4 = out["beats"]
    assert b1["num_scheduled_tokens"] == {"r1": 16, "r2": 64}
    assert b1["emitted"] == {"r1": [1]}
    assert b2["num_scheduled_tokens"] == {"r1": 1, "r2": 64}
    assert b2["emitted"] == {"r1": [2], "r2": [5]}
    assert b3["req_id_to_index"] == {"r1": 0, "r2": 1}
    assert b4["emitted"] == {"r2": [7]} and b4["r1_in_requests"] is False

    dest = Path(__file__).with_name("m11_hotloop.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for b in out["beats"]:
        print(b["beat"], "sched", b["num_scheduled_tokens"], "idx", b["req_id_to_index"],
              "inflight", b["in_flight_before_update"], "->", b["in_flight_after"],
              "emitted", b["emitted"])


if __name__ == "__main__":
    main()
