"""Driver for m8 (水位 watermark：watermark_blocks = int(watermark × num_blocks)，
仅 WAITING/PREEMPTED 准入且 has_scheduled_reqs 时计入 required_blocks 的
headroom；默认 0.0 关；RUNNING 增长不吃；首拍（running 空）不吃) — host run,
pin vLLM v0.27.1 (config/scheduler.py:L136-L141 + kv_cache_manager.py:
L168-L171 / L463-L470 / L521-L527 + benchmarks/kv_cache_watermark.sh:L5-L16).

Setup: pool 10 blocks, watermark 0.5 -> watermark_blocks = 5.
  A-1 first admission (running empty -> has_scheduled_reqs=False): r1's
      128-token prompt = 8 blocks admitted although 8+5=13 > 10 would refuse
      if the watermark were wrongly applied — the engine could never start.
  A-2 RUNNING growth: r1's decode takes its 9th block — status RUNNING, no
      watermark (otherwise normal decode would be throttled).
  A-3 admission with headroom: a 16-token `small` request needs 1 block,
      free = 1: required = 1+5 = 6 > 1 -> None -> stays WAITING (r1 keeps
      decoding normally).
  A-3' same setup with watermark=0.0 (default): required = 1 <= 1 ->
      admitted. The watermark is a trade-off knob, off by default.
Arithmetic: int(0.5*10)=5, int(0.3*10)=3, default config watermark 0.0.
"""
import json
import sys
from pathlib import Path

_CH = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_CH))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from implementation.kv_cache_manager import KVCacheManager  # noqa: E402
from implementation.output import ModelRunnerOutput  # noqa: E402
from implementation.request import Request, SamplingParams  # noqa: E402
from implementation.scheduler import Scheduler  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402


def make_request(req_id, prompt_len, base):
    return Request(
        request_id=req_id,
        prompt_token_ids=list(range(base, base + prompt_len)),
        sampling_params=SamplingParams(max_tokens=64),
    )


def step(sched, tokens_by_req):
    out = sched.schedule()
    req_ids = list(out.num_scheduled_tokens)
    mro = ModelRunnerOutput(
        req_ids=req_ids,
        req_id_to_index={rid: i for i, rid in enumerate(req_ids)},
        sampled_token_ids=[tokens_by_req.get(rid, []) for rid in req_ids],
    )
    sched.update_from_output(out, mro)
    return out


def decode_steady(num_gpu_blocks, watermark):
    """r1 prompt 128（8 块）prefill 完 + 1 decode（第 9 块），空闲 = num_gpu_blocks-9。"""
    sched = Scheduler(SchedulerConfig(watermark=watermark), max_model_len=4096,
                      num_gpu_blocks=num_gpu_blocks, block_size=16)
    r1 = make_request("r1", 128, 0)
    sched.add_request(r1)
    step(sched, {"r1": [1]})  # 首拍 prefill 8 块（running 空 → 水位不适用）
    step(sched, {"r1": [2]})  # decode 领第 9 块（RUNNING 增长 → 水位不适用）
    return sched, r1


def main():
    out = {
        "driver": "run_m8_watermark.py",
        "mechanism": "m8 水位 watermark 三限定（kv_cache_manager.py:L463-L470 + L521-L527；config/scheduler.py:L136-L141）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch11 implementation/：watermark 两处原文保留（准入门 + 分配门）",
        "arithmetic": {
            "watermark_0.5_x_10_blocks": KVCacheManager(
                num_gpu_blocks=10, block_size=16, max_model_len=4096,
                watermark=0.5).watermark_blocks,
            "watermark_0.3_x_10_blocks": KVCacheManager(
                num_gpu_blocks=10, block_size=16, max_model_len=4096,
                watermark=0.3).watermark_blocks,
            "watermark_0.3_x_100_blocks": KVCacheManager(
                num_gpu_blocks=100, block_size=16, max_model_len=4096,
                watermark=0.3).watermark_blocks,
            "default_config_watermark": SchedulerConfig().watermark,
            "formula": "watermark_blocks = int(watermark × num_blocks)（kv_cache_manager.py:L170-L171）",
        },
        "scenario": {"config": {"num_gpu_blocks": 10, "block_size": 16,
                                "watermark": 0.5, "r1_prompt": 128, "small_prompt": 16},
                     "beats": []},
    }
    assert out["arithmetic"]["watermark_0.5_x_10_blocks"] == 5
    assert out["arithmetic"]["watermark_0.3_x_10_blocks"] == 3
    assert out["arithmetic"]["watermark_0.3_x_100_blocks"] == 30
    assert out["arithmetic"]["default_config_watermark"] == 0.0

    alloc_log = []

    def beat(sched, label, note, tokens, reqs):
        calls = []
        orig = sched.kv_cache_manager.allocate_slots

        def rec(request, num_new_tokens, **kw):
            res = orig(request, num_new_tokens, **kw)
            calls.append({"req": request.request_id, "ask": num_new_tokens,
                          "ok": res is not None,
                          "full_sequence_must_fit": kw.get("full_sequence_must_fit", False),
                          "has_scheduled_reqs": kw.get("has_scheduled_reqs", True),
                          "status": request.status.name})
            return res

        sched.kv_cache_manager.allocate_slots = rec
        free_before = sched.kv_cache_manager.num_free_blocks
        o = step(sched, tokens)
        sched.kv_cache_manager.allocate_slots = orig
        alloc_log.append({"beat": label, "calls": list(calls)})
        out["scenario"]["beats"].append({
            "beat": label, "note": note,
            "watermark_blocks": sched.kv_cache_manager.watermark_blocks,
            "free_blocks_before": free_before,
            "free_blocks_after": sched.kv_cache_manager.num_free_blocks,
            "allocate_calls": list(calls),
            "num_scheduled_tokens": dict(o.num_scheduled_tokens),
            "statuses": {rid: r.status.name for rid, r in reqs.items()},
        })
        return o

    # A-1/A-2: build steady state on the watermark=0.5 scheduler, logging each beat
    sched = Scheduler(SchedulerConfig(watermark=0.5), max_model_len=4096,
                      num_gpu_blocks=10, block_size=16)
    r1 = make_request("r1", 128, 0)
    sched.add_request(r1)
    beat(sched, "A-1", "首拍准入（running 空 → has_scheduled_reqs=False）：8 块放行——若水位误计入 8+5=13>10 将永不起步",
         {"r1": [1]}, {"r1": r1})
    beat(sched, "A-2", "r1 decode 领第 9 块（status=RUNNING → 水位不适用）：1 ≤ 空闲 2 → OK",
         {"r1": [2]}, {"r1": r1})
    assert sched.kv_cache_manager.num_free_blocks == 1
    small = make_request("small", 16, 300)
    sched.add_request(small)
    o = beat(sched, "A-3", "准入 small（WAITING+在场者 → 水位计入）：required=1+5=6 > 空闲 1 → None → 留在 WAITING；r1 照常 decode",
             {"r1": [3]}, {"r1": r1, "small": small})
    assert "small" not in o.num_scheduled_tokens and small.status.name == "WAITING"
    assert o.num_scheduled_tokens == {"r1": 1}

    # A-3': watermark off (default 0.0) — same shape admits
    sched2, r1b = decode_steady(10, 0.0)
    assert sched2.kv_cache_manager.num_free_blocks == 1
    small2 = make_request("small2", 16, 400)
    sched2.add_request(small2)
    o = beat(sched2, "A-3'", "对照（watermark=0.0 默认关）：required=1+0=1 ≤ 空闲 1 → 放行准入",
             {"r1": [3], "small2": [9]}, {"r1": r1b, "small2": small2})
    assert "small2" in o.num_scheduled_tokens and small2.status.name == "RUNNING"

    dest = Path(__file__).with_name("m8_watermark.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    print("arithmetic:", out["arithmetic"])
    for b in out["scenario"]["beats"]:
        print(b["beat"], "sched", b["num_scheduled_tokens"],
              "calls", [(c["req"], c["ask"], c["ok"], c["status"],
                         c["full_sequence_must_fit"], c["has_scheduled_reqs"])
                        for c in b["allocate_calls"]],
              "free", b["free_blocks_before"], "->", b["free_blocks_after"])


if __name__ == "__main__":
    main()
