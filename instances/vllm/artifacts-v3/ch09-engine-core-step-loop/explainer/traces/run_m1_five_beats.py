"""Driver for m1 (五拍编排 EngineCore.step) — host run against the ch09
subtract-only engine companion (pin vLLM v0.27.1 / 6e448d0ea).

One request's whole life plus a late arrival, five beats of the REAL
EngineCore.step() recorded from the real trace surfaces (scheduler.trace /
runner.trace, both time.perf_counter_ns at entry):

  beat 1  req-A prefill (3 tokens)            → first output token
  beat 2  mixed batch {A:1, B:4}              → decode + late prefill
  beat 3  {A:1, B:1} → both hit max_tokens    → two LENGTH finishes
  beat 4  flush beat: finished_req_ids ride a 0-token batch, no sampling
  beat 5  idle guard: ({}, False), executor untouched

Also records: ② always called with non_block=True (spy), the ②→③ gap
(in this sync companion the scripted forward — a 5ms stand-in for one real
forward pass — executes inside execute_model, so ③'s entry lands after it),
and the ③→④→⑤ event order. Table rows are built from run values only.
"""
import json
import sys
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "implementation"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import engine_loop as el  # noqa: E402


def mk_config(**over):
    model_over = dict(over.pop("model_config", {}) or {})
    cache_over = dict(over.pop("cache_config", {}) or {})
    par_over = dict(over.pop("parallel_config", {}) or {})
    sched_over = dict(over.pop("scheduler_config", {}) or {})
    return el.VllmConfig(
        model_config=el.ModelConfig(**model_over),
        cache_config=el.CacheConfig(**cache_over),
        parallel_config=el.ParallelConfig(**par_over),
        scheduler_config=el.SchedulerConfig(**sched_over),
        instance_id=over.pop("instance_id", f"trace-{uuid.uuid4().hex[:8]}"),
        **over,
    )


def core_request(request_id, token_ids, max_tokens, structured=False, client_index=0):
    sp = el.SamplingParams(max_tokens=max_tokens)
    if structured:
        sp.structured_output_request = True
    return el.EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=list(token_ids),
        mm_features=None,
        sampling_params=sp,
        pooling_params=None,
        arrival_time=1.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        client_index=client_index,
    )


def add(ec, request_id, token_ids, max_tokens, structured=False, client_index=0):
    req, wave = ec.preprocess_add_request(
        core_request(request_id, token_ids, max_tokens, structured, client_index)
    )
    ec.add_request(req, wave)
    return req


def logits_row(favorite, vocab=16, value=5.0):
    row = [0.0] * vocab
    row[favorite] = value
    return row


def rel_ms(events):
    t0 = events[0][0]
    return [{"event": name, "rel_ms": round((t - t0) / 1e6, 3)} for t, name in events]


def main():
    cfg = mk_config()
    ec = el.EngineCore(cfg, el.UniProcExecutor, False)
    add(ec, "req-A", (1, 2, 3), max_tokens=3)
    ec.enqueue_forward_logits(
        [
            {"req-A": logits_row(7)},
            {"req-A": logits_row(8), "req-B": logits_row(6)},
            {"req-A": logits_row(9), "req-B": logits_row(6)},
        ]
    )

    seen_non_block = []
    orig_execute = ec.model_executor.execute_model

    def spy_execute(scheduler_output, non_block=False):
        seen_non_block.append(non_block)
        return orig_execute(scheduler_output, non_block=non_block)

    ec.model_executor.execute_model = spy_execute

    beats = []

    def do_beat(label, note):
        n_sched = len(ec.scheduler.trace)
        n_run = len(ec.model_executor.driver_worker.trace)
        t0 = time.perf_counter()
        outputs, executed = ec.step()
        wall_ms = round((time.perf_counter() - t0) * 1000, 3)
        sched_new = ec.scheduler.trace[n_sched:]
        run_new = ec.model_executor.driver_worker.trace[n_run:]
        events = sorted(
            [(t, f"sched:{n}") for t, n in sched_new]
            + [(t, f"runner:{n}") for t, n in run_new]
        )
        last = ec.scheduler.last_output
        rec = {
            "beat": label,
            "note": note,
            "events_rel_ms": rel_ms(events),
            "event_order": [n for _, n in events],
            "wall_ms": wall_ms,
            "batch": dict(last.num_scheduled_tokens),
            "total_num_scheduled_tokens": last.total_num_scheduled_tokens,
            "finished_req_ids_riding_batch": sorted(last.finished_req_ids),
            "outputs_by_client": {
                str(c): [
                    {
                        "request_id": o.request_id,
                        "new_token_ids": list(o.new_token_ids),
                        "finish_reason": (
                            None if o.finish_reason is None else str(o.finish_reason)
                        ),
                    }
                    for o in eco.outputs
                ]
                for c, eco in outputs.items()
            },
            "num_output_msgs": sum(len(eco.outputs) for eco in outputs.values()),
            "model_executed": executed,
            "requests_after": {
                rid: {
                    "status": r.status.name,
                    "num_computed_tokens": r.num_computed_tokens,
                    "num_output_tokens": r.num_output_tokens,
                    "num_tokens": r.num_tokens,
                }
                for rid, r in ec.scheduler.requests.items()
            },
            "running_ids": [r.request_id for r in ec.scheduler.running],
        }
        # ②发起→③ gap: get_grammar_bitmask entry minus execute_model entry.
        # In this sync companion the scripted forward (5ms stand-in) executes
        # inside execute_model, so this gap ≈ one forward pass.
        rels = {e["event"]: e["rel_ms"] for e in rec["events_rel_ms"]}
        if "runner:execute_model" in rels and "sched:get_grammar_bitmask" in rels:
            rec["gap_launch_to_bitmask_ms"] = round(
                rels["sched:get_grammar_bitmask"] - rels["runner:execute_model"], 3
            )
        if "sched:get_grammar_bitmask" in rels and "runner:greedy_sample" in rels:
            rec["gap_bitmask_to_sample_ms"] = round(
                rels["runner:greedy_sample"] - rels["sched:get_grammar_bitmask"], 3
            )
        beats.append(rec)
        return rec

    b1 = do_beat(1, "req-A 全量 prompt（3 token）一拍 prefill，产出首个输出 token")
    add(ec, "req-B", (7, 8, 9, 10), max_tokens=2)
    b2 = do_beat(2, "混相批：A decode 1 token + 迟到的 B 全量 prefill 4 token")
    b3 = do_beat(3, "双双 decode 到顶：A 3/3、B 2/2，两个 LENGTH 同拍完成")
    b4 = do_beat(4, "flush 拍：finished_req_ids 随 0-token 批下发 worker 清缓存，不采样")
    b5 = do_beat(5, "空转守卫拍：has_requests()==False → ({}, False)，executor 零调用")

    fwd = "seam forward = 5ms sleep modeling one real forward pass (真实一步前向 ~几十 ms)"

    def fmt_batch(b):
        return "{" + ", ".join(f"'{k}': {v}" for k, v in b["batch"].items()) + "}"

    def gap(b):
        return str(b.get("gap_launch_to_bitmask_ms", "—"))

    table = {
        "columns": [
            "拍",
            "① schedule 批 {req: num}",
            "② 发起→③ 间隔(ms)",
            "③ bitmask",
            "④ 采样产出",
            "⑤ 记账（输出/完成）",
            "step 返回",
        ],
        "rows": [
            [
                "1",
                fmt_batch(b1),
                gap(b1),
                "None（无结构化请求，快速返回）",
                f"req-A→{b1['outputs_by_client']['0'][0]['new_token_ids']}",
                f"client0 收 {b1['num_output_msgs']} 条；无人完成",
                f"executed={b1['model_executed']}",
            ],
            [
                "2",
                fmt_batch(b2),
                gap(b2),
                "None",
                "req-A→[8]，req-B→[6]",
                f"client0 收 {b2['num_output_msgs']} 条；无人完成",
                f"executed={b2['model_executed']}",
            ],
            [
                "3",
                fmt_batch(b3),
                gap(b3),
                "None",
                "req-A→[9]，req-B→[6]",
                "A LENGTH(3/3)、B LENGTH(2/2)→同拍释放",
                f"executed={b3['model_executed']}",
            ],
            [
                "4",
                "{}（flush：finished_ids={"
                + ", ".join(b4["finished_req_ids_riding_batch"])
                + "} 随批下发）",
                "—（0-token 批空跑，不前向）",
                "None",
                "跳过（model_output 非 None）",
                "finished 簿记冲刷，无输出",
                f"executed={b4['model_executed']}",
            ],
            [
                "5",
                "未到达（空转守卫先返回）",
                "—",
                "—",
                "—",
                "—",
                f"outputs={{}}，executed={b5['model_executed']}，executor 零调用",
            ],
        ],
    }

    out = {
        "driver": "run_m1_five_beats.py",
        "mechanism": "m1 五拍编排 EngineCore.step（vllm/v1/engine/core.py:L584-L614，impl 逐字）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch09 implementation/engine_loop.py 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "sync_mode_note": fwd,
        "config": {
            "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
            "max_model_len": cfg.model_config.max_model_len,
            "async_scheduling": cfg.scheduler_config.async_scheduling,
            "batch_queue_size": cfg.max_concurrent_batches,
        },
        "requests": {
            "req-A": {"prompt": [1, 2, 3], "max_tokens": 3, "arrives": "beat 1 前"},
            "req-B": {"prompt": [7, 8, 9, 10], "max_tokens": 2, "arrives": "beat 1 后"},
        },
        "scripted_logits_favorites": {
            "beat1": {"req-A": 7},
            "beat2": {"req-A": 8, "req-B": 6},
            "beat3": {"req-A": 9, "req-B": 6},
            "vocab": 16,
        },
        "non_block_spy": seen_non_block,
        "beats": beats,
        "beat1_event_order": b1["event_order"],
        "beat1_timeline_ms": b1["events_rel_ms"],
        "step_fn_binding": "step_fn == step（batch_queue_size=1 静态绑定，core.py:L231-L233）",
        "table": table,
    }

    dest = Path(__file__).with_name("m1_five_beats.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for b in beats:
        print(
            "beat",
            b["beat"],
            b["batch"],
            "total",
            b["total_num_scheduled_tokens"],
            "executed",
            b["model_executed"],
            "wall_ms",
            b["wall_ms"],
        )
    print("non_block spy:", seen_non_block)


if __name__ == "__main__":
    main()
