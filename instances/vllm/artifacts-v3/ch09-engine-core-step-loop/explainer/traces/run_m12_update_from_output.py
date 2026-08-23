"""Driver for m12 (⑤ update_from_output 逐请求热循环) — host run against the
ch09 subtract-only engine companion (pin vLLM v0.27.1 / 6e448d0ea).

Three requests, two clients, one mid-execution abort — every line of the ⑤
hot loop exercised on the real code:

  a  client 0, prompt 1 token, max_tokens 2 → finishes LENGTH at beat 2
  b  client 1, prompt 1 token, max_tokens 4 → finishes LENGTH at beat 4
  c  client 0, prompt 1 token, max_tokens 8 → ABORT lands via aborts_queue
     between ④ and ⑤ of beat 2 (the eager channel): c's row WAS sampled by
     the worker, but by update time the request is already freed → the real
     skip branch (scheduler.py:L1747-L1755 "aborted while the model is
     executing it") → no output, no leak.

Also captured per beat: the sampling-row positioning basis
(input_batch.req_ids == req_id_to_index), the client bucketing of outputs,
the finished_req_ids set that rides the NEXT beat's SchedulerOutput (worker
cache eviction protocol), and the merged single-batch abort call.
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


def core_request(request_id, token_ids, max_tokens, client_index):
    sp = el.SamplingParams(max_tokens=max_tokens)
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


def add(ec, request_id, token_ids, max_tokens, client_index):
    req, wave = ec.preprocess_add_request(
        core_request(request_id, token_ids, max_tokens, client_index)
    )
    ec.add_request(req, wave)
    return req


def logits_row(favorite, vocab=16, value=5.0):
    row = [0.0] * vocab
    row[favorite] = value
    return row


def main():
    cfg = mk_config()
    ec = el.EngineCore(cfg, el.UniProcExecutor, False)
    add(ec, "a", (1,), max_tokens=2, client_index=0)  # favorite 7
    add(ec, "b", (2,), max_tokens=4, client_index=1)  # favorite 6
    add(ec, "c", (3,), max_tokens=8, client_index=0)  # favorite 5

    ec.enqueue_forward_logits(
        [
            {"a": logits_row(7), "b": logits_row(6), "c": logits_row(5)},
            {"a": logits_row(8), "b": logits_row(6), "c": logits_row(5)},
            {"b": logits_row(6)},
            {"b": logits_row(6)},
        ]
    )

    beats = []

    def do_beat(label, note, abort_ids=None):
        if abort_ids:
            # the eager channel: lands between ④ and ⑤ (core.py:L606-L608)
            ec.aborts_queue.put_nowait(list(abort_ids))
        n_sched = len(ec.scheduler.trace)
        n_run = len(ec.model_executor.driver_worker.trace)
        n_finish = len(ec.scheduler.finish_calls)
        t0 = time.perf_counter()
        outputs, executed = ec.step()
        wall_ms = round((time.perf_counter() - t0) * 1000, 3)
        last = ec.scheduler.last_output
        beats.append(
            {
                "beat": label,
                "note": note,
                "abort_enqueued_before": abort_ids or [],
                "new_finish_calls": ec.scheduler.finish_calls[n_finish:],
                "batch": dict(last.num_scheduled_tokens),
                "total": last.total_num_scheduled_tokens,
                "finished_req_ids_riding_this_batch": sorted(last.finished_req_ids),
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
                "sampling_row_basis_input_batch_req_ids": list(
                    ec.model_executor.driver_worker.input_batch.req_ids
                ),
                "requests_after": {
                    rid: {
                        "status": r.status.name,
                        "num_output_tokens": r.num_output_tokens,
                        "client_index": r.client_index,
                    }
                    for rid, r in ec.scheduler.requests.items()
                },
                "model_executed": executed,
                "wall_ms": wall_ms,
                "sched_events": [n for _, n in ec.scheduler.trace[n_sched:]],
                "runner_events": [n for _, n in ec.model_executor.driver_worker.trace[n_run:]],
            }
        )

    do_beat(1, "三请求同拍 prefill：⑤ 逐请求定位行→append→判停，按 client 分桶")
    do_beat(2, "执行中 abort：④ 已把 c 的行采出来，⑤ 前 aborts 落地→c 跳过；a 到顶 LENGTH", abort_ids=["c"])
    do_beat(3, "finished_ids={a,c} 随本拍批下发 worker 清缓存；b 独舞")
    do_beat(4, "b 4/4 到顶 → LENGTH")
    do_beat(5, "flush 拍：finished_ids={b} 随 0-token 批下发")
    do_beat(6, "空转守卫拍")

    b = {x["beat"]: x for x in beats}

    def out_tokens(beat, rid):
        for msgs in beat["outputs_by_client"].values():
            for o in msgs:
                if o["request_id"] == rid:
                    return o["new_token_ids"], o["finish_reason"]
        return None, None

    ta, fa = out_tokens(b[2], "a")
    tb, fb = out_tokens(b[4], "b")
    table = {
        "columns": [
            "拍",
            "⑤ 输入批",
            "逐请求动作",
            "状态转移/回收",
            "分桶输出（client0 | client1）",
            "finished_ids 随下拍批",
        ],
        "rows": [
            [
                "1",
                "{a:1, b:1, c:1}",
                "3 请求定位行→append→判停（a→7, b→6, c→5）",
                "3×RUNNING 续跑",
                "a[7], c[5] | b[6]",
                "∅",
            ],
            [
                "2",
                "{a:1, b:1, c:1}（c 的行已被 ④ 采样）",
                "a: append→LENGTH→释放；b: append；c: 已 abort→跳过（行丢弃）",
                "a→LENGTH(2/2)、c→ABORTED；b 续",
                f"a{ta}+{fa} | b[6]",
                "{a, c}",
            ],
            [
                "3",
                "{b:1}（批已剔除 a/c）",
                "b: 定位行→append",
                "b 续（3/4）",
                "∅ | b[6]",
                "∅",
            ],
            [
                "4",
                "{b:1}",
                "b: append→LENGTH→释放",
                f"b→{fb}(4/4)",
                f"∅ | b{tb}+{fb}",
                "{b}",
            ],
            [
                "5",
                "{}（flush）",
                "无请求可记账，冲刷 finished 簿记",
                "—",
                "∅ | ∅",
                "∅",
            ],
            [
                "6",
                "未到达（空转守卫）",
                "—",
                "—",
                "—",
                "—",
            ],
        ],
    }

    out = {
        "driver": "run_m12_update_from_output.py",
        "mechanism": "m12 ⑤ update_from_output 逐请求热循环（scheduler.py:L1670-L1762 + L2014-L2029，impl 逐字骨架）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch09 implementation/engine_loop.py 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "config": {
            "max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
            "max_model_len": cfg.model_config.max_model_len,
            "vocab": 16,
        },
        "requests": {
            "a": {"prompt": [1], "max_tokens": 2, "client_index": 0, "favorite": 7},
            "b": {"prompt": [2], "max_tokens": 4, "client_index": 1, "favorite": 6},
            "c": {"prompt": [3], "max_tokens": 8, "client_index": 0, "favorite": 5},
        },
        "abort_schedule": "beat 2 前 aborts_queue.put(['c'])——急切通道在 ④⑤ 之间落地（core.py:L606-L608）",
        "woosuk_note": "scheduler.py:L1728-L1730 woosuk 自注：len(num_scheduled_tokens) 可上千，热循环是性能瓶颈，循环内避免昂贵操作",
        "beats": beats,
        "table": table,
    }

    dest = Path(__file__).with_name("m12_update_from_output.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for x in beats:
        print(
            "beat",
            x["beat"],
            x["batch"],
            "| outs",
            {c: [(o["request_id"], o["new_token_ids"]) for o in m] for c, m in x["outputs_by_client"].items()},
            "| finish_calls",
            x["new_finish_calls"],
            "| riding",
            x["finished_req_ids_riding_this_batch"],
        )


if __name__ == "__main__":
    main()
