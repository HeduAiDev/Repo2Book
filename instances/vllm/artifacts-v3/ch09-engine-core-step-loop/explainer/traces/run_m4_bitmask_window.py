"""Driver for m4 (grammar bitmask 窗口, F6 埋点) — host run against the ch09
subtract-only engine companion (pin vLLM v0.27.1 / 6e448d0ea).

Three scenarios, all on the real EngineCore.step / get_grammar_bitmask /
apply_grammar_bitmask code (the bitmask ROW itself crosses the ch30 seam —
tests script it, exactly like the real FSM compiler would supply it):

  main     structured "g", prompt 3 tokens, budget 8192 → prefill completes
           in ONE beat. The scripted logits row always favors token 5
           (logit 9.0; token 4 has 7.0, token 1 has 3.0); the scripted
           bitmask word only allows {1, 4}. Beats: prefill(mask→[4]) →
           decode [4] → decode [4] → LENGTH at 3/3 → flush → idle guard.
  probe    budget 2 < prompt 3 → the first beat is a NON-FINAL prefill
           chunk: ③ excludes the request (scheduler.py:L1654-L1659), the
           grammar manager is never consulted, no partial-prefill output.
  control  identical logits row, unstructured request → greedy picks the
           favorite 5 — the mask is the only variable.

Also records the main-beat-1 event timeline:
execute_model < get_grammar_bitmask < apply_bitmask < greedy_sample.

KNOWN SEAM GAP (flagged, not papered over): the companion's InputBatch seam
never refreshes num_computed_tokens for cached (already-admitted) requests,
so a request whose prefill spans MULTIPLE beats would keep its sampling row
cleared forever. The probe therefore stops after its first (chunk) beat —
matching tests/test_engine_loop.py::test_prefill_chunk_rows_excluded — and
multi-beat prefill completion is ch10's territory anyway (dossier boundary).
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

VOCAB = 8
ROW = [0.0, 3.0, 0.0, 0.0, 7.0, 9.0, 0.0, 0.0]  # favorite=5 @9.0; 4 @7.0; 1 @3.0
MASK_WORD = 0b00010010  # bits 1 and 4 set → allowed set {1, 4}


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


def core_request(request_id, token_ids, max_tokens, structured=False):
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
        client_index=0,
    )


def add(ec, request_id, token_ids, max_tokens, structured=False):
    req, wave = ec.preprocess_add_request(
        core_request(request_id, token_ids, max_tokens, structured)
    )
    ec.add_request(req, wave)
    return req


def beat_record(ec, outputs, executed, t0, n_sched, n_run, n_mgr):
    sched_new = ec.scheduler.trace[n_sched:]
    run_new = ec.model_executor.driver_worker.trace[n_run:]
    mgr_new = list(ec.structured_output_manager.trace[n_mgr:])
    events = sorted(
        [(t, f"sched:{n}") for t, n in sched_new]
        + [(t, f"runner:{n}") for t, n in run_new]
    )
    t0e = events[0][0] if events else t0
    last = ec.scheduler.last_output
    req_g = ec.scheduler.requests.get("g")
    req_p = ec.scheduler.requests.get("p")
    rec = {
        "event_order": [n for _, n in events],
        "timeline_rel_ms": [
            {"event": n, "rel_ms": round((t - t0e) / 1e6, 3)} for t, n in events
        ],
        "manager_events": mgr_new,
        "grammar_bitmask_called": "grammar_bitmask" in mgr_new,
        "wall_ms": round((time.perf_counter() - t0) * 1000, 3),
        "batch": dict(last.num_scheduled_tokens),
        "total": last.total_num_scheduled_tokens,
        "has_structured_output_requests": last.has_structured_output_requests,
        "g_is_prefill_chunk": (None if req_g is None else req_g.is_prefill_chunk),
        "g_num_computed_tokens": (None if req_g is None else req_g.num_computed_tokens),
        "g_num_output_tokens": (None if req_g is None else req_g.num_output_tokens),
        "p_is_prefill_chunk": (None if req_p is None else req_p.is_prefill_chunk),
        "p_num_computed_tokens": (None if req_p is None else req_p.num_computed_tokens),
        "outputs": [
            {
                "request_id": o.request_id,
                "new_token_ids": list(o.new_token_ids),
                "finish_reason": (
                    None if o.finish_reason is None else str(o.finish_reason)
                ),
            }
            for eco in outputs.values()
            for o in eco.outputs
        ],
        "model_executed": executed,
    }
    rels = {e["event"]: e["rel_ms"] for e in rec["timeline_rel_ms"]}
    if "runner:execute_model" in rels and "sched:get_grammar_bitmask" in rels:
        rec["gap_launch_to_bitmask_ms"] = round(
            rels["sched:get_grammar_bitmask"] - rels["runner:execute_model"], 3
        )
    if "sched:get_grammar_bitmask" in rels and "runner:apply_bitmask" in rels:
        rec["gap_bitmask_to_apply_ms"] = round(
            rels["runner:apply_bitmask"] - rels["sched:get_grammar_bitmask"], 3
        )
    return rec


def do_beat(ec, label, note):
    n_sched = len(ec.scheduler.trace)
    n_run = len(ec.model_executor.driver_worker.trace)
    n_mgr = len(ec.structured_output_manager.trace)
    t0 = time.perf_counter()
    outputs, executed = ec.step()
    rec = {"beat": label, "note": note}
    rec.update(beat_record(ec, outputs, executed, t0, n_sched, n_run, n_mgr))
    return rec


def main():
    out = {
        "driver": "run_m4_bitmask_window.py",
        "mechanism": "m4 grammar bitmask 窗口（core.py:L596-L604 + scheduler.py:L1646-L1668 + gpu_model_runner.py:L4582-L4586，impl 逐字）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch09 implementation/engine_loop.py 只做减法精简版（位掩码行经 ch30 边界 seam 注入；apply_grammar_bitmask 与 greedy argmax 为真码）",
        "mask_semantics": "位清零的 token 在采样前置 -inf（xgrammar 内核契约；host seam 同语义，容器内真内核接管）",
        "scripted_logits_row": list(ROW),
        "scripted_logits_note": "token 5 @9.0 最高、4 @7.0、1 @3.0，vocab=8",
        "scripted_bitmask_word0_binary": "0b00010010",
        "scripted_bitmask_allowed_set": [1, 4],
        "sync_mode_note": "同步教学版里 seam 前向（5ms 建模一步）在 execute_model 内同步完成，③ 因此排在 ② 返回之后；真实引擎 ② 发起即返回、③ 在前向进行中计算——本 trace 证明的是顺序约束（②发起 < ③ < ④ 应用 < 采样）与掩码压过 argmax 的数学",
        "seam_gap_note": "已知 seam 缺口（如实标注）：精简版 InputBatch seam 不为已入批（cached）请求刷新 num_computed_tokens，prefill 跨多拍的请求采样行会永远清空（tests 的多拍用例也只断言首拍）；排除探针因此止步首拍——多拍 prefill 的续拍收官本就归 ch10（dossier 域边界），本章场景取一拍收官的 prefill",
    }

    # ---- main scenario: single-beat prefill, masked all the way ----
    cfg = mk_config()
    ec = el.EngineCore(cfg, el.UniProcExecutor, False)
    add(ec, "g", (1, 2, 3), max_tokens=3, structured=True)
    ec.enqueue_forward_logits([{"g": list(ROW)}] * 3)
    # ③ computes on every beat the request is scheduled and not a chunk:
    # prefill(final) + decode + decode = 3 beats.
    for _ in range(3):
        ec.enqueue_grammar_bitmask([[MASK_WORD]])

    beats = []
    beats.append(do_beat(ec, 1, "一拍收官的 prefill（3/3）：③ 算掩码、④ 应用后采样"))
    beats.append(do_beat(ec, 2, "decode 拍：同一掩码继续约束"))
    beats.append(do_beat(ec, 3, "decode 拍：3/3 到顶 → LENGTH"))
    beats.append(do_beat(ec, 4, "flush 拍：finished_ids={g} 随 0-token 批下发，无采样"))
    beats.append(do_beat(ec, 5, "空转守卫拍"))
    main_beats = beats

    # ---- probe scenario: non-final prefill chunk → ③ excludes the request ----
    cfg2 = mk_config(scheduler_config={"max_num_batched_tokens": 2})
    ec2 = el.EngineCore(cfg2, el.UniProcExecutor, False)
    add(ec2, "p", (1, 2, 3), max_tokens=3, structured=True)
    ec2.enqueue_forward_logits([{"p": list(ROW)}])
    probe = do_beat(ec2, "probe", "预算 2 < prompt 3：首拍是非末块 prefill chunk（止步首拍，见 seam_gap_note）")
    probe["p_status_after"] = ec2.scheduler.requests["p"].status.name

    # ---- control: identical row, no structured marker → favorite wins ----
    ec3 = el.EngineCore(mk_config(), el.UniProcExecutor, False)
    add(ec3, "ctrl", (1, 2), max_tokens=1)
    ec3.enqueue_forward_logits([{"ctrl": list(ROW)}])
    ctrl_out, _ = ec3.step()
    control = {
        "request": "ctrl（同 logits 行、无结构化标记）",
        "batch": dict(ec3.scheduler.last_output.num_scheduled_tokens),
        "sampled": list(ctrl_out[0].outputs[0].new_token_ids),
        "manager_trace": list(ec3.structured_output_manager.trace),
    }

    b1, b2, b3 = main_beats[0], main_beats[1], main_beats[2]
    table = {
        "columns": [
            "场景/拍",
            "① 批",
            "is_prefill_chunk",
            "③ 掩码",
            "④ 采样（贪婪想选 5@9.0）",
            "输出/判定",
        ],
        "rows": [
            [
                "探针（预算 2）",
                "{p: 2}",
                "True（2/3 未完）",
                "排除（grammar manager 零调用）",
                "采样行被清空（无采样行）",
                "无输出（部分 prefill 不出活）",
            ],
            [
                "主场景拍 1",
                "{g: 3}（一拍收官）",
                "False（3/3）",
                "算出·允许集 {1, 4}",
                "5 被禁（置 -inf）→ 选 4（7.0）",
                f"client0 收 {b1['outputs'][0]['new_token_ids']}（首 token）",
            ],
            [
                "主场景拍 2",
                "{g: 1}",
                "False",
                "算出·允许集 {1, 4}",
                "选 4",
                f"{b2['outputs'][0]['new_token_ids']}",
            ],
            [
                "主场景拍 3",
                "{g: 1}",
                "False",
                "算出·允许集 {1, 4}",
                "选 4",
                f"{b3['outputs'][0]['new_token_ids']}→LENGTH（3/3）",
            ],
            [
                "对照",
                "{ctrl: 2}（无结构化标记）",
                "False",
                "None（快速返回）",
                f"无掩码 → 贪婪选 favorite → {control['sampled']}",
                "同一行 logits，掩码是唯一变量",
            ],
        ],
    }

    out.update(
        {
            "config": {
                "main_max_num_batched_tokens": cfg.scheduler_config.max_num_batched_tokens,
                "probe_max_num_batched_tokens": cfg2.scheduler_config.max_num_batched_tokens,
                "max_model_len": cfg.model_config.max_model_len,
            },
            "request": {"req_id": "g", "prompt_len": 3, "max_tokens": 3, "structured": True},
            "main_beats": main_beats,
            "beat1_event_order": b1["event_order"],
            "beat1_timeline_ms": b1["timeline_rel_ms"],
            "beat1_window": {
                "gap_launch_to_bitmask_ms": b1.get("gap_launch_to_bitmask_ms"),
                "gap_bitmask_to_apply_ms": b1.get("gap_bitmask_to_apply_ms"),
            },
            "probe_beat": probe,
            "control_unmasked": control,
            "table": table,
        }
    )

    dest = Path(__file__).with_name("m4_bitmask_window.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    for b in main_beats:
        print(
            "main beat",
            b["beat"],
            b["batch"],
            "mask" if b["grammar_bitmask_called"] else "no-mask",
            "out",
            [o["new_token_ids"] for o in b["outputs"]],
            "wall",
            b["wall_ms"],
        )
    print("probe:", probe["batch"], "chunk:", probe["p_is_prefill_chunk"],
          "computed:", probe["p_num_computed_tokens"],
          "mgr:", probe["manager_events"], "outs:", probe["outputs"])
    print("control sampled:", control["sampled"])
    print("beat1 window:", out["beat1_window"], "| order:", b1["event_order"])


if __name__ == "__main__":
    main()
