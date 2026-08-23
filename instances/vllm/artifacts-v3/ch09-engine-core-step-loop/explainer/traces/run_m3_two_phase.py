"""Driver for m3 (execute_model 两段式契约 + AsyncOutputFuture 只等 D2H) — host
run against the ch09 subtract-only engine companion (pin vLLM v0.27.1).

Three faces of the same contract, all on the real code:
  A. worker face  — execute_model returns None and stashes the 10-field
     ExecuteModelState; a second execute_model without an intervening
     sample_tokens raises the verbatim "State error" guard; sample_tokens
     unpacks, clears the state, greedily samples; a fresh execute_model
     afterwards works again.
  B. executor face — UniProcExecutor.execute_model(non_block=True) returns a
     done Future holding None; the misuse surfaces in-line through the
     executor (uniproc_executor.py:L117-L120 early-failure face).
  C. async face    — with async_scheduling=True, executor.sample_tokens
     wraps the output in AsyncOutputFuture; result() blocks on the D2H copy
     event (not on any computation) until the copy completes: held pending
     for 0.25s, unblocked within ~1ms of the event, second result() instant.
"""
import json
import sys
import threading
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


def logits_row(favorite, vocab=8, value=5.0):
    row = [0.0] * vocab
    row[favorite] = value
    return row


def sched_output(req_ids=("req-1",), tokens=(3,)):
    new_reqs = [
        el.NewRequestData(
            req_id=rid,
            prompt_token_ids=[1] * t,
            mm_features=[],
            sampling_params=None,
            pooling_params=None,
            block_ids=(),
            num_computed_tokens=0,
            lora_request=None,
        )
        for rid, t in zip(req_ids, tokens)
    ]
    return el.SchedulerOutput(
        scheduled_new_reqs=new_reqs,
        scheduled_cached_reqs=el.CachedRequestData(
            req_ids=[],
            resumed_req_ids=set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[],
            num_computed_tokens=[],
            num_output_tokens=[],
        ),
        num_scheduled_tokens=dict(zip(req_ids, tokens)),
        total_num_scheduled_tokens=sum(tokens),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )


def main():
    out = {
        "driver": "run_m3_two_phase.py",
        "mechanism": "m3 execute_model 两段式契约（gpu_model_runner.py:L4166-L4589 + uniproc_executor.py:L26-L131，impl 逐字）",
        "pin": "vLLM v0.27.1 (6e448d0ea)",
        "impl": "ch09 implementation/engine_loop.py 只做减法精简版（# SOURCE 锚 v0.27.1）",
        "parts": {},
    }

    # ---- A. worker face: stash → State error → unpack & clear → reusable ----
    runner = el.GPUModelRunner(mk_config())
    runner.enqueue_logits([{"req-1": logits_row(1)}])
    ret_a1 = runner.execute_model(sched_output())
    st = runner.execute_model_state
    part_a = {
        "execute_returns": ret_a1,  # None
        "state_fields": list(el.ExecuteModelState._fields),
        "state_num_fields": len(el.ExecuteModelState._fields),
        "stashed_logits_shape": list(st.logits.shape),
        "stashed_scheduler_output": dict(st.scheduler_output.num_scheduled_tokens),
        "state_error_message": None,
    }
    try:
        runner.execute_model(sched_output())
    except RuntimeError as e:
        part_a["state_error_message"] = str(e)
    out_a = runner.sample_tokens(None)
    part_a.update(
        {
            "state_after_sample": runner.execute_model_state,  # None (cleared)
            "sampled_token_ids": out_a.sampled_token_ids,
            "sampled_req_ids": list(out_a.req_ids),
        }
    )
    # consumed → a fresh execute_model is legal again (guard satisfied).
    # Realistic shape: the NEXT beat's batch carries a NEW request (req-2)
    # alongside req-1 still resident in the persistent batch (decoding) —
    # a request is "new" exactly once; later beats send it as cached.
    runner.enqueue_logits([{"req-1": logits_row(1), "req-2": logits_row(2)}])
    part_a["execute_after_consume_returns"] = runner.execute_model(
        sched_output(("req-2",), (2,))
    )
    out_a2 = runner.sample_tokens(None)
    part_a["sampled_token_ids_second"] = out_a2.sampled_token_ids
    part_a["second_req_ids"] = list(out_a2.req_ids)
    out["parts"]["A_worker_face"] = part_a

    # ---- B. executor face: non_block=True Future + in-line State error ----
    ex = el.UniProcExecutor(mk_config())
    ex.driver_worker.enqueue_logits([{"req-1": logits_row(1)}])
    fut = ex.execute_model(sched_output(), non_block=True)
    part_b = {
        "first_call_future_done": fut.done(),
        "first_call_result": fut.result(),  # None
        "in_line_state_error": None,
    }
    try:
        ex.execute_model(sched_output(), non_block=True)
    except RuntimeError as e:
        part_b["in_line_state_error"] = str(e)
    out["parts"]["B_executor_face"] = part_b

    # ---- C. async face: AsyncOutputFuture waits only the D2H event ----
    cfg_async = mk_config(scheduler_config={"async_scheduling": True})
    ex2 = el.UniProcExecutor(cfg_async)
    runner2 = ex2.driver_worker
    favorite = 4
    runner2.enqueue_logits([{"req-1": logits_row(favorite)}])
    runner2.execute_model(sched_output())
    t_fut = time.perf_counter()
    fut2 = ex2.sample_tokens(None, non_block=True)
    fut_build_ms = round((time.perf_counter() - t_fut) * 1000, 3)
    done_right_after_build = fut2.done()  # False: D2H not yet complete

    holder = {}
    t = threading.Thread(target=lambda: holder.setdefault("v", fut2.result()), daemon=True)
    t.start()
    pending_window_s = 0.25
    time.sleep(pending_window_s)
    still_blocked = "v" not in holder  # result() parked on the D2H event
    t_release = time.perf_counter()
    runner2.release_async_copies()  # the D2H copy completes
    t.join(10)
    unblock_ms = round((time.perf_counter() - t_release) * 1000, 3)
    t_second = time.perf_counter()
    again = fut2.result()
    second_result_ms = round((time.perf_counter() - t_second) * 1000, 3)
    part_c = {
        "future_type": type(fut2).__name__,
        "done_right_after_build": done_right_after_build,
        "fut_build_ms": fut_build_ms,
        "d2h_pending_window_s": pending_window_s,
        "result_still_blocked_after_window": still_blocked,
        "result_unblocked_ms_after_event": unblock_ms,
        "second_result_ms": second_result_ms,
        "second_result_instant": second_result_ms < 1.0,
        "sampled_token_ids": again.sampled_token_ids,
        "favorite": favorite,
    }
    out["parts"]["C_async_d2h_face"] = part_c

    # ---- table rows (numbers read off the run) ----
    out["table"] = {
        "columns": [
            "阶段",
            "动作",
            "execute_model_state",
            "返回",
            "判定",
        ],
        "rows": [
            [
                "② 第一段",
                f"execute_model(批{{'req-1': {part_a['stashed_scheduler_output']['req-1']}}})",
                f"暂存 {part_a['state_num_fields']} 字段（logits {part_a['stashed_logits_shape'][0]}×{part_a['stashed_logits_shape'][1]}）",
                "None",
                "前向算完、采样欠着",
            ],
            [
                "误用防御",
                "再来一次 execute_model",
                "非 None（上一拍未消费）",
                "RuntimeError：State error（原文见 trace）",
                "worker 自己炸，不产出错数据",
            ],
            [
                "④ 第二段",
                "sample_tokens(None)",
                "解包→清 None",
                f"sampled={part_a['sampled_token_ids']}（argmax=favorite {part_a['sampled_token_ids'][0][0]}）",
                "掩码位→贪心采样，态已清",
            ],
            [
                "再次 ②",
                "消费后再 execute_model（批含新 req-2）",
                "重新暂存（新批）",
                "None",
                f"合法：sampled={part_a['sampled_token_ids_second']}（req-1 续 decode + req-2 新入批）",
            ],
            [
                "④ 异步半边",
                "executor.sample_tokens(non_block=True)",
                "—",
                "AsyncOutputFuture（done=False）",
                "result() 只等 D2H 事件，不等计算",
            ],
            [
                "D2H 完成",
                f"挂起 {part_c['d2h_pending_window_s']}s 后事件置位",
                "—",
                f"置位后 {part_c['result_unblocked_ms_after_event']}ms 返回；二次 result() {part_c['second_result_ms']}ms",
                f"阻塞期间无返回={part_c['result_still_blocked_after_window']}；采样={part_c['sampled_token_ids']}",
            ],
        ],
    }

    dest = Path(__file__).with_name("m3_two_phase.json")
    with open(dest, "w", encoding="utf-8", newline="\n") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", dest)
    print("A:", part_a["state_num_fields"], "fields; err:", part_a["state_error_message"][:60])
    print("B:", part_b["first_call_future_done"], "| inline err ok:", part_b["in_line_state_error"] is not None)
    print("C:", part_c)


if __name__ == "__main__":
    main()
