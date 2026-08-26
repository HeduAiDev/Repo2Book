# ch12 m4 旗舰 worked example 驱动：一轮重叠心跳 e2e 四拍走（theory[2]：
# prompt=2、max_tokens=2、无 spec、num_sampled_tokens_per_step=1）。
# 真实链：EngineCore → UniProcExecutor → GPUWorker → GPUModelRunner（seam 前向
# 脚本 logits，greedy argmax）；D2H 完成由 release() 显式放行（HOST SEAM——
# 真实由 copy stream 硬件推进）。
# 观测重点：①盲调度证明（schedule 时 ph>0 且上一批 D2H 事件未完成）②队列水位
# 消长 ③占位账本 ④worker 影子（token_ids_cpu 行 -1 / input_ids.gpu 回填）。
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from implementation.core import EngineCore  # noqa: E402
from implementation.request import Request, SamplingParams  # noqa: E402
from implementation.scheduler_config import SchedulerConfig  # noqa: E402
from implementation.uniproc_executor import AsyncOutputFuture  # noqa: E402
from implementation.vllm_config import VllmConfig  # noqa: E402


def one_hot_row(token_id, vocab=16):
    row = [0.0] * vocab
    row[token_id] = 1.0
    return row


cfg = VllmConfig(scheduler_config=SchedulerConfig(), max_model_len=64)
cfg.check_and_set_default_async_scheduling()
engine = EngineCore(cfg)
req = Request(
    request_id="req-0",
    prompt_token_ids=[1, 2],
    sampling_params=SamplingParams(max_tokens=2),
)
engine.scheduler.add_request(req)
runner = engine.model_executor.driver_worker.model_runner
sched = engine.scheduler

runner.enqueue_logits([{"req-0": one_hot_row(7)}, {"req-0": one_hot_row(9)}])

trace = {
    "mechanism": "m4/m5 step_with_batch_queue 两态心跳 e2e 四拍（prompt=2, max_tokens=2, 无 spec）",
    "pin": "vLLM v0.27.1 (6e448d0ea)",
    "anchor": "vllm/v1/engine/core.py:L625-L739",
    "config": {
        "batch_queue_size": engine.batch_queue_size,
        "deque_maxlen": engine.batch_queue.maxlen,
        "step_fn": engine.step_fn.__name__,
        "async_scheduling": engine.async_scheduling,
        "scheduler_cls": type(sched).__name__,
    },
    "host_seam_note": (
        "HOST SEAM：D2H 拷贝完成事件由 release() 模拟（真实在 copy stream 上硬件推进）；"
        "前向是脚本 logits 行（ch17 seam）；HostEvent/HostCopyStream 站 CUDA event/stream 位。"
        "账本数值与控制流逐字对 pin。"
    ),
    "beats": [],
}

queue_labels = []  # 与 batch_queue 同步的批标签镜像（appendleft 进 / pop 出）


class LabelledDeque(type(engine.batch_queue)):
    """记账镜像：appendleft/pop 同步记批标签（A/B/C… 按调度序），不改语义。"""

    n_ever = 0

    def appendleft(self, x):
        type(self).n_ever += 1
        queue_labels.insert(0, chr(ord("A") + self.n_ever - 1))
        super().appendleft(x)

    def pop(self):
        queue_labels.pop()
        return super().pop()


engine.batch_queue = LabelledDeque(maxlen=engine.batch_queue.maxlen)

schedule_log = []
_orig_schedule = sched.schedule


def schedule_wrapped():
    state = {
        "ph_before_schedule": req.num_output_placeholders,
        "computed_before_schedule": req.num_computed_tokens,
        "output_token_ids_before": list(req.output_token_ids),
    }
    so = _orig_schedule()
    state["scheduled"] = dict(so.num_scheduled_tokens)
    state["total"] = so.total_num_scheduled_tokens
    if req.num_output_placeholders > 0:
        # early-stop 剪枝算术（scheduler.py:L488-L502）
        state["early_stop_lhs"] = (
            req.num_computed_tokens + 2 - req.num_output_placeholders
        )
        state["early_stop_rhs"] = req.num_prompt_tokens + req.max_tokens
    schedule_log.append(state)
    return so


sched.schedule = schedule_wrapped


def queue_snapshot():
    # deque: index0=最新 appendleft、末尾=最老；镜像同步，返回 最老→最新
    return list(reversed(queue_labels))


def oldest_d2h_pending():
    if not engine.batch_queue:
        return None
    fut = engine.batch_queue[-1][0]
    if isinstance(fut, AsyncOutputFuture):
        return not fut.async_output.async_copy_ready_event.is_set()
    return "plain_future(空批/失败路径)"


def req_snapshot():
    return {
        "computed": req.num_computed_tokens,
        "ph": req.num_output_placeholders,
        "in_flight": req.num_in_flight_tokens,
        "tws": req.num_tokens_with_spec,
        "real_computed_minus_ph": req.num_computed_tokens - req.num_output_placeholders,
        "output_token_ids": list(req.output_token_ids),
    }


beat_no = 0
while engine.has_work() and beat_no < 8:
    beat_no += 1
    rec = {
        "beat": beat_no,
        "queue_before": queue_snapshot(),
        "oldest_batch_d2h_pending_before_release": oldest_d2h_pending(),
    }
    runner.release_async_copies()
    outputs, executed = engine.step_fn()
    engine.post_step(model_executed=executed)
    rec["executed"] = executed
    rec["outputs"] = (
        None if outputs is None else {
            str(ci): [
                {"request_id": o.request_id, "new_token_ids": list(o.new_token_ids),
                 "finish_reason": o.finish_reason.name if o.finish_reason else None}
                for o in eco.outputs
            ]
            for ci, eco in outputs.items()
        }
    )
    rec["queue_after"] = queue_snapshot()
    rec["req_after"] = req_snapshot()
    rec["worker_token_ids_cpu_row_prefix"] = [
        int(v) for v in runner.input_batch.token_ids_cpu[0][:6]
    ]
    rec["worker_input_ids_gpu_prefix"] = [
        int(v) for v in runner.input_ids.gpu[:4]
    ]
    rec["prev_sampled_token_ids_cached"] = (
        runner.input_batch.prev_sampled_token_ids is not None
    )
    trace["beats"].append(rec)

trace["schedule_log"] = schedule_log
trace["schedule_log_note"] = (
    "early_stop_lhs/rhs 为 schedule() 返回后口径（本拍有排入的请求占位已 +1）。"
    "拍 3 的 4≥4 即剪枝判定现场：该拍请求被 continue 跳过、占位不再 +1，"
    "拍内判定用的正是 ph=1（拍 2 遗留）与 computed=3。"
)
trace["cache_blocks_calls"] = [
    list(c) for c in sched.kv_cache_manager.cache_blocks_calls
]
trace["terminal_state"] = {
    "has_work": engine.has_work(),
    "queue_len": len(engine.batch_queue),
    "req_finished": req.is_finished(),
    "finish_reason": req.get_finished_reason().name if req.is_finished() else None,
    "final_computed": req.num_computed_tokens,
    "final_ph": req.num_output_placeholders,
    "invariant_final_real_computed": (
        req.num_computed_tokens - req.num_output_placeholders
    ),
}
trace["invariant_check_per_beat"] = [
    {"beat": b["beat"], "computed": b["req_after"]["computed"],
     "ph": b["req_after"]["ph"], "real": b["req_after"]["real_computed_minus_ph"]}
    for b in trace["beats"]
]
trace["blind_schedule_proof"] = {
    "beat2_ph_before_schedule": schedule_log[1]["ph_before_schedule"] if len(schedule_log) > 1 else None,
    "beat2_output_tokens_before": schedule_log[1]["output_token_ids_before"] if len(schedule_log) > 1 else None,
    "beat2_note": (
        "拍 2 schedule() 时 ph=1>0 且 output_token_ids 为空——批 A 的 t7 尚未到账，"
        "调度器凭占位数盲排位置 3（同步版此处追赶公式=0、无 token 可排）"
    ),
    "beat2_d2h_pending_at_beat_start": trace["beats"][1]["oldest_batch_d2h_pending_before_release"],
}

out = os.path.join(os.path.dirname(__file__), "m4_e2e_heartbeat.json")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    json.dump(trace, f, ensure_ascii=False, indent=1)
print("wrote", out)
