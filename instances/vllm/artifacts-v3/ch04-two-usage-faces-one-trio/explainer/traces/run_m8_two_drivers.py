"""m8 driver — 两种驱动同一时刻的画面: 在线事件循环 vs 离线调用方线程裸循环。

A. 在线面 (AsyncLLM): 2 个并发请求 (chat-A/chat-B, DELTA) 各自一个 generate 消费者
   + 1 个 output_handler 后台任务; 引擎每拍一个 EngineCoreOutputs 整批喂两个信箱。
   期间用 asyncio.all_tasks() 抓任务清单快照。
B. 离线面 (LLM/LLMEngine): 三段实测——
   B1. step() 空转阻塞演示 (主线程卡在 outputs_queue.get(), core_client.py:L872-L882);
   B2. 裸 while has_unfinished: step() 循环, 请求 "1" 先完成、"0" 后完成 → sorted 还原;
       逐拍记录 step() 返回条数 (FINAL_ONLY: 中间拍返回 0 条);
   B3. 事务性批量提交: 第二个 add 失败 → 已加的 0 号双侧 abort, 消息序 [ADD, ABORT]。"""

import asyncio
import importlib.util
import json
import queue
import sys
import threading
import time
from pathlib import Path

_IMPL = Path(__file__).resolve().parent.parent.parent / "implementation" / "engine_faces.py"
_spec = importlib.util.spec_from_file_location("engine_faces", _IMPL)
ef = importlib.util.module_from_spec(_spec)
sys.modules["engine_faces"] = ef
_spec.loader.exec_module(ef)


def prompt(token_ids):
    return {"type": "token_ids", "prompt_token_ids": list(token_ids)}


class _EngineArgsStub:
    disable_log_stats = True

    def create_engine_config(self, usage_context=None):
        return ef.VllmConfig()


# ==================================================================
# A. 在线面: 事件循环驱动
# ==================================================================


async def online_face(trace):
    allm = ef.AsyncLLM(ef.VllmConfig(), ef.Executor, log_stats=False)
    engine = allm.engine_core.engine_core
    params = ef.SamplingParams(
        n=1, max_tokens=2, output_kind=ef.RequestOutputKind.DELTA
    )
    collected = {"chat-A": [], "chat-B": []}

    async def consumer_a():
        async for out in allm.generate(prompt([1, 2, 3]), params, "chat-A"):
            collected["chat-A"].append(out)

    async def consumer_b():
        async for out in allm.generate(prompt([4, 5]), params, "chat-B"):
            collected["chat-B"].append(out)

    ta = asyncio.ensure_future(consumer_a())
    tb = asyncio.ensure_future(consumer_b())

    deadline = time.monotonic() + 5
    while len(allm.output_processor.request_states) < 2 and time.monotonic() < deadline:
        await asyncio.sleep(0.001)

    # 任务清单快照: 事件循环里此刻都有谁
    def classify(t):
        try:
            qn = t.get_coro().__qualname__
        except AttributeError:
            qn = "?"
        if "output_handler" in qn:
            return "output_handler"
        if "consumer" in qn:
            return "generate_consumer"
        return f"other:{qn}"

    counts = {}
    for t in asyncio.all_tasks():
        c = classify(t)
        counts[c] = counts.get(c, 0) + 1
    trace["online"]["task_inventory_after_add"] = counts
    trace["online"]["chunk_size_default"] = ef.envs.VLLM_V1_OUTPUT_PROC_CHUNK_SIZE

    # 拍 1: 一个 EngineCoreOutputs 整批带两个请求的中间 token
    ridA, ridB = sorted(allm.output_processor.request_states)
    engine.emit_step_outputs(
        [
            (
                0,
                ef.EngineCoreOutputs(
                    outputs=[
                        ef.EngineCoreOutput(ridA, [101], None),
                        ef.EngineCoreOutput(ridB, [111], None),
                    ]
                ),
            )
        ]
    )
    deadline = time.monotonic() + 5
    while (
        len(collected["chat-A"]) < 1 or len(collected["chat-B"]) < 1
    ) and time.monotonic() < deadline:
        await asyncio.sleep(0.001)
    trace["online"]["beat_1"] = {
        "engine_output": "1 条 EngineCoreOutputs 整批 = [chat-A 内部id: token [101], chat-B 内部id: token [111]]",
        "chatA_yields": [list(o.outputs[0].token_ids) for o in collected["chat-A"]],
        "chatB_yields": [list(o.outputs[0].token_ids) for o in collected["chat-B"]],
        "who_moved": "output_handler 拉整批 → process_outputs demux → 两个信箱各 put 一封",
    }

    # 拍 2: 终拍
    engine.emit_step_outputs(
        [
            (
                0,
                ef.EngineCoreOutputs(
                    outputs=[
                        ef.EngineCoreOutput(ridA, [102], ef.FinishReason.LENGTH),
                        ef.EngineCoreOutput(ridB, [112], ef.FinishReason.LENGTH),
                    ]
                ),
            )
        ]
    )
    await asyncio.wait_for(ta, 5)
    await asyncio.wait_for(tb, 5)
    trace["online"]["beat_2_final"] = {
        "chatA_yields": [list(o.outputs[0].token_ids) for o in collected["chat-A"]],
        "chatB_yields": [list(o.outputs[0].token_ids) for o in collected["chat-B"]],
        "all_finished": [
            collected["chat-A"][-1].finished,
            collected["chat-B"][-1].finished,
        ],
        "all_ids_external": [
            sorted({o.request_id for o in collected[k]})[0] for k in ("chat-A", "chat-B")
        ],
        "ledgers_empty": {
            "request_states": len(allm.output_processor.request_states),
            "engine_requests": len(engine.requests),
        },
    }
    trace["online"]["driver_code"] = {
        "consume_loop": "out = q.get_nowait() or await q.get() (async_llm.py:L599, 每请求一个协程)",
        "handler": "单个 output_handler 后台任务 (async_llm.py:L657-L727)",
    }


# ==================================================================
# B. 离线面: 调用方线程驱动
# ==================================================================


def offline_face(trace):
    off = trace["offline"]

    # ---- B1: step() 空转阻塞演示 ----
    llm = ef.LLM(_EngineArgsStub())
    eng = llm.llm_engine.engine_core.engine_core
    # 离线面入口 _add_request 会强制 FINAL_ONLY (offline_utils.py:L559-L561)
    # —— 直接调 llm_engine.add_request 时按真实入口补上这枚章。
    params = ef.SamplingParams(
        n=1, max_tokens=2, output_kind=ef.RequestOutputKind.FINAL_ONLY
    )
    rid = llm.llm_engine.add_request("0", prompt([7]), params)
    box = {}

    def call_step():
        box["returns"] = llm.llm_engine.step()

    worker = threading.Thread(target=call_step, name="step_caller")
    worker.start()
    time.sleep(0.05)
    blocked = worker.is_alive()
    threads_during_block = sorted(t.name for t in threading.enumerate())
    eng.emit_step_outputs(
        [
            (
                0,
                ef.EngineCoreOutputs(
                    outputs=[ef.EngineCoreOutput(rid, [101], None)]
                ),
            )
        ]
    )
    worker.join(timeout=5)
    off["b1_step_blocks_until_engine_beat"] = {
        "blocked_after_50ms": blocked,
        "threads_during_block": threads_during_block,
        "step_return_len_intermediate": len(box["returns"]),
        "verdict": "FINAL_ONLY: 中间拍 step() 返回 0 条 (make_request_output 早退 None)",
        "note": "真实系统此处另有常驻 EngineCoreOutputQueueThread 守护线程喂 outputs_queue (core_client.py:L862-L867, Thread 名即 EngineCoreOutputQueueThread; companion 已删) 与独立引擎进程",
    }

    # 终拍: 再来一步拿到终帧
    worker2 = threading.Thread(
        target=lambda: box.__setitem__("final", llm.llm_engine.step())
    )
    worker2.start()
    time.sleep(0.02)
    eng.emit_step_outputs(
        [
            (
                0,
                ef.EngineCoreOutputs(
                    outputs=[ef.EngineCoreOutput(rid, [102], ef.FinishReason.LENGTH)]
                ),
            )
        ]
    )
    worker2.join(timeout=5)
    off["b1_final_step"] = {
        "step_return_len_final": len(box["final"]),
        "returned_ids": [o.request_id for o in box["final"]],
        "returned_tokens": [list(o.outputs[0].token_ids) for o in box["final"]],
        "has_unfinished_after": llm.llm_engine.has_unfinished_requests(),
    }
    llm.llm_engine.engine_core.shutdown()

    # ---- B2: 裸 while 循环 + 乱序完成 + sorted 还原 ----
    llm2 = ef.LLM(_EngineArgsStub())
    eng2 = llm2.llm_engine.engine_core.engine_core
    op2 = llm2.llm_engine.output_processor
    step_log = []
    _orig_step = llm2.llm_engine.step

    def rec_step():
        r = _orig_step()
        step_log.append(len(r))
        return r

    llm2.llm_engine.step = rec_step
    player_beats = []

    def player():
        counts, seen = {}, False
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                live = list(op2.request_states)
            except RuntimeError:
                time.sleep(0.001)
                continue
            if not live:
                if not seen:
                    time.sleep(0.002)
                    continue
                return
            seen = True
            outs, beat = [], []
            for rid_ in live:
                counts[rid_] = counts.get(rid_, 0) + 1
                finished = counts[rid_] >= 2 if rid_.startswith("1-") else counts[rid_] >= 3
                outs.append(
                    ef.EngineCoreOutput(
                        rid_,
                        [100 + counts[rid_]],
                        ef.FinishReason.LENGTH if finished else None,
                    )
                )
                beat.append(
                    {
                        "req": rid_.rpartition("-")[0],
                        "token": 100 + counts[rid_],
                        "finished": finished,
                    }
                )
            player_beats.append(beat)
            eng2.emit_step_outputs([(0, ef.EngineCoreOutputs(outputs=outs))])
            time.sleep(0.002)

    tp = threading.Thread(target=player, daemon=True)
    tp.start()
    params2 = [ef.SamplingParams(n=1, max_tokens=3) for _ in range(2)]
    outputs = llm2.generate([prompt((1,)), prompt((2,))], params2)
    tp.join(timeout=10)
    off["b2_bare_while_loop"] = {
        "player_beats": player_beats,
        "step_return_lens": step_log,
        "result_ids_in_return_order": [o.request_id for o in outputs],
        "result_tokens": [list(o.outputs[0].token_ids) for o in outputs],
        "all_finished": [o.finished for o in outputs],
        "completion_order_note": "请求 1 两拍先完成、请求 0 三拍后完成 → sorted(key=int(request_id)) 还原输入序 0,1",
        "driver_code": "while self.llm_engine.has_unfinished_requests(): step() (offline_utils.py:L594-L595)",
    }
    llm2.llm_engine.engine_core.shutdown()

    # ---- B3: 事务性批量提交 ----
    llm3 = ef.LLM(_EngineArgsStub())
    eng3 = llm3.llm_engine.engine_core.engine_core
    op3 = llm3.llm_engine.output_processor
    params3 = [ef.SamplingParams(n=1, max_tokens=2), object()]  # 第二个非法
    txn = {}
    try:
        llm3._render_and_add_requests([prompt((1,)), prompt((2,))], params3)
    except Exception as e:
        txn["exception"] = f"{type(e).__name__}"
    msgs = []
    while True:
        try:
            msgs.append(eng3.input_queue.get_nowait())
        except queue.Empty:
            break
    txn["message_sequence"] = [mt.name for mt, _ in msgs]
    txn["front_ledger_after_rollback"] = len(op3.request_states)
    eng3.emit_step_outputs([])
    txn["engine_ledger_after_rollback"] = len(eng3.requests)
    off["b3_transactional_batch_add"] = txn
    llm3.llm_engine.engine_core.shutdown()


def main():
    trace = {
        "mechanism": "m8 两种驱动",
        "companion": "implementation/engine_faces.py (pin vLLM v0.27.1)",
        "envs_VLLM_ENABLE_V1_MULTIPROCESSING": ef.envs.VLLM_ENABLE_V1_MULTIPROCESSING,
        "online": {},
        "offline": {},
    }
    asyncio.run(online_face(trace))
    offline_face(trace)

    dest = Path(__file__).resolve().parent / "m8_two_drivers.json"
    dest.write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(trace, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
