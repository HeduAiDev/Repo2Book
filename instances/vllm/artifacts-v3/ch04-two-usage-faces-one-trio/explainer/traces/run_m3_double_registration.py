"""m3 driver — add_request 双登记: 一个请求「chat-abc」的两本账逐轮实测。

在线面 AsyncLLM (companion, async_llm.py:L283-L435 逐字) 吃一个请求,
按 4 轮快照两本账:
  账本① = 本进程 OutputProcessor.request_states (RequestState 侧表 + 外→内映射)
  账本② = 引擎侧 EngineCore.requests (EngineCoreRequest 过界物)
引擎侧由 EngineCore.emit_step_outputs 扮演 (companion 的调度器接缝, ch9 的边界),
真 trace 须进容器加载完整引擎, 本章按 dossier delete 项 1 走 stub。"""

import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

_IMPL = Path(__file__).resolve().parent.parent.parent / "implementation" / "engine_faces.py"
_spec = importlib.util.spec_from_file_location("engine_faces", _IMPL)
ef = importlib.util.module_from_spec(_spec)
sys.modules["engine_faces"] = ef
_spec.loader.exec_module(ef)


def prompt(token_ids):
    return {"type": "token_ids", "prompt_token_ids": list(token_ids)}


def ledger1(allm):
    op = allm.output_processor
    return {
        "request_states_size": len(op.request_states),
        "entries": {
            rid: {
                "external_req_id": st.external_req_id,
                "queue_request_id": st.queue.request_id,
                "queue_aggregate": st.queue.aggregate,
                "output_kind": st.output_kind.name,
                "max_tokens_param": st.max_tokens_param,
                "prompt_token_ids": list(st.prompt_token_ids),
                "detokenizer": type(st.detokenizer).__name__,
            }
            for rid, st in op.request_states.items()
        },
        "external_to_internal": {k: list(v) for k, v in op.external_req_ids.items()},
    }


def ledger2(engine):
    return {
        "requests_size": len(engine.requests),
        "entries": {
            rid: {
                "request_id": r.request_id,
                "external_req_id": r.external_req_id,
                "client_index": r.client_index,
                "prompt_token_ids": list(r.prompt_token_ids),
                "sampling_n": r.sampling_params.n,
                "sampling_output_kind": r.sampling_params.output_kind.name,
                "sampling_max_tokens": r.sampling_params.max_tokens,
            }
            for rid, r in engine.requests.items()
        },
    }


async def main():
    trace = {
        "mechanism": "m3 add_request 双登记",
        "companion": "implementation/engine_faces.py (pin vLLM v0.27.1)",
        "params": {
            "external_request_id": "chat-abc",
            "prompt_token_ids": [1, 2, 3],
            "output_kind": "DELTA",
            "n": 1,
            "max_tokens": 2,
        },
        "rounds": [],
    }
    allm = ef.AsyncLLM(ef.VllmConfig(), ef.Executor, log_stats=False)
    engine = allm.engine_core.engine_core
    params = ef.SamplingParams(
        n=1, max_tokens=2, output_kind=ef.RequestOutputKind.DELTA
    )
    collected = []

    async def consume():
        async for out in allm.generate(prompt([1, 2, 3]), params, "chat-abc"):
            collected.append(out)

    task = asyncio.ensure_future(consume())
    deadline = time.monotonic() + 5
    while not allm.output_processor.request_states and time.monotonic() < deadline:
        await asyncio.sleep(0.001)

    # ---- 轮 1: 双登记刚发生 (generate→add_request→_add_request 两行已执行) ----
    internal = next(iter(allm.output_processor.request_states))
    suffix = internal.rpartition("-")[2]
    crossed_msgs = []
    while True:
        try:
            crossed_msgs.append(engine.input_queue.get_nowait())
        except Exception:
            break
    for m in crossed_msgs:  # 看完放回, 引擎下一拍还要消费
        engine.input_queue.put(m)
    crossed_view = [
        {
            "message_type": mt.name,
            "request_id": r.request_id,
            "external_req_id": r.external_req_id,
            "client_index": r.client_index,
            "prompt_token_ids": list(r.prompt_token_ids),
        }
        for mt, r in crossed_msgs
    ]
    trace["rounds"].append(
        {
            "round": 1,
            "action": 'add_request("chat-abc") → _add_request 双登记两行 (async_llm.py:L420-L435)',
            "ledger1_this_process": ledger1(allm),
            "ledger2_engine_side": ledger2(engine),
            "boundary": {
                "input_queue_size": engine.input_queue.qsize(),
                "crossed_messages": crossed_view,
                "state": "已过界、未入引擎 (mp 路径中间态)",
            },
            "internal_id_suffix_len": len(suffix),
            "internal_id": internal,
        }
    )

    # ---- 轮 2: 引擎一拍 (排空 input_queue → 引擎侧账本落账) ----
    engine.emit_step_outputs([])
    trace["rounds"].append(
        {
            "round": 2,
            "action": "引擎一拍: emit_step_outputs([]) 先排空 input_queue (busy loop 同序, core.py:L1378-L1389)",
            "ledger1_this_process": ledger1(allm),
            "ledger2_engine_side": ledger2(engine),
        }
    )

    # ---- 轮 3: 中间拍 (token 101, 无 finish) ----
    engine.emit_step_outputs(
        [
            (
                0,
                ef.EngineCoreOutputs(
                    outputs=[
                        ef.EngineCoreOutput(internal, [101], None)
                    ]
                ),
            )
        ]
    )
    deadline = time.monotonic() + 5
    while len(collected) < 1 and time.monotonic() < deadline:
        await asyncio.sleep(0.001)
    trace["rounds"].append(
        {
            "round": 3,
            "action": "引擎产出 (client_index=0, EngineCoreOutput(token [101], 未 finish)) → sockets[0] → output_handler → process_outputs 查表",
            "ledger1_this_process": ledger1(allm),
            "ledger2_engine_side": ledger2(engine),
            "consumer_side": {
                "yielded_request_id": collected[0].request_id,
                "yielded_token_ids": list(collected[0].outputs[0].token_ids),
                "finished": collected[0].finished,
                "note": "出门时 request_id 已换回外部 id chat-abc (output_processor.py 出口反查)",
            },
        }
    )

    # ---- 轮 4: 终拍 (token 102 + finish) → 两侧清账 ----
    engine.emit_step_outputs(
        [
            (
                0,
                ef.EngineCoreOutputs(
                    outputs=[
                        ef.EngineCoreOutput(internal, [102], ef.FinishReason.LENGTH)
                    ]
                ),
            )
        ]
    )
    deadline = time.monotonic() + 5
    while len(collected) < 2 and time.monotonic() < deadline:
        await asyncio.sleep(0.001)
    await asyncio.wait_for(task, 5)
    trace["rounds"].append(
        {
            "round": 4,
            "action": "终拍: token [102] + finish_reason=LENGTH → 前端 _finish_request + 引擎侧 finished 清账",
            "ledger1_this_process": ledger1(allm),
            "ledger2_engine_side": ledger2(engine),
            "consumer_side": {
                "final_request_id": collected[-1].request_id,
                "final_token_ids": list(collected[-1].outputs[0].token_ids),
                "finished": collected[-1].finished,
                "finish_reason": collected[-1].outputs[0].finish_reason,
            },
        }
    )

    trace["summary"] = {
        "yields_total": len(collected),
        "all_ids_external": sorted({o.request_id for o in collected}),
        "both_ledgers_empty_at_end": (
            len(allm.output_processor.request_states) == 0
            and len(engine.requests) == 0
        ),
        "tokens_seen": [list(o.outputs[0].token_ids) for o in collected],
    }

    dest = Path(__file__).resolve().parent / "m3_double_registration.json"
    dest.write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(trace, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
