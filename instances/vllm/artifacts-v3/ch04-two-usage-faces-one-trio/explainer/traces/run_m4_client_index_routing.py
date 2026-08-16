"""m4 driver — client_index 端到端路由: 两个前端 × 一个引擎的微缩 many-to-many。

companion 里两个 AsyncLLM 各自构造时会各自 spawn 一个 in-process 引擎 stub;
真实系统里两个 API server 的 ROUTER/PULL 连的是同一个引擎进程(地址接线由
launch_core_engines 做, core.py:L1760-L1766 每前端一条 PUSH)。本驱动把前端 1
的 client.engine_core 重接到共享 stub 上——正是那段 ZMQ 接线的 in-process 对应,
差异已在 trace 的 caveats 里标注。

之后: 两个前端各发一个请求(盖章 0/1) → 引擎同拍产出两组输出 → sockets[client_index]
各回各家 → 终拍 → 两侧账本清空。"""

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


def stamps_view(engine):
    msgs = []
    while True:
        try:
            msgs.append(engine.input_queue.get_nowait())
        except Exception:
            break
    view = [
        {
            "message_type": mt.name,
            "request_id": r.request_id,
            "client_index": r.client_index,
            "external_req_id": r.external_req_id,
        }
        for mt, r in msgs
    ]
    for m in msgs:  # 看完放回, 引擎一拍还要消费
        engine.input_queue.put(m)
    return view


async def main():
    trace = {
        "mechanism": "m4 client_index 端到端路由 (F3)",
        "companion": "implementation/engine_faces.py (pin vLLM v0.27.1)",
        "caveats": [
            "companion 的 MPClient 构造会各自建 in-process 引擎 stub; 本驱动把前端 1 的 "
            "client.engine_core 重接到共享 stub 并 attach_output_socket(1, q1) —— 对应真实系统 "
            "launch_core_engines 的 ZMQ 地址接线(两前端 ROUTER/PULL 连同一引擎, #17546 many-to-many)",
            "emit_step_outputs 从事件循环线程调用是 companion 约束(真实 ZMQ 无此约束, impl-notes 已知偏差 2)",
        ],
        "topology": {
            "client_count": 2,
            "front_end_0": 'AsyncLLM(client_index=0) 发 "chat-A"',
            "front_end_1": 'AsyncLLM(client_index=1) 发 "chat-B"',
            "engine": "1 个共享 stub; sockets = [前端0 槽, 前端1 槽]",
        },
        "rounds": [],
    }

    allm0 = ef.AsyncLLM(
        ef.VllmConfig(), ef.Executor, log_stats=False,
        client_count=2, client_index=0,
    )
    allm1 = ef.AsyncLLM(
        ef.VllmConfig(), ef.Executor, log_stats=False,
        client_count=2, client_index=1,
    )
    shared = allm0.engine_core.engine_core
    allm1.engine_core.engine_core = shared  # ZMQ 地址接线的 in-process 对应
    shared.attach_output_socket(1, allm1.engine_core.outputs_queue)

    collected = {0: [], 1: []}

    async def consumer(allm, key, ext_id):
        async for out in allm.generate(prompt([9, 9]), params, ext_id):
            collected[key].append(out)

    params = ef.SamplingParams(
        n=1, max_tokens=2, output_kind=ef.RequestOutputKind.DELTA
    )
    t0 = asyncio.ensure_future(consumer(allm0, 0, "chat-A"))
    t1 = asyncio.ensure_future(consumer(allm1, 1, "chat-B"))

    deadline = time.monotonic() + 5
    while (
        not allm0.output_processor.request_states
        or not allm1.output_processor.request_states
    ) and time.monotonic() < deadline:
        await asyncio.sleep(0.001)

    # ---- 轮 1+2: 两前端各发一请求, 盖章过线 ----
    trace["rounds"].append(
        {
            "round": "1+2",
            "action": '前端 0 add_request("chat-A") 与前端 1 add_request("chat-B")',
            "stamps": stamps_view(shared),
            "engine_sockets": {"slot_count": len(shared.sockets)},
        }
    )

    ridA = next(iter(allm0.output_processor.request_states))
    ridB = next(iter(allm1.output_processor.request_states))

    # ---- 轮 3: 引擎同拍产出两组 → 按章回发 ----
    shared.emit_step_outputs(
        [
            (0, ef.EngineCoreOutputs(outputs=[ef.EngineCoreOutput(ridA, [201], None)])),
            (1, ef.EngineCoreOutputs(outputs=[ef.EngineCoreOutput(ridB, [202], None)])),
        ]
    )
    deadline = time.monotonic() + 5
    while (len(collected[0]) < 1 or len(collected[1]) < 1) and time.monotonic() < deadline:
        await asyncio.sleep(0.001)
    trace["rounds"].append(
        {
            "round": 3,
            "action": "引擎一拍产出 (0, A批) 与 (1, B批) → sockets[0]/sockets[1] 各回各家",
            "front0_received": {
                "request_id": collected[0][0].request_id,
                "token_ids": list(collected[0][0].outputs[0].token_ids),
                "finished": collected[0][0].finished,
            },
            "front1_received": {
                "request_id": collected[1][0].request_id,
                "token_ids": list(collected[1][0].outputs[0].token_ids),
                "finished": collected[1][0].finished,
            },
            "routing_note": "输出 IO 线程按 client_index 下标查表 O(1) (core.py:L1804 sockets[client_index])",
        }
    )

    # ---- 轮 4: 终拍 → 两侧清账 ----
    shared.emit_step_outputs(
        [
            (0, ef.EngineCoreOutputs(outputs=[ef.EngineCoreOutput(ridA, [203], ef.FinishReason.LENGTH)])),
            (1, ef.EngineCoreOutputs(outputs=[ef.EngineCoreOutput(ridB, [204], ef.FinishReason.LENGTH)])),
        ]
    )
    await asyncio.wait_for(t0, 5)
    await asyncio.wait_for(t1, 5)
    trace["rounds"].append(
        {
            "round": 4,
            "action": "终拍: 各带 finish_reason=LENGTH",
            "front0_final": {
                "request_id": collected[0][-1].request_id,
                "token_ids": list(collected[0][-1].outputs[0].token_ids),
                "finished": collected[0][-1].finished,
            },
            "front1_final": {
                "request_id": collected[1][-1].request_id,
                "token_ids": list(collected[1][-1].outputs[0].token_ids),
                "finished": collected[1][-1].finished,
            },
            "ledgers": {
                "front0_request_states": len(allm0.output_processor.request_states),
                "front1_request_states": len(allm1.output_processor.request_states),
                "engine_requests": len(shared.requests),
            },
        }
    )

    # ---- 单前端对照: 默认出生参数 ----
    plain = ef.AsyncLLM(ef.VllmConfig(), ef.Executor, log_stats=False)
    plain.shutdown()
    trace["single_front_end_control"] = {
        "default_client_count": plain.engine_core.client_count,
        "default_client_index": plain.engine_core.client_index,
        "note": "单前端时 client_index 恒 0 —— 一个看似冗余的 int 字段 (ch34 多前端时它是回程路由键)",
    }

    dest = Path(__file__).resolve().parent / "m4_client_index_routing.json"
    dest.write_text(json.dumps(trace, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(trace, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
