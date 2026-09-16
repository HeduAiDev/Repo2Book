# m15/m16 取证：内部 LB 打分选路（v0.27.1 重写版公式）+ abort 按引擎路由。
# 全部走真 get_core_engine_for_request / process_engine_outputs / abort_requests_async；
# score 列由驱动按源码公式（core_client.py:L1494-L1501）从调用瞬间的真实 client
# 状态计算（函数本身不回传中间分——标注 provenance_note）。
from __future__ import annotations

import asyncio
from collections import Counter

from _driver_common import clean_tmp, dump


def _req(req_id, dp_rank=None):
    from vllm.v1.engine import EngineCoreRequest

    return EngineCoreRequest(
        request_id=req_id,
        prompt_token_ids=[1],
        mm_features=None,
        sampling_params=None,
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=dp_rank,
    )


def _lb_client(client_count=1, client_index=0, n_engines=4):
    from vllm.v1.engine.core_client import DPLBAsyncMPClient

    c = DPLBAsyncMPClient.__new__(DPLBAsyncMPClient)
    c.client_count = client_count
    c.client_index = client_index
    c.reqs_in_flight = {}
    c.engine_inflight = Counter()
    c.current_wave = 0
    c.engines_running = True
    c.core_engines = [i.to_bytes(2, "little") for i in range(n_engines)]
    c.lb_engines = [[0, 0, 0.0] for _ in range(n_engines)]
    c.eng_start_index = 0
    return c


def _snap(c):
    return {
        "inflight": [c.engine_inflight[e] for e in c.core_engines],
        "lb_engines": [list(x) for x in c.lb_engines],
        "eng_start_index": c.eng_start_index,
    }


def _scores_from_state(c):
    """按 core_client.py:L1494-L1501 公式对当前状态逐引擎算分（驱动侧复算，
    输入全部来自真实 client 状态）。"""
    rows = []
    for i in range(len(c.core_engines)):
        waiting, running, kv = c.lb_engines[i]
        inflight = c.engine_inflight[c.core_engines[i]]
        s = max(c.client_count * inflight, waiting + running)
        if waiting:
            s += waiting * 6.0 * max(0.0, kv - 0.5)
        rows.append(s)
    return rows


def _choose_and_record(c, rid):
    before = _snap(c)
    scores = _scores_from_state(c)
    eng = c.get_core_engine_for_request(_req(rid))
    idx = int.from_bytes(eng, "little")
    return {
        "request": rid,
        "state_before": before,
        "scores_computed": scores,
        "chosen_engine": idx,
        "inflight_after": [c.engine_inflight[e] for e in c.core_engines],
        "start_index_after": c.eng_start_index,
        "registry": {k: int.from_bytes(v, "little") for k, v in c.reqs_in_flight.items()},
    }


def main() -> None:
    doc = {
        "mechanisms": ["m15", "m16"],
        "provenance_note": "chosen_engine/inflight/registry/eng_start_index 全部为真调用产物；"
        "scores_computed 为驱动按源码公式从调用前真实状态复算（函数不回传中间分）",
        "params": "4 引擎；场景 A client_count=2 突发 5 请求、快照全 0（100ms 窗口内）",
    }

    # ── 场景 A：突发 5 请求，快照全 0 → inflight 地板保证 round-robin ──
    c = _lb_client(client_count=2)
    rounds = [_choose_and_record(c, f"r{i}") for i in range(1, 6)]
    doc["scenario_a_burst"] = {
        "client_count": 2,
        "lb_engines_snapshot_all_zero": True,
        "rounds": rounds,
        "choices": [r["chosen_engine"] for r in rounds],
    }

    # ── 场景 D：finished 回收 + abort 按引擎路由（同一 client 上续跑）──
    from vllm.v1.engine import EngineCoreOutputs

    asyncio.run(
        type(c).process_engine_outputs(c, EngineCoreOutputs(finished_requests={"r5"}))
    )
    after_finish = {
        "finished": "r5",
        "inflight": [c.engine_inflight[e] for e in c.core_engines],
        "registry": {k: int.from_bytes(v, "little") for k, v in c.reqs_in_flight.items()},
    }
    aborts = []

    async def _noop():
        return None

    def _abort(req_ids, engine):
        aborts.append((tuple(req_ids), int.from_bytes(engine, "little")))
        return _noop()

    c._abort_requests = _abort
    c.resources = type("R", (), {"engine_dead": False})()
    asyncio.run(c.abort_requests_async(["r3"]))
    doc["scenario_d_reclaim_abort"] = {
        "after_finish": after_finish,
        "abort_r3_routed_to": aborts[0][1],
    }

    # ── 场景 B：KV 压力斜坡（4 waiting，kv 0.4 vs 1.0 vs 重载 100）──
    c2 = _lb_client()
    c2.lb_engines = [[4, 0, 0.4], [4, 0, 1.0], [100, 0, 0.0], [100, 0, 0.0]]
    doc["scenario_b_kv_ramp"] = _choose_and_record(c2, "kv1")

    # ── 场景 C：平局轮转（4 引擎全空，连续 2 请求）──
    c3 = _lb_client()
    doc["scenario_c_tie_rotation"] = {
        "rounds": [_choose_and_record(c3, "t1"), _choose_and_record(c3, "t2")],
    }

    # ── 场景 E：外部 LB——绑定的引擎原样返回（F4 的另一种用法）──
    from vllm.v1.engine.core_client import DPAsyncMPClient

    ext = DPAsyncMPClient.__new__(DPAsyncMPClient)
    ext.core_engines = [i.to_bytes(2, "little") for i in range(4)]
    ext.core_engine = ext.core_engines[2]
    eng = ext.get_core_engine_for_request(_req("e1"))
    doc["scenario_e_external_lb"] = {
        "bound_engine": 2,
        "returned_engine": int.from_bytes(eng, "little"),
        "unchanged": eng == ext.core_engine,
    }

    dump("m15", doc)
    clean_tmp()


if __name__ == "__main__":
    main()
