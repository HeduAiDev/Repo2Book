# m09 取证：推模式 NixlPushConnector（m9）——D 分配即 PUSH_REG 注册、
# P 的 nixl-push-writer 线程配对（随机后缀剥离）、WRITE 直写 D 预分配块、
# 双端完成判定、注册 watchdog、has_pending_push_work 保活。
from _driver_common import (
    KVCacheBlocksStub,
    dump,
    make_d_request,
    make_p_request,
    receipt_from_p,
)
from _pd_harness import block_checksum, fill_blocks, make_engine, pump_until


def main() -> None:
    doc = {
        "mechanisms": ["m9"],
        "params": "8-token prompt、块 [0,1] 填 3.0；推模式 NixlPushConnector 两台引擎；"
        "注册 watchdog 默认 480s",
    }
    p = make_engine("p", kv_role="kv_producer", connector="NixlPushConnector")
    d = make_engine("d", kv_role="kv_consumer", connector="NixlPushConnector")
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.push_worker import (
            _PUSH_WRITER_POLL_INTERVAL_MS,
        )
        from vllm.v1.request import RequestStatus

        receipt = receipt_from_p(p, "req-1", blocks=[0, 1], num_computed_tokens=8)
        fill_blocks(p.kv_caches, [0, 1], 3.0)
        fill_blocks(d.kv_caches, [0, 1], 0.0)

        # ── 1) D 分配即注册（不等自己被调度） ──
        req = make_d_request("req-1", receipt, num_tokens=8)
        d.scheduler.update_state_after_alloc(req, KVCacheBlocksStub([[0, 1]]), 8)
        meta = d.build_meta()
        reg = meta.push_registrations["req-1"]
        doc["d_registration"] = {
            "reg_keys": sorted(reg.keys()),
            "reg_key_count": len(reg),
            "decode_engine_id": reg["decode_engine_id"],
            "local_block_ids": reg["local_block_ids"],
            "remote_engine_id": reg["remote_engine_id"],
            "watchdog_timeout_s": d.scheduler._push_registration_timeout,
            "seeded_remote_block_ids": req.kv_transfer_params["remote_block_ids"],
            "do_remote_prefill_flipped": req.kv_transfer_params["do_remote_prefill"]
            is False,
        }
        d.worker_step(meta)  # 注册进 writer 收件箱 → 发 PUSH_REG notif 给 P

        # ── 2) P 终局：完成块交给 writer 配对 ──
        p_req = make_p_request("req-1")
        p_req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        p_req.num_computed_tokens = 8
        p_esched = p.core.scheduler
        p_esched.kv_cache_manager.set_blocks("req-1", ([0, 1],))
        p_esched.requests["req-1"] = p_req
        delay_free, receipt2 = p.scheduler.request_finished(p_req, ([0, 1],))
        p_meta = p.build_meta()
        doc["p_terminal"] = {
            "delay_free_blocks": delay_free,
            "finished_blocks_in_meta": p_meta.push_finished_blocks["req-1"],
            "has_pending_push_work": p.scheduler.has_pending_push_work(),
            "lease_stamp_s": p.scheduler._kv_lease_duration,
        }
        p.worker_step(p_meta)  # 完成块进 writer 收件箱

        # ── 3) writer 配对 → WRITE 直写 D 预分配块 ──
        pump_until(lambda: block_checksum(d.kv_caches, 0) != 0.0)
        doc["write_landed"] = {
            "d_block0_checksum": block_checksum(d.kv_caches, 0),
            "d_block1_checksum": block_checksum(d.kv_caches, 1),
            "p_block0_checksum": block_checksum(p.kv_caches, 0),
            "expected_block_checksum": 128 * 3.0,
            "writer_poll_interval_ms": _PUSH_WRITER_POLL_INTERVAL_MS,
            "writer_inboxes": 4,
            "d_has_local_handle": len(d.worker._recving_transfers.get("req-1", [])),
        }

        # ── 4) 双端完成判定 ──
        got = pump_until(
            lambda: (lambda r: r[0] if r[0] else None)(p.worker_step())
        )
        dr = pump_until(
            lambda: (lambda r: r[1] if r[1] else None)(d.worker_step())
        )
        doc["completion"] = {
            "p_done_sending": sorted(got or set()),
            "d_done_recving": sorted(dr or set()),
            "p_req_left_to_send": "req-1" not in p.worker._reqs_to_send,
            "d_completion_source": "P 的 WRITE 完成 notif（无本地 handle → 物化空条目上报）",
            "p_completion_source": "WRITE handle 轮询 DONE",
        }

        # ── 5) 配对机器：随机后缀剥离 ──
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
            get_base_request_id,
        )

        w = p.worker
        w._pending_d_registrations["req-a"] = {"request_id": "req-a"}
        w._push_finished_blocks["req-b"] = ([0],)
        doc["suffix_matching"] = {
            "strip": ["req-a-1a2b3c4d", get_base_request_id("req-a-1a2b3c4d")],
            "exact_key_first": w._pop_matching_registration("req-a")
            == {"request_id": "req-a"},
            "case_mismatch_no_match": w._pop_matching_registration("REQ-A") is None,
            "finished_blocks_exact": w._pop_matching_finished_blocks("req-b")
            == ("req-b", ([0],)),
            "finished_blocks_consumed_once": w._pop_matching_finished_blocks("req-b")
            is None,
            "why_suffix": "两腿 request_id 差 input_processor.assign_request_id "
            "追加的 8 位十六进制随机后缀",
        }

        # ── 6) 注册 watchdog：过期注册丢弃 ──
        import time

        from vllm.v1.outputs import KVConnectorOutput

        sched_d = d.scheduler
        # req-1 的推送已完成：update_connector_output 清它的 watchdog（维持
        # 「到期时刻按非降序插入」的账本不变量——watchdog 扫描按插入序早退）
        sched_d.update_connector_output(
            KVConnectorOutput(finished_recving={"req-1"})
        )
        sched_d._push_pending_registrations["req-stale"] = {"request_id": "req-stale"}
        sched_d._push_registration_deadlines["req-stale"] = time.perf_counter() - 1.0
        meta_wd = d.build_meta()
        doc["registration_watchdog"] = {
            "stale_registration_dropped": "req-stale"
            not in sched_d._push_pending_registrations,
            "not_packaged": "req-stale" not in meta_wd.push_registrations,
        }

        # ── 7) 保活：无活请求时引擎继续步进 ──
        p2 = make_engine("p2", kv_role="kv_producer", connector="NixlPushConnector")
        try:
            s2 = p2.core.scheduler
            before = s2.has_requests()
            s2.connector.connector_scheduler._finished_request_blocks["r"] = ([0],)
            after = s2.has_requests()
            doc["keepalive"] = {
                "no_requests_no_pending_work": before is False,
                "pending_push_work_keeps_engine_alive": after is True,
            }
        finally:
            p2.close()
    finally:
        p.close()
        d.close()

    dump("m09", doc)


if __name__ == "__main__":
    main()
