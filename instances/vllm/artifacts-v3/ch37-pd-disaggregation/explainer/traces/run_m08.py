# m08 取证：租约心跳与过期（m8）——30s 租约 / 心跳间隔=租约/6 / 续租 lease×2/3 只增不减 /
# scheduler_clock 跨进程重基准 / 到期收割自愈 / 真 HB notif 跨端续租。
from _driver_common import dump, make_d_request, receipt_from_p
from _pd_harness import make_engine, pump_until


def main() -> None:
    doc = {
        "mechanisms": ["m8"],
        "params": "租约默认 30s（kv_lease_duration）；D 等待队列里的请求按引擎分组发 "
        "HB: notif；重基准示例：scheduler_clock=99000、截止=100000",
    }
    p = make_engine("p", kv_role="kv_producer")
    d = make_engine("d", kv_role="kv_consumer")
    try:
        import time

        # ── 1) 三个常量 ──
        doc["constants"] = {
            "kv_lease_duration_s": p.scheduler._kv_lease_duration,
            "heartbeat_interval_s": d.scheduler._heartbeat_interval,
            "lease_extension_s": p.worker._lease_extension,
            "heartbeat_interval_formula": "kv_lease_duration // 6 = 30 // 6 = 5",
            "lease_extension_formula": "kv_lease_duration * 2 // 3 = 30 * 2 // 3 = 20",
            "heartbeats_per_lease": p.scheduler._kv_lease_duration
            // d.scheduler._heartbeat_interval,
            "renewal_rate_x": p.worker._lease_extension
            / d.scheduler._heartbeat_interval,
        }

        # ── 2) 重基准：调度器时钟域 → worker 时钟域 ──
        from vllm.v1.core.sched.output import SchedulerOutput

        w = d.worker
        meta = d.build_meta(SchedulerOutput())
        meta.scheduler_clock = 99000.0
        meta.reqs_to_send = {"req-1": 100000.0}
        w._reqs_to_process.add("req-1")
        t0 = time.perf_counter()
        w.start_load_kv(meta)
        rebased = w._reqs_to_send["req-1"]
        doc["rebase"] = {
            "scheduler_clock": 99000.0,
            "deadline_scheduler_domain": 100000.0,
            "remaining_ttl_s": 1000.0,
            "rebased_minus_t0_s": round(rebased - t0, 1),
            "in_range_990_1010": 990 < rebased - t0 < 1010,
            "note": "remaining = now_local + (deadline − scheduler_clock)；广播延迟只延长租约",
        }

        # ── 3) 旧式 metadata（scheduler_clock=0）：截止时刻原样保留 ──
        meta_legacy = d.build_meta(SchedulerOutput())
        meta_legacy.scheduler_clock = 0.0
        meta_legacy.reqs_to_send = {"req-2": 100000.0}
        w._reqs_to_process.add("req-2")
        w.start_load_kv(meta_legacy)
        doc["legacy_no_clock"] = {"deadline_kept": w._reqs_to_send["req-2"]}

        # ── 4) 真 HB notif 跨端续租（D 发、P 收、只增不减） ──
        receipt = receipt_from_p(p, "req-1", blocks=[0, 1], num_computed_tokens=8)
        req = make_d_request("req-1", receipt, num_tokens=8)
        d.scheduler.on_new_request(req)
        # D 侧先把到 P 的 agent 建好（心跳的预握手）
        d.worker._ensure_handshake(
            p.engine_id,
            p.scheduler.side_channel_host,
            p.scheduler.side_channel_port,
            1,
        ).result(timeout=10)
        # P 侧：req-1 钉着、只剩 1s 租约
        p.worker._reqs_to_process.add("req-1")
        before = time.perf_counter() + 1.0
        p.worker._reqs_to_send["req-1"] = before
        hb_meta = d.build_meta()  # 打包心跳（节流门内第一次）
        d.worker_connector._connector_metadata = hb_meta
        d.worker_connector.start_load_kv(hb_meta)  # → send_notif("HB:req-1")
        # P 的引擎线程翻收件箱（get_finished）→ _handle_heartbeat 续租
        def _p_poll():
            p.worker_connector.get_finished(set())
            return p.worker._reqs_to_send.get("req-1", before) > before + 19

        arrived = pump_until(_p_poll, timeout=5.0)
        after = p.worker._reqs_to_send["req-1"]
        # 紧接着第二条心跳：max(old, new) 只增不减
        d.worker_connector.start_load_kv(hb_meta)
        time.sleep(0.01)
        _p_poll()
        after2 = p.worker._reqs_to_send["req-1"]
        doc["heartbeat_renewal"] = {
            "hb_message": "HB:req-1",
            "old_lease_remaining_s": 1.0,
            "new_expiry_formula": "time.perf_counter() + _lease_extension(20)",
            "deadline_movement_s": round(after - before, 2),
            "hb_arrived_and_extended": arrived,
            "second_hb_never_shortens": after2 >= after,
            "second_delta_s": round(after2 - after, 3),
        }

        # ── 5) 心跳节流与停跳 ──
        doc["heartbeat_packaging"] = {
            "first_meta_has_hb": sorted(
                hb_meta.heartbeat_by_engine[p.engine_id].req_ids
            ),
            "immediate_second_meta_throttled": d.build_meta().heartbeat_by_engine == {},
            "throttle_interval_s": d.scheduler._heartbeat_interval,
        }
        from vllm.v1.outputs import KVConnectorOutput

        d.scheduler.update_connector_output(
            KVConnectorOutput(finished_recving={"req-1"})
        )
        doc["heartbeat_packaging"]["stops_after_finished_recving"] = (
            d.scheduler._heartbeat_by_engine == {}
        )

        # ── 6) 到期收割：没人来读也放块（自愈） ──
        w_p = p.worker
        # 收割按插入序早退（账本不变量：到期时刻按非降序插入）——先清掉未到期的
        w_p._reqs_to_send.pop("req-1", None)
        w_p._reqs_to_send["req-dead"] = time.perf_counter() - 1.0
        w_p._reqs_to_process.add("req-dead")
        done_sending, _ = w_p.get_finished()
        doc["expiry_harvest"] = {
            "deadline_in_past_s": -1.0,
            "done_sending": sorted(done_sending),
            "req_left_to_send": "req-dead" not in w_p._reqs_to_send,
            "note": "get_finished 按到期序扫描，过期即强制 done_sending——D 失联时 P 显存自愈",
        }
    finally:
        p.close()
        d.close()

    dump("m08", doc)


if __name__ == "__main__":
    main()
