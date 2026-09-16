# m07 取证：完成回收·双端不对称（m7）——D 轮询 handle（本地 DONE）vs P 收 notif
# （对端送达）；KVOutputAggregator 按 expected_finished_count 聚 TP；调度器两侧落点。
from _driver_common import (
    KVCacheBlocksStub,
    dump,
    make_d_request,
    make_p_request,
)
from _pd_harness import fill_blocks, make_engine, pump_until


def main() -> None:
    doc = {
        "mechanisms": ["m7"],
        "params": "8-token prompt、块 [0,1]；TP=1（expected_finished_count=1）为主线，"
        "TP=2 聚合器对照（expected_finished_count=2）",
    }
    p = make_engine("p", kv_role="kv_producer")
    d = make_engine("d", kv_role="kv_consumer")
    try:
        from vllm.v1.request import RequestStatus

        # ── P 侧终局（同 m6 前奏，压缩为三行） ──
        p_req = make_p_request("req-1")
        p.scheduler.update_state_after_alloc(p_req, None, 0)
        p.worker_step(p.build_meta())
        fill_blocks(p.kv_caches, [0, 1], 7.0)
        p_req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        p_req.num_computed_tokens = 8
        receipt = p.scheduler.request_finished(p_req, ([0, 1],))[1]
        p.worker_step(p.build_meta())

        # ── D 侧登记 + 首遇握手 ──
        d_req = make_d_request("req-1", receipt, num_tokens=8)
        d.scheduler.update_state_after_alloc(d_req, KVCacheBlocksStub([[0, 1]]), 8)
        d.worker_step()
        pump_until(lambda: d.worker._remote_agents.get(p.engine_id))

        # ── 1) D 端：本地 handle 轮询出 DONE（完成信号源一） ──
        meta = d.build_meta()
        d.worker_connector._connector_metadata = meta
        d.worker_connector.start_load_kv(meta)  # 排空 _ready_requests → READ
        doc["d_side"] = {
            "handles_in_flight": len(d.worker._recving_transfers.get("req-1", [])),
            "poll_source": "check_xfer_state(handle) 本地轮询",
        }
        ds_d, dr_d = d.worker_connector.get_finished(set())
        doc["d_side"].update(
            {
                "done_recving": sorted(dr_d),
                "done_sending_at_d": sorted(ds_d),
            }
        )

        # ── 2) P 端：完成源=对端 notif（本地无 handle 可轮询——不对称的根） ──
        # seam 的 notif 随传输完成即时入 P 的收件箱；真 RDMA 下同样由网卡送达，
        # P 的引擎线程只在 get_finished 里翻收件箱。
        doc["p_side"] = {
            "p_local_handles": len(p.worker._recving_transfers),
            "p_completion_source": "_get_new_notifs 翻对端送达的收件箱",
        }
        got_p = pump_until(
            lambda: (lambda r: r[0] if r[0] else None)(
                p.worker_connector.get_finished(set())
            )
        )
        ds_again = p.worker_connector.get_finished(set())
        doc["p_side"].update(
            {
                "done_sending_once_notif_polled": sorted(got_p or set()),
                "done_sending_on_next_poll": sorted(ds_again[0]),
                "consumers_per_producer_symmetric_tp": 1,
                "notif_accounting": "notif_id=req:tp_size，对称 TP 下 1 条 notif 即放块",
                "req_left_to_send": "req-1" not in p.worker._reqs_to_send,
            }
        )

        # ── 3) KVOutputAggregator：TP 各 worker 计数归零才上报 ──
        from vllm.distributed.kv_transfer.kv_connector.utils import KVOutputAggregator
        from vllm.v1.outputs import KVConnectorOutput, ModelRunnerOutput

        def out(finished_recving):
            return ModelRunnerOutput(
                req_ids=["r1"],
                req_id_to_index={"r1": 0},
                sampled_token_ids=[[0]],
                kv_connector_output=KVConnectorOutput(
                    finished_recving=set(finished_recving) or None
                ),
            )

        agg2 = KVOutputAggregator(expected_finished_count=2)
        first = agg2.aggregate([out({"r1"}), out(set())])
        second = agg2.aggregate([out({"r1"}), out({"r1"})])
        third = agg2.aggregate([out(set()), out({"r1"})])
        agg1 = KVOutputAggregator(expected_finished_count=1)
        one = agg1.aggregate([out({"r1"})])
        doc["aggregator"] = {
            "tp2_expected_finished_count": 2,
            "tp2_first_worker_only_reports": first.kv_connector_output.finished_recving
            is None,
            "tp2_both_report": sorted(
                second.kv_connector_output.finished_recving or set()
            ),
            "tp2_late_second_worker_alone_completes": sorted(
                third.kv_connector_output.finished_recving or set()
            ),
            "tp1_expected_finished_count": 1,
            "tp1_single_report_suffices": sorted(
                one.kv_connector_output.finished_recving or set()
            ),
            "why": "TP 各 worker 各持一份 KV 分片各报一次；只等一个 rank 就放块"
            "=其余 rank 数据还在被读",
        }

        # ── 4) 调度器两侧落点 ──
        from vllm.v1.outputs import KVConnectorOutput as KCO
        from vllm.v1.request import RequestStatus as RS

        d.core.scheduler.requests["req-1"] = d_req
        d_req.status = RS.WAITING_FOR_REMOTE_KVS
        d.core.scheduler._update_from_kv_xfer_finished(KCO(finished_recving={"req-1"}))
        p_esched = p.core.scheduler
        p_esched.kv_cache_manager.set_blocks("req-1", ([0, 1],))
        p_esched.requests["req-1"] = p_req
        p_esched._update_from_kv_xfer_finished(KCO(finished_sending={"req-1"}))
        doc["scheduler_sinks"] = {
            "d_promoted_to_finished_recving": "req-1"
            in d.core.scheduler.finished_recving_kv_req_ids,
            "p_blocks_freed_on_finished_sending": "req-1" in p_esched.kv_blocks_freed,
            "p_request_removed": "req-1" not in p_esched.requests,
        }
    finally:
        p.close()
        d.close()

    dump("m07", doc)


if __name__ == "__main__":
    main()
