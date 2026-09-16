# m11 取证：传输失败处理（m11）——_handle_failed_transfer 把块标 invalid + 请求入
# 失败队列 → 并入 done_recving（一次性上报，取走即清空）；P 侧 WRITE 失败特例。
from _driver_common import KVCacheBlocksStub, dump, make_d_request, receipt_from_p
from _pd_harness import make_engine, pump_until


def main() -> None:
    doc = {
        "mechanisms": ["m11"],
        "params": "8-token prompt、D 本地块 [0,1]；失败源=传输 setup 异常"
        "（_read_blocks 的 except 路径 / _pop_done_transfers 的异常路径共用同一入口）",
    }
    p = make_engine("p", kv_role="kv_producer")
    d = make_engine("d", kv_role="kv_consumer")
    try:
        # ── 1) D 登记待收（元数据先落账，供失败回滚用） ──
        receipt = receipt_from_p(p, "req-1", blocks=[0, 1], num_computed_tokens=8)
        req = make_d_request("req-1", receipt, num_tokens=8)
        d.scheduler.update_state_after_alloc(req, KVCacheBlocksStub([[0, 1]]), 8)
        d.worker_step()  # start_load_kv：_recving_metadata 落账（握手可能仍在飞）
        pump_until(lambda: "req-1" in d.worker._recving_metadata)
        meta_stored = d.worker._recving_metadata["req-1"]
        doc["failure_input"] = {
            "local_blocks_on_record": meta_stored.local_block_ids,
            "blocks_count": len(meta_stored.local_block_ids[0]),
        }

        # ── 2) 触发失败（真入口：握手失败/传输 setup 失败都汇到这里） ──
        d.worker._handle_failed_transfer("req-1", None)
        doc["after_failure"] = {
            "failed_queue_size": d.worker._failed_recv_reqs.qsize(),
        }

        # ── 3) 上报：失败请求并入 done_recving + 坏块一次性取走 ──
        done_sending, done_recving = d.worker_connector.get_finished(set())
        first_take = d.worker_connector.get_block_ids_with_load_errors()
        second_take = d.worker_connector.get_block_ids_with_load_errors()
        doc["reporting"] = {
            "done_recving": sorted(done_recving),
            "done_sending_still_empty": sorted(done_sending) == [],
            "invalid_blocks_first_take": sorted(first_take),
            "invalid_blocks_second_take": sorted(second_take),
            "take_and_clear": not second_take,
            "next_step_for_scheduler": "ch16 失败回滚：fail=FINISHED_ERROR / "
            "recompute=第一个坏块截断重算+补登记清零",
        }

        # ── 4) P 侧对照：WRITE 失败为何不能走同一条路 ──
        p.worker._handle_failed_transfer("req-p-write", None)
        doc["p_side_write_failure"] = {
            "p_recving_metadata_entries": len(p.worker._recving_metadata),
            "p_invalid_blocks_reported": sorted(
                p.worker_connector.get_block_ids_with_load_errors()
            ),
            "note": "P 推模式 WRITE 纯出站、没有 recv 元数据可标 invalid——"
            "只能弃单等租约/watchdog（push_worker._xfer_blocks 的 except 路径）",
        }
    finally:
        p.close()
        d.close()

    dump("m11", doc)


if __name__ == "__main__":
    main()
