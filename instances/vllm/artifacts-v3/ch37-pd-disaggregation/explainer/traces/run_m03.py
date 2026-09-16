# m03 取证：P 终局·租约钉住与回执（m3）——双条件交接 + delay_free + 30s 租约戳 + 回执字典
# + 引擎侧块生命周期（_free_request 延迟释放 → get_finished 报完成才放块）。
from _driver_common import dump, make_p_request
from _pd_harness import make_engine


def main() -> None:
    doc = {
        "mechanisms": ["m3"],
        "params": "16-token prompt、P 块 [0,1,2,3]；三种终局对照："
        "LENGTH_CAPPED / STOPPED / ABORTED；租约默认 30s",
    }
    p = make_engine("p", kv_role="kv_producer")
    try:
        import time

        from vllm.v1.request import RequestStatus

        sched = p.scheduler

        # ── 1) 正常终局（LENGTH_CAPPED，disaggregator 发单的 max_tokens=1 触发） ──
        req = make_p_request("req-p")
        req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        req.num_computed_tokens = 16
        t0 = time.perf_counter()
        delay_free, receipt = sched.request_finished(req, ([0, 1, 2, 3],))
        deadline = sched._reqs_need_send["req-p"]
        doc["terminal_length_capped"] = {
            "kv_lease_duration": sched._kv_lease_duration,
            "delay_free_blocks": delay_free,
            "deadline_minus_t0_s": round(deadline - t0, 3),
            "in_reqs_need_send": "req-p" in sched._reqs_need_send,
            "receipt_keys": len(receipt),
            "remote_block_ids": receipt["remote_block_ids"],
            "remote_num_tokens": receipt["remote_num_tokens"],
            "remote_request_id": receipt["remote_request_id"],
            "tp_size": receipt["tp_size"],
            "blocks_pinned": len(receipt["remote_block_ids"][0]),
        }

        # ── 2) 正常终局（STOPPED——命中 stop 串同样交接） ──
        req2 = make_p_request("req-stop")
        req2.status = RequestStatus.FINISHED_STOPPED
        req2.num_computed_tokens = 12
        delay_free2, receipt2 = sched.request_finished(req2, ([0, 1],))
        doc["terminal_stopped"] = {
            "delay_free_blocks": delay_free2,
            "remote_num_tokens": receipt2["remote_num_tokens"],
            "remote_block_ids": receipt2["remote_block_ids"],
        }

        # ── 3) 异常终局（ABORTED）不交接 ──
        req3 = make_p_request("req-abort")
        req3.status = RequestStatus.FINISHED_ABORTED
        delay_free3, receipt3 = sched.request_finished(req3, ([0, 1],))
        doc["terminal_aborted"] = {
            "delay_free_blocks": delay_free3,
            "receipt_is_none": receipt3 is None,
            "in_reqs_not_processed": "req-abort" in sched._reqs_not_processed,
            "in_reqs_need_send": "req-abort" in sched._reqs_need_send,
        }

        # ── 4) 引擎侧块生命周期：_free_request 摘回执、块延迟释放 ──
        esched = p.core.scheduler
        esched.kv_cache_manager.set_blocks("req-p", ([0, 1, 2, 3],))
        esched.requests["req-p"] = req
        kv_xfer_params, _ec = esched._free_request(req)
        doc["engine_side_lifecycle"] = {
            "free_request_returns_receipt": kv_xfer_params is not None,
            "receipt_out_of_engine_keys": len(kv_xfer_params or {}),
            "blocks_freed_immediately": "req-p" in esched.kv_blocks_freed,
            "request_still_tracked": "req-p" in esched.requests,
        }
        # worker 报完成（D 已读完）→ 调度器放块
        from vllm.v1.outputs import KVConnectorOutput

        esched._update_from_kv_xfer_finished(
            KVConnectorOutput(finished_sending={"req-p"})
        )
        doc["engine_side_lifecycle"].update(
            {
                "blocks_freed_after_finished_sending": "req-p"
                in esched.kv_blocks_freed,
                "request_removed": "req-p" not in esched.requests,
            }
        )
        # 回执随响应出引擎的真源码锚（本章精简版把响应体段收窄为返回值）
        doc["engine_side_lifecycle"]["receipt_export_real_anchor"] = (
            "vllm/v1/core/sched/scheduler.py:L1901-L1935"
        )
    finally:
        p.close()

    dump("m03", doc)


if __name__ == "__main__":
    main()
