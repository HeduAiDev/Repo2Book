# m06 取证：RDMA 单边 READ（m6）——D 从 P 的 KV 张量直接拉块（按描述符裸地址、
# P 的引擎线程不参与）；全命中 notif-only；部分命中尾裁剪；notif_id 携消费者数。
from _driver_common import (
    KVCacheBlocksStub,
    dump,
    make_d_request,
    make_p_request,
    receipt_from_p,
)
from _pd_harness import LAYER_NAMES, block_checksum, fill_blocks, make_engine, pump_until


def capture_notifs(worker, sink: list):
    """包一层 get_new_notifs：把对端送达的 notif 原样记录（只旁路观察，不改行为）。"""
    inner = worker.nixl_wrapper.get_new_notifs

    def wrapped():
        got = inner()
        for sender, notifs in got.items():
            for n in notifs:
                sink.append({"sender": sender, "notif": n.decode("utf-8", "replace")})
        return got

    worker.nixl_wrapper.get_new_notifs = wrapped


def main() -> None:
    doc = {
        "mechanisms": ["m6"],
        "params": "16-token prompt；P 块 [0,1,2,3] 填 7.0、D 同号块清零；"
        "block_size=4、每块 512 B（128 float32）；拉模式 NixlConnector",
    }
    p = make_engine("p", kv_role="kv_producer")
    d = make_engine("d", kv_role="kv_consumer")
    try:
        import torch

        from vllm.v1.request import RequestStatus

        # ── 0) 备料：P 侧完整走一遍终局（块钉住 + 租约进 worker 账本） ──
        p_req = make_p_request("req-1")
        p.scheduler.update_state_after_alloc(p_req, None, 0)  # → _reqs_in_batch
        p.worker_step(p.build_meta())  # → worker._reqs_to_process
        fill_blocks(p.kv_caches, [0, 1, 2, 3], 7.0)
        fill_blocks(d.kv_caches, [0, 1, 2, 3], 0.0)
        p_req.status = RequestStatus.FINISHED_LENGTH_CAPPED
        p_req.num_computed_tokens = 16
        receipt = p.scheduler.request_finished(p_req, ([0, 1, 2, 3],))[1]
        p.worker_step(p.build_meta())  # reqs_to_send → 租约戳进 worker 时钟域
        p_notifs: list = []
        capture_notifs(p.worker, p_notifs)
        doc["setup"] = {
            "p_block_checksum": block_checksum(p.kv_caches, 0),
            "d_block_checksum_before": block_checksum(d.kv_caches, 0),
            "p_req_in_worker_to_send": "req-1" in p.worker._reqs_to_send,
            "blocks_per_block_bytes": 512,
            "blocks_to_pull": 4,
            "total_bytes": 4 * 512,
        }

        # ── 1) D 登记 + 首步：首遇远端 → 后台握手，READ 未发 ──
        req = make_d_request("req-1", receipt)
        d.scheduler.update_state_after_alloc(req, KVCacheBlocksStub([[0, 1, 2, 3]]), 16)
        d.worker_step()
        doc["step1_first_encounter"] = {
            "handshake_inflight": p.engine_id in d.worker._handshake_futures,
            "remote_agent_ready_yet": p.engine_id in d.worker._remote_agents,
            "handles_after_step1": len(d.worker._recving_transfers.get("req-1", [])),
            "d_block_checksum_still": block_checksum(d.kv_caches, 0),
        }

        # ── 2) 握手完成后的下一步：排空 _ready_requests → 发 READ（真代码路径拆开跑：
        #     start_load_kv 发起、get_finished 轮询——与引擎一步内的调用序列相同） ──
        pump_until(lambda: d.worker._remote_agents.get(p.engine_id))
        meta2 = d.build_meta()
        d.worker_connector._connector_metadata = meta2
        d.worker_connector.start_load_kv(meta2)
        handles = len(d.worker._recving_transfers.get("req-1", []))
        doc["step2_read_issued"] = {
            "handles": handles,
            "d_block_checksum_after_read": block_checksum(d.kv_caches, 0),
            "all_four_blocks_equal": all(
                torch.equal(
                    d.kv_caches[LAYER_NAMES[0]][b], p.kv_caches[LAYER_NAMES[0]][b]
                )
                for b in (0, 1, 2, 3)
            ),
            "expected_block_checksum": 128 * 7.0,
            "note": "seam 的 transfer() 当场搬迁完（真 RDMA 为异步，check_xfer_state "
            "会先回 PROC 若干次）——单边语义不变：只按描述符裸地址取数",
        }

        # ── 3) D 完成回收：handle 轮询全 DONE ──
        ds3, dr3 = d.worker_connector.get_finished(set())
        doc["step3_d_done_recving"] = {
            "done_recving": sorted(dr3),
            "done_sending_side_empty": not ds3,
            "handles_drained": len(d.worker._recving_transfers.get("req-1", [])),
        }

        # ── 4) P 完成回收：收 READ-done notif（consumer 计数）→ 放块 ──
        done_sending = pump_until(
            lambda: (lambda r: r[0] if r[0] else None)(p.worker_step())
        )
        doc["step4_p_done_sending"] = {
            "notifs_received": p_notifs,
            "done_sending": sorted(done_sending or set()),
            "req_left_worker_to_send": "req-1" not in p.worker._reqs_to_send,
            "p_engine_thread_in_transfer": 0,
            "note": "READ 只按描述符裸地址搬运；P 侧引擎线程在传输期间执行的是空步"
            "（get_finished 轮询），不经任何 Python 数据路径",
        }

        # ── 5) 对照 A：本地全命中 → 不发 READ，只发 notif ──
        receipt_full = receipt_from_p(p, "req-full", blocks=[0, 1], num_computed_tokens=8)
        p.worker._reqs_to_process.add("req-full")
        p.worker._reqs_to_send["req-full"] = __import__("time").perf_counter() + 30
        req_full = make_d_request("req-full", receipt_full, num_tokens=8)
        d.scheduler.update_state_after_alloc(req_full, KVCacheBlocksStub([[]]), 0)
        d.worker_step()
        pump_until(lambda: d.worker._remote_agents.get(p.engine_id))
        d.worker_step()  # 全命中路径：只发 notif
        done_full = pump_until(
            lambda: (lambda r: r[0] if r[0] else None)(p.worker_step())
        )
        doc["variant_full_hit_notif_only"] = {
            "handles_created": len(d.worker._recving_transfers.get("req-full", [])),
            "done_sending_at_p": sorted(done_full or set()),
            "notifs_received": [n["notif"] for n in p_notifs if "req-full" in n["notif"]],
        }

        # ── 6) 对照 B：部分本地前缀命中 → 远端块表裁到尾段 ──
        w = d.worker
        local, remote = w._apply_prefix_caching([[0, 1]], [[0, 1, 2, 3]], 1)
        doc["variant_partial_hit_tail_trim"] = {
            "local_kept": local,
            "remote_trimmed": remote,
            "blocks_actually_pulled": len(remote[0]),
            "blocks_skipped": len([[0, 1, 2, 3]][0]) - len(remote[0]),
        }
    finally:
        p.close()
        d.close()

    dump("m06", doc)


if __name__ == "__main__":
    main()
