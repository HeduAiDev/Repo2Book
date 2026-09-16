# m04 取证：D 查命中·全 prompt 异步拉（m4）——(N−computed, True) 三档 +
# do_remote_decode 骨架的 kv_recompute_threshold 边界 + update_state_after_alloc 一次性翻转。
from _driver_common import (
    KVCacheBlocksStub,
    dump,
    make_d_request,
    make_p_request,
)
from _pd_harness import make_engine


def make_p_leg_request(request_id: str, num_tokens: int, remote_num_tokens: int):
    """P 腿（多轮回拉方向）请求：do_remote_decode=True + 远端坐标（骨架的输入）。"""
    from vllm.sampling_params import SamplingParams
    from vllm.v1.request import Request

    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=SamplingParams(
            max_tokens=8,
            extra_args={
                "kv_transfer_params": {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                    "remote_block_ids": [[0, 1, 2, 3]],
                    "remote_engine_id": "engine-d",
                    "remote_request_id": "req-old",
                    "remote_host": "127.0.0.1",
                    "remote_port": 5600,
                    "tp_size": 1,
                    "remote_num_tokens": remote_num_tokens,
                }
            },
        ),
    )


def main() -> None:
    doc = {
        "mechanisms": ["m4"],
        "params": "D 腿 16-token prompt；P 腿骨架 128-token prompt 对照 "
        "remote_num_tokens=32/64/128 三档；kv_recompute_threshold 默认 64",
    }
    p = make_engine("p", kv_role="kv_producer")
    d = make_engine("d", kv_role="kv_consumer")
    try:
        # ── 1) P 引擎真终局产回执，D 腿请求挂载 ──
        from _driver_common import receipt_from_p

        receipt = receipt_from_p(p, "req-1", blocks=[0, 1, 2, 3], num_computed_tokens=16)
        req = make_d_request("req-1", receipt)
        sched = d.scheduler

        doc["d_leg_hit"] = {
            "computed_0": list(sched.get_num_new_matched_tokens(req, 0)),
            "computed_12": list(sched.get_num_new_matched_tokens(req, 12)),
            "computed_16": list(sched.get_num_new_matched_tokens(req, 16)),
            "prompt_tokens": 16,
        }

        # ── 2) P 腿骨架（do_remote_decode）：短拉不划算判定 ──
        doc["kv_recompute_threshold"] = sched.kv_recompute_threshold
        r32 = make_p_leg_request("r32", 128, 32)
        r64 = make_p_leg_request("r64", 128, 64)
        r128 = make_p_leg_request("r128", 128, 128)
        doc["p_leg_skeleton"] = {
            "remote_32": list(sched.get_num_new_matched_tokens(r32, 0)),
            "remote_64_boundary": list(sched.get_num_new_matched_tokens(r64, 0)),
            "remote_128": list(sched.get_num_new_matched_tokens(r128, 0)),
            "note": "count<threshold(64) → (0, False) 本地重算更划算；=64 过界即拉",
        }

        # ── 3) update_state_after_alloc：登记未哈希块 + 一次性翻转 ──
        blocks = KVCacheBlocksStub([[0, 1, 2, 3]])
        sched.update_state_after_alloc(req, blocks, 16)
        first = sched._reqs_need_recv["req-1"]
        doc["register_need_recv"] = {
            "registered_block_ids": first[1],
            "do_remote_prefill_flipped": req.kv_transfer_params["do_remote_prefill"]
            is False,
            "remote_blocks_processed": req.kv_transfer_params[
                "_remote_blocks_processed"
            ]
            is True,
        }
        # 重入（抢占后重调度）：块表变了也不重发
        sched.update_state_after_alloc(req, KVCacheBlocksStub([[4, 5]]), 16)
        doc["register_need_recv"].update(
            {
                "reentry_block_ids_unchanged": sched._reqs_need_recv["req-1"][1]
                == first[1],
                "reentry_blocks_seen": [[4, 5]],
            }
        )

        # ── 4) 本地全命中对照：未哈希块为空（m6 的 notif-only 输入） ──
        receipt_full = receipt_from_p(p, "req-full", blocks=[0, 1], num_computed_tokens=8)
        req_full = make_d_request("req-full", receipt_full, num_tokens=8)
        sched.update_state_after_alloc(req_full, KVCacheBlocksStub([[]]), 0)
        doc["full_local_hit"] = {
            "registered_block_ids": sched._reqs_need_recv["req-full"][1],
            "note": "空列表 = 本地全命中（远端块不用拉，只发 notif）",
        }

        # ── 5) 部分本地前缀命中对照：本地 2 块已缓存（m6 尾裁剪的输入） ──
        receipt_part = receipt_from_p(p, "req-part", blocks=[0, 1, 2, 3], num_computed_tokens=16)
        req_part = make_d_request("req-part", receipt_part)
        sched.update_state_after_alloc(req_part, KVCacheBlocksStub([[0, 1]]), 8)
        doc["partial_local_hit"] = {
            "local_unhashed": sched._reqs_need_recv["req-part"][1],
            "remote_blocks": req_part.kv_transfer_params["remote_block_ids"],
        }

        # ── 6) build_connector_meta 把账本换成 ReqMeta（一次交接即清账） ──
        meta = d.build_meta()
        doc["meta_handoff"] = {
            "reqs_in_meta": sorted(meta.reqs_to_recv.keys()),
            "req_meta_remote_engine": meta.reqs_to_recv["req-1"].remote.engine_id,
            "req_meta_remote_blocks": meta.reqs_to_recv["req-1"].remote.block_ids,
            "scheduler_need_recv_cleared": sched._reqs_need_recv == {},
        }
    finally:
        p.close()
        d.close()

    dump("m04", doc)


if __name__ == "__main__":
    main()
