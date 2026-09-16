# m05 取证：带外握手（m5）——side channel 待命 / 真 ZMQ REQ→ROUTER / compat hash 门 /
# RTT 中点时钟偏移 / 单飞 future / hash 因子敏感性。
from _driver_common import dump
from _pd_harness import make_engine, make_vllm_config, pump_until


def main() -> None:
    doc = {
        "mechanisms": ["m5"],
        "params": "P/D 两台引擎（同机、真 ZMQ tcp 套接字）；worker 各注册 1 个 KV 区域"
        "（8 块 × 512 B）；握手走 base_worker._nixl_handshake 真 ZMQ REQ",
    }
    p = make_engine("p", kv_role="kv_producer")
    d = make_engine("d", kv_role="kv_consumer")
    try:
        import time

        # ── 1) P 启动待命：ROUTER 监听线程 + 常驻握手元数据 ──
        sched = p.scheduler
        payload = p.worker_connector.get_handshake_metadata()
        # agent_metadata_bytes 是 msgpack 编码的 NixlAgentMetadata，解开取字段
        import msgspec

        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
            NixlAgentMetadata,
        )

        am = msgspec.msgpack.decode(payload.agent_metadata_bytes, type=NixlAgentMetadata)
        doc["p_side_channel_standby"] = {
            "listener_thread_alive": bool(
                sched._nixl_handshake_listener_t
                and sched._nixl_handshake_listener_t.is_alive()
            ),
            "listener_name": sched._nixl_handshake_listener_t.name,
            "compat_hash_len": len(payload.compatibility_hash),
            "compat_hash_prefix": payload.compatibility_hash[:8],
            "matches_worker_hash": payload.compatibility_hash == p.worker.compat_hash,
            "agent_num_blocks": am.num_blocks,
            "agent_block_lens": am.block_lens,
            "agent_block_size": am.block_size,
            "agent_kv_cache_layout": am.kv_cache_layout,
            "agent_attn_backend": am.attn_backend_name,
            "agent_physical_blocks_per_logical": am.physical_blocks_per_logical_kv_block,
            "two_phase_payload_fields": ["compatibility_hash", "agent_metadata_bytes"],
        }

        # ── 2) D→P 真 ZMQ 握手：compat 门 + 时钟偏移 ──
        d_worker = d.worker
        p_sched = p.scheduler
        t0 = time.perf_counter()
        agents, clock_offset = d_worker._nixl_handshake(
            p_sched.side_channel_host,
            p_sched.side_channel_port,
            p.worker.world_size,
            p.engine_id,
        )
        rtt_ms = (time.perf_counter() - t0) * 1000
        doc["d_to_p_handshake"] = {
            "agents": {str(k): v for k, v in agents.items()},
            "clock_offset_s": round(clock_offset, 6),
            "clock_offset_abs_lt_1s_same_host": abs(clock_offset) < 1.0,
            "whole_handshake_wall_ms": round(rtt_ms, 3),
            "descs_registered_by_add_remote_agent": p.engine_id
            in d_worker.dst_xfer_side_handles,
            "agent_map_lands_via_done_callback": p.engine_id in d_worker._remote_agents,
            "note": "直呼 _nixl_handshake 只落描述符；_remote_agents/_engine_clock_offset"
            "由 _ensure_handshake 的完成回调落账（见下）",
        }

        # ── 3) 不兼容门：hash 不匹配在解 agent metadata 之前就炸 ──
        d_worker.compat_hash = "0" * 64
        err = None
        try:
            d_worker._nixl_handshake(
                p_sched.side_channel_host,
                p_sched.side_channel_port,
                p.worker.world_size,
                p.engine_id,
            )
        except RuntimeError as e:
            err = str(e).splitlines()[0]
        doc["compat_gate_mismatch"] = {
            "forced_hash_len": 64,
            "error_first_line": err[:64],
            "gate_before_agent_decode": err is not None
            and "compatibility hash mismatch" in err,
        }
        # 还原（用 P 的真 hash 注册回去，供后续机制复用）
        d_worker.compat_hash = p.worker.compat_hash
        # 走一遍 _ensure_handshake：完成回调把 agent 表与时钟偏移落账
        d.worker._ensure_handshake(
            p.engine_id, p_sched.side_channel_host, p_sched.side_channel_port, 1
        ).result(timeout=10)
        doc["d_to_p_handshake"]["agent_map_lands_via_done_callback"] = (
            p.engine_id in d_worker._remote_agents
        )
        doc["d_to_p_handshake"]["engine_clock_offset_recorded"] = (
            p.engine_id in d_worker._engine_clock_offset
        )

        # ── 4) 单飞：同一远端只有一个在飞的握手 ──
        d2 = None
        from _pd_harness import make_engine as mk

        d2 = mk("d2", kv_role="kv_consumer")
        try:
            fut = d2.worker._ensure_handshake(
                p.engine_id, p_sched.side_channel_host, p_sched.side_channel_port, 1
            )
            fut2 = d2.worker._ensure_handshake(
                p.engine_id, p_sched.side_channel_host, p_sched.side_channel_port, 1
            )
            fut.result(timeout=10)
            again = d2.worker._ensure_handshake(
                p.engine_id, p_sched.side_channel_host, p_sched.side_channel_port, 1
            )
            doc["single_flight"] = {
                "first_returns_future": fut is not None,
                "second_inflight_returns_future": fut2 is not None,
                "after_completion_returns_none": again is None,
                "remote_agent_registered": p.engine_id in d2.worker._remote_agents,
            }
        finally:
            d2.close()

        # ── 5) hash 因子敏感性：动一个因子、hash 必变 ──
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
            NIXL_CONNECTOR_VERSION,
            compute_nixl_compatibility_hash,
        )

        base_cfg = make_vllm_config(engine_id="e", kv_role="kv_producer")
        other_model = make_vllm_config(engine_id="e", kv_role="kv_producer")
        other_model.model_config.model = "dummy-v2"
        h_base = compute_nixl_compatibility_hash(base_cfg, "FLASH_ATTN", False)
        h_model = compute_nixl_compatibility_hash(other_model, "FLASH_ATTN", False)
        h_backend = compute_nixl_compatibility_hash(base_cfg, "TRITON_ATTN", False)
        doc["hash_factors"] = {
            "factor_count": 10,
            "vllm_version": "0.27.1",
            "nixl_connector_version": NIXL_CONNECTOR_VERSION,
            "hash_len": len(h_base),
            "model_change_flips_hash": h_base != h_model,
            "backend_change_flips_hash": h_base != h_backend,
            "factors_not_in_hash": "tensor_parallel_size / block_size / kv_cache_layout"
            "（运行期在 _validate_remote_agent_handshake 校验，支持异构部署）",
        }
    finally:
        p.close()
        d.close()

    dump("m05", doc)


if __name__ == "__main__":
    main()
