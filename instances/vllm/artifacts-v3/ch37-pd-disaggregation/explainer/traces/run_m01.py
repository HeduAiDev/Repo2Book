# m01 取证：双实例部署形态（m1）——注册名解析 / kv_role 门 / engine_id / facade 半边 / HND。
from _driver_common import dump, make_p_request
from _pd_harness import (
    BLOCK_SIZE,
    HEAD_SIZE,
    LAYER_NAMES,
    NUM_BLOCKS,
    NUM_KV_HEADS,
    make_engine,
    make_kv_config,
    make_vllm_config,
)


def main() -> None:
    doc = {
        "mechanisms": ["m1"],
        "params": "两台完整引擎（P=kv_role:kv_producer、D=kv_role:kv_consumer），"
        f"block_size={BLOCK_SIZE}、num_kv_heads={NUM_KV_HEADS}、head_size={HEAD_SIZE}、"
        f"num_blocks={NUM_BLOCKS}、1 层（{LAYER_NAMES[0]}）、dtype=float32",
    }

    # ── 1) factory 注册名解析（3 个名字、1 个别名） ──
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
        NixlConnector,
        NixlPullConnector,
        NixlPushConnector,
    )

    resolution = {}
    for name, cls in (
        ("NixlConnector", NixlPullConnector),
        ("NixlPullConnector", NixlPullConnector),
        ("NixlPushConnector", NixlPushConnector),
    ):
        cfg = make_vllm_config(engine_id="e", kv_role="kv_producer", connector=name)
        resolution[name] = KVConnectorFactory.get_connector_class(
            cfg.kv_transfer_config
        ).__name__
    doc["factory_resolution"] = {
        "names": 3,
        "alias_pairs": 1,
        "NixlConnector is NixlPullConnector": NixlConnector is NixlPullConnector,
        "resolved": resolution,
    }

    # ── 2) kv_role 部署门：缺省 / 非法值 ──
    from vllm.config.kv_transfer import KVTransferConfig

    gate = {}
    try:
        KVTransferConfig(kv_connector="NixlConnector")
    except ValueError as e:
        gate["missing_kv_role"] = str(e).splitlines()[0][:48]
    try:
        KVTransferConfig(kv_connector="NixlConnector", kv_role="kv_watcher")
    except ValueError as e:
        gate["unsupported_kv_watcher"] = str(e).splitlines()[0][:48]
    a = KVTransferConfig(kv_connector="NixlConnector", kv_role="kv_producer")
    b = KVTransferConfig(kv_connector="NixlConnector", kv_role="kv_consumer")
    gate["default_engine_id_is_uuid4"] = {
        "a_len": len(a.engine_id),
        "b_len": len(b.engine_id),
        "distinct": a.engine_id != b.engine_id,
        "a_is_producer": a.is_kv_producer,
        "b_is_consumer": b.is_kv_consumer,
    }
    doc["kv_role_gate"] = gate

    # ── 3) 真两台引擎：facade 按 role 只建半边 + side channel 待命 ──
    p = make_engine("p", kv_role="kv_producer")
    d = make_engine("d", kv_role="kv_consumer")
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        from vllm.distributed.kv_transfer.kv_connector.v1.nixl.connector import (
            NixlPullConnector,
        )

        cfg = make_vllm_config(engine_id="e", kv_role="kv_producer")
        kv_config = make_kv_config()
        sched = NixlPullConnector(cfg, KVConnectorRole.SCHEDULER, kv_config)
        worker = NixlPullConnector(cfg, KVConnectorRole.WORKER, kv_config)
        # 对 P/D 两台引擎真建的 4 个 facade 逐一数非空半边（全局口径），
        # ad-hoc 的 sched/worker 两个演示对象只作 role 分支展示。
        objs = [
            p.scheduler_connector,
            p.worker_connector,
            d.scheduler_connector,
            d.worker_connector,
        ]
        doc["facade_halves"] = {
            "per_engine_connector_copies": 2,
            "SCHEDULER_role": {
                "connector_scheduler": type(sched.connector_scheduler).__name__,
                "connector_worker": None,
            },
            "WORKER_role": {
                "connector_scheduler": None,
                "connector_worker": type(worker.connector_worker).__name__,
            },
            "total_connector_objects": len(objs),
            "non_none_halves": sum(
                1
                for o in objs
                for half in (o.connector_scheduler, o.connector_worker)
                if half is not None
            ),
        }

        doc["engines"] = {
            "p": {
                "engine_id": p.engine_id,
                "kv_role": p.vllm_config.kv_transfer_config.kv_role,
                "side_channel_host": p.scheduler.side_channel_host,
                "side_channel_port": p.scheduler.side_channel_port,
                "handshake_listener_alive": bool(
                    p.scheduler._nixl_handshake_listener_t
                    and p.scheduler._nixl_handshake_listener_t.is_alive()
                ),
                "worker_compat_hash_len": len(p.worker.compat_hash),
            },
            "d": {
                "engine_id": d.engine_id,
                "kv_role": d.vllm_config.kv_transfer_config.kv_role,
                "side_channel_port": d.scheduler.side_channel_port,
                "distinct_ports": p.scheduler.side_channel_port
                != d.scheduler.side_channel_port,
            },
        }

        # ── 4) 布局门：非 MLA 强制 HND；MLA 回退 None ──
        doc["required_kvcache_layout"] = {
            "non_mla": NixlPullConnector.get_required_kvcache_layout(cfg),
            "mla": None,  # 占位，下面真算
        }
        cfg_mla = make_vllm_config(engine_id="e", kv_role="kv_producer")
        cfg_mla.model_config.use_mla = True
        doc["required_kvcache_layout"]["mla"] = NixlPullConnector.get_required_kvcache_layout(
            cfg_mla
        )

        # ── 5) kv_both 已弃用但仍构建（告警路径可观察） ──
        both = make_engine("both", kv_role="kv_both")
        try:
            doc["kv_both_deprecated"] = {
                "still_builds_scheduler_half": both.scheduler_connector.connector_scheduler
                is not None,
                "warning_anchor": "nixl/connector.py:L93-L99",
            }
        finally:
            both.close()

        # ── 6) HND 张量几何（m6 的字节账预置） ──
        spec = p.kv_config.kv_cache_groups[0].kv_cache_spec
        cache = p.kv_caches[LAYER_NAMES[0]]
        doc["hnd_geometry"] = {
            "cache_shape": list(cache.shape),
            "num_blocks": NUM_BLOCKS,
            "block_size": BLOCK_SIZE,
            "num_kv_heads": NUM_KV_HEADS,
            "head_size": HEAD_SIZE,
            "page_size_bytes_per_block": spec.page_size_bytes,
            "block_numel": int(cache[0].numel()),
            "dtype_bytes": 4,
        }

        # P 引擎最小烟测：kv_role 门通过 + 注册名落到 Pull
        doc["smoke_p_engine_role"] = {
            "scheduler_connector_class": type(p.scheduler_connector).__name__,
            "worker_connector_class": type(p.worker_connector).__name__,
        }
    finally:
        p.close()
        d.close()

    dump("m01", doc)


if __name__ == "__main__":
    main()
