# m12 取证：混合模型传输剪裁（m12）——get_exchange_clipped_blocks 的 SWA 尾剪裁
# （窗口内尾块才过线）+ host buffer 旁路的旗标面（宿主 CPU 平台恒关）。
from _driver_common import dump
from _pd_harness import make_engine, make_kv_config, make_vllm_config


def main() -> None:
    import torch

    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        KVCacheTensor,
        SlidingWindowSpec,
    )

    doc = {
        "mechanisms": ["m12"],
        "params": "混合配置：full 组 + SWA 组（sliding_window=10 tokens、block_size=4、"
        "同几何 2 头×8 维×float32 → 每块 512 B）；对照纯 full 配置",
    }

    fa = FullAttentionSpec(
        block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float32
    )
    sw = SlidingWindowSpec(
        block_size=4,
        num_kv_heads=2,
        head_size=8,
        dtype=torch.float32,
        sliding_window=10,
    )
    tensor_bytes = fa.page_size_bytes * 8
    hybrid_cfg = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[
            KVCacheTensor(size=tensor_bytes, shared_by=["layer.0.attn"]),
            KVCacheTensor(size=tensor_bytes, shared_by=["layer.0.attn.swa"]),
        ],
        kv_cache_groups=[
            KVCacheGroupSpec(layer_names=["layer.0.attn"], kv_cache_spec=fa),
            KVCacheGroupSpec(layer_names=["layer.0.attn.swa"], kv_cache_spec=sw),
        ],
    )
    plain_cfg = make_kv_config()

    # ── 1) 直接构建调度器半边（不引 ZMQ 监听——listener 只在握手元数据注入时启动） ──
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.pull_scheduler import (
        NixlPullConnectorScheduler,
    )

    vc = make_vllm_config(engine_id="e-hyb", kv_role="kv_producer")
    hybrid_sched = NixlPullConnectorScheduler(vc, "e-hyb", hybrid_cfg)
    plain_sched = NixlPullConnectorScheduler(vc, "e-plain", plain_cfg)

    doc["configs"] = {
        "plain_is_hma_required": plain_sched._is_hma_required,
        "hybrid_is_hma_required": hybrid_sched._is_hma_required,
        "hybrid_blocks_per_sw": hybrid_sched.blocks_per_sw,
        "sw_window_tokens": 10,
        "cdiv_formula": "cdiv(10, 4) + 1 = 4（+1 保守吸收窗口与块边界不对齐）",
        "fa_page_bytes": fa.page_size_bytes,
        "sw_page_bytes": sw.page_size_bytes,
    }

    # ── 2) 剪裁：full 组全保留、SWA 组只留窗口内尾块 ──
    full_blocks = list(range(8))
    before = (list(full_blocks), list(full_blocks))
    after = hybrid_sched.get_exchange_clipped_blocks(before)
    doc["clipping"] = {
        "before": {"full_group": before[0], "swa_group": before[1]},
        "after": {"full_group": list(after[0]), "swa_group": list(after[1])},
        "full_kept": len(after[0]),
        "swa_kept": len(after[1]),
        "swa_dropped": len(before[1]) - len(after[1]),
        "plain_passthrough": list(
            plain_sched.get_exchange_clipped_blocks((list(full_blocks),))[0]
        ),
        "invariant": "交接的是『可恢复计算所需的最小集』：SWA 组窗口外的头块"
        "不再被任何注意力读到，过线就是白搬",
    }

    # ── 3) host buffer 旁路的旗标面（宿主观察；不引 ZMQ——同上直构） ──
    from vllm.platforms import current_platform

    vc_cpu = make_vllm_config(
        engine_id="e-cpu", kv_role="kv_producer", kv_buffer_device="cpu"
    )
    cpu_sched = NixlPullConnectorScheduler(vc_cpu, "e-cpu", plain_cfg)
    doc["host_buffer_flag"] = {
        "configured_kv_buffer_device": "cpu",
        "host_platform_use_host_buffer": cpu_sched.use_host_buffer,
        "host_platform_device_type": current_platform.device_type,
        "pin_behavior_on_cuda": "真 CUDA 平台此处为 True——KV 先落主机缓冲再传输"
        "（save_kv_to_host / sync_recved_kv_to_device，非 CUDA 平台必经的实现体"
        "已按减法计划删除，旗标与判定保留）",
        "pin_source": "base_scheduler.py:L88-L93",
    }

    dump("m12", doc)


if __name__ == "__main__":
    main()
