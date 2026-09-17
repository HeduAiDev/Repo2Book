# ch38《KV 池化》精简版测试 —— TDD（先于实现）。
#
# 测的是精简版复现真实 vLLM v0.27.1 (6e448d0ea) 的**可观察行为**
# （锚点 = vllm/... 行号，基线 v0.27.1 现核）。
#
# 行为清单（按 dossier.mechanisms 对账）：
# - m1 池的构成：OffloadingConnector best-effort（requires_kv_delivery=False）、
#   build_offloading_config→OffloadingSpecFactory.create_spec 按 role 建半边、
#   spec_name 注册表（CPU/Tiering/out-of-tree/未知拒绝）、HND 布局要求
# - m2 CPU 池开张：num_blocks = cpu_bytes_to_use // round_up(
#   worker_kv_bytes_per_block×world_size×blocks_per_chunk, PAGESIZE)、
#   cpu_page_size_per_worker、缺 cpu_bytes_to_use 拒启
# - m3 满块采集：『每 chunk 取最后一个 GPU 块』（1 5 6 7 2 4 9 3 8 → 6 4 8）、
#   block_id=0 跳过、offload_prompt_only / max_offload_tokens 双封顶、
#   next_stored_chunk_idx 增量游标
# - m4 池准入：store_threshold 频次过滤、已存去重、驱逐不足 None 拒收、
#   CachePolicy(LRU/ARC).evict 只逐空闲+保护集、touch 刷新鲜、
#   complete_store ref_cnt -1→0 翻转、失败回收
# - m5 延迟一步提交：prepare_store_kv 只入队、start_kv_transfers 才提交
#   （NOTE(orozery)）、wait_for_layer_load/save_kv_layer/wait_for_save 全空
# - m6 DMA 描述符：compute_sub_block_ptrs 指针数学（含 skip_count 半块跳越）
# - m7 四态查找：HIT/HIT_PENDING(defer→None)/RETRY(不 break)/MISS(break)、
#   SWA 从尾数连续窗口、多组收敛重扫、touch 含 GPU 命中块
# - m8 加载登记：update_state_after_alloc→load job、GPULoadStoreSpec
#   group_sizes/block_indices 半块对齐
# - m9 完成回收：completed_jobs {job:count} 聚满 world_size 才结算、
#   store 永不发 finished_sending
# - m10 flush 围栏：块复用 ∩ _block_id_to_pending_jobs → jobs_to_flush →
#   handle_preemptions 抢先提交 + wait
# - m11 分层池：primary 先查→secondary 命中即 promotion→RETRY→
#   on_schedule_end 批量 submit_load→complete_write→HIT；cascade 全层下推；
#   store_threshold 分层禁用 raise
# - m12 chunk 粒度：block_size/blocks_per_chunk 二选一、FileMapper 三级分桶命名
# - m13 P2P：三角色键解析、PYTHONHASHSEED 未设拒启、协议消息校验
# - m14 生态：MooncakeStore lookup_async None=稍后再问、LMCache 双路 lazy
#   import、MultiConnector 列表顺序首个命中数>0 者获加载权
# - m15 事件面：BlockStored/BlockRemoved 翻译（self-describing 与占位两态）
# - m16 请求级旋钮：max_offload_tokens / kv_load_tiers TierFilter
# - WC6 init_none_hash：PYTHONHASHSEED 固定→可复现种子
#
# 运行：cd instances/vllm/artifacts-v3/ch38-kv-pooling && python -m pytest tests/ -q
from __future__ import annotations

import ctypes
import os
import threading

import numpy as np
import pytest
import torch
import zmq

from _kv_harness import (
    BLOCK_SIZE,
    Engine,
    FakeBlock,
    HostOffloadingWorker,
    LAYER_NAMES,
    NUM_BLOCKS,
    assemble_engine,
    block_checksum,
    fill_block_hashes,
    make_blocks,
    make_kv_caches,
    make_kv_config,
    make_request,
    make_scheduler_output,
    make_vllm_config,
)


def _ctx(req_id: str = "r1"):
    from vllm.v1.kv_offload.base import ReqContext

    return ReqContext(req_id=req_id)


def _key(seed: int, group: int = 0):
    from vllm.v1.kv_offload.base import make_offload_key

    return make_offload_key(seed.to_bytes(32, "big"), group)


# ═══════════════════════ m1 池的构成与契约再嵌套 ═══════════════════════


class TestFacade:
    def test_requires_kv_delivery_false_best_effort(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
            OffloadingConnector,
        )

        # property 求值（不建实例读旗标）
        assert OffloadingConnector.requires_kv_delivery.fget(None) is False
        # best-effort 的另一面：prefer_cross_layer_blocks=True（跨层单张量布局）
        assert OffloadingConnector.prefer_cross_layer_blocks.fget(None) is True

    def test_role_split_builds_only_one_half(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
            OffloadingConnector,
        )

        kv_config = make_kv_config()
        vllm_config = make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20})
        sched = OffloadingConnector(vllm_config, KVConnectorRole.SCHEDULER, kv_config)
        work = OffloadingConnector(vllm_config, KVConnectorRole.WORKER, kv_config)
        assert sched.connector_scheduler is not None and sched.connector_worker is None
        assert work.connector_worker is not None and work.connector_scheduler is None

    def test_required_kvcache_layout_hnd(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
            OffloadingConnector,
        )

        assert OffloadingConnector.get_required_kvcache_layout(None) == "HND"

    def test_spec_factory_registry(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
            build_offloading_config,
        )
        from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
        from vllm.v1.kv_offload.factory import OffloadingSpecFactory
        from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

        kv_config = make_kv_config()
        cfg = build_offloading_config(
            make_vllm_config(
                extra_config={"cpu_bytes_to_use": 1 << 20, "spec_name": "CPUOffloadingSpec"}
            ),
            kv_config,
        )
        assert isinstance(OffloadingSpecFactory.create_spec(cfg), CPUOffloadingSpec)

        cfg = build_offloading_config(
            make_vllm_config(
                extra_config={
                    "cpu_bytes_to_use": 1 << 20,
                    "spec_name": "TieringOffloadingSpec",
                    "secondary_tiers": [],
                }
            ),
            kv_config,
        )
        assert isinstance(OffloadingSpecFactory.create_spec(cfg), TieringOffloadingSpec)

        # 默认 spec_name = CPUOffloadingSpec（不写也走 CPU）
        cfg = build_offloading_config(
            make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20}), kv_config
        )
        assert isinstance(OffloadingSpecFactory.create_spec(cfg), CPUOffloadingSpec)

    def test_spec_factory_unknown_spec_rejected(self):
        from vllm.v1.kv_offload.factory import OffloadingSpecFactory

        with pytest.raises(ValueError, match="Unsupported spec type"):
            OffloadingSpecFactory.get_spec_cls({"spec_name": "NoSuchSpec"})

    def test_spec_factory_out_of_tree_module_path(self):
        import _kv_harness as harness
        from vllm.v1.kv_offload.factory import OffloadingSpecFactory

        # out-of-tree：未注册名 + spec_module_path → importlib 加载（警告 + 取类）
        cls = OffloadingSpecFactory.get_spec_cls(
            {"spec_name": "OutTreeSpecSentinel", "spec_module_path": "_kv_harness"}
        )
        assert cls is harness.OutTreeSpecSentinel

    def test_factory_six_eco_registrations(self):
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

        # 六条生态注册行（懒加载：仅按名取类时才 import 模块）
        for name in (
            "OffloadingConnector",
            "MultiConnector",
            "MooncakeConnector",
            "MooncakeStoreConnector",
            "LMCacheConnectorV1",
            "NixlConnector",
        ):
            assert name in KVConnectorFactory._registry
        # 本章树内存在的三类可真加载
        for name in ("OffloadingConnector", "MultiConnector", "MooncakeStoreConnector"):
            assert KVConnectorFactory.get_connector_class_by_name(name).__name__ == name
        # 已删条目（删除项 6 的十条）不再注册
        with pytest.raises(ValueError):
            KVConnectorFactory.get_connector_class_by_name("DecodeBenchConnector")


# ═══════════════════════ m2 CPU 池开张：定价公式 ═══════════════════════


def _make_cpu_spec(cpu_bytes: int, world_size: int = 2, blocks_per_chunk: int = 2,
                   worker_kv_bytes_per_block: int = 512):
    from vllm.v1.kv_offload.config import (
        OffloadingCacheConfig,
        OffloadingConfig,
        OffloadingGroupConfig,
        OffloadingParallelConfig,
    )

    from vllm.v1.kv_offload.config import OffloadingModelConfig

    return OffloadingConfig(
        groups=(OffloadingGroupConfig(tokens_per_block=16, layer_names=tuple(LAYER_NAMES)),),
        model=OffloadingModelConfig(name="dummy-model", dtype="float32"),
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        enable_kv_cache_events=False,
        extra_config={
            "cpu_bytes_to_use": cpu_bytes,
            "blocks_per_chunk": blocks_per_chunk,
        },
        engine_id="e0",
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=blocks_per_chunk),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=world_size,
            tp_size=world_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            is_parallelism_agnostic=True,
        ),
    )


class TestCpuPoolSizing:
    def test_num_blocks_formula(self):
        import mmap as _mmap

        from vllm.utils.math_utils import round_up
        from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

        cpu_bytes = 100 << 20
        spec = CPUOffloadingSpec(_make_cpu_spec(cpu_bytes, world_size=2, blocks_per_chunk=2))
        # num_blocks = cpu_bytes // round_up(worker_bytes*world*chunks, PAGESIZE)
        aligned = round_up(512 * 2 * 2, _mmap.PAGESIZE)
        assert spec.num_blocks == cpu_bytes // aligned
        # 每 worker 的 CPU 页 = 对齐前的 chunk 字节 ÷ 拷贝数
        assert spec.cpu_page_size_per_worker == 512 * 2 * 2 // 2
        assert spec.kv_bytes_per_chunk == aligned

    def test_missing_cpu_bytes_rejected(self):
        from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

        cfg = _make_cpu_spec(1 << 20)
        object.__setattr__(cfg, "extra_config", {"blocks_per_chunk": 2})
        with pytest.raises(Exception, match="cpu_bytes_to_use"):
            CPUOffloadingSpec(cfg)


# ═══════════════════════ m3 满块采集（『每 chunk 取最后一个 GPU 块』） ═══════════════════════


class TestStoreCollection:
    def _engine(self, **kw):
        cfg = make_kv_config(**kw.get("kv", {}))
        vcfg = make_vllm_config(
            extra_config=kw.get("extra", {"cpu_bytes_to_use": 1 << 20}),
            world_size=kw.get("world_size", 1),
        )
        return assemble_engine(vcfg, cfg, make_kv_caches(cfg))

    def test_last_gpu_block_per_chunk_and_incremental_cursor(self):
        # blocks_per_chunk=3、GPU 块 1 5 6 7 2 4 9 3 8 → 采集 6 4 8（源码注释原例）
        from vllm.v1.kv_offload.base import get_offload_block_hash

        eng = self._engine(
            extra={
                "cpu_bytes_to_use": 1 << 20,
                "blocks_per_chunk": 3,
                "offload_prompt_only": False,
            }
        )
        req = make_request("r1", list(range(36)))  # 9 块 × block_size 4
        eng.scheduler.on_new_request(req)
        fill_block_hashes(req, 12)  # hashes_per_chunk=3 → 4 chunk 键
        block_ids = [1, 5, 6, 7, 2, 4, 9, 3, 8]
        out = make_scheduler_output(
            new_reqs=[{"req_id": "r1", "block_ids": (block_ids,)}],
            num_scheduled_tokens={"r1": 36},
        )
        meta = eng.scheduler.build_connector_meta(out)
        assert len(meta.store_jobs) == 1
        (job,) = meta.store_jobs.values()
        src_spec = job.src_spec
        # 『每 chunk 取最后一个 GPU 块』决定**哪些 chunk 入池**（3 chunk 全入）；
        # 搬运则整 chunk 逐块走：src = 全部 9 块，首块 index=0
        assert list(src_spec.block_ids) == [1, 5, 6, 7, 2, 4, 9, 3, 8]
        assert list(src_spec.group_sizes) == [9]
        assert list(src_spec.block_indices) == [0]

        # 游标推进：decode 再长 12 token（+3 块）→ 只采新 chunk [11,12,13]
        from vllm.v1.request import RequestStatus

        req.status = RequestStatus.RUNNING
        req.num_computed_tokens = 36
        req._all_token_ids.extend(range(100, 112))  # decode 长 12 token
        out2 = make_scheduler_output(
            cached_reqs=[{"req_id": "r1", "block_ids": ([11, 12, 13],)}],
            num_scheduled_tokens={"r1": 12},
        )
        meta2 = eng.scheduler.build_connector_meta(out2)
        (job2,) = meta2.store_jobs.values()
        assert list(job2.src_spec.block_ids) == [11, 12, 13]
        assert list(job2.src_spec.group_sizes) == [3]

    def test_block_id_zero_skipped(self):
        eng = self._engine()
        req = make_request("r1", list(range(16)))
        eng.scheduler.on_new_request(req)
        fill_block_hashes(req, 4)
        out = make_scheduler_output(
            new_reqs=[{"req_id": "r1", "block_ids": ([1, 0, 3, 4],)}],
            num_scheduled_tokens={"r1": 16},
        )
        meta = eng.scheduler.build_connector_meta(out)
        (job,) = meta.store_jobs.values()
        # chunk1 的尾块是 0（SWA/陈旧占位）→ 该 chunk 不入池；
        # chunk0/2/3 入池，搬运 [1, 3, 4]
        assert list(job.src_spec.block_ids) == [1, 3, 4]
        assert list(job.src_spec.group_sizes) == [3]

    def test_offload_prompt_only_and_max_offload_tokens_caps(self):
        eng = self._engine()
        # 8 块 prompt（32 token）+ 已 decode 4 token（36 token、9 块）。
        # make_request 把整表 36 token 记作 prompt；手工改回「prompt 32 + decode 4」
        # 的口径（Request 属性可写，_calc_num_offloadable_tokens 只读它）。
        req = make_request("r1", list(range(32)) + [100, 101, 102, 103])
        req.num_prompt_tokens = 32
        req.kv_transfer_params = {"max_offload_tokens": 8}  # 只卸前 2 块
        eng.scheduler.on_new_request(req)
        fill_block_hashes(req, 9)
        req.num_computed_tokens = 36
        out = make_scheduler_output(
            new_reqs=[{"req_id": "r1", "block_ids": (list(range(1, 10)),)}],
            num_scheduled_tokens={"r1": 4},
        )
        meta = eng.scheduler.build_connector_meta(out)
        (job,) = meta.store_jobs.values()
        assert list(job.src_spec.block_ids) == [1, 2]  # max_offload_tokens=8 → 2 块

        # 对照：无 max 旋钮 → offload_prompt_only（默认 True）卸 prompt 32 token = 8 块
        # （另起引擎：r1 已落位的同键会被 prepare_store 去重，混跑测不出满 8 块）
        eng2 = self._engine()
        req2 = make_request("r2", list(range(32)) + [100, 101, 102, 103])
        req2.num_prompt_tokens = 32
        eng2.scheduler.on_new_request(req2)
        fill_block_hashes(req2, 9)
        req2.num_computed_tokens = 36
        out2 = make_scheduler_output(
            new_reqs=[{"req_id": "r2", "block_ids": (list(range(1, 10)),)}],
            num_scheduled_tokens={"r2": 4},
        )
        meta2 = eng2.scheduler.build_connector_meta(out2)
        (job2,) = meta2.store_jobs.values()
        assert list(job2.src_spec.block_ids) == list(range(1, 9))  # 32 token = 8 块

    def test_prepare_store_none_skips_batch(self):
        # 页 512B → PAGESIZE 对齐后每 chunk 4096B；4096B 预算 = 池 1 块
        eng = self._engine(extra={"cpu_bytes_to_use": 4096, "blocks_per_chunk": 1})
        # 池只装 1 块且被 ref_cnt 钉住 → 新 store 驱逐失败 → None → 本批放弃
        eng.manager.prepare_store([_key(0)], _ctx())
        eng.manager.complete_store([_key(0)], _ctx())
        eng.manager.prepare_load([_key(0)], _ctx())
        req = make_request("r1", list(range(8)))
        eng.scheduler.on_new_request(req)
        fill_block_hashes(req, 2)
        out = make_scheduler_output(
            new_reqs=[{"req_id": "r1", "block_ids": ([1, 2],)}],
            num_scheduled_tokens={"r1": 8},
        )
        meta = eng.scheduler.build_connector_meta(out)
        assert meta.store_jobs == {}


# ═══════════════════════ m4 CPU 池准入与驱逐 ═══════════════════════


class TestCpuManager:
    def _mgr(self, num_blocks=4, store_threshold=1, policy="lru"):
        from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

        return CPUOffloadingManager(
            num_blocks=num_blocks,
            cache_policy=policy,
            enable_events=True,
            store_threshold=store_threshold,
        )

    def test_store_lifecycle_four_states(self):
        m = self._mgr()
        k = _key(0)
        assert m.lookup(k, _ctx()).name == "MISS"
        out = m.prepare_store([k], _ctx())
        assert out.keys_to_store == [k]
        assert m.lookup(k, _ctx()).name == "HIT_PENDING"  # ref_cnt=-1 写入中
        m.complete_store([k], _ctx())
        assert m.lookup(k, _ctx()).name == "HIT"  # 翻转可读
        # 已存块去重：再 store 同键 → keys_to_store 空
        out2 = m.prepare_store([k], _ctx())
        assert out2.keys_to_store == []

    def test_eviction_lru_order_and_touch(self):
        m = self._mgr(num_blocks=3)
        keys = [_key(i) for i in range(3)]
        for k in keys:
            m.prepare_store([k], _ctx())
        m.complete_store(keys, _ctx())
        # k0 最旧 → 被逐；touch(k0) 后 k1 变最旧
        m.touch([_key(0)], _ctx())
        out = m.prepare_store([_key(9)], _ctx())
        assert out.evicted_keys == [_key(1)]
        assert m.lookup(_key(1), _ctx()).name == "MISS"
        assert m.lookup(_key(0), _ctx()).name == "HIT"

    def test_eviction_failure_returns_none(self):
        m = self._mgr(num_blocks=2)
        keys = [_key(i) for i in range(2)]
        for k in keys:
            m.prepare_store([k], _ctx())
        m.complete_store(keys, _ctx())
        m.prepare_load(keys, _ctx())  # ref_cnt=1 → 全部不可逐
        assert m.prepare_store([_key(9)], _ctx()) is None

    def test_input_keys_protected_from_eviction(self):
        m = self._mgr(num_blocks=2)
        keys = [_key(i) for i in range(2)]
        for k in keys:
            m.prepare_store([k], _ctx())
        m.complete_store(keys, _ctx())
        # 输入里已存的 k0 必须留下；腾地只能逐 k1
        out = m.prepare_store([_key(0), _key(9)], _ctx())
        assert _key(0) in out.keys_to_store or _key(0) not in out.evicted_keys
        assert out.evicted_keys == [_key(1)]

    def test_prepare_load_pins_ref_cnt(self):
        m = self._mgr(num_blocks=2)
        keys = [_key(i) for i in range(2)]
        for k in keys:
            m.prepare_store([k], _ctx())
        m.complete_store(keys, _ctx())
        spec = m.prepare_load([_key(0)], _ctx())
        assert list(spec.block_ids) == [0]
        # k0 被钉住 → 只能逐 k1；完成后解钉
        out = m.prepare_store([_key(9)], _ctx())
        assert out.evicted_keys == [_key(1)]
        m.complete_load([_key(0)], _ctx())
        out = m.prepare_store([_key(10)], _ctx())
        assert out.evicted_keys == [_key(0)]

    def test_complete_store_failure_removes_block(self):
        m = self._mgr()
        k = _key(0)
        m.prepare_store([k], _ctx())
        m.complete_store([k], _ctx(), success=False)
        assert m.lookup(k, _ctx()).name == "MISS"

    def test_store_threshold_frequency_filter(self):
        m = self._mgr(num_blocks=4, store_threshold=2)
        k = _key(0)
        m.lookup(k, _ctx())  # 计数 1
        out = m.prepare_store([k], _ctx())
        assert out.keys_to_store == []  # 不足阈值 → 过滤
        m.lookup(k, _ctx())  # 计数 2
        out = m.prepare_store([k], _ctx())
        assert out.keys_to_store == [k]

    def test_reset_cache_clears_all(self):
        m = self._mgr()
        keys = [_key(i) for i in range(2)]
        for k in keys:
            m.prepare_store([k], _ctx())
        m.complete_store(keys, _ctx())
        m.reset_cache()
        assert m.lookup(_key(0), _ctx()).name == "MISS"

    def test_events_removed_and_stored(self):
        m = self._mgr(num_blocks=1)
        k = _key(0)
        m.prepare_store([k], _ctx())
        m.complete_store([k], _ctx())
        stored = list(m.take_events())
        assert len(stored) == 1 and not stored[0].removed
        m.prepare_store([_key(1)], _ctx())  # 池 1 块 → 逐 k
        removed = list(m.take_events())
        assert len(removed) == 1 and removed[0].removed


class TestCachePolicies:
    def test_lru_evict_atomic_and_protected(self):
        from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

        p = LRUCachePolicy(cache_capacity=4)
        keys = [_key(i) for i in range(4)]
        for i, k in enumerate(keys):
            p.insert(k, type("B", (), {"ref_cnt": 0, "block_id": i})())
        evicted = p.evict(2, protected={keys[0]})
        assert [k for k, _ in evicted] == [keys[1], keys[2]]
        assert p.evict(10, protected=set()) is None  # 不够逐 → None 且不动状态
        assert p.get(keys[0]) is not None

    def test_arc_adapts_target_on_ghost_hits(self):
        from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy

        p = ARCCachePolicy(cache_capacity=4)
        keys = [_key(i) for i in range(4)]
        for i, k in enumerate(keys):
            p.insert(k, type("B", (), {"ref_cnt": 0, "block_id": i, "is_ready": True})())
        # 逐两个进 B1 ghost → 再 touch B1 命中 → target_t1_size 上升
        p.evict(2, protected=set())
        p.touch([keys[0]], _ctx())  # keys[0] 现在在 B1
        assert p.target_t1_size > 0

    def test_policy_factory_out_of_tree(self):
        from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
        from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory
        from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

        assert CachePolicyFactory.get_cache_policy_cls("lru") is LRUCachePolicy
        assert CachePolicyFactory.get_cache_policy_cls("arc") is ARCCachePolicy
        with pytest.raises(ValueError, match="out-of-tree"):
            CachePolicyFactory.get_cache_policy_cls("MyTinyPolicy")

        import _kv_harness as harness

        cls = CachePolicyFactory.get_cache_policy_cls(
            "MyTinyPolicy", "_kv_harness"
        )
        assert cls is harness.MyTinyPolicy


# ═══════════════════════ m5+m9+m10 一个块进出池的一生（两半边全链路） ═══════════════════════


class TestBlockLifecycle:
    def test_full_round_trip(self):
        eng = assemble_engine(
            make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20}),
            make_kv_config(),
            make_kv_caches(make_kv_config()),
        )
        layer0 = eng.kv_caches[LAYER_NAMES[0]]

        # ── 存：prompt 16 token = 4 块 ──
        req = make_request("r1", list(range(16)))
        eng.scheduler.on_new_request(req)
        hashes = fill_block_hashes(req, 4)
        n_hit, async_load = eng.scheduler.get_num_new_matched_tokens(req, 0)
        assert (n_hit, async_load) == (0, False)  # 空池无命中
        out = make_scheduler_output(
            new_reqs=[{"req_id": "r1", "block_ids": ([1, 2, 3, 4],)}],
            num_scheduled_tokens={"r1": 16},
        )
        meta = eng.scheduler.build_connector_meta(out)
        assert len(meta.store_jobs) == 1

        # 每块写入互不相同的字节（写入 GPU 张量 = 写入 canonical 视图同源）
        for i, bid in enumerate([1, 2, 3, 4]):
            layer0[bid] = torch.full_like(layer0[bid], 0x30 + i)

        # 延迟一步：get_finished 只入队不提交（m5）
        eng.worker.prepare_store_kv(meta)
        assert eng.host_worker.submitted == []
        eng.worker.get_finished(set())
        assert eng.host_worker.submitted == []

        # 下一步 start_kv_transfers 开头提交 → 立即完成（host DMA 同步）
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
            OffloadingConnectorMetadata,
        )

        next_meta = OffloadingConnectorMetadata(load_jobs={}, store_jobs={})
        eng.worker.start_kv_transfers(next_meta)
        assert eng.host_worker.submitted == [(0, "store")]

        finished_sending, finished_recving = eng.worker.get_finished(set())
        assert finished_sending == set()  # store 永不发 finished_sending（m9）
        wmeta = eng.worker.build_connector_worker_meta()
        assert wmeta.completed_jobs == {0: 1}

        # 调度器结算：world_size=1 → 聚满 → complete_store
        from vllm.v1.outputs import KVConnectorOutput

        eng.scheduler.update_connector_output(
            KVConnectorOutput(kv_connector_worker_meta=wmeta)
        )
        assert eng.scheduler.has_pending_push_work() is False  # 账清

        # ── 查：同前缀新请求 → 16 token 全命中 ──
        req2 = make_request("r2", list(range(16)))
        eng.scheduler.on_new_request(req2)
        fill_block_hashes(req2, 4, seed=0)
        n_hit2, _ = eng.scheduler.get_num_new_matched_tokens(req2, 0)
        assert n_hit2 == 16

        # ── 载：命中块 → load job → CPU→GPU DMA → finished_recving ──
        eng.scheduler.update_state_after_alloc(req2, make_blocks([[4, 5, 6, 7]]), 16)
        out2 = make_scheduler_output(cached_reqs=[{"req_id": "r2"}], num_scheduled_tokens={"r2": 1})
        meta2 = eng.scheduler.build_connector_meta(out2)
        assert set(meta2.load_jobs) == {1}
        job = meta2.load_jobs[1]
        assert list(job.dst_spec.block_ids) == [4, 5, 6, 7]
        assert list(job.dst_spec.group_sizes) == [4]
        assert list(job.src_spec.block_ids) == [0, 1, 2, 3]  # CPU 池槽位

        eng.worker.start_kv_transfers(meta2)
        _, recv = eng.worker.get_finished(set())
        assert recv == {"r2"}  # load 发 finished_recving 接 ch16 提升路径
        wmeta2 = eng.worker.build_connector_worker_meta()
        eng.scheduler.update_connector_output(
            KVConnectorOutput(kv_connector_worker_meta=wmeta2)
        )

        # 字节级验证：目的块 4-7 == CPU 池槽 0-3（搬运后的权威副本）
        page = layer0.shape[1]
        cpu_tensor = eng.host_worker.cpu_tensors[0]
        for i, b in enumerate([4, 5, 6, 7]):
            assert block_checksum(layer0, b, page) == bytes(cpu_tensor[i].numpy())
        # CPU 池留的就是当初写入 GPU 块 1-4 的字节
        for i, a in enumerate([1, 2, 3, 4]):
            assert bytes(cpu_tensor[i].numpy()) == bytes(
                torch.full((page,), 0x30 + i, dtype=torch.int8).numpy()
            )

    def test_deferred_store_note_and_noop_layer_hooks(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
            OffloadingConnector,
        )

        # ch16 逐层契约的『不逐层』填法：三钩子全空
        assert OffloadingConnector.wait_for_layer_load(None, "layer") is None
        assert OffloadingConnector.save_kv_layer(None, "l", torch.zeros(1), None) is None
        assert OffloadingConnector.wait_for_save(None) is None

    def test_flush_fence_submits_and_waits_on_block_reuse(self):
        eng = assemble_engine(
            make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20}),
            make_kv_config(),
            make_kv_caches(make_kv_config()),
        )
        # 请求带在飞 store 结束（job 0 未结算）→ request_finished 登记盯防块
        req = make_request("r1", list(range(8)))
        eng.scheduler.on_new_request(req)
        fill_block_hashes(req, 2)
        from vllm.v1.request import RequestStatus

        req.status = RequestStatus.FINISHED_STOPPED
        out = make_scheduler_output(
            new_reqs=[{"req_id": "r1", "block_ids": ([1, 2],)}],
            num_scheduled_tokens={"r1": 8},
            finished_req_ids={"r1"},
        )
        meta = eng.scheduler.build_connector_meta(out)
        assert len(meta.store_jobs) == 1
        # 终局触发：finished 请求的 store 未完 → 本步已收 jobs_to_flush
        assert 0 in (meta.jobs_to_flush or set())
        # 真实步序：get_finished 里 prepare_store_kv 把 store job 排进延迟队列
        eng.worker.prepare_store_kv(meta)
        assert eng.host_worker.submitted == []
        # 复用块 1 的新分配 → 盯防账本命中 → flush
        req2 = make_request("r2", [1, 2, 3, 4])
        fill_block_hashes(req2, 1)
        eng.scheduler.on_new_request(req2)
        out2 = make_scheduler_output(
            new_reqs=[{"req_id": "r2", "block_ids": ([1],)}],
            num_scheduled_tokens={"r2": 4},
        )
        meta2 = eng.scheduler.build_connector_meta(out2)
        assert 0 in meta2.jobs_to_flush

        # worker 侧围栏：抢先提交 + wait（host 替身记录提交序）
        waits: list[set] = []
        orig_wait = eng.host_worker.wait
        eng.host_worker.wait = lambda ids: waits.append(set(ids))  # type: ignore[method-assign]
        eng.worker.handle_preemptions(meta2)
        assert eng.host_worker.submitted == [(0, "store")]
        assert waits == [{0}]

    def test_preemption_triggers_flush(self):
        eng = assemble_engine(
            make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20}),
            make_kv_config(),
            make_kv_caches(make_kv_config()),
        )
        req = make_request("r1", list(range(8)))
        eng.scheduler.on_new_request(req)
        fill_block_hashes(req, 2)
        out = make_scheduler_output(
            new_reqs=[{"req_id": "r1", "block_ids": ([1, 2],)}],
            num_scheduled_tokens={"r1": 8},
        )
        meta = eng.scheduler.build_connector_meta(out)
        assert len(meta.store_jobs) == 1
        out2 = make_scheduler_output(
            cached_reqs=[{"req_id": "r1", "block_ids": ([],)}],
            num_scheduled_tokens={"r1": 0},
            preempted_req_ids={"r1"},
        )
        meta2 = eng.scheduler.build_connector_meta(out2)
        assert 0 in meta2.jobs_to_flush

    def test_completed_jobs_needs_world_size_reports(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
            OffloadingWorkerMetadata,
        )

        w1 = OffloadingWorkerMetadata()
        w2 = OffloadingWorkerMetadata()
        w1.mark_completed(0)
        w2.mark_completed(0)
        w2.mark_completed(7)
        merged = w1.aggregate(w2)
        assert merged.completed_jobs == {0: 2, 7: 1}

        eng = assemble_engine(
            make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20}, world_size=2),
            make_kv_config(),
            make_kv_caches(make_kv_config()),
        )
        req = make_request("r1", list(range(8)))
        eng.scheduler.on_new_request(req)
        fill_block_hashes(req, 2)
        out = make_scheduler_output(
            new_reqs=[{"req_id": "r1", "block_ids": ([1, 2],)}],
            num_scheduled_tokens={"r1": 8},
        )
        meta = eng.scheduler.build_connector_meta(out)
        job_ids = set(meta.store_jobs)
        from vllm.v1.outputs import KVConnectorOutput

        # 第 1 个 worker 报告 → pending 2-1=1 → 不结算
        half1 = OffloadingWorkerMetadata()
        half1.mark_completed(*job_ids)
        eng.scheduler.update_connector_output(
            KVConnectorOutput(kv_connector_worker_meta=half1)
        )
        assert eng.scheduler.has_pending_push_work() is True
        key0 = next(iter(job_ids))
        assert eng.scheduler.manager.lookup(_key_of(eng, 0), _ctx()).name == "HIT_PENDING"
        # 第 2 个 worker 报告 → 聚满 world_size 结算（aggregate 只用于跨 worker 求和）
        half2 = OffloadingWorkerMetadata()
        half2.mark_completed(*job_ids)
        eng.scheduler.update_connector_output(
            KVConnectorOutput(kv_connector_worker_meta=half2)
        )
        assert eng.scheduler.manager.lookup(_key_of(eng, 0), _ctx()).name == "HIT"

    def test_request_finished_returns_false_and_registers_watch(self):
        eng = assemble_engine(
            make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20}),
            make_kv_config(),
            make_kv_caches(make_kv_config()),
        )
        req = make_request("r1", list(range(8)))
        eng.scheduler.on_new_request(req)
        ok, params = eng.scheduler.request_finished(req)
        assert ok is False and params is None  # 不接管块释放
        # 未登记过的请求：立即走 on_new/on_finished 直通（返回 False）
        stray = make_request("rX", [1])
        ok2, _ = eng.scheduler.request_finished(stray)
        assert ok2 is False


def _key_of(eng: Engine, chunk_idx: int):
    """取 r1 第 chunk_idx 块的 OffloadKey（读调度器档案）。"""
    st = eng.scheduler._req_status["r1"]
    return st.group_states[0].offload_keys[chunk_idx]


# ═══════════════════════ m6 DMA 描述符数学 ═══════════════════════


class TestDescriptorMath:
    def test_compute_sub_block_ptrs_1to1(self):
        from vllm.v1.kv_offload.cpu.gpu_worker import compute_sub_block_ptrs

        t = torch.arange(200, dtype=torch.int8)
        out = np.empty(2, dtype=np.uint64)
        compute_sub_block_ptrs(np.array([0, 1]), 1, out, t.view(2, 100))
        assert int(out[0]) == t.data_ptr()
        assert int(out[1]) == t.data_ptr() + 100

    def test_compute_sub_block_ptrs_expansion_and_skip(self):
        from vllm.v1.kv_offload.cpu.gpu_worker import compute_sub_block_ptrs

        # blocks_per_chunk=2：CPU 块 k 的两个子块指针 = base + k*行距 + j*页
        t = torch.arange(600, dtype=torch.int8)
        view = t.view(3, 200)  # 3 个 CPU 块、行距 200
        out = np.empty(3, dtype=np.uint64)
        # 块 [1,2] 展开成 4 个子块指针，skip=1 取后 3 个：
        # [b1 第二子块, b2 第一子块, b2 第二子块]
        compute_sub_block_ptrs(np.array([1, 2]), 2, out, view, skip_count=1)
        base = t.data_ptr()
        assert int(out[0]) == base + 200 + 100
        assert int(out[1]) == base + 400
        assert int(out[2]) == base + 400 + 100
        # 无 skip：块 [1,2] 的前 2 个子块
        out2 = np.empty(2, dtype=np.uint64)
        compute_sub_block_ptrs(np.array([1, 2]), 2, out2, view, skip_count=0)
        assert int(out2[0]) == base + 200
        assert int(out2[1]) == base + 200 + 100

    def test_host_swap_blocks_batch_moves_bytes(self):
        # host seam：描述符三缓冲 → memmove 语义（真源是 C++ DMA kernel）
        from vllm import _custom_ops as ops

        src = torch.arange(64, dtype=torch.int8)
        dst = torch.zeros(64, dtype=torch.int8)
        ops.swap_blocks_batch(
            torch.tensor([src.data_ptr(), src.data_ptr() + 32], dtype=torch.int64),
            torch.tensor([dst.data_ptr() + 32, dst.data_ptr()], dtype=torch.int64),
            torch.tensor([32, 32], dtype=torch.int64),
            is_src_access_order_any=True,
        )
        assert bytes(dst[32:64].numpy()) == bytes(src[:32].numpy())
        assert bytes(dst[0:32].numpy()) == bytes(src[32:].numpy())

    def test_transfer_dataclass_and_descriptor_buffers(self):
        from vllm.v1.kv_offload.cpu.gpu_worker import (
            Transfer,
            _new_descriptor_buffers,
        )

        s, d, z = _new_descriptor_buffers(4)
        assert s.dtype == torch.int64 and s.shape == (4,) and z.shape == (4,)


# ═══════════════════════ m7 四态查找 / SWA / 收敛 ═══════════════════════


class _ScriptedManager:
    """按 key 前缀回放四态的假账本（只用于调度器查找算法的驱动）。"""

    def __init__(self, results: dict[bytes, str]):
        self.results = results
        from vllm.v1.kv_offload.base import LookupResult

        self._lr = LookupResult
        self.touched: list[bytes] = []

    def lookup(self, key, ctx):
        return self._lr[self.results.get(key, "MISS")]

    def touch(self, keys, ctx):
        self.touched.extend(keys)


class TestLookupFourStates:
    def _sched(self, kv_cfg=None):
        cfg = kv_cfg or make_kv_config()
        vcfg = make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20})
        eng = assemble_engine(vcfg, cfg, make_kv_caches(cfg))
        return eng.scheduler

    def test_maximal_prefix_hit_miss_break(self):
        s = self._sched()
        keys = [_key(i) for i in range(4)]
        s.manager = _ScriptedManager(
            {keys[0]: "HIT", keys[1]: "HIT", keys[3]: "HIT"}
        )
        n = s._maximal_prefix_lookup(keys, _ctx(), None, None, 0)
        assert n == 2  # MISS 之后必 MISS → 早停

    def test_hit_pending_defers_whole_lookup(self):
        s = self._sched()
        keys = [_key(i) for i in range(3)]
        s.manager = _ScriptedManager({keys[0]: "HIT", keys[1]: "HIT_PENDING"})
        assert s._maximal_prefix_lookup(keys, _ctx(), None, None, 0) is None

    def test_retry_keeps_scanning_then_defers(self):
        s = self._sched()
        keys = [_key(i) for i in range(3)]
        s.manager = _ScriptedManager({keys[0]: "RETRY", keys[1]: "HIT", keys[2]: "MISS"})
        seen: list[bytes] = []
        real_lookup = s.manager.lookup

        def spy(key, ctx):
            seen.append(key)
            return real_lookup(key, ctx)

        s.manager.lookup = spy
        assert s._maximal_prefix_lookup(keys, _ctx(), None, None, 0) is None
        # RETRY 不 break：继续扫到 MISS（让 manager 踢异步查询）
        assert seen == keys

    def test_lookup_entry_none_when_hit_pending(self):
        s = self._sched()
        eng_req = make_request("r1", list(range(16)))
        s.on_new_request(eng_req)
        fill_block_hashes(eng_req, 4)
        s.manager = _ScriptedManager({_key(0): "HIT_PENDING"})
        n, load_async = s.get_num_new_matched_tokens(eng_req, 0)
        assert n is None and load_async is False  # 稍后再问

    def test_lookup_delayed_when_in_flight_jobs(self):
        s = self._sched()
        req = make_request("r1", list(range(16)))
        s.on_new_request(req)
        fill_block_hashes(req, 4)
        st = s._req_status["r1"]
        st.transfer_jobs.add(99)
        n, _ = s.get_num_new_matched_tokens(req, 0)
        assert n is None  # 在飞互斥：单请求 load/store 不可并发

    def test_touch_includes_gpu_hit_chunks(self):
        s = self._sched()
        req = make_request("r1", list(range(16)))
        s.on_new_request(req)
        hashes = fill_block_hashes(req, 4)
        s.manager = _ScriptedManager({})
        s.get_num_new_matched_tokens(req, 8)  # GPU 已算 8 token（2 块）
        # touch 覆盖全部 offload_keys（含 GPU 前缀缓存命中的前 2 块）
        s.manager.touched.clear()
        s._touch(s._req_status["r1"])
        assert set(s.manager.touched) == {
            make_offload_key_for(h) for h in hashes
        }

    def test_sliding_window_lookup_from_tail(self):
        s = self._sched()
        keys = [_key(i) for i in range(6)]
        # 尾部 [k3,k4] 连续命中、窗口=2 → 返回 end idx 5
        s.manager = _ScriptedManager(
            {keys[3]: "HIT", keys[4]: "HIT", keys[5]: "MISS"}
        )
        assert s._sliding_window_lookup(keys, 2, _ctx()) == 5
        # 中断 → 从头连续窗口才算
        s.manager = _ScriptedManager(
            {keys[3]: "HIT", keys[5]: "HIT"}
        )
        assert s._sliding_window_lookup(keys, 2, _ctx()) == 0

    def test_multi_group_convergence_reruns(self):
        # full 组先查到长命中，SWA 组收紧 → full 组重扫
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
            KVCacheTensor,
            SlidingWindowSpec,
        )

        full = FullAttentionSpec(
            block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float32
        )
        swa = SlidingWindowSpec(
            block_size=4, num_kv_heads=2, head_size=8, dtype=torch.float32,
            sliding_window=4,
        )
        cfg = KVCacheConfig(
            num_blocks=8,
            kv_cache_tensors=[
                KVCacheTensor(size=full.page_size_bytes * 8, shared_by=["f.0"]),
                KVCacheTensor(size=swa.page_size_bytes * 8, shared_by=["s.0"]),
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=["f.0"], kv_cache_spec=full),
                KVCacheGroupSpec(layer_names=["s.0"], kv_cache_spec=swa),
            ],
        )
        vcfg = make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20})
        eng = assemble_engine(vcfg, cfg, make_kv_caches(cfg))
        s = eng.scheduler
        req = make_request("r1", list(range(16)))  # 4 chunk
        s.on_new_request(req)
        fill_block_hashes(req, 4)
        # full 组 4 chunk 全 HIT（16 token）；SWA 组（窗口 1 chunk）尾部 chunk
        # MISS、倒数第二 HIT → 连续窗口止于 idx2 → 只能跳过 3 chunk = 12 token
        full_keys = [
            make_offload_key_for((0 * 1_000_003 + i).to_bytes(32, "big"), 0)
            for i in range(4)
        ]
        swa_keys = [
            make_offload_key_for((0 * 1_000_003 + i).to_bytes(32, "big"), 1)
            for i in range(4)
        ]
        results = {k: "HIT" for k in full_keys}
        results[swa_keys[2]] = "HIT"  # 尾扫窗口止于 idx 2
        s.manager = _ScriptedManager(results)
        n, _ = s.get_num_new_matched_tokens(req, 0)
        assert n == 12  # min(16, SWA 可支撑 3 chunk)


def make_offload_key_for(block_hash: bytes, group: int = 0):
    from vllm.v1.kv_offload.base import make_offload_key

    return make_offload_key(block_hash, group)


# ═══════════════════════ m11 分层池 ═══════════════════════


class _FakeShmRegion:
    """SharedOffloadRegion 的 host 替身：create_kv_memoryview 盖在 numpy 池上。"""

    def __init__(self, num_blocks: int, chunk_bytes: int):
        self.pool = np.zeros((num_blocks, chunk_bytes), dtype=np.uint8)
        self.total_size_bytes = num_blocks * chunk_bytes

    def create_kv_memoryview(self) -> memoryview:
        return memoryview(self.pool)


@pytest.fixture
def tiering_env(tmp_path, monkeypatch):
    """装配真 TieringOffloadingManager + 真 FileSystemTierManager（mmap 换 host 池）。"""
    import vllm.v1.kv_offload.tiering.spec as tiering_spec_mod
    from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

    created: list[_FakeShmRegion] = []

    def fake_shm(engine_id, num_blocks, rank, kv_bytes_per_block, cpu_page_size):
        region = _FakeShmRegion(num_blocks, kv_bytes_per_block)
        created.append(region)
        return region

    monkeypatch.setattr(tiering_spec_mod, "SharedOffloadRegion", fake_shm)

    cfg = make_kv_config(num_blocks=8)
    vcfg = make_vllm_config(
        extra_config={
            "cpu_bytes_to_use": 1 << 20,
            "spec_name": "TieringOffloadingSpec",
            "secondary_tiers": [
                {"type": "fs", "root_dir": str(tmp_path / "kvfs"), "n_read_threads": 2,
                 "n_write_threads": 2, "locality": "LOCAL"}
            ],
        }
    )
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
        build_offloading_config,
    )
    from vllm.v1.kv_offload.factory import OffloadingSpecFactory

    spec = TieringOffloadingSpec(build_offloading_config(vcfg, cfg))
    manager = spec.get_manager()
    return manager, created[0], tmp_path


class TestTiering:
    def test_primary_gateway_promotion_and_cascade(self, tiering_env):
        from vllm.v1.kv_offload.base import LookupResult, ScheduleEndContext

        manager, region, tmp_path = tiering_env
        ctx = _ctx("r1")
        k = _key(42)

        # ① store：primary 准入 → complete_store → cascade 到 fs 层
        manager.on_new_request(ctx)
        out = manager.prepare_store([k], ctx)
        assert out.keys_to_store == [k]
        pool = region.pool
        pool[out.store_spec.block_ids[0]][:16] = np.arange(16, dtype=np.uint8)
        manager.complete_store([k], ctx)
        assert manager.lookup(k, ctx) is LookupResult.HIT
        # 等 fs 线程把 cascade 落盘
        import time

        deadline = time.monotonic() + 5
        files = []
        while time.monotonic() < deadline:
            files = list((tmp_path / "kvfs").rglob("*.bin"))
            if files:
                break
            manager._maybe_process_finished_jobs()
            time.sleep(0.01)
        assert files, "cascade 应把块写进 fs 层"

        # ② primary 清空（模拟主层全部被逐）→ fs 异步查询 RETRY → 命中即 promotion
        manager.reset_cache()
        end_ctx = ScheduleEndContext(new_req_ids=[], preempted_req_ids=[])
        r1 = manager.lookup(k, ctx)  # fs 批查询入队 → RETRY
        assert r1 is LookupResult.RETRY
        manager.on_schedule_end(end_ctx)  # flush 查询批
        r2 = manager.lookup(k, ctx)  # fs HIT → promotion 占位 → RETRY
        assert r2 is LookupResult.RETRY
        manager.on_schedule_end(end_ctx)  # 批量 submit_load（fs→primary）
        deadline = time.monotonic() + 5
        while manager.lookup(k, ctx) is not LookupResult.HIT:
            manager.on_schedule_end(end_ctx)
            assert time.monotonic() < deadline, "promotion 应完成并转 HIT"
            time.sleep(0.01)
        # promotion 后主层槽里有那 16 个字节
        hit_rows = [row for row in pool if row[0] == 0]
        assert any(bytes(row[:16]) == bytes(np.arange(16, dtype=np.uint8)) for row in hit_rows)

    def test_tiering_rejects_store_threshold(self, tmp_path, monkeypatch):
        import vllm.v1.kv_offload.tiering.spec as tiering_spec_mod
        from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

        monkeypatch.setattr(
            tiering_spec_mod,
            "SharedOffloadRegion",
            lambda *a, **k: _FakeShmRegion(1, 4096),
        )
        cfg = make_kv_config()
        vcfg = make_vllm_config(
            extra_config={
                "cpu_bytes_to_use": 1 << 20,
                "spec_name": "TieringOffloadingSpec",
                "store_threshold": 2,
                "secondary_tiers": [],
            }
        )
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
            build_offloading_config,
        )

        spec = TieringOffloadingSpec(build_offloading_config(vcfg, cfg))
        with pytest.raises(ValueError, match="store_threshold"):
            spec.get_manager()

    def test_secondary_tier_factory_types(self):
        from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
        from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
        from vllm.v1.kv_offload.tiering.p2p.manager import P2PSecondaryTierManager

        assert (
            SecondaryTierFactory.get_tier_class({"type": "fs"})
            is FileSystemTierManager
        )
        assert (
            SecondaryTierFactory.get_tier_class({"type": "p2p"})
            is P2PSecondaryTierManager
        )
        with pytest.raises(ValueError, match="type"):
            SecondaryTierFactory.get_tier_class({})
        with pytest.raises(ValueError, match="Unknown secondary tier type"):
            SecondaryTierFactory.get_tier_class({"type": "obj"})

    def test_fs_tier_file_layout_and_roundtrip(self, tmp_path):
        from vllm.v1.kv_offload.tiering.base import JobMetadata
        from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

        cfg = make_kv_config()
        vcfg = make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20})
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
            build_offloading_config,
        )
        from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

        spec = CPUOffloadingSpec(build_offloading_config(vcfg, cfg))
        pool = np.zeros((4, 64), dtype=np.uint8)
        tier = FileSystemTierManager(
            offloading_spec=spec,
            primary_kv_view=memoryview(pool),
            tier_type="fs",
            root_dir=str(tmp_path),
            n_read_threads=1,
            n_write_threads=1,
        )
        k = _key(7)
        pool[0][:8] = np.arange(8, dtype=np.uint8)
        tier.submit_store(JobMetadata(job_id=0, keys=[k], block_ids=np.array([0]),
                                      is_promotion=False, req_context=_ctx()))
        import time

        deadline = time.monotonic() + 5
        while not list(tier.get_finished_jobs()):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        fname = tier.file_mapper.get_file_name(k)
        assert os.path.exists(fname)
        # 三级分桶：<root>/<model>_<digest>_r0/<hhh>/<hh>_g0/<hash>.bin
        rel = os.path.relpath(fname, str(tmp_path)).replace("\\", "/")
        parts = rel.split("/")
        h = (7).to_bytes(32, "big").hex()
        assert len(parts) == 4
        assert parts[0].endswith("_r0")
        assert parts[1] == h[:3] and parts[2] == f"{h[3:5]}_g0" and parts[3] == f"{h}.bin"
        # load：文件 → 主层槽 2
        pool[2].fill(0)
        tier.submit_load(JobMetadata(job_id=1, keys=[k], block_ids=np.array([2]),
                                     is_promotion=True, req_context=_ctx()))
        deadline = time.monotonic() + 5
        while not list(tier.get_finished_jobs()):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        assert bytes(pool[2][:8]) == bytes(np.arange(8, dtype=np.uint8))


class TestFileMapper:
    def test_three_level_bucket_naming(self, tmp_path):
        from vllm.v1.kv_offload.file_mapper import FileMapper

        fm = FileMapper(
            root_dir=str(tmp_path),
            model_name="org/name",
            tokens_per_hash=16,
            blocks_per_file=1,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            rank=3,
            dtype="float32",
        )
        name = fm.get_file_name(_key(5))
        h = (5).to_bytes(32, "big").hex()
        assert name.replace("\\", "/").endswith(
            f"_r3/{h[:3]}/{h[3:5]}_g0/{h}.bin"
        )
        assert "org_name" in name  # HF 路径斜杠折平


# ═══════════════════════ m12 chunk 粒度配置 ═══════════════════════


class TestChunkGranularityConfig:
    def test_block_size_maps_to_blocks_per_chunk(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
            build_offloading_config,
        )

        cfg = make_kv_config()
        built = build_offloading_config(
            make_vllm_config(extra_config={"cpu_bytes_to_use": 1, "block_size": 8}),
            cfg,
        )
        assert built.cache.blocks_per_chunk == 2  # 8 token / 4 token 每块
        assert built.worker_kv_bytes_per_block == (
            cfg.kv_cache_tensors[0].size // cfg.num_blocks
        )

    def test_block_size_and_blocks_per_chunk_mutually_exclusive(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
            build_offloading_config,
        )

        with pytest.raises(ValueError, match="only one"):
            build_offloading_config(
                make_vllm_config(
                    extra_config={
                        "cpu_bytes_to_use": 1,
                        "block_size": 8,
                        "blocks_per_chunk": 2,
                    }
                ),
                make_kv_config(),
            )


# ═══════════════════════ m13 P2P 层 ═══════════════════════


class TestP2P:
    def test_role_key_parsing(self):
        from vllm.v1.kv_offload.tiering.p2p.manager import (
            P2PDestInfo,
            P2PSourceInfo,
            _annotate_req_context,
            _parse_dest,
            _parse_source,
        )

        consumer = {
            "remote_kv_source": {
                "kv_request_id": "kq-1",
                "remote_host": "10.0.0.2",
                "remote_port": 5710,
            }
        }
        src = _parse_source(consumer)
        assert src == P2PSourceInfo(
            kv_request_id="kq-1", peer_id="10.0.0.2:5710", do_probe=True
        )
        pd_consumer = {
            "remote_prefiller": {
                "kv_request_id": "kq-2",
                "remote_host": "10.0.0.3",
                "remote_port": 5710,
            }
        }
        assert _parse_source(pd_consumer).do_probe is False  # PD 不 probe
        producer = {"remote_decoder": {"kv_request_id": "kq-3"}}
        assert _parse_dest(producer) == P2PDestInfo(kv_request_id="kq-3")
        assert _parse_source({}) is None and _parse_dest({}) is None

        ctx = _ctx("r9")
        ctx.kv_transfer_params = consumer
        _annotate_req_context(ctx)
        assert ctx.get_state(P2PSourceInfo) is not None

    def test_pythonhashseed_gate(self, monkeypatch):
        from vllm.v1.kv_offload.tiering.p2p.manager import P2PSecondaryTierManager

        cfg = make_kv_config()
        vcfg = make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20})
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
            build_offloading_config,
        )
        from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

        spec = CPUOffloadingSpec(build_offloading_config(vcfg, cfg))
        pool = memoryview(np.zeros((2, 64), dtype=np.uint8))
        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
        with pytest.raises(ValueError, match="PYTHONHASHSEED"):
            P2PSecondaryTierManager(spec, pool, tier_type="p2p")

    def test_lookup_pd_hit_and_symmetric_miss_without_session(self, monkeypatch):
        from vllm.v1.kv_offload.base import LookupResult, ReqContext
        from vllm.v1.kv_offload.tiering.p2p.manager import P2PSecondaryTierManager

        cfg = make_kv_config()
        vcfg = make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20})
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
            build_offloading_config,
        )
        from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

        spec = CPUOffloadingSpec(build_offloading_config(vcfg, cfg))
        pool = memoryview(np.zeros((2, 64), dtype=np.uint8))
        monkeypatch.setenv("PYTHONHASHSEED", "0")
        tier = P2PSecondaryTierManager(spec, pool, tier_type="p2p")

        pd_ctx = ReqContext(
            req_id="d1",
            kv_transfer_params={
                "remote_prefiller": {
                    "kv_request_id": "kq",
                    "remote_host": "h",
                    "remote_port": 1,
                }
            },
        )
        tier.on_new_request(pd_ctx)
        # PD consumer：对端 prefiller 有全部块 → 立即 HIT
        assert tier.lookup(_key(0), pd_ctx) is LookupResult.HIT
        # 普通请求（无角色键）→ MISS
        assert tier.lookup(_key(0), _ctx()) is LookupResult.MISS

    def test_protocol_message_validation(self):
        from vllm.v1.kv_offload.tiering.p2p.session.protocol import (
            FetchMsg,
            LookupMsg,
            LookupRespMsg,
            TransferDoneMsg,
        )

        LookupMsg.validate(
            {"kv_request_id": "k", "keys": [b"a"], "round_seq": 0}
        )
        with pytest.raises(ValueError):
            LookupMsg.validate({"kv_request_id": "k", "keys": "notalist", "round_seq": 0})
        FetchMsg.validate(
            {"kv_request_id": "k", "keys": [b"a"], "block_indexes": [3], "round_seq": 1}
        )
        with pytest.raises(ValueError, match="mismatch"):
            FetchMsg.validate(
                {"kv_request_id": "k", "keys": [b"a", b"b"], "block_indexes": [1], "round_seq": 0}
            )
        with pytest.raises(ValueError):
            FetchMsg.validate(
                {"kv_request_id": "k", "keys": [b"a"], "block_indexes": [-1], "round_seq": 0}
            )
        TransferDoneMsg.validate({"kv_request_id": "k", "success": True, "round_seq": 0})
        with pytest.raises(ValueError):
            TransferDoneMsg.validate({"kv_request_id": "k", "success": "yes", "round_seq": 0})
        LookupRespMsg.validate({"kv_request_id": "k", "keys": [b"a"], "hits": [True]})
        with pytest.raises(ValueError, match="mismatch"):
            LookupRespMsg.validate({"kv_request_id": "k", "keys": [b"a"], "hits": []})


# ═══════════════════════ m14 生态对照 ═══════════════════════


class _MooncakeServer:
    """测试侧 ZMQ REP：按脚本回 lookup 结果（LOOKUP_MSG 帧）。"""

    def __init__(self, delay: float, hits: dict[str, int]):
        import zmq

        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.protocol import (
            LOOKUP_MSG,
        )

        self._LOOKUP_MSG = LOOKUP_MSG
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REP)
        self.port = self.sock.bind_to_random_port("tcp://127.0.0.1")
        self.delay = delay
        self.hits = hits
        self.requests: list[int] = []
        self.running = True
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        import time

        while self.running:
            try:
                frames = self.sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.005)
                continue
            assert bytes(frames[0]) == self._LOOKUP_MSG
            num_tokens = int.from_bytes(frames[1], "big")
            self.requests.append(num_tokens)
            if self.delay:
                time.sleep(self.delay)
            self.sock.send(self.hits.get(str(num_tokens), 0).to_bytes(4, "big"))

    def close(self):
        self.running = False
        self.sock.close(linger=0)
        self.ctx.term()


class TestMooncakeStore:
    def test_lookup_async_none_then_result(self, monkeypatch):
        import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker as wmod
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
            MooncakeStoreScheduler,
        )

        server = _MooncakeServer(delay=0.35, hits={"16": 12})
        try:
            monkeypatch.setattr(
                wmod, "get_zmq_rpc_path_lookup",
                lambda vcfg: f"tcp://127.0.0.1:{server.port}", raising=True,
            )
            vcfg = make_vllm_config(
                extra_config={"lookup_async": True}, connector="MooncakeStoreConnector"
            )
            sched = MooncakeStoreScheduler(vcfg, make_kv_config())
            req = make_request("r1", list(range(16)))
            fill_block_hashes(req, 4)
            # 查询在后台线程未就绪 → None=稍后再问
            first, load_async = sched.get_num_new_matched_tokens(req, 0)
            assert first is None and load_async is False
            # 稍后一步再问 → 就绪 → 命中差值 (12-0, load_async=True)
            import time

            time.sleep(0.5)
            second, la2 = sched.get_num_new_matched_tokens(req, 0)
            assert second == 12 and la2 is True
            assert sched.load_specs["r1"].kvpool_cached_tokens == 12
            sched.client.close()
        finally:
            server.close()

    def test_lookup_sync_and_partial_alignment(self, monkeypatch):
        import vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker as wmod
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import (
            MooncakeStoreScheduler,
        )

        server = _MooncakeServer(delay=0.0, hits={"16": 12, "8": 3})
        try:
            monkeypatch.setattr(
                wmod, "get_zmq_rpc_path_lookup",
                lambda vcfg: f"tcp://127.0.0.1:{server.port}",
            )
            vcfg = make_vllm_config(
                extra_config={"lookup_async": False}, connector="MooncakeStoreConnector"
            )
            sched = MooncakeStoreScheduler(vcfg, make_kv_config())
            req = make_request("r1", list(range(16)))
            fill_block_hashes(req, 4)
            n, la = sched.get_num_new_matched_tokens(req, 0)
            assert (n, la) == (12, True)
            # 本地已算 12 → 外部命中 12 → 无需再分配 → 0
            req2 = make_request("r2", list(range(16)))
            fill_block_hashes(req2, 4)
            n2, _ = sched.get_num_new_matched_tokens(req2, 12)
            assert n2 == 0
            sched.client.close()
        finally:
            server.close()

    def test_worker_io_all_in_get_finished_contract(self):
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
            MooncakeStoreWorker,
        )

        w = MooncakeStoreWorker.__new__(MooncakeStoreWorker)
        w.start_load_kv(None)  # No-op: loads are issued in get_finished() for overlap.
        w.wait_for_save(None)  # No-op: stores are issued in get_finished() for overlap.
        # 完整实现体（双传输线程 + MooncakeDistributedStore）未进精简版：get_finished
        # 的 I/O 下发须 CUDA/Mooncake 运行时（impl-notes seam 清单）。
        with pytest.raises(NotImplementedError):
            w.get_finished(set(), None)


class TestLMCacheConnector:
    def _cfg(self, use_native: bool):
        return make_vllm_config(
            extra_config={"use_native": use_native}, connector="LMCacheConnectorV1"
        )

    def test_dual_path_lazy_import(self, monkeypatch):
        import sys
        import types

        from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
            LMCacheConnectorV1,
        )

        made = []

        class NativeImpl:
            def __init__(self, vllm_config, role, parent):
                made.append(("native", vllm_config.kv_transfer_config.kv_role, parent))

        class LatestImpl:
            def __init__(self, vllm_config, role, parent):
                made.append(("latest", vllm_config.kv_transfer_config.kv_role, parent))

        native_mod = types.ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration"
        )
        adapter = types.ModuleType(
            "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter"
        )
        adapter.LMCacheConnectorV1Impl = NativeImpl
        native_mod.vllm_v1_adapter = adapter

        ext_pkg = types.ModuleType("lmcache")
        ext_int = types.ModuleType("lmcache.integration")
        ext_vllm = types.ModuleType("lmcache.integration.vllm")
        ext_adapter = types.ModuleType("lmcache.integration.vllm.vllm_v1_adapter")
        ext_adapter.LMCacheConnectorV1Impl = LatestImpl
        ext_int.vllm = ext_vllm
        ext_vllm.vllm_v1_adapter = ext_adapter

        for name, mod in {
            "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration": native_mod,
            "vllm.distributed.kv_transfer.kv_connector.v1.lmcache_integration.vllm_v1_adapter": adapter,
            "lmcache": ext_pkg,
            "lmcache.integration": ext_int,
            "lmcache.integration.vllm": ext_vllm,
            "lmcache.integration.vllm.vllm_v1_adapter": ext_adapter,
        }.items():
            monkeypatch.setitem(sys.modules, name, mod)

        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

        c1 = LMCacheConnectorV1(
            self._cfg(True), KVConnectorRole.WORKER, make_kv_config()
        )
        c2 = LMCacheConnectorV1(
            self._cfg(False), KVConnectorRole.WORKER, make_kv_config()
        )
        assert made[0][0] == "native" and made[1][0] == "latest"
        assert c1._lmcache_engine is not c2._lmcache_engine

    def test_missing_external_package_raises(self, monkeypatch):
        import sys

        monkeypatch.delitem(sys.modules, "lmcache", raising=False)
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
            LMCacheConnectorV1,
        )

        with pytest.raises(ImportError):
            LMCacheConnectorV1(self._cfg(False), KVConnectorRole.WORKER, make_kv_config())


class TestMultiConnector:
    def test_first_connector_with_matched_tokens_wins(self, monkeypatch):
        import _kv_harness as harness
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

        for name in ("FakePoolA", "FakePoolB"):
            try:
                KVConnectorFactory.register_connector(name, "_kv_harness", name)
            except ValueError:
                pass  # 前一用例已注册（注册表是进程级单例）
        harness.FakePoolA.hits = 0
        harness.FakePoolB.hits = 5
        vcfg = make_vllm_config(
            extra_config={
                "connectors": [
                    {"kv_connector": "FakePoolA", "kv_role": "kv_both"},
                    {"kv_connector": "FakePoolB", "kv_role": "kv_both"},
                ]
            },
            connector="MultiConnector",
        )
        vcfg.scheduler_config.disable_hybrid_kv_cache_manager = True
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
            MultiConnector,
        )

        mc = MultiConnector(vcfg, KVConnectorRole.SCHEDULER, make_kv_config())
        req = make_request("r1", list(range(8)))
        fill_block_hashes(req, 2)
        n, la = mc.get_num_new_matched_tokens(req, 0)
        assert (n, la) == (5, False)  # A=0 → B=5 获加载权（列表顺序）
        # 后续更长命中不覆盖首个非零者
        harness.FakePoolA.hits = 20
        req2 = make_request("r2", list(range(8)))
        fill_block_hashes(req2, 2)
        n2, _ = mc.get_num_new_matched_tokens(req2, 0)
        assert n2 == 20  # 这次 A 先非零 → A 获胜
        # 分配只喂选中的 connector
        blocks = make_blocks([[1, 2]])
        mc.update_state_after_alloc(req, blocks, 5)
        calls = [c for c in (harness.FakePoolA.alloc_calls, harness.FakePoolB.alloc_calls)]
        assert harness.FakePoolB.alloc_calls[-1] == ("r1", 5)
        assert harness.FakePoolA.alloc_calls[-1] == ("r1", 0)

    def test_none_from_any_child_defers(self, monkeypatch):
        import _kv_harness as harness
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

        try:
            KVConnectorFactory.register_connector("FakePoolA", "_kv_harness", "FakePoolA")
        except ValueError:
            pass
        harness.FakePoolA.hits = None  # 仍在查
        vcfg = make_vllm_config(
            extra_config={"connectors": [{"kv_connector": "FakePoolA", "kv_role": "kv_both"}]},
            connector="MultiConnector",
        )
        vcfg.scheduler_config.disable_hybrid_kv_cache_manager = True
        from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
        from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
            MultiConnector,
        )

        mc = MultiConnector(vcfg, KVConnectorRole.SCHEDULER, make_kv_config())
        req = make_request("r1", [1, 2, 3])
        n, la = mc.get_num_new_matched_tokens(req, 0)
        assert n is None and la is False


# ═══════════════════════ m15 事件面 ═══════════════════════


class TestEvents:
    def test_self_describing_store_and_remove_events(self):
        eng = assemble_engine(
            make_vllm_config(
                extra_config={"cpu_bytes_to_use": 1 << 20},
                enable_kv_events=True,
                self_describing_kv_events=True,
            ),
            make_kv_config(),
            make_kv_caches(make_kv_config()),
        )
        req = make_request("r1", list(range(16)))
        eng.scheduler.on_new_request(req)
        hashes = fill_block_hashes(req, 4)
        out = make_scheduler_output(
            new_reqs=[{"req_id": "r1", "block_ids": ([1, 2, 3, 4],)}],
            num_scheduled_tokens={"r1": 16},
        )
        meta = eng.scheduler.build_connector_meta(out)
        (job,) = meta.store_jobs.values()
        # complete_store → BlockStored（带 chunk 哈希族/token_ids/block_size）
        from vllm.v1.outputs import KVConnectorOutput

        eng.worker.prepare_store_kv(meta)
        eng.worker.start_kv_transfers(
            __import__(
                "vllm.distributed.kv_transfer.kv_connector.v1.offloading.common",
                fromlist=["OffloadingConnectorMetadata"],
            ).OffloadingConnectorMetadata(load_jobs={}, store_jobs={})
        )
        eng.worker.get_finished(set())
        wmeta = eng.worker.build_connector_worker_meta()
        eng.scheduler.update_connector_output(
            KVConnectorOutput(kv_connector_worker_meta=wmeta)
        )
        events = list(eng.scheduler.take_events())
        assert events, "应产出 BlockStored 事件"
        stored = [e for e in events if type(e).__name__ == "BlockStored"]
        assert stored
        first = stored[0]
        assert first.block_hashes and first.token_ids  # self-describing 载荷
        assert first.block_size == 4  # tokens_per_hash
        assert first.medium == "CPU"

    def test_placeholder_events_without_self_describing(self):
        eng = assemble_engine(
            make_vllm_config(
                extra_config={"cpu_bytes_to_use": 1 << 20}, enable_kv_events=True
            ),
            make_kv_config(),
            make_kv_caches(make_kv_config()),
        )
        req = make_request("r1", list(range(16)))
        eng.scheduler.on_new_request(req)
        fill_block_hashes(req, 4)
        out = make_scheduler_output(
            new_reqs=[{"req_id": "r1", "block_ids": ([1, 2, 3, 4],)}],
            num_scheduled_tokens={"r1": 16},
        )
        meta = eng.scheduler.build_connector_meta(out)
        from vllm.v1.outputs import KVConnectorOutput

        eng.worker.prepare_store_kv(meta)
        eng.worker.start_kv_transfers(
            __import__(
                "vllm.distributed.kv_transfer.kv_connector.v1.offloading.common",
                fromlist=["OffloadingConnectorMetadata"],
            ).OffloadingConnectorMetadata(load_jobs={}, store_jobs={})
        )
        eng.worker.get_finished(set())
        wmeta = eng.worker.build_connector_worker_meta()
        eng.scheduler.update_connector_output(
            KVConnectorOutput(kv_connector_worker_meta=wmeta)
        )
        events = list(eng.scheduler.take_events())
        stored = [e for e in events if type(e).__name__ == "BlockStored"]
        assert stored and stored[0].token_ids == [] and stored[0].block_size == 0


# ═══════════════════════ m16 请求级旋钮 ═══════════════════════


class TestRequestKnobs:
    def test_kv_load_tiers_parsed_into_tier_filter(self):
        from vllm.v1.kv_offload.base import Medium, TierFilter

        eng = assemble_engine(
            make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20}),
            make_kv_config(),
            make_kv_caches(make_kv_config()),
        )
        req = make_request(
            "r1",
            [1] * 8,
            kv_transfer_params={
                "kv_load_tiers": [
                    {"medium": "cpu"},
                    {"medium": "storage", "locality": "local"},
                ]
            },
        )
        eng.scheduler.on_new_request(req)
        f = eng.scheduler._req_status["r1"].req_context.load_tier_filter
        assert f is not TierFilter.ALL
        from vllm.v1.kv_offload.base import Locality

        assert f.allows(Medium.CPU, None) is True
        assert f.allows(Medium.STORAGE, Locality.LOCAL) is True
        assert f.allows(Medium.STORAGE, Locality.REMOTE) is False  # 限 LOCAL
        # 非法条目跳过、缺省 ALL
        req2 = make_request("r2", [1], kv_transfer_params={"kv_load_tiers": [{"medium": "bogus"}]})
        eng.scheduler.on_new_request(req2)
        f2 = eng.scheduler._req_status["r2"].req_context.load_tier_filter
        assert f2.allows(Medium.CPU, None) is True and f2.allows(Medium.STORAGE, None) is True

    def test_tier_filter_semantics(self):
        from vllm.v1.kv_offload.base import Locality, Medium, TierFilter

        assert TierFilter.ALL.allows(Medium.STORAGE, Locality.REMOTE) is True
        empty = TierFilter(matchers=())
        assert empty.allows(Medium.CPU, None) is False  # 显式空 = 全拒

    def test_max_offload_tokens_invalid_value_ignored(self):
        eng = assemble_engine(
            make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20}),
            make_kv_config(),
            make_kv_caches(make_kv_config()),
        )
        req = make_request("r1", list(range(16)), kv_transfer_params={"max_offload_tokens": "big"})
        eng.scheduler.on_new_request(req)
        st = eng.scheduler._req_status["r1"]
        assert st.max_offload_tokens is None  # 非法值 → 忽略


# ═══════════════════════ WC6 跨进程哈希前提 ═══════════════════════


class TestNoneHash:
    def test_seed_makes_none_hash_reproducible(self, monkeypatch):
        import vllm.v1.core.kv_cache_utils as kcu

        def sha_prefix(seed: str) -> bytes:
            import hashlib

            return hashlib.sha256(seed.encode()).digest()

        monkeypatch.setenv("PYTHONHASHSEED", "0")
        kcu.init_none_hash(sha_prefix)
        h1 = kcu.NONE_HASH
        kcu.init_none_hash(sha_prefix)
        assert kcu.NONE_HASH == h1  # 同种子 → 同链式种子

        monkeypatch.delenv("PYTHONHASHSEED", raising=False)
        kcu.init_none_hash(sha_prefix)
        h2 = kcu.NONE_HASH
        assert h2 != h1 and len(h2) == 32  # 未设 → os.urandom(32) 随机

    def test_offload_key_packing(self):
        from vllm.v1.kv_offload.base import (
            get_offload_block_hash,
            get_offload_group_idx,
            make_offload_key,
        )

        k = make_offload_key(b"\x01" * 32, 7)
        assert isinstance(k, bytes)
        assert get_offload_block_hash(k) == b"\x01" * 32
        assert get_offload_group_idx(k) == 7


# ═══════════════════════ reset_cache 全清钩子 ═══════════════════════


class TestResetCache:
    def test_scheduler_reset_clears_ledger_and_cursor(self):
        eng = assemble_engine(
            make_vllm_config(extra_config={"cpu_bytes_to_use": 1 << 20}),
            make_kv_config(),
            make_kv_caches(make_kv_config()),
        )
        req = make_request("r1", list(range(8)))
        eng.scheduler.on_new_request(req)
        fill_block_hashes(req, 2)
        out = make_scheduler_output(
            new_reqs=[{"req_id": "r1", "block_ids": ([1, 2],)}],
            num_scheduled_tokens={"r1": 8},
        )
        meta = eng.scheduler.build_connector_meta(out)
        assert meta.store_jobs
        eng.worker.prepare_store_kv(meta)
        eng.worker.start_kv_transfers(
            __import__(
                "vllm.distributed.kv_transfer.kv_connector.v1.offloading.common",
                fromlist=["OffloadingConnectorMetadata"],
            ).OffloadingConnectorMetadata(load_jobs={}, store_jobs={})
        )
        eng.worker.get_finished(set())
        wmeta = eng.worker.build_connector_worker_meta()
        from vllm.v1.outputs import KVConnectorOutput

        eng.scheduler.update_connector_output(
            KVConnectorOutput(kv_connector_worker_meta=wmeta)
        )
        key0 = _key_of(eng, 0)
        assert eng.manager.lookup(key0, _ctx()).name == "HIT"

        eng.scheduler.reset_cache()
        assert eng.manager.lookup(key0, _ctx()).name == "MISS"
        # 活跃请求游标归零 → 重新从 chunk 0 卸
        assert eng.scheduler._req_status["r1"].group_states[0].next_stored_chunk_idx == 0
        assert eng.scheduler.has_pending_push_work() is False
