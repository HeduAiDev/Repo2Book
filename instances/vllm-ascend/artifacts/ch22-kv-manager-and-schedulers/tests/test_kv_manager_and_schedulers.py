"""ch22 —— KV cache 管理与调度器的 NPU 特化：验证精简版复现 vllm-ascend
single_type_kv_cache_manager.py / scheduler_dynamic_batch.py / recompute_scheduler.py /
profiling_chunk_predictor.py 的可观察控制流（不是精简版自洽）：
  - manager 重映射：压缩 MLA→CompressAttentionManager + admission cap；非压缩→复用 vLLM；
  - 压缩缩放：分配 //=compress_ratio、前缀命中按 logical_block_size 粒度；
  - BudgetRefiner 查表 / 恒等；二次延迟模型解 chunk size；
  - RecomputeSchedulerConfig 选类、register 补登记、recomputed_reqs 回吐、AsyncRecompute MRO。
真实 NPU 物理分配由 vLLM 基类替身承接——只验复用 vs 特化的边界控制流。
"""
import types


# ======================= (1) get_manager_for_kv_cache_spec 重映射 ======================= #

def _bp():
    return types.SimpleNamespace(null_block=object())


def test_compressed_mla_selects_compress_manager_and_sets_admission_cap(env):
    kvm, cdiv = env.kvm, env.cdiv
    spec = env.MLAAttentionSpec(compress_ratio=4, block_size=16)
    mgr = kvm.get_manager_for_kv_cache_spec(
        spec, max_num_batched_tokens=1000, max_model_len=128, block_pool=_bp()
    )
    # 压缩 MLA → 选 CompressAttentionManager（昇腾唯一新增 manager）
    assert isinstance(mgr, kvm.CompressAttentionManager)
    assert mgr.compress_ratio == 4
    # admission cap = cdiv(max_model_len // compress_ratio, block_size) + 1
    expected = cdiv(128 // 4, 16) + 1
    assert mgr._kwargs["max_admission_blocks_per_request"] == expected == 3


def test_uncompressed_mla_reuses_vllm_full_manager(env):
    kvm = env.kvm
    spec = env.MLAAttentionSpec(compress_ratio=1, block_size=16)
    mgr = kvm.get_manager_for_kv_cache_spec(spec, max_model_len=128, block_pool=_bp())
    # compress_ratio==1 → 原样复用 vLLM FullAttentionManager，不开子类
    assert isinstance(mgr, env.FullAttentionManager)
    assert not isinstance(mgr, kvm.CompressAttentionManager)
    assert "max_admission_blocks_per_request" not in mgr._kwargs


def test_sliding_window_cap_re_applied_after_factory_override(env):
    kvm = env.kvm
    spec = env.SlidingWindowSpec(cap=7)
    mgr = kvm.get_manager_for_kv_cache_spec(
        spec, max_num_batched_tokens=1000, max_model_len=128, block_pool=_bp()
    )
    # 覆盖 vLLM 工厂后，须在此重设 SWA 的 admission cap（否则原 cap 代码成死代码）
    assert mgr._kwargs["max_admission_blocks_per_request"] == 7


def test_full_attention_spec_has_no_admission_cap(env):
    kvm = env.kvm
    spec = env.FullAttentionSpec()
    mgr = kvm.get_manager_for_kv_cache_spec(spec, max_model_len=128, block_pool=_bp())
    assert isinstance(mgr, env.FullAttentionManager)
    assert "max_admission_blocks_per_request" not in mgr._kwargs


# ======================= (2) CompressAttentionManager 压缩缩放 ======================= #

def test_get_num_blocks_divides_tokens_by_compress_ratio(env):
    kvm = env.kvm
    spec = env.MLAAttentionSpec(compress_ratio=4, block_size=16)
    mgr = kvm.CompressAttentionManager(spec, _bp())
    # 父类替身回显它收到的 num_tokens —— 子类应先 //= compress_ratio
    got = mgr.get_num_blocks_to_allocate("r0", 40, [], 0, 40)
    assert got == 40 // 4 == 10


def test_find_longest_cache_hit_matches_by_logical_block_size(env):
    kvm = env.kvm
    block_pool = types.SimpleNamespace(get_cached_block=lambda h, gids: [object() for _ in gids])
    # compress_ratio=4, block_size=16 → logical_block_size=64; max_length=200 → 3 逻辑块
    spec = env.MLAAttentionSpec(compress_ratio=4, block_size=16)
    computed = kvm.CompressAttentionManager.find_longest_cache_hit(
        block_hashes=[1, 2, 3, 4, 5],
        max_length=200,
        kv_cache_group_ids=[0],
        block_pool=block_pool,
        kv_cache_spec=spec,
        use_eagle=False,
        alignment_tokens=64,
    )
    assert len(computed[0]) == 200 // 64 == 3
    # 压缩比变小 → 逻辑块更小 → 命中更多块（粒度随 compress_ratio 缩放）
    spec2 = env.MLAAttentionSpec(compress_ratio=2, block_size=16)
    computed2 = kvm.CompressAttentionManager.find_longest_cache_hit(
        block_hashes=[1, 2, 3, 4, 5], max_length=200, kv_cache_group_ids=[0],
        block_pool=block_pool, kv_cache_spec=spec2, use_eagle=False, alignment_tokens=32,
    )
    assert len(computed2[0]) == 5  # logical=32 → max 6，但只 5 个哈希被缓存


# ======================= (3) BudgetRefiner 动态预算 ======================= #

def test_budget_refiner_identity_when_disabled(env):
    br = env.dyn.BudgetRefiner(default_budget=100, slo_limit=-1)
    assert br.enabled is False
    assert br.refine_budget([], 4096) == 4096  # 未配 SLO → 恒等，零开销


def test_align_key_picks_smallest_key_geq_value(env):
    br = env.dyn.BudgetRefiner(default_budget=100, slo_limit=-1)
    assert br._align_key(5, [2, 4, 6, 8]) == 6
    assert br._align_key(8, [2, 4, 6, 8]) == 8
    assert br._align_key(100, [2, 4]) is None


def test_read_lookup_table_disables_when_csv_absent(env):
    # 真实行为：slo_limit>0 先置 enabled=True，但 profile_table.csv 缺失 → 回退禁用
    br = env.dyn.BudgetRefiner(default_budget=100, slo_limit=5)
    assert br.enabled is False
    assert br.refine_budget([], 4096) == 4096


def test_get_max_budget_and_refine_lookup(env):
    br = env.dyn.BudgetRefiner(default_budget=100, slo_limit=5)
    # csv 缺失会自动禁用；手动模拟「查表已就绪」以验 _get_max_budget/refine_budget 控制流
    br.enabled = True
    # 注入查表（真实由 _read_lookup_table 从 profile_table.csv 加载，已减法）
    br.lookup = {(64, 2): 256}
    br.context_keys = {64}
    br.dnum_keys = {2}
    assert br._get_max_budget(50, 2) == 256
    assert br._get_max_budget(999, 2) == 100  # ctx 对齐失败 → default

    def req(tok, computed, prompt):
        return types.SimpleNamespace(num_tokens_with_spec=tok, num_computed_tokens=computed, num_prompt_tokens=prompt)

    running = [req(50, 10, 5), req(50, 10, 5)]  # 两个 decode 请求（computed>=prompt）
    assert br.refine_budget(running, 9999) == 256


# ======================= (4) 二次延迟模型预测 chunk size ======================= #

def test_chunk_predictor_fit_needs_min_points(env):
    p = env.pred.ChunkSizePredictor()
    assert p.fit([1, 2, 3], [1.0, 2.0, 3.0]) is False  # <8 点


def test_chunk_predictor_solves_quadratic(env):
    p = env.pred.ChunkSizePredictor(smooth_factor=1.0, min_chunk=64)
    # f(l)=a l^2 + b l + c, a=1e-6, b=1e-3 → f(10000)-f(0)=100+10=110
    p.quadratic_coeff_a = 1e-6
    p.linear_coeff_b = 1e-3
    p.constant_coeff_c = 0.0
    p.target_latency = 110.0
    p.is_ready = True
    out = p.predict(num_computed_tokens=0, base_chunk_size=10000, page_size=64)
    assert out is not None
    assert out % 64 == 0          # 对齐到 max(page_size, 64)
    assert abs(out - 10000) < 64  # 解出的 chunk ≈ 10000


def test_chunk_predictor_returns_none_when_not_ready(env):
    p = env.pred.ChunkSizePredictor()
    p.target_latency = 10.0
    p.is_ready = False
    assert p.predict(0, 8192, 128) is None


def test_profiling_chunk_manager_readiness_gate(env):
    mgr = env.pred.ProfilingChunkManager(base_chunk_size=8192, page_size=128)
    # 未 profiling → 门控返回 None / 0.0
    assert mgr.is_ready is False
    assert mgr.predict_chunk_size(num_computed_tokens=100, target_time=1.0) is None
    assert mgr.predict_time(num_new_tokens=10, num_computed_tokens=100) == 0.0
    # 标记就绪后走基础二次模型
    mgr.predictor.quadratic_coeff_a = 1e-6
    mgr.predictor.linear_coeff_b = 1e-3
    mgr.predictor.target_latency = 110.0
    mgr.predictor.is_ready = True
    mgr._profiling_done = True
    assert mgr.is_ready is True
    assert mgr.predict_chunk_size(num_computed_tokens=0, target_time=1.0) is not None
    assert mgr.predict_time(num_new_tokens=100, num_computed_tokens=0) > 0


# ======================= (5) RecomputeScheduler 系列 ======================= #

def test_recompute_config_selects_class_by_async(env):
    rec, SchedulerConfig = env.rec, env.SchedulerConfig

    def vllm_config(async_scheduling):
        return types.SimpleNamespace(
            scheduler_config=SchedulerConfig(async_scheduling=async_scheduling, max_num_batched_tokens=2048),
            model_config=types.SimpleNamespace(max_model_len=128, is_encoder_decoder=False),
        )

    sync_cfg = rec.RecomputeSchedulerConfig.initialize_from_config(vllm_config(False))
    assert sync_cfg.scheduler_cls.endswith("RecomputeScheduler")
    assert not sync_cfg.scheduler_cls.endswith("AsyncRecomputeScheduler")
    assert sync_cfg.max_model_len == 128

    async_cfg = rec.RecomputeSchedulerConfig.initialize_from_config(vllm_config(True))
    assert async_cfg.scheduler_cls.endswith("AsyncRecomputeScheduler")


def test_register_ascend_mla_spec_patches_map(env):
    rec, smm, MLA = env.rec, env.spec_manager_map, env.MLAAttentionSpec
    smm.pop(MLA, None)
    assert MLA not in smm
    rec.register_ascend_mla_spec_in_manager()
    assert MLA in smm  # 补登记，绕过子进程 unpickle 的 KeyError
    assert smm[MLA] is env.FullAttentionManager


def test_recompute_dataclasses(env):
    rec = env.rec
    info = rec.RecomputeReqInfo("rid-1", [7, 8, 9], 3)
    assert info.request_id == "rid-1"
    assert info.output_token_ids == [7, 8, 9]
    assert info.client_index == 3
    out = rec.RecomputeSchedulerOutput(recomputed_reqs=[info])
    assert out.recomputed_reqs == [info]
    # 默认无 recomputed_reqs 时为 None（普通调度步）
    assert rec.RecomputeSchedulerOutput().recomputed_reqs is None


def test_update_from_output_emits_recomputed_stop_reason(env):
    rec = env.rec
    obj = rec.RecomputeScheduler.__new__(rec.RecomputeScheduler)
    so = rec.RecomputeSchedulerOutput(
        num_scheduled_tokens={},
        recomputed_reqs=[rec.RecomputeReqInfo("rid-9", [], 0)],
    )
    result = rec.RecomputeScheduler.update_from_output(obj, so, types.SimpleNamespace())
    assert 0 in result
    eco = result[0].outputs[0]
    assert eco.request_id == "rid-9"
    assert eco.stop_reason == "recomputed"      # PD proxy 据此改投他节点重算
    assert eco.finish_reason == "STOP"
    assert eco.new_token_ids == []


def test_async_recompute_scheduler_mro(env):
    rec = env.rec
    mro_names = [c.__name__ for c in rec.AsyncRecomputeScheduler.__mro__]
    # 多继承组合：异步调度 + recompute 调度
    assert mro_names[:3] == ["AsyncRecomputeScheduler", "AsyncScheduler", "RecomputeScheduler"]
    assert "Scheduler" in mro_names
