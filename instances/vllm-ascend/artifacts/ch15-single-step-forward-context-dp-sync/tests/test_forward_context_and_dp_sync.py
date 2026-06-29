"""ch15 TDD —— 验证 vllm-ascend 真实代码（忠实减法进 ../implementation）的可观察行为：

1. select_moe_comm_method：按 是否 MoE / 是否 EP / soc 版本 / token 数 vs mc2 容量
   选定 MoECommType（ALLGATHER/MC2/ALLTOALL/FUSED_MC2 / None）。
2. set_ascend_forward_context：`with set_forward_context(基座)` 包住后，往同一个
   forward_context 上注入昇腾专属字段（moe_comm_type/method、flash_comm_v1/v2、
   mmrs_fusion、padded_num_tokens、mc2_mask）。
3. _sync_metadata_across_dp + _post_process_cudagraph_mode：把 num_tokens + cudagraph_mode
   打包进 [2, dp] 一次 all_reduce(sum 即广播)，解包取 max token 数 + min cudagraph_mode。

昇腾算子/真实 all_reduce 不在 host 跑——这些都是纯 Python 控制流，故可断言其与真实仓一致。
"""
import types

import pytest
import torch


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def make_vllm_config(
    *,
    is_moe=True,
    enable_ep=False,
    world_size_across_dp=2,
    pipeline_parallel_size=1,
    num_experts=16,
    data_parallel_size=1,
    moe_quantize=None,
    num_experts_per_tok=2,
):
    hf = types.SimpleNamespace(
        moe_quantize=moe_quantize,
        num_experts_per_tok=num_experts_per_tok,
    )
    hf.to_dict = lambda: ({"num_experts": num_experts} if is_moe else {"num_layers": 4})
    model_config = types.SimpleNamespace(
        hf_text_config=hf,
        get_num_experts=lambda: num_experts,
        is_encoder_decoder=False,
    )
    parallel_config = types.SimpleNamespace(
        enable_expert_parallel=enable_ep,
        world_size_across_dp=world_size_across_dp,
        pipeline_parallel_size=pipeline_parallel_size,
        data_parallel_size=data_parallel_size,
        is_moe_model=is_moe,
        data_parallel_rank=0,
    )
    compilation_config = types.SimpleNamespace(static_forward_context={})
    return types.SimpleNamespace(
        model_config=model_config,
        parallel_config=parallel_config,
        compilation_config=compilation_config,
        speculative_config=None,
    )


# --------------------------------------------------------------------------- #
# 1. select_moe_comm_method
# --------------------------------------------------------------------------- #
def test_non_moe_returns_none(env):
    mods, knobs, _ = env
    cfg = make_vllm_config(is_moe=False)
    assert mods.afc.select_moe_comm_method(10, cfg) is None


def test_no_expert_parallel_falls_back_to_allgather(env):
    mods, knobs, _ = env
    cfg = make_vllm_config(is_moe=True, enable_ep=False)
    assert mods.afc.select_moe_comm_method(10, cfg) is mods.afc.MoECommType.ALLGATHER


def test_a2_picks_mc2_when_capacity_and_large_ep(env):
    mods, knobs, _ = env
    knobs.soc = mods.utils.AscendDeviceType.A2
    mods.utils._ascend_device_type = knobs.soc
    mods.afc._mc2_tokens_capacity = 512
    # ep_world_size = world_size_across_dp // pp = 16; experts/device = 256//16=16 <=24
    cfg = make_vllm_config(is_moe=True, enable_ep=True, world_size_across_dp=16, num_experts=256)
    knobs.ep_world_size = 2  # get_ep_group().world_size != 1 so we don't short-circuit
    assert mods.afc.select_moe_comm_method(100, cfg) is mods.afc.MoECommType.MC2
    # tokens exceed capacity → ALLGATHER
    assert mods.afc.select_moe_comm_method(1000, cfg) is mods.afc.MoECommType.ALLGATHER


def test_a3_mc2_within_capacity_else_alltoall(env):
    mods, knobs, _ = env
    knobs.soc = mods.utils.AscendDeviceType.A3
    mods.utils._ascend_device_type = knobs.soc
    mods.afc._mc2_tokens_capacity = 512
    knobs.ep_world_size = 8
    knobs.ascend_config.enable_fused_mc2 = 0  # plain MC2 / ALLTOALL
    cfg = make_vllm_config(is_moe=True, enable_ep=True)
    assert mods.afc.select_moe_comm_method(100, cfg) is mods.afc.MoECommType.MC2
    assert mods.afc.select_moe_comm_method(1000, cfg) is mods.afc.MoECommType.ALLTOALL


def test_unsupported_soc_raises(env):
    mods, knobs, _ = env

    class _Bad:
        pass

    mods.utils._ascend_device_type = _Bad()
    mods.afc._mc2_tokens_capacity = 512
    knobs.ep_world_size = 8
    cfg = make_vllm_config(is_moe=True, enable_ep=True)
    with pytest.raises(ValueError):
        mods.afc.select_moe_comm_method(100, cfg)


# --------------------------------------------------------------------------- #
# 2. set_ascend_forward_context —— 包基座 + 注入昇腾字段
# --------------------------------------------------------------------------- #
def test_wraps_base_and_injects_ascend_fields(env):
    mods, knobs, _ = env
    knobs.tp_world_size = 4
    # 注册一个 ALLGATHER 的通信方法实例，验证 get_moe_comm_method 被取用
    sentinel_method = object()
    mods.mcm._MoECommMethods[mods.afc.MoECommType.ALLGATHER] = sentinel_method
    cfg = make_vllm_config(is_moe=True, enable_ep=False, data_parallel_size=1)

    seen = {}
    with mods.afc.set_ascend_forward_context(
        attn_metadata=None,
        vllm_config=cfg,
        num_tokens=10,
        model_instance=None,
    ):
        fc = mods.fwd.get_forward_context()
        seen["fc"] = fc
        # 基座建好（additional_kwargs 由 current_platform 注入）
        assert fc.additional_kwargs == {"injected_by_platform": True}
        # 昇腾注入字段
        assert fc.moe_comm_type is mods.afc.MoECommType.ALLGATHER
        assert fc.moe_comm_method is sentinel_method
        # MoE 模型分支强制 mmrs_fusion=False（tp<=8 的默认 True 被 MoE 分支覆盖）
        assert fc.mmrs_fusion is False
        assert fc.num_tokens == 10
        assert fc.flash_comm_v1_enabled is False  # enable_sp 默认 False
        assert fc.flashcomm_v2_enabled is False
        # padded_num_tokens = ceil(10/4)*4 = 12
        assert fc.padded_num_tokens == 12
    # 退出后 forward context 还原为 None
    with pytest.raises(AssertionError):
        mods.fwd.get_forward_context()


def test_mc2_mask_sliced_to_padded_num_tokens(env):
    mods, knobs, _ = env
    knobs.tp_world_size = 4
    mods.mcm._MoECommMethods[mods.afc.MoECommType.ALLGATHER] = object()
    # 预留 mc2 掩码缓冲（set_mc2_mask 的产物，本章只读切片）
    mods.afc._reserved_mc2_mask = torch.zeros(64, dtype=torch.bool)
    cfg = make_vllm_config(is_moe=True, enable_ep=False)

    with mods.afc.set_ascend_forward_context(
        attn_metadata=None,
        vllm_config=cfg,
        num_tokens=10,
        num_actual_tokens=7,
        model_instance=None,
    ):
        fc = mods.fwd.get_forward_context()
        # padded_num_tokens = ceil(10/4)*4 = 12；切片长度 12，前 7 个 True 其余 False
        assert fc.mc2_mask.shape[0] == 12
        assert bool(fc.mc2_mask[:7].all())
        assert not bool(fc.mc2_mask[7:].any())


def test_flashcomm_v1_threshold_for_dense_model(env):
    mods, knobs, _ = env
    knobs.tp_world_size = 2
    knobs.ascend_config.enable_flashcomm1 = True  # enable_sp -> True
    mods.mcm._MoECommMethods[mods.afc.MoECommType.ALLGATHER] = object()
    cfg = make_vllm_config(is_moe=False, enable_ep=False)  # dense → 1000 阈值生效

    with mods.afc.set_ascend_forward_context(
        attn_metadata=None, vllm_config=cfg, num_tokens=500, model_instance=None
    ):
        # dense + num_tokens<=1000 → flashcomm_v1 关
        assert mods.fwd.get_forward_context().flash_comm_v1_enabled is False

    # reset cached _ENABLE_SP for second evaluation
    mods.utils._ENABLE_SP = None
    mods.utils._IS_MOE_MODEL = None
    knobs.ascend_config.enable_flashcomm1 = True
    with mods.afc.set_ascend_forward_context(
        attn_metadata=None, vllm_config=cfg, num_tokens=2000, model_instance=None
    ):
        # dense + num_tokens>1000 → flashcomm_v1 开
        assert mods.fwd.get_forward_context().flash_comm_v1_enabled is True


# --------------------------------------------------------------------------- #
# 3. _sync_metadata_across_dp + _post_process_cudagraph_mode
# --------------------------------------------------------------------------- #
def _make_runner(env, *, dp_size, dp_rank, dp_allreduce_on_npu=False):
    mods, knobs, _ = env
    runner = object.__new__(mods.mr.NPUModelRunner)
    runner.dp_size = dp_size
    runner.dp_rank = dp_rank
    runner.vllm_config = make_vllm_config(data_parallel_size=dp_size)
    runner.parallel_config = runner.vllm_config.parallel_config
    runner.ascend_config = types.SimpleNamespace(dp_allreduce_on_npu=dp_allreduce_on_npu)
    return runner, mods


def test_post_process_takes_min_cudagraph_mode(env):
    mods, knobs, _ = env
    CG = mods.cfg.CUDAGraphMode
    # 第二行是 cudagraph_mode：一卡 FULL(2)、一卡 NONE(0) → min = NONE
    packed = torch.tensor([[10, 14], [CG.FULL.value, CG.NONE.value]], dtype=torch.int32)
    assert mods.mr._post_process_cudagraph_mode(packed) == CG.NONE.value


def test_sync_dp_size_one_passthrough(env):
    runner, mods = _make_runner(env, dp_size=1, dp_rank=0)
    CG = mods.cfg.CUDAGraphMode
    out = runner._sync_metadata_across_dp(num_tokens=10, cudagraph_mode=CG.FULL)
    assert out == (10, None, CG.FULL)


def test_sync_packs_and_unpacks_two_ranks(env, monkeypatch):
    runner, mods = _make_runner(env, dp_size=2, dp_rank=0)
    CG = mods.cfg.CUDAGraphMode
    # 强制不跳过 all_reduce，走打包路径
    monkeypatch.setattr(mods.mr, "should_skip_allreduce_across_dp_group", lambda *a, **k: False)

    # 模拟另一卡(rank1)的贡献：num_tokens=14, cudagraph_mode=NONE(0)
    other = torch.tensor([[0, 14], [0, CG.NONE.value]], dtype=torch.int32)

    def fake_all_reduce(t, group=None):
        t += other  # sum-allreduce == 把各卡各填己列后汇齐广播

    monkeypatch.setattr(mods.mr, "dist", types.SimpleNamespace(all_reduce=fake_all_reduce))

    # 本卡 rank0：num_tokens=10, cudagraph_mode=FULL(2)
    max_tokens, after_pad, synced_mode = runner._sync_metadata_across_dp(
        num_tokens=10, cudagraph_mode=CG.FULL, allow_dp_padding=True
    )
    # token 取 max；cudagraph_mode 取 min(任一 NONE 则全 NONE)
    assert max_tokens == 14
    assert synced_mode == CG.NONE
    # allow_dp_padding=True → 全卡对齐到 max
    assert after_pad.tolist() == [14, 14]


def test_sync_without_padding_keeps_per_rank_tokens(env, monkeypatch):
    runner, mods = _make_runner(env, dp_size=2, dp_rank=0)
    CG = mods.cfg.CUDAGraphMode
    monkeypatch.setattr(mods.mr, "should_skip_allreduce_across_dp_group", lambda *a, **k: False)
    other = torch.tensor([[0, 14], [0, CG.FULL.value]], dtype=torch.int32)

    def fake_all_reduce(t, group=None):
        t += other

    monkeypatch.setattr(mods.mr, "dist", types.SimpleNamespace(all_reduce=fake_all_reduce))
    max_tokens, after_pad, synced_mode = runner._sync_metadata_across_dp(
        num_tokens=10, cudagraph_mode=CG.FULL, allow_dp_padding=False
    )
    assert max_tokens == 14
    # 两卡都 FULL → min 仍 FULL
    assert synced_mode == CG.FULL
    # 不 padding → 保留各卡真实 token 数 [10, 14]
    assert after_pad.tolist() == [10, 14]


# --------------------------------------------------------------------------- #
# 4. execute_model 派发骨架 —— 主线顺序 + set_ascend_forward_context 包住前向
# --------------------------------------------------------------------------- #
class _Model:
    """可调用的模型桩：__call__ 返回 hidden_states；compute_logits 返回 logits。"""

    def __init__(self):
        self.hidden = torch.zeros(4, 8)

    def __call__(self, **kw):
        return self.hidden

    def compute_logits(self, x):
        return torch.zeros(x.shape[0], 16)


def _cm(value=None):
    import contextlib

    @contextlib.contextmanager
    def _c(*a, **k):
        yield value

    return _c


def test_execute_model_dispatch_order(env, monkeypatch):
    mods, knobs, _ = env
    mr = mods.mr

    # 注入 execute_model 主干引用、被减法折叠的符号（真实出处见 model_runner 顶部 SUBTRACTED 注释）
    monkeypatch.setattr(mr, "ExecuteModelState", lambda *a, **k: ("state", a), raising=False)
    monkeypatch.setattr(mr, "EMPTY_MODEL_RUNNER_OUTPUT", object(), raising=False)

    cfg = make_vllm_config(is_moe=True, enable_ep=False, data_parallel_size=1)
    cfg.model_config.enable_return_routed_experts = False
    cfg.model_config.enforce_eager = True

    runner = object.__new__(mr.NPUModelRunner)
    order = []

    # ---- 状态/标志 ---- #
    runner.vllm_config = cfg
    runner.model_config = cfg.model_config
    runner.parallel_config = cfg.parallel_config
    runner.speculative_config = None
    runner.ascend_config = types.SimpleNamespace(
        profiling_chunk_config=types.SimpleNamespace(need_timing=False)
    )
    runner.execute_model_state = None
    runner.use_async_scheduling = False
    runner.num_spec_tokens = 0
    runner._draft_token_ids = None
    runner.pcp_size = 1
    runner._has_sinks = False
    runner.is_pooling_model = False
    runner.enable_enpu = False
    runner.model = _Model()
    runner.input_batch = types.SimpleNamespace(num_reqs=1, req_ids=[0])

    # ---- 子步骤桩：记录调用顺序 ---- #
    def rec(name, ret):
        def _fn(*a, **k):
            order.append(name)
            return ret
        return _fn

    runner._start_dump_data = rec("dump", None)
    runner.synchronize_input_prep = _cm()
    runner._update_states = rec("update_states", None)
    runner._prepare_inputs = rec(
        "prepare_inputs", (torch.tensor([0]), None, 4, None)
    )
    batch_desc = mr.BatchDescriptor(num_tokens=4)
    runner._determine_batch_execution_and_padding = rec(
        "determine_padding", (mods.cfg.CUDAGraphMode.NONE, batch_desc, False, None, None)
    )
    runner._build_attention_metadata = rec("build_attn", ({}, None))
    runner._preprocess = rec(
        "preprocess", (torch.tensor([1, 2, 3, 4]), None, torch.zeros(4), None, {}, None)
    )
    runner.maybe_get_kv_connector_output = _cm("kvc")
    runner._model_forward = rec("model_forward", runner.model.hidden)
    runner._update_full_graph_params_if_needed = rec("full_graph", None)

    scheduler_output = types.SimpleNamespace(
        total_num_scheduled_tokens=4,
        num_scheduled_tokens={0: 4},
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        disable_profiling_timing=False,
    )

    out = runner.execute_model(scheduler_output)

    # execute_model 返回 None（采样在 sample_tokens 里做），执行态已存
    assert out is None
    assert runner.execute_model_state is not None
    # 主线顺序：输入整形 → 批/padding(含 DP 同步) → 建注意力元数据 → 预处理 → 前向
    assert order == [
        "dump",
        "update_states",
        "prepare_inputs",
        "determine_padding",
        "build_attn",
        "preprocess",
        "model_forward",
    ]


def test_sync_routes_through_npu_group_when_enabled(env, monkeypatch):
    runner, mods = _make_runner(env, dp_size=2, dp_rank=0, dp_allreduce_on_npu=True)
    CG = mods.cfg.CUDAGraphMode
    monkeypatch.setattr(mods.mr, "should_skip_allreduce_across_dp_group", lambda *a, **k: False)
    used = {}

    def fake_all_reduce(t, group=None):
        used["group"] = group
        t += torch.tensor([[0, 12], [0, CG.NONE.value]], dtype=torch.int32)

    monkeypatch.setattr(mods.mr, "dist", types.SimpleNamespace(all_reduce=fake_all_reduce))

    # host 无 NPU：把 device="npu" 的分配落到 cpu，只验「选中 npu device group + 走 .cpu() 回拷」
    # 的路由决策（真实 npu 张量/all_reduce 不在 host 跑——见本章运行约束）。
    real_zeros = torch.zeros

    def zeros_coerce(*a, **k):
        if k.get("device") == "npu":
            k["device"] = "cpu"
        return real_zeros(*a, **k)

    monkeypatch.setattr(torch, "zeros", zeros_coerce)
    max_tokens, after_pad, synced_mode = runner._sync_metadata_across_dp(
        num_tokens=8, cudagraph_mode=CG.FULL, allow_dp_padding=True
    )
    # 走 NPU device group 规避 CPU 脏数据
    assert used["group"] == "npu-dp-group"
    assert max_tokens == 12
    assert synced_mode == CG.NONE
