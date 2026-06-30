"""ch26 —— FusedMoE 算子与 batch-invariant 一致性，纯 Python 控制流的 host 验证。

测的是『精简版复现 vllm-ascend 真实可观察行为』，不是精简版自洽：
  - f10 回收：MoECommType 枚举 → setup_moe_comm_method 建好的 *CommImpl 实例（三选一/一注册表）
  - f10 起点：select_moe_comm_method 按 EP/soc/token 选枚举
  - 算子二分：MoECommMethod.fused_experts 的 dispatch→mlp→combine 三段顺序
  - f3 回收：async_all_to_all 不等长 split 的输出形状代数
  - prepare 的 pad/TP 切片 与 DP all-gather 形状契约
  - AscendMoERunner._fused_output_is_reduced 按 comm_type 决定是否二次 all-reduce
  - batch-invariant：override_envs_for_invariance 的 env 覆盖 + reduce_sum 确定性回退
"""
import os
import types

import torch




# ---------- f10 回收：注册表三选一 ---------- #
def test_setup_registers_four_impls_when_ep_gt_1(env):
    MoECommType = env.fc.MoECommType
    env.moe_comm.setup_moe_comm_method(env.make_moe_config(ep_size=4))

    got = {ct: env.moe_comm.get_moe_comm_method(ct) for ct in MoECommType}
    assert type(got[MoECommType.ALLGATHER]).__name__ == "AllGatherCommImpl"
    assert type(got[MoECommType.MC2]).__name__ == "MC2CommImpl"
    assert type(got[MoECommType.ALLTOALL]).__name__ == "AlltoAllCommImpl"
    assert type(got[MoECommType.FUSED_MC2]).__name__ == "FusedMC2CommImpl"


def test_setup_registers_only_allgather_when_ep_eq_1(env):
    MoECommType = env.fc.MoECommType
    env.moe_comm._MoECommMethods.clear()
    env.moe_comm.setup_moe_comm_method(env.make_moe_config(ep_size=1))

    assert type(env.moe_comm.get_moe_comm_method(MoECommType.ALLGATHER)).__name__ == "AllGatherCommImpl"
    # EP=1 不需要跨卡重分发，MC2/All2All/FusedMC2 不注册。
    assert env.moe_comm.get_moe_comm_method(MoECommType.MC2) is None
    assert env.moe_comm.get_moe_comm_method(MoECommType.ALLTOALL) is None


def test_each_impl_pairs_dispatcher_and_prepare_finalize(env):
    """『二分』实证：每个 *CommImpl 配一对 (TokenDispatcherWith*, PrepareAndFinalizeWith*)。"""
    env.moe_comm.setup_moe_comm_method(env.make_moe_config(ep_size=4))
    MoECommType = env.fc.MoECommType

    mc2 = env.moe_comm.get_moe_comm_method(MoECommType.MC2)
    assert type(mc2.token_dispatcher).__name__ == "TokenDispatcherWithMC2"
    assert type(mc2.prepare_finalize).__name__ == "PrepareAndFinalizeWithMC2"

    a2a = env.moe_comm.get_moe_comm_method(MoECommType.ALLTOALL)
    assert type(a2a.token_dispatcher).__name__ == "TokenDispatcherWithAll2AllV"
    assert type(a2a.prepare_finalize).__name__ == "PrepareAndFinalizeWithAll2All"

    ag = env.moe_comm.get_moe_comm_method(MoECommType.ALLGATHER)
    assert type(ag.token_dispatcher).__name__ == "TokenDispatcherWithAllGather"
    assert type(ag.prepare_finalize).__name__ == "PrepareAndFinalizeWithAllGather"


# ---------- f10 起点：select_moe_comm_method ---------- #
def _vllm_config(knobs):
    return types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(enable_expert_parallel=knobs.enable_expert_parallel),
        model_config=types.SimpleNamespace(),
    )


def test_select_no_ep_falls_back_to_allgather(env):
    env.knobs.enable_expert_parallel = False
    ct = env.fc.select_moe_comm_method(num_tokens=10, vllm_config=_vllm_config(env.knobs))
    assert ct is env.fc.MoECommType.ALLGATHER


def test_select_a3_within_capacity_picks_mc2_else_alltoall(env):
    env.knobs.enable_expert_parallel = True
    env.knobs.soc = "A3"
    cfg = _vllm_config(env.knobs)
    # 容量内 → MC2；容量外 → ALLTOALL（capacity 默认 512）
    assert env.fc.select_moe_comm_method(num_tokens=8, vllm_config=cfg) is env.fc.MoECommType.MC2
    assert env.fc.select_moe_comm_method(num_tokens=5000, vllm_config=cfg) is env.fc.MoECommType.ALLTOALL


def test_select_non_moe_returns_none(env):
    env.knobs.is_moe = False
    assert env.fc.select_moe_comm_method(num_tokens=8, vllm_config=_vllm_config(env.knobs)) is None


# ---------- 算子二分：dispatch→mlp→combine 三段顺序 ---------- #
def test_fused_experts_runs_three_stage_pipeline(env):
    env.moe_comm.setup_moe_comm_method(env.make_moe_config(ep_size=4))
    impl = env.moe_comm.get_moe_comm_method(env.fc.MoECommType.ALLGATHER)
    env.extra_ctx.moe_comm_method = impl

    rec = env.rec

    class _Disp:
        def token_dispatch(self, token_dispatch_input):
            rec.rec("dispatch")
            return env.Kw(combine_metadata="cm", group_list_type=1, group_list="gl")

        def token_combine(self, hidden_states, combine_metadata):
            rec.rec("combine")
            return "routed_out"

    impl.token_dispatcher = _Disp()

    fe_input = env.Kw(topk_ids="tids", routing=env.Kw(log2phy=None), swiglu_limit=0.0)
    result = impl.fused_experts(fe_input)

    # 三段流水线严格顺序：先按专家重分发，再每专家 MLP，最后聚回。
    assert rec.calls == ["dispatch", "mlp", "combine"]
    assert result.routed_out == "routed_out"
    assert result.expert_tokens == "gl"


# ---------- f3 回收：async_all_to_all 形状代数 ---------- #
def test_async_all_to_all_equal_split_shape(env):
    cu = env.comm_utils
    captured = {}

    def _fake_a2a(out, inp, output_split_sizes, input_split_sizes, group, async_op):
        captured["out_shape"] = tuple(out.shape)
        return "handle"

    cu.dist = types.SimpleNamespace(all_to_all_single=_fake_a2a)

    x = torch.zeros(6, 4)
    _, a2a_out, handle = cu.async_all_to_all(x, None, None, group="ep")
    # 等长 split → empty_like(input)
    assert tuple(a2a_out.shape) == (6, 4)
    assert handle == "handle"


def test_async_all_to_all_unequal_split_shape(env):
    cu = env.comm_utils
    cu.dist = types.SimpleNamespace(all_to_all_single=lambda *a, **k: "handle")

    x = torch.zeros(6, 4)
    # 各卡收到的 token 数不均：output_split_sizes 决定输出行数（all2all-v）。
    _, a2a_out, _ = cu.async_all_to_all(x, output_split_sizes=[2, 3, 5], input_split_sizes=[1, 2, 3], group="ep")
    assert tuple(a2a_out.shape) == (10, 4)  # sum([2,3,5]) == 10


# ---------- prepare 的 pad / TP 切片 ---------- #
def test_all2all_prepare_pads_to_tp_and_slices(env):
    env.knobs.tp_size = 4
    env.knobs.tp_rank = 0
    PF = env.prepare_finalize.PrepareAndFinalizeWithAll2All
    pf = PF(env.make_moe_config())

    hidden = torch.arange(3 * 8, dtype=torch.float32).reshape(3, 8)
    logits = torch.zeros(3, 8)
    out = pf.prepare(hidden, logits)
    # pad 3→4 行后按 tp_size=4 切片，rank0 取 1 行。
    assert out.padded_hidden_states_shape == torch.Size([4, 8])
    assert out.hidden_states.shape[0] == 1
    assert out.mc2_mask is None  # All2All 路径不用 mc2_mask


def test_allgather_prepare_dp_all_gather_invoked(env):
    calls = []
    dp_group = types.SimpleNamespace(all_gather=lambda x, d: (calls.append("ag"), x)[1])
    cfg = env.make_moe_config(dp_size=2, dp_group=dp_group, pcp_size=1)
    env.extra_ctx.max_tokens_across_dp = 5

    PF = env.prepare_finalize.PrepareAndFinalizeWithAllGather
    pf = PF(cfg)
    hidden = torch.zeros(3, 8)
    logits = torch.zeros(3, 8)
    out = pf.prepare(hidden, logits)
    # DP>1：pad 到 max_tokens_across_dp 后对 hidden/logits 各 all-gather 一次。
    assert calls == ["ag", "ag"]
    assert out.hidden_states.shape[0] == 5  # padded to max_tokens_across_dp


# ---------- _fused_output_is_reduced 旋钮 ---------- #
def test_fused_output_is_reduced_by_comm_type(env):
    Runner = env.fused.AscendMoERunner
    r = Runner()
    MoECommType = env.fc.MoECommType
    env.extra_ctx.flash_comm_v1_enabled = False

    for ct in (MoECommType.MC2, MoECommType.ALLTOALL, MoECommType.FUSED_MC2):
        env.extra_ctx.moe_comm_type = ct
        assert r._fused_output_is_reduced is True  # finalize 内已 all-reduce，别二次 reduce

    env.extra_ctx.moe_comm_type = MoECommType.ALLGATHER
    assert r._fused_output_is_reduced is False  # AllGather 每卡 partial，仍需 all-reduce


def test_runner_use_dp_chunking_forces_forward_impl(env):
    r = env.fused.AscendMoERunner()
    assert r.use_dp_chunking is False  # 强制走 forward_impl，不走 FlashInfer Cutlass chunked


# ---------- batch-invariant ---------- #
def test_override_envs_sets_deterministic_flags(env):
    bi = env.batch_invariant
    # 预置非确定性默认值
    env.ascend_cfg.weight_nz_mode = 1
    env.ascend_cfg.enable_matmul_allreduce = True
    os.environ.pop("HCCL_DETERMINISTIC", None)
    os.environ.pop("LCCL_DETERMINISTIC", None)

    bi.override_envs_for_invariance()

    assert env.ascend_cfg.weight_nz_mode == 0
    assert env.ascend_cfg.enable_matmul_allreduce is False
    assert os.environ["HCCL_DETERMINISTIC"] == "strict"
    assert os.environ["LCCL_DETERMINISTIC"] == "1"


def test_reduce_sum_cpu_falls_back_to_torch_sum(env):
    bi = env.batch_invariant
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    # cpu 张量走 torch.sum 回退（npu 内核仅 NPU 可用），数值须与 torch.sum 一致。
    assert torch.equal(bi.reduce_sum(x, dim=1), x.sum(dim=1))
