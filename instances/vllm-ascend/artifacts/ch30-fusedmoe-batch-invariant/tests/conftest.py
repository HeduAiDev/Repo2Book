"""ch26 测试脚手架：host 无 NPU/CANN，在 sys.modules 桩掉 torch_npu / vllm 重运行时依赖 /
vllm_ascend 周边模块，再把（已减法的）implementation/ 模块按**规范模块名**注册进去，
让它们彼此 import 解析到精简版。

可在 host 验证、与真仓一致的纯 Python 控制流：
  (1) setup_moe_comm_method/get_moe_comm_method —— EP>1 注册 4 种 *CommImpl、EP=1 只注册 AllGather
      （回收 f10：MoECommType 枚举 → 真正建好的实例）；
  (2) select_moe_comm_method —— 无 EP→ALLGATHER；按 soc/token 数选 MC2/ALLTOALL（f10 起点）；
  (3) MoECommMethod.fused_experts —— dispatch→mlp→combine 三段流水线的调用顺序；
  (4) async_all_to_all —— 等长 vs 不等长 split 的输出形状代数（回收 f3）；
  (5) PrepareAndFinalize*.prepare —— pad 到 TP / DP all-gather 的形状契约；
  (6) AscendMoERunner._fused_output_is_reduced —— 按 comm_type 决定是否二次 all-reduce；
  (7) batch_invariant override_envs_for_invariance / reduce_sum —— env 覆盖 + 确定性 reduce。
真实 torch_npu / 集合通信算子由「记录调用」替身承接——只验入参/分流，不真算（昇腾才有内核）。
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


def _load(relpath, modname):
    spec = importlib.util.spec_from_file_location(modname, IMPL_DIR / relpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    if "." in modname:
        parent = modname.rsplit(".", 1)[0]
        if parent in sys.modules:
            setattr(sys.modules[parent], modname.rsplit(".", 1)[1], mod)
    spec.loader.exec_module(mod)
    return mod


class _Stubs:
    def __init__(self):
        self.added = []

    def mod(self, dotted):
        parts = dotted.split(".")
        for i in range(len(parts)):
            name = ".".join(parts[: i + 1])
            if name not in sys.modules:
                m = types.ModuleType(name)
                sys.modules[name] = m
                self.added.append(name)
                if i > 0:
                    setattr(sys.modules[".".join(parts[:i])], parts[i], m)
        return sys.modules[dotted]

    def cleanup(self):
        for n in reversed(self.added):
            sys.modules.pop(n, None)


class _Kw:
    """记录构造 kwargs 的通用占位对象（替 MoEPrepareOutput / *DispatchOutput / metadata）。"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class _Recorder:
    def __init__(self):
        self.calls = []

    def rec(self, name):
        self.calls.append(name)


@pytest.fixture
def env():
    stubs = _Stubs()
    rec = _Recorder()

    # ---- 可由测试改写的旋钮 ---- #
    knobs = types.SimpleNamespace(
        ep_size=4,
        enable_expert_parallel=True,
        ep_world_size=4,
        soc="A3",
        is_moe=True,
        enable_fused_mc2=0,
        tp_size=1,
        tp_rank=0,
        dp_size=1,
    )

    # ---- torch_npu 替身 ---- #
    sys.modules["torch_npu"] = types.ModuleType("torch_npu")
    stubs.added.append("torch_npu")

    # ---- vllm 顶层 + 子模块 ---- #
    stubs.mod("vllm")
    envs = stubs.mod("vllm.envs")
    envs.VLLM_BATCH_INVARIANT = False
    logger_mod = stubs.mod("vllm.logger")
    logger_mod.logger = types.SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None)
    triton_utils = stubs.mod("vllm.triton_utils")
    triton_utils.HAS_TRITON = False
    triton_utils.tl = types.SimpleNamespace(constexpr=None)
    triton_utils.triton = types.SimpleNamespace(jit=lambda f: f, cdiv=lambda a, b: (a + b - 1) // b)

    cfg = stubs.mod("vllm.config")
    cfg.VllmConfig = type("VllmConfig", (), {})
    cfg.get_current_vllm_config = lambda: types.SimpleNamespace()

    # vllm.distributed (+ parallel_state)
    def _ep_group():
        return types.SimpleNamespace(world_size=knobs.ep_world_size, rank_in_group=0, device_group=object())

    dist_mod = stubs.mod("vllm.distributed")
    dist_mod.get_ep_group = _ep_group
    dist_mod.get_dp_group = lambda: types.SimpleNamespace(world_size=knobs.dp_size)
    dist_mod.get_tp_group = lambda: types.SimpleNamespace(world_size=knobs.tp_size)
    dist_mod.get_tensor_model_parallel_world_size = lambda: knobs.tp_size
    dist_mod.get_tensor_model_parallel_rank = lambda: knobs.tp_rank
    dist_mod.tensor_model_parallel_all_reduce = lambda x: x
    ps = stubs.mod("vllm.distributed.parallel_state")
    ps.get_ep_group = _ep_group
    ps.get_tensor_model_parallel_world_size = lambda: knobs.tp_size
    ps.get_tensor_model_parallel_rank = lambda: knobs.tp_rank
    ps.get_dp_group = lambda: types.SimpleNamespace(reduce_scatter=lambda x, d: x)

    fwd = stubs.mod("vllm.forward_context")
    fwd.get_forward_context = lambda: types.SimpleNamespace(input_ids=None)

    # vllm.model_executor.layers.fused_moe (+ config/layer/runner)
    fm = stubs.mod("vllm.model_executor.layers.fused_moe")
    fm.FusedMoEConfig = type("FusedMoEConfig", (), {})
    fm.FusedMoE = type("FusedMoE", (), {})
    fm.UnquantizedFusedMoEMethod = type("UnquantizedFusedMoEMethod", (), {"__init__": lambda self, moe=None: None})
    stubs.mod("vllm.model_executor.layers.fused_moe.config").FusedMoEConfig = fm.FusedMoEConfig
    layer_mod = stubs.mod("vllm.model_executor.layers.fused_moe.layer")
    layer_mod.FusedMoE = fm.FusedMoE
    layer_mod.UnquantizedFusedMoEMethod = fm.UnquantizedFusedMoEMethod
    runner_mod = stubs.mod("vllm.model_executor.layers.fused_moe.runner.moe_runner")
    runner_mod.MoERunner = type("MoERunner", (), {})

    # ---- vllm_ascend 周边 ---- #
    stubs.mod("vllm_ascend")

    class _AscendCfg:
        weight_nz_mode = 1
        enable_matmul_allreduce = True
        enable_fused_mc2 = property(lambda self: knobs.enable_fused_mc2)
        enable_shared_expert_dp = False
        ascend_fusion_config = types.SimpleNamespace(fusion_ops_gmmswigluquant=False)
        eplb_config = types.SimpleNamespace(dynamic_eplb=False)
        ascend_compilation_config = types.SimpleNamespace(enable_static_kernel=False)

    _ascend_cfg = _AscendCfg()
    ascfg = stubs.mod("vllm_ascend.ascend_config")
    ascfg.get_ascend_config = lambda: _ascend_cfg

    # AscendDeviceType / device probes / is_moe_model
    class _Soc:
        A2 = "A2"
        A3 = "A3"
        A5 = "A5"
        _310P = "310P"

    utils = stubs.mod("vllm_ascend.utils")
    utils.AscendDeviceType = _Soc
    utils.get_ascend_device_type = lambda: getattr(_Soc, knobs.soc if knobs.soc != "310P" else "_310P")
    utils.is_moe_model = lambda cfg: knobs.is_moe
    utils.maybe_trans_nz = lambda x: x
    utils.ACL_FORMAT_FRACTAL_NZ = 29

    # quant_type
    qt = stubs.mod("vllm_ascend.quantization.quant_type")

    class QuantType:
        NONE = "none"
        W8A8 = "w8a8"
        MXFP8 = "mxfp8"
        MXFP4 = "mxfp4"
        W4A8MXFP = "w4a8mxfp"

    qt.QuantType = QuantType

    # moe_mlp.unified_apply_mlp（记录 mlp 被调）
    mlp = stubs.mod("vllm_ascend.ops.fused_moe.moe_mlp")

    def _unified_apply_mlp(mlp_compute_input=None):
        rec.rec("mlp")
        return ("mlp_out", None)

    mlp.unified_apply_mlp = _unified_apply_mlp

    # device_op.DeviceOperator
    devop = stubs.mod("vllm_ascend.device.device_op")
    devop.DeviceOperator = types.SimpleNamespace(npu_moe_init_routing=lambda *a, **k: (None, None, None, None))

    # moe_runtime_args（dataclass 占位 + builders）
    mra = stubs.mod("vllm_ascend.ops.fused_moe.moe_runtime_args")
    for nm in (
        "MoEFusedExpertsInput", "MoEMlpComputeInput", "MoEPrepareOutput",
        "MoEAllGatherCombineMetadata", "MoEAllToAllCombineMetadata", "MoEMC2CombineMetadata",
        "MoETokenDispatchInput", "MoETokenDispatchOutput",
    ):
        setattr(mra, nm, _Kw)
    import typing

    mra.TMoECombineMetadata = typing.TypeVar("TMoECombineMetadata")
    mra.build_token_dispatch_input = lambda **k: _Kw(**k)
    mra.build_mlp_compute_input = lambda **k: _Kw(**k)
    mra.build_fused_experts_input = lambda **k: _Kw(**k)

    # ---- 加载精简版（按依赖顺序）---- #
    fc = _load("ascend_forward_context.py", "vllm_ascend.ascend_forward_context")
    # _EXTRA_CTX 是 ch15 的容器（本章已减法）；测试注入一个可改写的占位。
    fc._EXTRA_CTX = types.SimpleNamespace(
        moe_comm_method=None, moe_comm_type=None, flash_comm_v1_enabled=False,
        in_profile_run=False, mc2_mask=None, padded_num_tokens=0, max_tokens_across_dp=0,
    )
    stubs.mod("vllm_ascend.ops")
    stubs.mod("vllm_ascend.ops.fused_moe")
    _load("comm_utils.py", "vllm_ascend.ops.fused_moe.comm_utils")
    _load("prepare_finalize.py", "vllm_ascend.ops.fused_moe.prepare_finalize")
    _load("token_dispatcher.py", "vllm_ascend.ops.fused_moe.token_dispatcher")
    moe_comm = _load("moe_comm_method.py", "vllm_ascend.ops.fused_moe.moe_comm_method")
    _load("experts_selector.py", "vllm_ascend.ops.fused_moe.experts_selector")
    fused = _load("fused_moe.py", "vllm_ascend.ops.fused_moe.fused_moe")
    bi = _load("batch_invariant.py", "vllm_ascend.batch_invariant")

    mods = types.SimpleNamespace(
        fc=fc,
        moe_comm=moe_comm,
        fused=fused,
        token_dispatcher=sys.modules["vllm_ascend.ops.fused_moe.token_dispatcher"],
        prepare_finalize=sys.modules["vllm_ascend.ops.fused_moe.prepare_finalize"],
        comm_utils=sys.modules["vllm_ascend.ops.fused_moe.comm_utils"],
        batch_invariant=bi,
        ascend_cfg=_ascend_cfg,
        extra_ctx=fc._EXTRA_CTX,
        knobs=knobs,
        rec=rec,
        Kw=_Kw,
        make_moe_config=make_moe_config,
    )
    try:
        yield mods
    finally:
        stubs.cleanup()


def make_moe_config(**over):
    """构造一个 setup_moe_comm_method 能吃的 moe_config 占位。"""
    base = dict(
        ep_size=4, experts_per_token=2, num_experts=8, num_local_experts=2,
        dp_size=1, pcp_size=1, tp_group=types.SimpleNamespace(device_group=object()),
    )
    base.update(over)
    return types.SimpleNamespace(**base)
