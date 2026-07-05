"""ch15 测试脚手架：在 sys.modules 桩掉 vllm / vllm_ascend 的重运行时依赖，
再把（已减法的）implementation/ 模块按**规范模块名**注册进去，让它们彼此 import
解析到精简版本身。昇腾 NPU/CANN 不在 host 真跑，但本章三处核心——
select_moe_comm_method 决策 / set_ascend_forward_context 注入 / _sync_metadata_across_dp
打包——都是纯 Python，故可在 host 验证其与真实仓一致的可观察控制流。
"""
import enum
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


class _CUDAGraphMode(enum.IntEnum):
    NONE = 0
    PIECEWISE = 1
    FULL = 2


def _load(filename, modname):
    """按规范模块名加载精简版文件并登记进 sys.modules（含父包链接）。"""
    spec = importlib.util.spec_from_file_location(modname, IMPL_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    if "." in modname:
        parent = modname.rsplit(".", 1)[0]
        if parent in sys.modules:
            setattr(sys.modules[parent], modname.rsplit(".", 1)[1], mod)
    spec.loader.exec_module(mod)
    return mod


class _Stubs:
    """记录注入的 sys.modules 名字，测后清理。"""

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


def _make_ascend_config(**over):
    """昇腾配置桩：默认值可被单测覆盖。"""
    base = dict(
        enable_flashcomm1=False,
        enable_flashcomm2_parallel_size=0,
        enable_fused_mc2=0,
        enable_mc2_hierarchy_comm=False,
        recompute_scheduler_enable=False,
        dp_allreduce_on_npu=False,
        enable_reduce_sample=False,
    )
    base.update(over)
    ns = types.SimpleNamespace(**base)
    ns.finegrained_tp_config = types.SimpleNamespace(lmhead_tensor_parallel_size=0)
    return ns


@pytest.fixture
def env():
    """搭桩 + 加载五个精简版模块，返回 (modules, knobs)。"""
    stubs = _Stubs()

    # ---- 可被测试调节的旋钮 ---- #
    knobs = types.SimpleNamespace(
        tp_world_size=1,
        dp_world_size=1,
        ep_world_size=1,
        ascend_config=_make_ascend_config(),
        soc=None,        # AscendDeviceType 成员，加载后填
        mc2_capacity=512,
        additional_kwargs={"injected_by_platform": True},
    )

    # ---- vllm.config ---- #
    cfg = stubs.mod("vllm.config")
    cfg.CUDAGraphMode = _CUDAGraphMode
    cfg.VllmConfig = type("VllmConfig", (), {})
    cfg.ParallelConfig = type("ParallelConfig", (), {})
    cfg.get_current_vllm_config = lambda: None

    # ---- vllm.platforms.current_platform（昇腾注入 additional_kwargs 的入口）---- #
    plat = stubs.mod("vllm.platforms")
    plat.current_platform = types.SimpleNamespace(
        set_additional_forward_context=lambda **kw: dict(knobs.additional_kwargs)
    )

    # ---- vllm.distributed 组 ---- #
    dist_mod = stubs.mod("vllm.distributed")
    dist_mod.get_dp_group = lambda: types.SimpleNamespace(
        world_size=knobs.dp_world_size,
        device_group="npu-dp-group",
        cpu_group="cpu-dp-group",
    )
    dist_mod.get_ep_group = lambda: types.SimpleNamespace(world_size=knobs.ep_world_size)
    dist_mod.get_tensor_model_parallel_world_size = lambda: knobs.tp_world_size
    ps = stubs.mod("vllm.distributed.parallel_state")
    ps.get_dp_group = dist_mod.get_dp_group
    ps.get_pp_group = lambda: types.SimpleNamespace(is_last_rank=True)
    kvt = stubs.mod("vllm.distributed.kv_transfer")
    kvt.has_kv_transfer_group = lambda: False

    # ---- 其它 vllm.* ---- #
    seqm = stubs.mod("vllm.sequence")
    seqm.IntermediateTensors = type("IntermediateTensors", (), {})
    v1u = stubs.mod("vllm.v1.utils")
    from contextlib import nullcontext as _nc

    v1u.record_function_or_nullcontext = lambda *_a, **_k: _nc()
    gmr = stubs.mod("vllm.v1.worker.gpu_model_runner")
    gmr.GPUModelRunner = type("GPUModelRunner", (), {})

    # ---- vllm_ascend.ascend_config ---- #
    acfg = stubs.mod("vllm_ascend.ascend_config")
    acfg.get_ascend_config = lambda: knobs.ascend_config

    # ---- vllm_ascend.ops.rotary_embedding ---- #
    rope = stubs.mod("vllm_ascend.ops.rotary_embedding")
    rope.update_cos_sin = lambda *_a, **_k: None

    # ---- 按依赖顺序加载精简版（覆盖任何同名桩）---- #
    stubs.mod("vllm_ascend")
    stubs.mod("vllm_ascend.ops")
    stubs.mod("vllm_ascend.ops.fused_moe")
    stubs.mod("vllm_ascend.worker")

    fwd = _load("forward_context.py", "vllm.forward_context")
    utils = _load("utils.py", "vllm_ascend.utils")
    afc = _load("ascend_forward_context.py", "vllm_ascend.ascend_forward_context")
    mcm = _load("moe_comm_method.py", "vllm_ascend.ops.fused_moe.moe_comm_method")
    mr = _load("model_runner_v1.py", "vllm_ascend.worker.model_runner_v1")

    # default soc = A2
    knobs.soc = utils.AscendDeviceType.A2
    utils._ascend_device_type = knobs.soc

    mods = types.SimpleNamespace(fwd=fwd, utils=utils, afc=afc, mcm=mcm, mr=mr, cfg=cfg)
    try:
        yield mods, knobs, stubs
    finally:
        stubs.cleanup()
        # 清掉精简版自身在 sys.modules 的登记，避免污染其它测试
        for n in (
            "vllm.forward_context",
            "vllm_ascend.utils",
            "vllm_ascend.ascend_forward_context",
            "vllm_ascend.ops.fused_moe.moe_comm_method",
            "vllm_ascend.worker.model_runner_v1",
        ):
            sys.modules.pop(n, None)
