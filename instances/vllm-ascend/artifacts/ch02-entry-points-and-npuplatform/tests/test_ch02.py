"""TDD tests — 复现 vllm-ascend OOT 平台发现/顶替的真实可观察行为。

测的是‘精简版复现目标代码仓的可观察控制流’，不是自洽：
  - register() 返回 qualname **字符串**（而非类），故意不 import；
  - load_plugins_by_group 只 plugin.load() 出回调函数、**不调用**它；
  - resolve_current_platform_cls_qualname：OOT 优先于 builtin、≥2 个 OOT 报错；
  - current_platform 懒加载成单例（只解析一次）；
  - NPUPlatform 身份替换类属性 + 一批返回 qualname 的工厂钩子；
  - get_attn_backend_cls 按 (use_mla,use_sparse[,use_compress]) 查表、310P 走独立表；
  - check_and_update_config 把 worker_cls 'auto'→NPUWorker qualname（按分代分流）；
  - AscendDeviceType / is_310p 设备分代横切。

昇腾真实运行依赖 torch_npu/CANN（host 无），故测纯 Python 控制流；真正的 import
那一刻（resolve_obj_by_qualname → import vllm_ascend.platform）在真机才发生。
"""
import importlib
import importlib.metadata
import sys
import types
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

IMPL = Path(__file__).resolve().parents[1] / "implementation"
sys.path.insert(0, str(IMPL))

import vllm_ascend_init
import vllm_ascend_platform
import vllm_ascend_utils
import vllm_platforms
import vllm_plugins
from vllm_ascend_platform import NPUPlatform
from vllm_ascend_utils import AscendDeviceType
from vllm_import_utils import resolve_obj_by_qualname
from vllm_interface import AttentionBackendEnum, Platform, PlatformEnum


# --------------------------------------------------------------------------- #
# entry-point 发现的桩：把 importlib.metadata.entry_points 换成返回合成 EntryPoint，
# 它的 .load() 会 import 我们扁平的 vllm_ascend_init 模块、取出 register 函数。
# --------------------------------------------------------------------------- #
def _patch_entry_points(monkeypatch, eps):
    def fake_entry_points(group=None):
        return [e for e in eps if e.group == group]

    monkeypatch.setattr(importlib.metadata, "entry_points", fake_entry_points)


@pytest.fixture(autouse=True)
def _reset_state():
    # 每个用例前重置懒加载单例与设备分代缓存，避免互相污染。
    vllm_platforms._current_platform = None
    vllm_ascend_utils._ascend_device_type = None
    vllm_ascend_init._GLOBAL_PATCH_APPLIED = False
    yield
    vllm_platforms._current_platform = None
    vllm_ascend_utils._ascend_device_type = None


# --------------------------------------------------------------------------- #
# register / entry points
# --------------------------------------------------------------------------- #
def test_register_returns_qualname_string_not_class():
    out = vllm_ascend_init.register()
    assert out == "vllm_ascend.platform.NPUPlatform"
    assert isinstance(out, str)  # 是字符串，不是类对象 → 故意不 import


def test_load_plugins_by_group_loads_func_but_does_not_call_it(monkeypatch):
    eps = [EntryPoint(name="ascend", value="vllm_ascend_init:register",
                      group="vllm.platform_plugins")]
    _patch_entry_points(monkeypatch, eps)

    plugins = vllm_plugins.load_plugins_by_group("vllm.platform_plugins")
    assert set(plugins) == {"ascend"}
    # 关键：load() 拿到的是 register 函数本身（callable），尚未被调用
    assert plugins["ascend"] is vllm_ascend_init.register
    assert callable(plugins["ascend"])


def test_load_plugins_empty_group_returns_empty(monkeypatch):
    _patch_entry_points(monkeypatch, [])
    assert vllm_plugins.load_plugins_by_group("vllm.platform_plugins") == {}


# --------------------------------------------------------------------------- #
# resolve_current_platform_cls_qualname — OOT 优先 / 唯一性
# --------------------------------------------------------------------------- #
def test_oot_priority_over_builtin(monkeypatch):
    # builtin 全返 None（host 无硬件），ascend 无条件返回字符串 → OOT 命中
    eps = [EntryPoint(name="ascend", value="vllm_ascend_init:register",
                      group="vllm.platform_plugins")]
    _patch_entry_points(monkeypatch, eps)

    qn = vllm_platforms.resolve_current_platform_cls_qualname()
    assert qn == "vllm_ascend.platform.NPUPlatform"


def test_oot_beats_activated_builtin(monkeypatch):
    # 即便某个 builtin 也‘激活’（探测到硬件），OOT 分支在 elif 链最前 → 仍取 OOT
    monkeypatch.setattr(vllm_platforms, "cuda_platform_plugin",
                        lambda: "vllm.platforms.cuda.CudaPlatform")
    eps = [EntryPoint(name="ascend", value="vllm_ascend_init:register",
                      group="vllm.platform_plugins")]
    _patch_entry_points(monkeypatch, eps)

    assert vllm_platforms.resolve_current_platform_cls_qualname() == \
        "vllm_ascend.platform.NPUPlatform"


def test_two_oot_plugins_raise(monkeypatch):
    eps = [
        EntryPoint(name="ascend", value="vllm_ascend_init:register",
                   group="vllm.platform_plugins"),
        EntryPoint(name="ascend2", value="vllm_ascend_init:register",
                   group="vllm.platform_plugins"),
    ]
    _patch_entry_points(monkeypatch, eps)
    with pytest.raises(RuntimeError, match="Only one platform plugin"):
        vllm_platforms.resolve_current_platform_cls_qualname()


def test_no_platform_falls_back_to_unspecified(monkeypatch):
    _patch_entry_points(monkeypatch, [])  # 无 OOT、builtin 全 None
    assert vllm_platforms.resolve_current_platform_cls_qualname() == \
        "vllm.platforms.interface.UnspecifiedPlatform"


# --------------------------------------------------------------------------- #
# resolve_obj_by_qualname / current_platform 懒加载单例
# --------------------------------------------------------------------------- #
def test_resolve_obj_by_qualname_imports_and_returns_class():
    cls = resolve_obj_by_qualname("vllm_ascend_platform.NPUPlatform")
    assert cls is NPUPlatform


def test_current_platform_lazy_singleton_resolved_once(monkeypatch):
    calls = {"n": 0}

    def counting_resolve():
        calls["n"] += 1
        # 用扁平模块的 qualname，使 resolve_obj_by_qualname 在 host 上可解析
        return "vllm_ascend_platform.NPUPlatform"

    monkeypatch.setattr(vllm_platforms, "resolve_current_platform_cls_qualname",
                        counting_resolve)
    vllm_platforms._current_platform = None

    first = vllm_platforms.current_platform   # 触发 __getattr__ → 解析+实例化
    second = vllm_platforms.current_platform  # 命中缓存
    assert first is second                    # 单例
    assert isinstance(first, NPUPlatform)
    assert calls["n"] == 1                     # 只解析一次


# --------------------------------------------------------------------------- #
# NPUPlatform 身份替换类属性
# --------------------------------------------------------------------------- #
def test_npuplatform_identity_attrs():
    assert NPUPlatform._enum is PlatformEnum.OOT
    assert NPUPlatform.device_name == "npu"
    assert NPUPlatform.device_type == "npu"
    assert NPUPlatform.dispatch_key == "PrivateUse1"  # PyTorch 外部后端派发键
    assert NPUPlatform().is_out_of_tree() is True


# --------------------------------------------------------------------------- #
# 工厂钩子：返回 vllm_ascend.* qualname，且区别于 vLLM 基类默认
# --------------------------------------------------------------------------- #
def test_factory_hooks_return_ascend_qualnames():
    assert NPUPlatform.get_device_communicator_cls() == \
        "vllm_ascend.distributed.device_communicators.npu_communicator.NPUCommunicator"
    assert NPUPlatform.get_static_graph_wrapper_cls() == \
        "vllm_ascend.compilation.acl_graph.ACLGraphWrapper"
    assert NPUPlatform.get_punica_wrapper() == \
        "vllm_ascend.lora.punica_npu.PunicaWrapperNPU"
    assert NPUPlatform.get_compile_backend() == \
        "vllm_ascend.compilation.compiler_interface.AscendCompiler"
    assert NPUPlatform.get_pass_manager_cls() == \
        "vllm_ascend.compilation.graph_fusion_pass_manager.GraphFusionPassManager"


def test_hooks_override_base_defaults():
    # 对照 vLLM 基类默认 qualname：钩子确实顶替了 builtin 实现
    assert Platform.get_device_communicator_cls() == \
        "vllm.distributed.device_communicators.base_device_communicator.DeviceCommunicatorBase"
    assert NPUPlatform.get_device_communicator_cls() != \
        Platform.get_device_communicator_cls()


# --------------------------------------------------------------------------- #
# get_attn_backend_cls — 按 key 查表分发；310P 走独立表
# --------------------------------------------------------------------------- #
def _attn_cfg(use_mla, use_sparse, use_compress=False):
    return types.SimpleNamespace(use_mla=use_mla, use_sparse=use_sparse,
                                 use_compress=use_compress)


@pytest.mark.parametrize("cfg,expected", [
    (_attn_cfg(False, False, False), "vllm_ascend.attention.attention_v1.AscendAttentionBackend"),
    (_attn_cfg(True, False, False), "vllm_ascend.attention.mla_v1.AscendMLABackend"),
    (_attn_cfg(True, True, False), "vllm_ascend.attention.sfa_v1.AscendSFABackend"),
    (_attn_cfg(True, False, True), "vllm_ascend.attention.dsa_v1.AscendDSABackend"),
])
def test_get_attn_backend_cls_non_310p(cfg, expected):
    vllm_ascend_utils._ascend_device_type = AscendDeviceType.A2  # 构建期烙印=A2(非310P)
    out = NPUPlatform.get_attn_backend_cls(selected_backend=None,
                                           attn_selector_config=cfg)
    assert out == expected


def test_get_attn_backend_cls_310p_uses_dedicated_map():
    vllm_ascend_utils._ascend_device_type = AscendDeviceType._310P
    out = NPUPlatform.get_attn_backend_cls(selected_backend=None,
                                           attn_selector_config=_attn_cfg(False, False))
    assert out == "vllm_ascend._310p.attention.attention_v1.AscendAttentionBackend310"


def test_get_attn_backend_cls_fa3_branch_disabled_in_companion():
    # _validate_fa3_backend 在精简版降为 False → 即便 selected_backend 是 FLASH_ATTN，
    # 也不会走 FA3 分支，落回查表主干（这是本章删除计划批准的占位）
    vllm_ascend_utils._ascend_device_type = AscendDeviceType.A2
    out = NPUPlatform.get_attn_backend_cls(
        selected_backend=AttentionBackendEnum.FLASH_ATTN,
        attn_selector_config=_attn_cfg(False, False))
    assert out == "vllm_ascend.attention.attention_v1.AscendAttentionBackend"


# --------------------------------------------------------------------------- #
# worker_cls 改写：'auto' → NPUWorker qualname（按分代分流）
# --------------------------------------------------------------------------- #
def _vllm_config(worker_cls="auto", enable_sp=True, xlite=False):
    return types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(worker_cls=worker_cls),
        compilation_config=types.SimpleNamespace(
            pass_config=types.SimpleNamespace(enable_sp=enable_sp)),
        ascend_config=types.SimpleNamespace(
            xlite_graph_config=types.SimpleNamespace(enabled=xlite)),
    )


def test_worker_cls_default_rewritten_to_npuworker():
    vllm_ascend_utils._ascend_device_type = AscendDeviceType.A2
    cfg = _vllm_config()
    NPUPlatform.check_and_update_config(cfg)
    assert cfg.parallel_config.worker_cls == "vllm_ascend.worker.worker.NPUWorker"


def test_worker_cls_310p_rewritten_to_npuworker310():
    vllm_ascend_utils._ascend_device_type = AscendDeviceType._310P
    cfg = _vllm_config()
    NPUPlatform.check_and_update_config(cfg)
    assert cfg.parallel_config.worker_cls == "vllm_ascend._310p.worker_310p.NPUWorker310"


def test_worker_cls_not_auto_is_left_untouched():
    vllm_ascend_utils._ascend_device_type = AscendDeviceType.A2
    cfg = _vllm_config(worker_cls="my.custom.Worker")
    NPUPlatform.check_and_update_config(cfg)
    assert cfg.parallel_config.worker_cls == "my.custom.Worker"  # 非 'auto' 不改写


# --------------------------------------------------------------------------- #
# 设备分代横切：AscendDeviceType / is_310p / 运行期复核映射
# --------------------------------------------------------------------------- #
def test_is_310p_reflects_device_generation():
    vllm_ascend_utils._ascend_device_type = AscendDeviceType.A2
    assert vllm_ascend_utils.is_310p() is False
    vllm_ascend_utils._ascend_device_type = AscendDeviceType._310P
    assert vllm_ascend_utils.is_310p() is True


def test_get_ascend_device_type_is_cached():
    vllm_ascend_utils._ascend_device_type = AscendDeviceType.A3
    assert vllm_ascend_utils.get_ascend_device_type() is AscendDeviceType.A3


@pytest.mark.parametrize("soc,expected", [
    (222, AscendDeviceType.A2),
    (252, AscendDeviceType.A3),
    (202, AscendDeviceType._310P),
    (260, AscendDeviceType.A5),
])
def test_check_ascend_device_type_soc_mapping(soc, expected):
    # 运行期复核：soc_version 区间 → 分代；与构建期烙印一致则放行
    vllm_ascend_utils._ascend_device_type = expected
    vllm_ascend_utils.check_ascend_device_type(soc_version=soc)  # 不应 assert


def test_check_ascend_device_type_mismatch_raises():
    vllm_ascend_utils._ascend_device_type = AscendDeviceType.A2
    with pytest.raises(AssertionError):
        vllm_ascend_utils.check_ascend_device_type(soc_version=202)  # 310P 硬件 vs A2 包


def test_check_ascend_device_type_unknown_soc_raises():
    vllm_ascend_utils._ascend_device_type = AscendDeviceType.A2
    with pytest.raises(RuntimeError, match="Can not support soc_version"):
        vllm_ascend_utils.check_ascend_device_type(soc_version=999)


# --------------------------------------------------------------------------- #
# general_plugins：先 _ensure_global_patch（幂等）再注册
# --------------------------------------------------------------------------- #
def test_ensure_global_patch_is_idempotent():
    vllm_ascend_init._GLOBAL_PATCH_APPLIED = False
    vllm_ascend_init._ensure_global_patch()
    assert vllm_ascend_init._GLOBAL_PATCH_APPLIED is True
    vllm_ascend_init._ensure_global_patch()  # 第二次直接返回
    assert vllm_ascend_init._GLOBAL_PATCH_APPLIED is True


def test_register_connector_applies_global_patch_first():
    vllm_ascend_init._GLOBAL_PATCH_APPLIED = False
    vllm_ascend_init.register_connector()
    assert vllm_ascend_init._GLOBAL_PATCH_APPLIED is True
