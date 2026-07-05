"""ch23 —— CustomOp 的 OOT 顶替：验证精简版复现 vllm-ascend 的可观察控制流。

测的是「精简版与真仓控制流一致」，不是精简版自洽：
  注册表总开关建表 + register_oot 写入 op_registry_oot / 幂等闸只生效一次 /
  __new__ 换身 / dispatch_forward 换头（oot vs native）/ forward_oot 的融合 vs 回退二分 /
  register_oot 把 op.name 覆盖成类名键。
真实 torch_npu / _C_ascend 算子由记录替身承接（host 无 NPU）。
"""
import torch


# ---------- (1) 注册表总开关：建表 + 遍历 register_oot 写入 op_registry_oot ----------

def test_register_builds_table_and_populates_oot_registry(env):
    mods, knobs = env
    co = mods.custom_op
    assert co.op_registry_oot == {}  # 注册前为空

    mods.utils.register_ascend_customop()

    # 建出的 REGISTERED_ASCEND_OPS 表：键是 vLLM 类名字符串
    table = mods.utils.REGISTERED_ASCEND_OPS
    assert set(table.keys()) == {"QuickGELU", "SiluAndMul", "RMSNorm", "FusedMoE"}
    # 表里每一项都被逐个写进基座全局 op_registry_oot（一处调用，全模型算子换头）
    for name, op_cls in table.items():
        assert co.op_registry_oot[name] is op_cls


def test_register_oot_overrides_name_with_class_key(env):
    """register_oot 把 op.name 设成注册键（类名 'RMSNorm'），而非基座 register 的 lowercase 'rms_norm'。"""
    mods, _ = env
    mods.utils.register_ascend_customop()
    AscendRMSNorm = mods.ascend_layernorm.AscendRMSNorm
    assert AscendRMSNorm.name == "RMSNorm"          # OOT 注册键 = 类名
    assert mods.layernorm.RMSNorm.name == "rms_norm"  # 基座 in-tree 注册键不变


# ---------- (2) 幂等闸：只生效一次 ----------

def test_idempotent_gate_registers_once(env):
    mods, _ = env
    co = mods.custom_op
    mods.utils.register_ascend_customop()
    assert mods.utils._ASCEND_CUSTOMOP_IS_REIGISTERED is True
    n_first = len(co.op_registry_oot)

    # 再次调用：幂等闸直接 return，不重复注册（否则 register_oot 的 assert 会炸）
    mods.utils.register_ascend_customop()
    assert len(co.op_registry_oot) == n_first


# ---------- (3) __new__ 换身：模型写 RMSNorm() 实际造出 Ascend 子类 ----------

def test_new_swaps_instance_to_ascend_subclass(env):
    mods, _ = env
    mods.utils.register_ascend_customop()
    RMSNorm = mods.layernorm.RMSNorm
    AscendRMSNorm = mods.ascend_layernorm.AscendRMSNorm

    # 模型代码照常写 RMSNorm(...)，不知昇腾存在
    obj = RMSNorm(8)
    # __new__ 命中 op_registry_oot['RMSNorm'] → 真正被实例化的是昇腾子类（换身）
    assert type(obj) is AscendRMSNorm
    assert isinstance(obj, RMSNorm)  # 身（继承/接口）不变


def test_new_keeps_class_when_not_registered(env):
    """未注册（注册前）的类不被换身：__new__ 走 op_cls_to_instantiate = cls。"""
    mods, _ = env
    RMSNorm = mods.layernorm.RMSNorm
    obj = RMSNorm(8)
    assert type(obj) is RMSNorm  # 注册前：仍是基座类自己


# ---------- (4) dispatch_forward 换头：oot vs native ----------

def test_dispatch_binds_forward_oot_on_oot_platform(env):
    mods, knobs = env
    mods.utils.register_ascend_customop()
    knobs.is_out_of_tree = True
    knobs.custom_ops[:] = ["all"]  # default_on → enabled

    obj = mods.layernorm.RMSNorm(8)  # 换身为 AscendRMSNorm
    # enabled 且 is_out_of_tree() → _forward_method 绑到 forward_oot（换头）
    assert obj._forward_method == obj.forward_oot


def test_dispatch_falls_back_to_native_when_not_enabled(env):
    mods, knobs = env
    mods.utils.register_ascend_customop()
    knobs.custom_ops[:] = ["none"]  # default_on → False，未 enabled

    obj = mods.layernorm.RMSNorm(8)
    # 未 enabled → 编译 forward_native（精简版 maybe_compile 直接返回 fn）
    assert obj._forward_method == obj.forward_native


def test_silu_only_overrides_forward_oot(env):
    """标本一：AscendSiluAndMul 只覆 forward_oot —— forward_native/forward_cuda 仍是基座的（身不变）。"""
    mods, knobs = env
    mods.utils.register_ascend_customop()
    SiluAndMul = mods.activation.SiluAndMul
    AscendSiluAndMul = mods.ascend_activation.AscendSiluAndMul

    obj = SiluAndMul()
    assert type(obj) is AscendSiluAndMul
    # forward_native 来自基座（未被昇腾覆写）
    assert "AscendSiluAndMul" not in obj.forward_native.__qualname__
    # forward_oot 来自昇腾子类（换头点）
    assert "AscendSiluAndMul" in obj.forward_oot.__qualname__

    # 真正前向：走 forward_oot → 一行 npu_swiglu 顶替
    x = torch.randn(2, 8)
    obj._forward_method(x)
    assert "torch_npu.npu_swiglu" in mods.rec.names()


# ---------- (5) forward_oot 内 enable_custom_op() 真二分：融合算子 vs 原子算子回退 ----------

def test_forward_oot_fallback_to_atomic_ops_when_lib_absent(env):
    """host 无编译产物 → enable_custom_op()=False → 走 torch_npu.npu_add_rms_norm 原子算子回退。"""
    mods, _ = env
    mods.utils.register_ascend_customop()
    assert mods.utils.enable_custom_op() is False  # import vllm_ascend_C 失败

    obj = mods.layernorm.RMSNorm(8)  # AscendRMSNorm
    x = torch.randn(4, 8)
    residual = torch.randn(4, 8)
    obj.forward_oot(x, residual)

    names = mods.rec.names()
    assert "torch_npu.npu_add_rms_norm" in names          # 原子算子回退分支
    assert "_C_ascend.npu_add_rms_norm_bias" not in names  # 未走融合分支


def test_forward_oot_uses_fused_kernel_when_lib_present(env):
    """模拟编译产物在位（_CUSTOM_OP_ENABLED=True）→ 走 _C_ascend.npu_add_rms_norm_bias 融合 kernel。"""
    mods, _ = env
    mods.utils.register_ascend_customop()
    mods.utils._CUSTOM_OP_ENABLED = True  # 模拟「AscendC 融合算子库在位」
    assert mods.utils.enable_custom_op() is True

    obj = mods.layernorm.RMSNorm(8)  # AscendRMSNorm
    x = torch.randn(4, 8)
    residual = torch.randn(4, 8)
    obj.forward_oot(x, residual)

    names = mods.rec.names()
    assert "_C_ascend.npu_add_rms_norm_bias" in names      # 融合 kernel 分支
    assert "torch_npu.npu_add_rms_norm" not in names        # 未走回退


def test_enable_custom_op_cached(env):
    """惰性 + 缓存：结果只判一次，写进 _CUSTOM_OP_ENABLED。"""
    mods, _ = env
    assert mods.utils._CUSTOM_OP_ENABLED is None
    first = mods.utils.enable_custom_op()
    assert mods.utils._CUSTOM_OP_ENABLED is first  # 已缓存
    assert mods.utils.enable_custom_op() is first
