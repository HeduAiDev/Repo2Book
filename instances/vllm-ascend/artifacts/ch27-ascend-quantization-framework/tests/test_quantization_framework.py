"""ch27 —— 昇腾量化框架「注册表 + 适配器」范式的纯 Python 控制流测试。

测的是精简版复现真仓的**可观察行为**（注册/查表/分发/形状/转交），
NPU 量化算子由替身记录调用——不真算（host 无 CANN）。
"""
import pytest
import torch


# ---------------------------------------------------------------------------
# (1) scheme 注册表：register_scheme 落表 / get_scheme_class 查表 / 重复注册报错
# ---------------------------------------------------------------------------
def test_registry_register_and_lookup(quant_pkg):
    reg = quant_pkg.registry

    @reg.register_scheme("FOO_TYPE", "linear")
    class _FooScheme:
        pass

    assert reg.get_scheme_class("FOO_TYPE", "linear") is _FooScheme
    assert ("FOO_TYPE", "linear") in reg._SCHEME_REGISTRY


def test_registry_unknown_returns_none(quant_pkg):
    assert quant_pkg.registry.get_scheme_class("NOPE", "linear") is None


def test_registry_duplicate_raises(quant_pkg):
    reg = quant_pkg.registry

    @reg.register_scheme("DUP_TYPE", "linear")
    class _A:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @reg.register_scheme("DUP_TYPE", "linear")
        class _B:
            pass


def test_w8a8_scheme_is_registered(quant_pkg):
    # methods/__init__ import w8a8_dynamic 即触发其 @register_scheme 落表
    reg = quant_pkg.registry
    assert reg.get_scheme_class("W8A8_DYNAMIC", "linear") is quant_pkg.w8a8.AscendW8A8DynamicLinearMethod
    assert reg.get_scheme_class("W8A8_DYNAMIC", "moe") is quant_pkg.w8a8.AscendW8A8DynamicFusedMoEMethod


# ---------------------------------------------------------------------------
# (2) 逐层解析：get_linear_quant_type 融合层各 shard 一致性校验
# ---------------------------------------------------------------------------
def test_get_linear_quant_type_plain(quant_pkg):
    ms = quant_pkg.modelslim
    qd = {"model.layers.0.mlp.down_proj.weight": "W8A8_DYNAMIC"}
    assert ms.get_linear_quant_type(qd, "model.layers.0.mlp.down_proj", {}) == "W8A8_DYNAMIC"


def test_get_linear_quant_type_fused_consistent(quant_pkg):
    ms = quant_pkg.modelslim
    mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    qd = {
        "m.gate_proj.weight": "W8A8_DYNAMIC",
        "m.up_proj.weight": "W8A8_DYNAMIC",
    }
    assert ms.get_linear_quant_type(qd, "m.gate_up_proj", mapping) == "W8A8_DYNAMIC"


def test_get_linear_quant_type_fused_mismatch_raises(quant_pkg):
    ms = quant_pkg.modelslim
    mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    qd = {
        "m.gate_proj.weight": "W8A8_DYNAMIC",
        "m.up_proj.weight": "W4A8_DYNAMIC",
    }
    with pytest.raises(ValueError, match="same quant type"):
        ms.get_linear_quant_type(qd, "m.gate_up_proj", mapping)


# ---------------------------------------------------------------------------
# (3) create_scheme_for_layer：解析 -> 查注册表 -> 实例化 / 未支持则报错
# ---------------------------------------------------------------------------
def test_create_scheme_for_layer_returns_instance(quant_pkg):
    ms = quant_pkg.modelslim
    qd = {"m.down_proj.weight": "W8A8_DYNAMIC"}
    scheme = ms.create_scheme_for_layer(qd, "m.down_proj", "linear", {})
    assert isinstance(scheme, quant_pkg.w8a8.AscendW8A8DynamicLinearMethod)


def test_create_scheme_for_layer_unsupported_raises(quant_pkg):
    ms = quant_pkg.modelslim
    qd = {"m.down_proj.weight": "NO_SUCH_TYPE"}
    with pytest.raises(NotImplementedError, match="NO_SUCH_TYPE"):
        ms.create_scheme_for_layer(qd, "m.down_proj", "linear", {})


# ---------------------------------------------------------------------------
# (4) get_quant_method：按 layer 类型分发到对应 wrapper（FLOAT 层跳过）
# ---------------------------------------------------------------------------
def _make_config(quant_pkg, quant_description):
    return quant_pkg.modelslim.AscendModelSlimConfig(quant_config=quant_description)


def test_get_quant_method_linear_dispatch(quant_pkg):
    LinearBase = quant_pkg.linear_mod.LinearBase
    layer = type("MyLinear", (LinearBase,), {})()
    cfg = _make_config(quant_pkg, {"model.q_proj.weight": "W8A8_DYNAMIC"})
    method = cfg.get_quant_method(layer, "model.q_proj")
    assert isinstance(method, quant_pkg.method_adapters.AscendLinearMethod)
    assert isinstance(method.quant_method, quant_pkg.w8a8.AscendW8A8DynamicLinearMethod)


def test_get_quant_method_float_layer_skipped(quant_pkg):
    LinearBase = quant_pkg.linear_mod.LinearBase
    layer = type("MyLinear", (LinearBase,), {})()
    cfg = _make_config(quant_pkg, {"model.q_proj.weight": "FLOAT"})
    method = cfg.get_quant_method(layer, "model.q_proj")
    # FLOAT 层回退到未量化方法，而非 AscendLinearMethod
    import vllm_ascend.ops.linear as opl

    assert isinstance(method, opl.AscendUnquantizedLinearMethod)


def test_get_quant_method_moe_dispatch(quant_pkg):
    FusedMoE = quant_pkg.fused_moe_mod.FusedMoE
    layer = type("MyMoE", (FusedMoE,), {})()
    layer.moe_config = object()
    cfg = _make_config(quant_pkg, {"model.experts.weight": "W8A8_DYNAMIC"})
    method = cfg.get_quant_method(layer, "model.experts")
    assert isinstance(method, quant_pkg.method_adapters.AscendFusedMoEMethod)
    assert isinstance(method.quant_method, quant_pkg.w8a8.AscendW8A8DynamicFusedMoEMethod)


# ---------------------------------------------------------------------------
# (5) 「先删后替换」：compressed-tensors / fp8 从 QUANTIZATION_METHODS 删原版再同名注册
# ---------------------------------------------------------------------------
def test_compressed_tensors_remove_then_replace(quant_pkg):
    # import 时已执行 _remove_quantization_method() + @register_quantization_config
    assert "compressed-tensors" not in quant_pkg.QUANTIZATION_METHODS
    assert quant_pkg.registered["compressed-tensors"] is quant_pkg.compressed.AscendCompressedTensorsConfig


def test_fp8_remove_then_replace_and_alias(quant_pkg):
    assert "fp8" not in quant_pkg.QUANTIZATION_METHODS
    assert "deepseek_v4_fp8" not in quant_pkg.QUANTIZATION_METHODS
    # fp8 与别名 deepseek_v4_fp8 复用同一个 Config
    assert quant_pkg.registered["fp8"] is quant_pkg.fp8.AscendFp8Config
    assert quant_pkg.registered["deepseek_v4_fp8"] is quant_pkg.fp8.AscendFp8Config


# ---------------------------------------------------------------------------
# (6) scheme 决定量化权重/scale 的形状：int8 [out,in] + per-channel [out,1]
# ---------------------------------------------------------------------------
def test_w8a8_get_weight_int8_shape(quant_pkg):
    scheme = quant_pkg.w8a8.AscendW8A8DynamicLinearMethod()
    wd = scheme.get_weight(input_size=8, output_size=4, params_dtype=torch.float16)
    assert set(wd) == {"weight"}
    assert wd["weight"].shape == (4, 8)
    assert wd["weight"].dtype == torch.int8


def test_w8a8_get_perchannel_param_shape(quant_pkg):
    scheme = quant_pkg.w8a8.AscendW8A8DynamicLinearMethod()
    pc = scheme.get_perchannel_param(output_size=4, params_dtype=torch.float16)
    assert set(pc) == {"weight_scale", "weight_offset"}
    assert pc["weight_scale"].shape == (4, 1)
    assert pc["weight_offset"].shape == (4, 1)


# ---------------------------------------------------------------------------
# (7) 适配器只搬运：AscendLinearMethod.create_weights 把 scheme 返回的参数逐个注册
# ---------------------------------------------------------------------------
def test_linear_wrapper_registers_scheme_params(quant_pkg):
    scheme = quant_pkg.w8a8.AscendW8A8DynamicLinearMethod()
    wrapper = quant_pkg.method_adapters.AscendLinearMethod(scheme)
    layer = torch.nn.Module()
    wrapper.create_weights(
        layer,
        input_size_per_partition=8,
        output_partition_sizes=[4],
        input_size=8,
        output_size=4,
        params_dtype=torch.float16,
        weight_loader=None,
    )
    # scheme.get_weight + get_perchannel_param 返回的三个参数都被 register_parameter
    assert layer.weight.shape == (4, 8)
    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale.shape == (4, 1)
    assert layer.weight_offset.shape == (4, 1)
    # wrapper 给权重打的方向属性（适配器搬运时设的）
    assert layer.weight.input_dim == 1
    assert layer.weight.output_dim == 0


# ---------------------------------------------------------------------------
# (8) 全链终点 apply：npu_dynamic_quant -> npu_quant_matmul 的调用次序与入参
# ---------------------------------------------------------------------------
def test_w8a8_apply_calls_npu_ops_in_order(quant_pkg):
    quant_pkg.npu_rec.calls.clear()
    scheme = quant_pkg.w8a8.AscendW8A8DynamicLinearMethod()
    layer = torch.nn.Module()
    layer.weight = torch.zeros(8, 4, dtype=torch.int8)
    layer.weight_scale = torch.ones(4)
    x = torch.ones(2, 8, dtype=torch.float16)

    out = scheme.apply(layer, x)

    names = [c[0] for c in quant_pkg.npu_rec.calls]
    assert names == ["npu_dynamic_quant", "npu_quant_matmul"]
    # 动态量化收到激活 x
    assert quant_pkg.npu_rec.calls[0][1] is x
    # 量化 matmul 收到 layer.weight 与 layer.weight_scale
    mm = quant_pkg.npu_rec.calls[1]
    assert mm[2] is layer.weight
    assert mm[3] is layer.weight_scale
    assert mm[6] == x.dtype  # output_dtype 回到激活 dtype
    assert out.shape == (2, 4)
