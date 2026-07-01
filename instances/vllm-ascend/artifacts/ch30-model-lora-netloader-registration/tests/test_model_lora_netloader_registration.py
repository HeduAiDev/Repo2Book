"""ch30 —— 模型 / LoRA / netloader 的昇腾接入（注册 + 全局类替换 trick + loader 扩展点）。

测的是精简版**复现真仓可观察行为**，不是自洽：
  变体(1) 模型注册：register_model() 的两条 arch→「<module>:<class>」映射；
                     _ModelRegistry.register_model 的字符串懒加载 / 非法格式 raise / 重名覆盖。
  变体(2) LoRA：refresh_all_lora_classes 的确定顺序追加；can_replace_layer 的严格 type 匹配；
                from_layer 在全局元组被追加后命中昇腾 LoRA 类；PunicaWrapperNPU.__init__ 的
                device/rank 二选一绑 op；lora_ops 薄壳的参数 reorder 转调。
  变体(3) netloader：@register_model_loader 注册 + get_model_loader 分发 + 非子类 raise；
                load_model 的 source 无效/弹性失败 → revert_to_default；revert_to_default 量化分支。
重 NPU 路径（真 kernel、网络传输、整模型前向）不真跑。
"""

import types

import pytest
import torch


# ===========================================================================
# 变体(1) 模型注册
# ===========================================================================
def test_register_model_maps_arch_to_lazy_module_class_string(env):
    # 注入 SUBTRACTED 的懒加载表示哨兵，观察 register_model 把架构名映射成 (module, class)。
    env.vllm_registry._LazyRegisteredModel = lambda module, cls: ("lazy", module, cls)

    env.models_register.register_model()

    models = env.ModelRegistry.models
    assert models["DeepseekV4ForCausalLM"] == ("lazy", "vllm_ascend.models.deepseek_v4", "AscendDeepseekV4ForCausalLM")
    assert models["DeepSeekV4MTPModel"] == ("lazy", "vllm_ascend.models.deepseek_v4_mtp", "DeepSeekV4MTP")


def test_registry_string_must_be_module_colon_class(env):
    env.vllm_registry._LazyRegisteredModel = lambda module, cls: ("lazy", module, cls)
    reg = env.ModelRegistry
    # 合法 <module>:<class>
    reg.register_model("Foo", "pkg.mod:Cls")
    assert reg.models["Foo"] == ("lazy", "pkg.mod", "Cls")
    # 非法：缺冒号 / 多冒号 → ValueError
    with pytest.raises(ValueError):
        reg.register_model("Bad", "pkg.mod.Cls")


def test_registry_same_arch_overwrites(env):
    env.vllm_registry._LazyRegisteredModel = lambda module, cls: ("lazy", module, cls)
    reg = env.ModelRegistry
    reg.register_model("Dup", "a.b:C1")
    reg.register_model("Dup", "a.b:C2")  # 已注册同名 → 覆盖（昇腾顶替同名模型的机制）
    assert reg.models["Dup"] == ("lazy", "a.b", "C2")


# ===========================================================================
# 变体(2) LoRA —— 全局类替换 trick
# ===========================================================================
def test_refresh_appends_four_ascend_classes_in_deterministic_order(env):
    vllm_utils = env.vllm_lora_utils
    lu = env.lora_utils
    before = vllm_utils._all_lora_classes
    n0 = len(before)

    lu.refresh_all_lora_classes()

    after = vllm_utils._all_lora_classes
    assert len(after) == n0 + 4
    # 原有的全部保留在前；4 个昇腾类按确定顺序追加在尾部。
    assert after[:n0] == before
    assert after[n0:] == (
        lu.AscendQKVParallelLinearWithLoRA,
        lu.AscendMergedQKVParallelLinearWithLoRA,
        lu.AscendMergedQKVParallelLinearWithShardedLoRA,
        lu.AscendQKVParallelLinearWithShardedLoRA,
    )


def test_can_replace_layer_strict_type_and_packed_len(env):
    lu = env.lora_utils
    AscendQKV = env.ops_linear.AscendQKVParallelLinear
    ascend_layer = AscendQKV()
    other_layer = object()

    # 单切片（packed 长度 1）只被非 merged 的两个类认领；type 必须严格是 AscendQKVParallelLinear。
    assert lu.AscendQKVParallelLinearWithLoRA.can_replace_layer(
        source_layer=ascend_layer, lora_config=None, packed_modules_list=[0], model_config=None
    )
    assert not lu.AscendQKVParallelLinearWithLoRA.can_replace_layer(
        source_layer=ascend_layer, lora_config=None, packed_modules_list=[0, 1, 2], model_config=None
    )
    # 非昇腾层不命中（哪怕 packed 长度对）——昇腾类只认昇腾 QKV，零侵入。
    assert not lu.AscendQKVParallelLinearWithLoRA.can_replace_layer(
        source_layer=other_layer, lora_config=None, packed_modules_list=[0], model_config=None
    )
    # merged 三切片（packed 长度 3）走 merged 类。
    assert lu.AscendMergedQKVParallelLinearWithLoRA.can_replace_layer(
        source_layer=ascend_layer, lora_config=None, packed_modules_list=[0, 1, 2], model_config=None
    )


def test_from_layer_picks_ascend_class_after_global_replacement(env):
    """全局类替换的可观察效果：refresh 后 vLLM from_layer 能选出昇腾 LoRA 类。"""
    vllm_utils = env.vllm_lora_utils
    lu = env.lora_utils
    AscendQKV = env.ops_linear.AscendQKVParallelLinear

    layer = AscendQKV()
    # 追加前：候选元组里没有任何类能认领昇腾 QKV 层 → 原样返回。
    assert vllm_utils.from_layer(layer, max_loras=1, lora_config=None, packed_modules_list=[0]) is layer

    lu.refresh_all_lora_classes()
    # 追加后：from_layer 命中 AscendQKVParallelLinearWithLoRA。
    replaced = vllm_utils.from_layer(layer, max_loras=1, lora_config=None, packed_modules_list=[0])
    assert isinstance(replaced, lu.AscendQKVParallelLinearWithLoRA)


# ===========================================================================
# 变体(2) LoRA —— PunicaWrapperNPU 的 device/rank 二选一绑 op
# ===========================================================================
def _make_wrapper(env, lora_config=None):
    return env.punica_npu.PunicaWrapperNPU(8, 2, "cpu", lora_config=lora_config)


def test_default_npu_binds_ascend_lora_ops(env):
    w = _make_wrapper(env)  # 910 + 无 lora_config → 走昇腾自定义 kernel 薄壳
    assert w.bgmv_shrink is env.lora_ops.bgmv_shrink
    assert w.sgmv_expand is env.lora_ops.sgmv_expand


def test_310p_falls_back_to_vllm_torch_ops(env):
    env.set_device(env.AscendDeviceType._310P)
    w = _make_wrapper(env)
    assert w.bgmv_shrink is env.torch_ops_mod.bgmv_shrink
    assert w.sgmv_shrink is env.torch_ops_mod.sgmv_shrink


def test_large_rank_falls_back_to_vllm_torch_ops(env):
    lora_config = types.SimpleNamespace(max_lora_rank=128)
    w = _make_wrapper(env, lora_config=lora_config)  # rank>=128 → 退回 torch_ops
    assert w.bgmv_expand is env.torch_ops_mod.bgmv_expand


def test_small_rank_keeps_ascend_ops(env):
    lora_config = types.SimpleNamespace(max_lora_rank=64)
    w = _make_wrapper(env, lora_config=lora_config)  # rank<128 → 昇腾 kernel
    assert w.bgmv_expand is env.lora_ops.bgmv_expand


# ===========================================================================
# 变体(2) LoRA —— lora_ops 薄壳：参数 reorder 后转调 torch.ops._C_ascend.*
# ===========================================================================
def test_bgmv_shrink_reorders_indices_before_output(env):
    out = torch.zeros(3, 5)
    env.lora_ops.bgmv_shrink("inp", "A", out, "idx", 2.0)
    name, args = env.kernel_calls[-1]
    assert name == "bgmv_shrink"
    # 源签名 (inputs, lora_a, output, indices, scaling) → kernel (inputs, lora_a, indices, output, scaling)
    assert args == ("inp", "A", "idx", out, 2.0)


def test_bgmv_expand_drops_add_inputs_and_passes_full_slice(env):
    out = torch.zeros(3, 7)
    env.lora_ops.bgmv_expand("inp", "B", out, "idx", add_inputs=True)
    name, args = env.kernel_calls[-1]
    assert name == "bgmv_expand"
    # add_inputs 被丢弃；固定传 offset=0 / size=output.size(1)=7。
    assert args == ("inp", "B", "idx", out, 0, 7)


# ===========================================================================
# 变体(3) netloader —— 注册到 loader 扩展点
# ===========================================================================
def test_netloader_registered_into_load_format_table(env):
    table = env.vllm_model_loader._LOAD_FORMAT_TO_MODEL_LOADER
    assert table["netloader"] is env.netloader.ModelNetLoaderElastic
    # 是 BaseModelLoader 的子类（薄壳继承范式）。
    assert issubclass(env.netloader.ModelNetLoaderElastic, env.base_loader.BaseModelLoader)


def test_get_model_loader_dispatches_by_load_format(env):
    lc = sys_loadconfig("netloader")
    loader = env.vllm_model_loader.get_model_loader(lc)
    assert isinstance(loader, env.netloader.ModelNetLoaderElastic)


def test_register_model_loader_rejects_non_base_subclass(env):
    with pytest.raises(ValueError):
        env.vllm_model_loader.register_model_loader("bogus")(object)


def sys_loadconfig(load_format, quantization=None):
    import vllm.config.load as cfgload

    lc = cfgload.LoadConfig()
    lc.load_format = load_format
    lc.model_loader_extra_config = {}
    return lc


# ===========================================================================
# 变体(3) netloader —— load_model 弹性加载 + 失败回退
# ===========================================================================
def _vllm_config(quantization=None):
    device_config = types.SimpleNamespace(device="cpu", device_type="cpu")
    parallel_config = types.SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1)
    return types.SimpleNamespace(device_config=device_config, parallel_config=parallel_config)


def _model_config(quantization=None):
    return types.SimpleNamespace(model="/models/dsv4", dtype="bf16", quantization=quantization, runner_type="generate")


def _new_loader(env):
    return env.netloader.ModelNetLoaderElastic(sys_loadconfig("netloader"))


def test_load_model_no_source_reverts_to_default(env):
    loader = _new_loader(env)  # source 默认 None
    model = loader.load_model(_vllm_config(), _model_config())
    assert model.tag == "default_loaded"  # 走了 DefaultModelLoader
    assert model.evaled


def test_load_model_elastic_success_returns_skeleton(env):
    loader = _new_loader(env)
    loader.source = [{"device_id": 0}]  # 本 rank(0) 在 source 列表 → 走弹性加载
    model = loader.load_model(_vllm_config(), _model_config())
    assert model.tag == "empty_skeleton"  # initialize_model 的骨架被 elastic_load 原样填充
    assert model.evaled


def test_load_model_elastic_failure_falls_back(env):
    loader = _new_loader(env)
    loader.source = [{"device_id": 0}]
    # netloader.py 顶部 `from .load import elastic_load` 在导入时绑定该名，故打 netloader 模块名。
    env.netloader.elastic_load = lambda model=None, **k: None  # 弹性拉取失败
    model = loader.load_model(_vllm_config(), _model_config())
    assert model.tag == "default_loaded"  # 优雅回退 DefaultModelLoader


# ===========================================================================
# 变体(3) netloader —— revert_to_default 的量化分支
# ===========================================================================
def test_revert_to_default_no_quant_uses_default_load_model(env):
    loader = _new_loader(env)
    model, need_process = loader.revert_to_default(_model_config(quantization=None), _vllm_config(), _vllm_config().device_config)
    assert model.tag == "default_loaded"
    assert need_process is False


def test_revert_to_default_with_quant_loads_weights_and_flags_process(env):
    loader = _new_loader(env)
    model, need_process = loader.revert_to_default(
        _model_config(quantization="w8a8"), _vllm_config(), _vllm_config().device_config
    )
    # 量化路径：建空骨架 → load_weights → eval；并标记 need_process_weights_after_loading。
    assert model.tag == "empty_skeleton"
    assert model.weights_loaded
    assert need_process is True
