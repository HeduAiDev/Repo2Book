"""ch27 测试脚手架：host 无 NPU/CANN，在 sys.modules 桩掉 torch_npu / vllm 重运行时依赖 /
vllm_ascend 周边模块，再把（已减法的）implementation/ 模块按**规范模块名**
（vllm_ascend.quantization.*）注册进去，让它们彼此 import 解析到精简版。

可在 host 验证、与真仓一致的纯 Python 控制流：
  (1) registry —— register_scheme 落表 / get_scheme_class 查表 / 重复注册 ValueError；
  (2) get_linear_quant_type —— 融合层各 shard 量化类型一致校验（不一致即 ValueError）；
  (3) create_scheme_for_layer —— 解析 quant_type → 查注册表 → 实例化 scheme（未支持则 NotImplementedError）；
  (4) AscendModelSlimConfig.get_quant_method —— 按 layer 类型四岔分发到 wrapper（FLOAT 层跳过）；
  (5) compressed-tensors / fp8 的「先删后替换」—— 从 QUANTIZATION_METHODS 删原版再同名注册；
  (6) AscendW8A8DynamicLinearMethod.get_weight/get_perchannel_param —— int8 [out,in] 权重 + per-channel scale；
  (7) AscendLinearMethod.create_weights —— wrapper 把 scheme 返回的参数 dict 逐个 register_parameter（适配器只搬运）；
  (8) AscendW8A8DynamicLinearMethod.apply —— 调 npu_dynamic_quant → npu_quant_matmul 的调用次序与入参。
真实 torch_npu 量化算子由「记录调用」替身承接——只验入参/分流，不真算（昇腾才有内核）。
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


# ----------------------------------------------------------------------------
# 通用工具
# ----------------------------------------------------------------------------
class _Recorder:
    """记录算子/装饰器调用的通用替身。"""

    def __init__(self):
        self.calls = []


class _Obj:
    """记录构造 kwargs 的通用占位对象。"""

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _mod(stubs, dotted):
    parts = dotted.split(".")
    for i in range(len(parts)):
        name = ".".join(parts[: i + 1])
        if name not in sys.modules:
            m = types.ModuleType(name)
            sys.modules[name] = m
            stubs.append(name)
            if i > 0:
                setattr(sys.modules[".".join(parts[:i])], parts[i], m)
    return sys.modules[dotted]


# ----------------------------------------------------------------------------
# vLLM / torch_npu / vllm_ascend 周边依赖的桩
# ----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def quant_pkg():
    stubs: list[str] = []
    added_pkgs: list[str] = []

    # regex -> 标准库 re
    import re as _re

    sys.modules["regex"] = _re
    stubs.append("regex")

    # transformers
    tf = _mod(stubs, "transformers")
    tf.PretrainedConfig = type("PretrainedConfig", (), {})

    # torch_npu —— 量化算子记录替身
    npu_rec = _Recorder()
    tnpu = _mod(stubs, "torch_npu")

    def npu_dynamic_quant(x):
        npu_rec.calls.append(("npu_dynamic_quant", x))
        quantized_x = torch.zeros(x.shape, dtype=torch.int8)
        pertoken_scale = torch.ones(x.shape[0])  # per-token，1-D
        return quantized_x, pertoken_scale

    def npu_quant_matmul(quantized_x, weight, weight_scale, pertoken_scale=None, bias=None, output_dtype=None):
        npu_rec.calls.append(
            ("npu_quant_matmul", quantized_x, weight, weight_scale, pertoken_scale, bias, output_dtype)
        )
        return torch.full((quantized_x.shape[0], 4), 7.0, dtype=output_dtype or torch.float32)

    def npu_format_cast(t, fmt):
        npu_rec.calls.append(("npu_format_cast", t, fmt))
        return t

    tnpu.npu_dynamic_quant = npu_dynamic_quant
    tnpu.npu_quant_matmul = npu_quant_matmul
    tnpu.npu_format_cast = npu_format_cast

    # vllm.config
    vconf = _mod(stubs, "vllm.config")
    vconf.CompilationMode = _Obj(VLLM_COMPILE="vllm_compile")

    def get_current_vllm_config():
        return _Obj(
            compilation_config=_Obj(mode="eager"),
            model_config=_Obj(enforce_eager=True, dtype=torch.float16, hf_config=_Obj(model_type="test_model")),
        )

    vconf.get_current_vllm_config = get_current_vllm_config

    # vllm.logger
    vlog = _mod(stubs, "vllm.logger")

    class _Logger:
        def __getattr__(self, _):
            return lambda *a, **k: None

    vlog.logger = _Logger()

    # vllm.model_executor.layers.attention_layer_base
    alb = _mod(stubs, "vllm.model_executor.layers.attention_layer_base")
    alb.AttentionLayerBase = type("AttentionLayerBase", (), {})

    # vllm.model_executor.layers.linear
    lin = _mod(stubs, "vllm.model_executor.layers.linear")
    lin.LinearBase = type("LinearBase", (), {})
    lin.LinearMethodBase = type("LinearMethodBase", (), {})
    lin.RowParallelLinear = type("RowParallelLinear", (lin.LinearBase,), {})
    lin.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})

    # vllm.model_executor.layers.fused_moe (+ .config)
    fm = _mod(stubs, "vllm.model_executor.layers.fused_moe")
    fm.FusedMoE = type("FusedMoE", (), {})

    class FusedMoEMethodBase:
        def __init__(self, moe_config=None):
            self.moe_config = moe_config

    fm.FusedMoEMethodBase = FusedMoEMethodBase
    fm.FusedMoeWeightScaleSupported = _Obj(CHANNEL=_Obj(value="channel"), GROUP=_Obj(value="group"))
    fmc = _mod(stubs, "vllm.model_executor.layers.fused_moe.config")
    fmc.FusedMoEConfig = type("FusedMoEConfig", (), {})

    # vllm.model_executor.layers.quantization (+ base_config / kv_cache)
    vq = _mod(stubs, "vllm.model_executor.layers.quantization")
    vq.QUANTIZATION_METHODS = ["awq", "compressed-tensors", "fp8", "deepseek_v4_fp8"]
    registered: dict = {}

    def register_quantization_config(name):
        def deco(cls):
            registered[name] = cls
            return cls

        return deco

    vq.register_quantization_config = register_quantization_config
    vq._registered = registered

    bc = _mod(stubs, "vllm.model_executor.layers.quantization.base_config")

    class QuantizationConfig:
        def __init__(self, *a, **k):
            self.packed_modules_mapping = {}

        def __repr__(self):
            return "QuantizationConfig()"

    bc.QuantizationConfig = QuantizationConfig
    bc.QuantizeMethodBase = type("QuantizeMethodBase", (), {})

    kvc = _mod(stubs, "vllm.model_executor.layers.quantization.kv_cache")
    kvc.BaseKVCacheMethod = type("BaseKVCacheMethod", (), {})

    # vocab_parallel_embedding
    vpe = _mod(stubs, "vllm.model_executor.layers.vocab_parallel_embedding")
    vpe.UnquantizedEmbeddingMethod = type("UnquantizedEmbeddingMethod", (), {})
    vpe.VocabParallelEmbedding = type("VocabParallelEmbedding", (), {})

    # models.utils.WeightsMapper / parameter / utils
    mu = _mod(stubs, "vllm.model_executor.models.utils")
    mu.WeightsMapper = type("WeightsMapper", (), {})
    par = _mod(stubs, "vllm.model_executor.parameter")
    par.PerTensorScaleParameter = type("PerTensorScaleParameter", (), {"__init__": lambda self, **k: None})
    meu = _mod(stubs, "vllm.model_executor.utils")

    def set_weight_attrs(param, d):
        for k, v in d.items():
            setattr(param, k, v)

    meu.set_weight_attrs = set_weight_attrs

    # vllm_ascend.utils
    au = _mod(stubs, "vllm_ascend.utils")
    au.ASCEND_QUANTIZATION_METHOD = "ascend"
    au.COMPRESSED_TENSORS_METHOD = "compressed-tensors"
    au.FP8_METHOD = "fp8"
    au.ACL_FORMAT_FRACTAL_NZ = 29
    au.maybe_trans_nz = lambda w: w
    au.enable_dsa_cp_with_layer_shard = lambda: False

    # vllm_ascend.ascend_config
    ac = _mod(stubs, "vllm_ascend.ascend_config")
    ac.get_ascend_config = lambda: _Obj(
        multistream_overlap_gate=False, eplb_config=_Obj(dynamic_eplb=False), enable_fused_mc2=0
    )

    # vllm_ascend.ascend_forward_context
    afc = _mod(stubs, "vllm_ascend.ascend_forward_context")
    afc._EXTRA_CTX = _Obj(moe_comm_method=None)

    # vllm_ascend.distributed.parallel_state
    ps = _mod(stubs, "vllm_ascend.distributed.parallel_state")
    ps.get_mc2_group = lambda: object()  # .device_group 缺失 -> AttributeError -> 走 except 分支

    # vllm_ascend.ops.* （FLOAT 跳过分支 / select_experts / build_fused_experts_input）
    opl = _mod(stubs, "vllm_ascend.ops.linear")
    opl.AscendUnquantizedLinearMethod = type("AscendUnquantizedLinearMethod", (), {})
    opf = _mod(stubs, "vllm_ascend.ops.fused_moe.fused_moe")
    opf.AscendUnquantizedFusedMoEMethod = type(
        "AscendUnquantizedFusedMoEMethod", (), {"__init__": lambda self, cfg=None: None}
    )
    es = _mod(stubs, "vllm_ascend.ops.fused_moe.experts_selector")
    es.select_experts = lambda **k: (torch.ones(2, 2), torch.zeros(2, 2, dtype=torch.long))
    mra = _mod(stubs, "vllm_ascend.ops.fused_moe.moe_runtime_args")
    mra.build_fused_experts_input = lambda **k: _Obj(**k)

    # vllm_ascend.device.mxfp_compat
    mx = _mod(stubs, "vllm_ascend.device.mxfp_compat")
    mx.FLOAT8_E8M0FNU_DTYPE = "float8_e8m0fnu"
    mx.FLOAT4_E2M1FN_X2_DTYPE = "float4_e2m1fn_x2"
    mx.ensure_mxfp4_dtype_available = lambda *a, **k: None
    mx.ensure_mxfp8_scale_dtype_available = lambda *a, **k: None

    # ------------------------------------------------------------------
    # 按规范模块名加载 implementation/ 各模块
    # ------------------------------------------------------------------
    def _ensure_pkg(dotted, path):
        m = types.ModuleType(dotted)
        m.__path__ = [str(path)]
        m.__package__ = dotted
        sys.modules[dotted] = m
        added_pkgs.append(dotted)
        parent = dotted.rsplit(".", 1)
        if len(parent) == 2 and parent[0] in sys.modules:
            setattr(sys.modules[parent[0]], parent[1], m)
        return m

    def _load(relpath, dotted, is_pkg=False):
        path = IMPL_DIR / relpath
        if is_pkg:
            spec = importlib.util.spec_from_file_location(
                dotted, path, submodule_search_locations=[str(path.parent)]
            )
        else:
            spec = importlib.util.spec_from_file_location(dotted, path)
        m = importlib.util.module_from_spec(spec)
        sys.modules[dotted] = m
        added_pkgs.append(dotted)
        parent = dotted.rsplit(".", 1)
        if len(parent) == 2 and parent[0] in sys.modules:
            setattr(sys.modules[parent[0]], parent[1], m)
        spec.loader.exec_module(m)
        return m

    # 包命名空间
    _ensure_pkg("vllm_ascend.quantization", IMPL_DIR)
    _ensure_pkg("vllm_ascend.quantization.methods", IMPL_DIR / "methods")

    # 叶子模块（依赖顺序）
    _load("quant_type.py", "vllm_ascend.quantization.quant_type")
    _load("quant_parser.py", "vllm_ascend.quantization.quant_parser")
    _load("methods/base.py", "vllm_ascend.quantization.methods.base")
    _load("methods/registry.py", "vllm_ascend.quantization.methods.registry")
    _load("methods/w8a8_dynamic.py", "vllm_ascend.quantization.methods.w8a8_dynamic")
    _load("methods/__init__.py", "vllm_ascend.quantization.methods", is_pkg=True)
    _load("method_adapters.py", "vllm_ascend.quantization.method_adapters")
    _load("modelslim_config.py", "vllm_ascend.quantization.modelslim_config")
    _load("compressed_tensors_config.py", "vllm_ascend.quantization.compressed_tensors_config")
    _load("fp8_config.py", "vllm_ascend.quantization.fp8_config")

    ns = _Obj(
        npu_rec=npu_rec,
        registered=registered,
        QUANTIZATION_METHODS=vq.QUANTIZATION_METHODS,
        quant_type=sys.modules["vllm_ascend.quantization.quant_type"],
        quant_parser=sys.modules["vllm_ascend.quantization.quant_parser"],
        registry=sys.modules["vllm_ascend.quantization.methods.registry"],
        base=sys.modules["vllm_ascend.quantization.methods.base"],
        w8a8=sys.modules["vllm_ascend.quantization.methods.w8a8_dynamic"],
        methods=sys.modules["vllm_ascend.quantization.methods"],
        method_adapters=sys.modules["vllm_ascend.quantization.method_adapters"],
        modelslim=sys.modules["vllm_ascend.quantization.modelslim_config"],
        compressed=sys.modules["vllm_ascend.quantization.compressed_tensors_config"],
        fp8=sys.modules["vllm_ascend.quantization.fp8_config"],
        linear_mod=lin,
        fused_moe_mod=fm,
    )

    yield ns

    for n in reversed(added_pkgs):
        sys.modules.pop(n, None)
    for n in reversed(stubs):
        sys.modules.pop(n, None)
