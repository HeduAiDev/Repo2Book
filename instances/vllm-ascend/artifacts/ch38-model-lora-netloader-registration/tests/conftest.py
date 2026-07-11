"""ch30 测试脚手架：host 无 NPU/CANN/vllm/transformers，在 sys.modules 桩掉 vllm 周边
基类/工具，再把（已减法的）implementation/ 模块按**规范模块名**注册进去，让三组扩展点的
纯 Python 控制流可在 host 验证、且与真仓一致：

  变体(1) 模型注册：
    - register_model() → ModelRegistry.register_model 两次，架构名→「<module>:<class>」字符串；
    - _ModelRegistry.register_model：字符串走懒加载、非法格式 raise、已注册同名 warn+覆盖。
  变体(2) LoRA：
    - refresh_all_lora_classes()：把 4 个 Ascend 类按确定顺序追加进 vllm.lora.utils._all_lora_classes；
    - can_replace_layer：仅当 type(source_layer) is AscendQKVParallelLinear 且 packed 长度匹配才 True；
    - from_layer：遍历全局元组，遇昇腾 QKV 层时命中昇腾 LoRA 类（全局类替换 trick 的可观察效果）；
    - PunicaWrapperNPU.__init__：按 device(310P)/rank(≥128) 二选一绑 torch_ops 或昇腾 lora_ops；
    - lora_ops.bgmv_shrink/bgmv_expand 薄壳：参数 reorder 后转调 torch.ops._C_ascend.*。
  变体(3) netloader：
    - @register_model_loader('netloader') 写进 _LOAD_FORMAT_TO_MODEL_LOADER + get_model_loader 分发 +
      非 BaseModelLoader 子类 raise；
    - load_model：source 无效 → revert_to_default；elastic_load 返回 None → 回退；
    - revert_to_default：quantization None 走 DefaultModelLoader.load_model，否则 load_weights 分支。

重 NPU 路径（真 bgmv/sgmv kernel、网络弹性传输、整模型前向）不真跑，只验上述控制流分流。
"""

import importlib.util
import sys
import types
from contextlib import contextmanager
from pathlib import Path

import pytest

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


def _permissive(name, **methods):
    ns = {"__init__": lambda self, *a, **k: None}
    ns.update(methods)
    return type(name, (), ns)


def _dummy_logger():
    def _noop(*a, **k):
        return None

    return types.SimpleNamespace(info=_noop, warning=_noop, error=_noop, info_once=_noop, debug=_noop)


@pytest.fixture
def env():
    stubs = _Stubs()

    @contextmanager
    def _nullctx(*a, **k):
        yield None

    # ---- vllm.logger ---- #
    vlog = stubs.mod("vllm.logger")
    vlog.init_logger = lambda *a, **k: _dummy_logger()
    vlog.logger = _dummy_logger()

    # ---- transformers（host 未安装） ---- #
    tf = stubs.mod("transformers")
    tf.PretrainedConfig = type("PretrainedConfig", (), {})

    # ---- vllm.config ---- #
    cfg = stubs.mod("vllm.config")
    cfg.LoRAConfig = type("LoRAConfig", (), {})
    cfg.LoadConfig = type("LoadConfig", (), {})
    cfg.ModelConfig = type("ModelConfig", (), {})
    cfg.VllmConfig = type("VllmConfig", (), {})
    cfgload = stubs.mod("vllm.config.load")
    cfgload.LoadConfig = cfg.LoadConfig

    # ---- torch.ops._C_ascend（记录薄壳转调） ---- #
    import torch

    calls = []

    class _OpRec:
        def __getattr__(self, name):
            def _rec(*args):
                calls.append((name, args))
                return ("kernel", name, args)

            return _rec

    _saved_ops = getattr(torch.ops, "_C_ascend", None)
    torch.ops._C_ascend = _OpRec()

    # ===================== 变体(2) LoRA 基类 / 工具 ===================== #
    lly = stubs.mod("vllm.lora.layers")
    for cls in (
        "QKVParallelLinearWithLoRA",
        "MergedQKVParallelLinearWithLoRA",
        "MergedQKVParallelLinearWithShardedLoRA",
        "QKVParallelLinearWithShardedLoRA",
        "VocabParallelEmbeddingWithLoRA",
    ):
        # 基类带一个 can_replace_layer 默认返回 False，便于断言「昇腾子类重写后才命中」。
        setattr(lly, cls, type(cls, (), {"can_replace_layer": classmethod(lambda c, **k: False),
                                          "__init__": lambda self, layer=None: None,
                                          "create_lora_weights": lambda self, *a, **k: None}))
    llb = stubs.mod("vllm.lora.layers.base")
    llb.BaseLayerWithLoRA = type("BaseLayerWithLoRA", (), {})
    llu = stubs.mod("vllm.lora.layers.utils")
    # 直通装饰器：保留昇腾 can_replace_layer 的纯 type 判定（真仓装饰器只多加 sharded 维度校验）。
    llu._fully_sharded_can_replace = lambda f: f
    llu._not_fully_sharded_can_replace = lambda f: f

    ops_linear = stubs.mod("vllm_ascend.ops.linear")
    ops_linear.AscendQKVParallelLinear = type("AscendQKVParallelLinear", (), {})

    # punica 基类 + 昇腾 utils 依赖
    pwb = stubs.mod("vllm.lora.punica_wrapper.punica_base")
    pwb.PunicaWrapperBase = _permissive("PunicaWrapperBase")
    asu = stubs.mod("vllm_ascend.utils")

    class _AscendDeviceType:
        _310P = "310P"
        _910 = "910"

    asu.AscendDeviceType = _AscendDeviceType
    asu._device_type = _AscendDeviceType._910
    asu.get_ascend_device_type = lambda: asu._device_type

    # vLLM torch_ops 退路（6 个哨兵函数，便于断言「大 rank / 310P 时绑的是这一组」）
    vto = stubs.mod("vllm.lora.ops.torch_ops")
    for fn in ("bgmv_expand", "bgmv_expand_slice", "bgmv_shrink", "sgmv_expand", "sgmv_expand_slice", "sgmv_shrink"):
        setattr(vto, fn, (lambda _n: (lambda *a, **k: ("torch_ops", _n)))(fn))

    # ===================== 变体(3) loader 依赖 ===================== #
    class _FakeModel:
        def __init__(self, tag):
            self.tag = tag
            self.evaled = False
            self.weights_loaded = False

        def eval(self):
            self.evaled = True
            return self

    # torch.distributed.get_rank（host 无分布式初始化）
    _saved_get_rank = getattr(torch.distributed, "get_rank", None)
    torch.distributed.get_rank = lambda: 0

    base_loader = stubs.mod("vllm.model_executor.model_loader.base_loader")

    class _BaseModelLoader:
        def __init__(self, load_config=None):
            self.load_config = load_config

    base_loader.BaseModelLoader = _BaseModelLoader
    default_loader = stubs.mod("vllm.model_executor.model_loader.default_loader")

    class _DefaultModelLoader(_BaseModelLoader):
        def __init__(self, load_config=None):
            self.load_config = load_config
            self.loaded = False

        def load_model(self, vllm_config=None, model_config=None, prefix=""):
            self.loaded = True
            return _FakeModel("default_loaded")

        def load_weights(self, model, model_config):
            model.weights_loaded = True

    default_loader.DefaultModelLoader = _DefaultModelLoader

    ml_utils = stubs.mod("vllm.model_executor.model_loader.utils")
    ml_utils.initialize_model = lambda **k: _FakeModel("empty_skeleton")
    ml_utils.process_weights_after_loading = lambda *a, **k: None
    tu = stubs.mod("vllm.utils.torch_utils")
    tu.set_default_torch_dtype = _nullctx

    # netloader 的 `from .load import elastic_load`
    stubs.mod("vllm_ascend")
    stubs.mod("vllm_ascend.model_loader")
    stubs.mod("vllm_ascend.lora")
    stubs.mod("vllm_ascend.models")
    netpkg = stubs.mod("vllm_ascend.model_loader.netloader")
    loadmod = stubs.mod("vllm_ascend.model_loader.netloader.load")

    def _default_elastic_load(model=None, **k):
        return model  # 默认：弹性加载成功，原样返回骨架

    loadmod.elastic_load = _default_elastic_load

    # ---- 载入（已减法的）精简版，按规范模块名注册 ---- #
    vllm_registry = _load("vllm_registry.py", "_ch30_vllm_registry")
    # 把 ModelRegistry 暴露成 `vllm.ModelRegistry`（models_register.py 从这里 import）
    vmod = sys.modules["vllm"]
    vmod.ModelRegistry = vllm_registry.ModelRegistry

    vllm_lora_utils = _load("vllm_lora_utils.py", "vllm.lora.utils")
    vllm_model_loader = _load("vllm_model_loader.py", "vllm.model_executor.model_loader")

    lora_utils = _load("lora_utils.py", "vllm_ascend.lora.utils")
    lora_ops = _load("lora_ops.py", "vllm_ascend.lora.lora_ops")
    punica_npu = _load("punica_npu.py", "vllm_ascend.lora.punica_npu")
    models_register = _load("models_register.py", "_ch30_models_register")
    netloader = _load("netloader.py", "vllm_ascend.model_loader.netloader.netloader")

    yield types.SimpleNamespace(
        vllm_registry=vllm_registry,
        ModelRegistry=vllm_registry.ModelRegistry,
        vllm_lora_utils=vllm_lora_utils,
        vllm_model_loader=vllm_model_loader,
        lora_utils=lora_utils,
        lora_ops=lora_ops,
        punica_npu=punica_npu,
        models_register=models_register,
        netloader=netloader,
        ops_linear=ops_linear,
        asu=asu,
        AscendDeviceType=_AscendDeviceType,
        torch_ops_mod=vto,
        DefaultModelLoader=_DefaultModelLoader,
        base_loader=base_loader,
        FakeModel=_FakeModel,
        loadmod=loadmod,
        kernel_calls=calls,
        set_device=lambda d: setattr(asu, "_device_type", d),
        set_elastic_load=lambda fn: setattr(loadmod, "elastic_load", fn),
    )

    if _saved_ops is not None:
        torch.ops._C_ascend = _saved_ops
    if _saved_get_rank is not None:
        torch.distributed.get_rank = _saved_get_rank
    stubs.cleanup()
