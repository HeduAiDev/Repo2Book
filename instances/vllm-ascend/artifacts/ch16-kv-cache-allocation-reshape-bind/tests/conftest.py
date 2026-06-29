"""ch16 测试脚手架：在 sys.modules 桩掉 vllm / vllm_ascend 的重运行时依赖（NPU/CANN/分布式
在 host 不可用），再把（已减法的）implementation/ 模块按**规范模块名**注册进去，让它们彼此
import 解析到精简版本身。昇腾 NPU 显存分配/物理布局不在 host 真跑，但本章核心——对齐算术
（_align_up/_align_memory）、int8 裸分配、K/V 字节拆分（calc_split_factor）、as_strided 跨步重排
（_adjust_kv_layout）、bind 三分支派发、kernel_block_sizes 装配——都是纯 Python/CPU torch，
可在 host 验证其与真实仓一致的可观察控制流。
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


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


# ---- KVCacheSpec 家族的轻量桩（只保留本章控制流分派需要的字段/类型）---- #
class _AttentionSpec:
    def __init__(self, block_size=128, num_kv_heads=1, head_size=128, dtype=torch.float16,
                 page_size_bytes=None, **kw):
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.dtype = dtype
        self._page_size_bytes = page_size_bytes
        for k, v in kw.items():
            setattr(self, k, v)

    @property
    def page_size_bytes(self):
        if self._page_size_bytes is not None:
            return self._page_size_bytes
        return self.block_size * self.num_kv_heads * self.head_size * torch.empty(
            (), dtype=self.dtype).element_size() * 2


class _MLAAttentionSpec(_AttentionSpec):
    pass


class _SlidingWindowMLASpec(_MLAAttentionSpec):
    pass


class _MambaSpec:
    def __init__(self, shapes, dtypes, page_size_bytes, block_size=128):
        self.shapes = shapes
        self.dtypes = dtypes
        self._page_size_bytes = page_size_bytes
        self.block_size = block_size

    @property
    def page_size_bytes(self):
        return self._page_size_bytes


class _EncoderOnlyAttentionSpec:
    def __init__(self, block_size=128):
        self.block_size = block_size


class _UniformTypeKVCacheSpecs:
    def __init__(self, kv_cache_specs):
        self.kv_cache_specs = kv_cache_specs


@pytest.fixture
def env():
    stubs = _Stubs()

    # ---- vllm.* 运行时依赖桩 ---- #
    cfg = stubs.mod("vllm.config")
    cfg.get_layers_from_vllm_config = lambda *_a, **_k: {}

    ec = stubs.mod("vllm.distributed.ec_transfer")
    ec.has_ec_transfer = lambda: False
    ec.get_ec_transfer = lambda: types.SimpleNamespace(is_producer=False)

    kvt = stubs.mod("vllm.distributed.kv_transfer")
    kvt.has_kv_transfer_group = lambda: False
    kvt.get_kv_transfer_group = lambda: None

    attn = stubs.mod("vllm.model_executor.layers.attention")
    attn.Attention = type("Attention", (), {})
    attn.MLAAttention = type("MLAAttention", (), {})

    alb = stubs.mod("vllm.model_executor.layers.attention_layer_base")
    alb.AttentionLayerBase = type("AttentionLayerBase", (), {})

    mamba = stubs.mod("vllm.model_executor.layers.mamba.abstract")
    mamba.MambaBase = type("MambaBase", (), {})

    ehs = stubs.mod("vllm.model_executor.models.extract_hidden_states")
    ehs.CacheOnlyAttentionLayer = type("CacheOnlyAttentionLayer", (), {})

    mu = stubs.mod("vllm.model_executor.models.utils")
    mu.extract_layer_index = lambda name: int("".join(c for c in name.split(".")[-1] if c.isdigit()) or 0)

    mm = stubs.mod("vllm.utils.math_utils")
    mm.cdiv = lambda a, b: -(-a // b)

    tu = stubs.mod("vllm.utils.torch_utils")
    tu.get_dtype_size = lambda d: torch.empty((), dtype=d).element_size()

    kci = stubs.mod("vllm.v1.kv_cache_interface")
    kci.AttentionSpec = _AttentionSpec
    kci.MLAAttentionSpec = _MLAAttentionSpec
    kci.SlidingWindowMLASpec = _SlidingWindowMLASpec
    kci.MambaSpec = _MambaSpec
    kci.EncoderOnlyAttentionSpec = _EncoderOnlyAttentionSpec
    kci.UniformTypeKVCacheSpecs = _UniformTypeKVCacheSpecs
    kci.KVCacheConfig = type("KVCacheConfig", (), {})
    kci.KVCacheSpec = type("KVCacheSpec", (), {})

    cpu = stubs.mod("vllm.v1.worker.cp_utils")
    cpu.get_total_cp_world_size = lambda: 1

    gmr = stubs.mod("vllm.v1.worker.gpu_model_runner")
    gmr.GPUModelRunner = type("GPUModelRunner", (), {})

    wutils = stubs.mod("vllm.v1.worker.utils")
    # 默认实现可被各测试用 monkeypatch 覆盖以记录调用
    wutils.bind_kv_cache = lambda *_a, **_k: None
    wutils.select_common_block_size = lambda kv_block, backends: kv_block

    # ---- vllm_ascend.* ---- #
    qutils = stubs.mod("vllm_ascend.quantization.utils")
    qutils.enable_fa_quant = lambda *_a, **_k: False

    nib = stubs.mod("vllm_ascend.worker.npu_input_batch")

    class _NPUInputBatch:
        def __init__(self, **kw):
            self.kwargs = kw
            self.logitsprocs = kw.get("logitsprocs")

    nib.NPUInputBatch = _NPUInputBatch

    # ---- 加载精简版（覆盖任何同名桩）---- #
    stubs.mod("vllm_ascend")
    stubs.mod("vllm_ascend.worker")

    utils = _load("utils.py", "vllm_ascend.utils")
    mr = _load("model_runner_v1.py", "vllm_ascend.worker.model_runner_v1")

    handles = types.SimpleNamespace(
        utils=utils, mr=mr, kci=kci, wutils=wutils, nib=nib, attn=attn, ehs=ehs,
        AttentionSpec=_AttentionSpec, MLAAttentionSpec=_MLAAttentionSpec,
        MambaSpec=_MambaSpec, EncoderOnlyAttentionSpec=_EncoderOnlyAttentionSpec,
        UniformTypeKVCacheSpecs=_UniformTypeKVCacheSpecs, stubs=stubs,
    )
    try:
        yield handles
    finally:
        stubs.cleanup()
        for n in ("vllm_ascend.utils", "vllm_ascend.worker.model_runner_v1"):
            sys.modules.pop(n, None)
