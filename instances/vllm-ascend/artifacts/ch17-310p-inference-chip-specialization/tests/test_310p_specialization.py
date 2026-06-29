"""TDD tests for ch17 — 310P 推理芯片：受限硬件上的全栈特化.

这些测试验证 vllm-ascend 真实代码（忠实减法进 ../implementation）的*可观察行为*——
聚焦 dossier 标注为 host 可跑的纯 Python / NumPy / torch(CPU) 控制流：

  1. block_table.py  : 310P 用 CPU NumPy 算 slot = block_number*block_size + offset（替换
                       基类 Triton kernel）；_to_numpy 的 "must be CPU" 守卫；2/3 参规整。
  2. kv_block_zeroer : 去 Triton 的 KV 清零——init_meta 收集 (k,v) 张量 + logical_page_ratio，
                       zero_block_ids 切片 .zero_()。
  3. sharded_state_loader : save_model 永远单 part；generate_quant_description 产出
                       parameters_type_map.json（int dtype→量化类型，否则 FLOAT）。
  4. patch_distributed : 310P 用 all_gather 模拟 int64 all_reduce(sum/max)；非 int64 / CPU
                       broadcast 走原生 fn；group_src 优先级。
  5. utils.py        : "Ascend310P3" 子串 → AscendDeviceType._310P → is_310p()==True 总开关。
  6. platform.py     : is_310p 时 worker_cls→NPUWorker310 且不开 custom_ops；backend_map_310
                       只有 (False,False)，MLA 不支持。

昇腾 NPU/CANN 不在 host 真跑（torch_npu / Triton 算子 / 真实显存），但上述减法保留的控制流
本身是纯 Python——所以在 sys.modules 里桩掉 vllm / vllm_ascend / 设备相关命名空间，import
（已减法的）实现，断言行为与真实仓一致。重型 runner 子类（model_runner_310p / npu_input_batch
/ worker_310p）触运行时无法 host 实例化，改以源码级结构断言验证"子类化覆写点 + must_keep 符号"。
"""

import ast
import importlib.util
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
class ModRegistry:
    """在 sys.modules 注册假的点分模块，自动清理；不覆盖已存在的真实模块。"""

    def __init__(self):
        self.added = []

    def module(self, dotted):
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

    def define(self, dotted, **attrs):
        m = self.module(dotted)
        for k, v in attrs.items():
            setattr(m, k, v)
        return m

    def cleanup(self):
        for n in reversed(self.added):
            sys.modules.pop(n, None)


_counter = [0]


def _load(filename, modname=None):
    _counter[0] += 1
    name = modname or f"_impl_ch17_{_counter[0]}_{Path(filename).stem}"
    spec = importlib.util.spec_from_file_location(name, IMPL_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _Buffer:
    """模拟 310 BlockTable 的 CpuGpuBuffer：有 .np 数组与 copy_to_gpu(n) 记录。"""

    def __init__(self, n, dtype=np.int64):
        self.np = np.zeros(n, dtype=dtype)
        self.copied_n = None

    def copy_to_gpu(self, n=None):
        self.copied_n = n


# --------------------------------------------------------------------------- #
# 1. block_table.py — CPU NumPy slot_mapping
# --------------------------------------------------------------------------- #
@pytest.fixture
def block_table_cls():
    reg = ModRegistry()
    reg.define("vllm.utils.math_utils", cdiv=lambda a, b: -(-a // b))
    reg.define("vllm.v1.attention.backends.utils", PAD_SLOT_ID=-1)
    reg.define("vllm.v1.kv_cache_interface", KVCacheGroupSpec=object)
    reg.define("vllm.v1.worker.cp_utils", get_total_cp_world_size=lambda: 1)

    class AscendBlockTable:  # 昇腾独立基类的桩（不继承 vLLM BlockTable）
        pass

    class AscendMultiGroupBlockTable:
        pass

    reg.define(
        "vllm_ascend.worker.block_table",
        BlockTable=AscendBlockTable,
        MultiGroupBlockTable=AscendMultiGroupBlockTable,
    )
    mod = _load("block_table.py")
    yield mod.BlockTable
    reg.cleanup()


def _make_bt(block_table_cls, block_size=4, table=None, slot_len=16):
    bt = object.__new__(block_table_cls)
    bt.block_size = block_size
    bt.max_num_blocks_per_req = 2
    bt.blocks_per_phys_block = 1
    table = np.array([[10, 11], [20, 21]], dtype=np.int32) if table is None else table
    bt.block_table = types.SimpleNamespace(np=table)
    bt.slot_mapping = _Buffer(slot_len)
    return bt


def test_slot_mapping_numpy_formula(block_table_cls):
    """slot = block_number*block_size + (position % block_size)，全在 CPU 算（无 Triton）。"""
    bt = _make_bt(block_table_cls, block_size=4)
    req_indices = np.array([0, 0, 1], dtype=np.int64)
    positions = np.array([0, 5, 3], dtype=np.int64)
    bt._compute_slot_mapping_numpy(req_indices, positions)
    # block_numbers = [10, 11, 20]；offsets = [0, 1, 3]
    assert list(bt.slot_mapping.np[:3]) == [40, 45, 83]
    assert bt.slot_mapping.copied_n == 3


def test_slot_mapping_numpy_empty(block_table_cls):
    bt = _make_bt(block_table_cls)
    bt._compute_slot_mapping_numpy(np.array([], dtype=np.int64), np.array([], dtype=np.int64))
    assert bt.slot_mapping.copied_n == 0


def test_compute_slot_mapping_dispatches_2arg(block_table_cls):
    bt = _make_bt(block_table_cls, block_size=4)
    bt.compute_slot_mapping(np.array([0, 1], dtype=np.int64), np.array([0, 0], dtype=np.int64))
    # req0 pos0 → 10*4+0=40 ; req1 pos0 → 20*4+0=80
    assert list(bt.slot_mapping.np[:2]) == [40, 80]


def test_to_numpy_accepts_cpu_tensor_and_ndarray(block_table_cls):
    f = block_table_cls._to_numpy
    out = f(np.array([1, 2], dtype=np.int32))
    assert out.dtype == np.int64 and list(out) == [1, 2]
    out2 = f(torch.tensor([3, 4], dtype=torch.int32))
    assert out2.dtype == np.int64 and list(out2) == [3, 4]


def test_to_numpy_rejects_non_cpu_tensor(block_table_cls):
    """310P 规避 device 算术/D2H：device 张量直接 raise（用 meta 设备模拟非 CPU）。"""
    meta = torch.zeros(3, dtype=torch.int64, device="meta")
    with pytest.raises(TypeError, match="must be computed from CPU"):
        block_table_cls._to_numpy(meta)


def test_normalize_3arg_builds_req_indices(block_table_cls):
    bt = object.__new__(block_table_cls)
    req, pos = bt._normalize_slot_mapping_inputs(
        2, np.array([0, 2, 5], dtype=np.int64), np.arange(5, dtype=np.int64)
    )
    assert list(req) == [0, 0, 1, 1, 1]
    assert list(pos) == [0, 1, 2, 3, 4]


def test_normalize_3arg_token_count_mismatch_raises(block_table_cls):
    bt = object.__new__(block_table_cls)
    with pytest.raises(ValueError, match="different token counts"):
        bt._normalize_slot_mapping_inputs(2, np.array([0, 2, 5], dtype=np.int64), np.arange(4, dtype=np.int64))


def test_normalize_bad_argcount_raises(block_table_cls):
    bt = object.__new__(block_table_cls)
    with pytest.raises(TypeError, match="2 or 3 positional"):
        bt._normalize_slot_mapping_inputs(1)


# --------------------------------------------------------------------------- #
# 2. kv_block_zeroer.py — 去 Triton 的切片清零
# --------------------------------------------------------------------------- #
@pytest.fixture
def kv_zeroer_mod():
    reg = ModRegistry()

    class FullAttentionSpec:
        def __init__(self, block_size):
            self.block_size = block_size

    class KVBlockZeroer:  # vLLM 基类的桩
        def __init__(self, device, pin_memory):
            self.device = device
            self.pin_memory = pin_memory

    reg.define("vllm.v1.kv_cache_interface", FullAttentionSpec=FullAttentionSpec)
    reg.define("vllm.v1.worker.utils", AttentionGroup=object, KVBlockZeroer=KVBlockZeroer)
    mod = _load("kv_block_zeroer.py")
    mod._FullAttentionSpec = FullAttentionSpec  # 暴露给测试构造 spec
    yield mod
    reg.cleanup()


def _group(spec, layer_names, gid=0):
    return types.SimpleNamespace(kv_cache_spec=spec, kv_cache_group_id=gid, layer_names=layer_names)


def test_init_meta_collects_kv_and_ratio(kv_zeroer_mod):
    Spec = kv_zeroer_mod._FullAttentionSpec
    z = kv_zeroer_mod.AscendKVBlockZeroer310(torch.device("cpu"), pin_memory=False)
    k, v = torch.ones(8), torch.ones(8)
    ctx = {"l0": types.SimpleNamespace(kv_cache=(k, v))}
    z.init_meta(
        attn_groups_iter=[_group(Spec(block_size=16), ["l0"])],
        kernel_block_sizes=[[4]],
        cache_dtype="float16",
        runner_only_attn_layers=set(),
        static_forward_context=ctx,
    )
    assert z._logical_page_ratio == 16 // 4  # spec.block_size // kernel_bs
    assert z._kv_tensors == [k, v]


def test_init_meta_dedups_by_data_ptr(kv_zeroer_mod):
    Spec = kv_zeroer_mod._FullAttentionSpec
    z = kv_zeroer_mod.AscendKVBlockZeroer310(torch.device("cpu"), pin_memory=False)
    shared = torch.ones(8)
    ctx = {"l0": types.SimpleNamespace(kv_cache=(shared, shared))}  # k 与 v 同一张量
    z.init_meta([_group(Spec(16), ["l0"])], [[4]], "float16", set(), ctx)
    assert len(z._kv_tensors) == 1  # data_ptr 去重


def test_init_meta_skips_non_full_attention_spec(kv_zeroer_mod):
    z = kv_zeroer_mod.AscendKVBlockZeroer310(torch.device("cpu"), pin_memory=False)
    other_spec = types.SimpleNamespace(block_size=16)  # 非 FullAttentionSpec
    z.init_meta([_group(other_spec, ["l0"])], [[4]], "float16", set(), {})
    assert z._kv_tensors == []


def test_zero_block_ids_slices_to_zero(kv_zeroer_mod):
    z = kv_zeroer_mod.AscendKVBlockZeroer310(torch.device("cpu"), pin_memory=False)
    kv = torch.ones(12)
    z._kv_tensors = [kv]
    z._logical_page_ratio = 4
    z.zero_block_ids([1])  # start=4, end=8
    expected = torch.ones(12)
    expected[4:8] = 0
    assert torch.equal(kv, expected)


def test_zero_block_ids_noop_when_empty(kv_zeroer_mod):
    z = kv_zeroer_mod.AscendKVBlockZeroer310(torch.device("cpu"), pin_memory=False)
    z._kv_tensors = [torch.ones(4)]
    z.zero_block_ids([])  # 空 block_ids → no-op
    assert torch.equal(z._kv_tensors[0], torch.ones(4))


# --------------------------------------------------------------------------- #
# 3. sharded_state_loader_310p.py — 单 part + parameters_type_map.json
# --------------------------------------------------------------------------- #
@pytest.fixture
def loader_cls(tmp_path):
    reg = ModRegistry()

    class ShardedStateLoader:
        DEFAULT_PATTERN = "model-rank-{rank}-part-{part}.safetensors"

        @staticmethod
        def _filter_subtensors(sd):
            return sd

    reg.define("vllm.config.load", LoadConfig=object)
    reg.define("vllm.model_executor.layers.quantization.base_config", QuantizationConfig=object)
    reg.define("vllm.model_executor.model_loader", ShardedStateLoader=ShardedStateLoader)
    reg.define("vllm.distributed", get_tensor_model_parallel_rank=lambda: 0)
    mod = _load("sharded_state_loader_310p.py")
    yield mod.ShardedStateLoader310
    reg.cleanup()


class _Model:
    def __init__(self, sd):
        self._sd = sd

    def state_dict(self):
        return self._sd


def test_generate_quant_description_int_vs_float(loader_cls, tmp_path):
    sd = {
        "a.weight": torch.ones(2, 2, dtype=torch.int8),
        "b.weight": torch.ones(2, 2, dtype=torch.float16),
        "c.bias": torch.ones(2, dtype=torch.int32),
        "d.scale": torch.ones(2, dtype=torch.float32),
    }
    quant_config = types.SimpleNamespace(quant_description={"model_quant_type": "W8A8"})
    loader_cls.generate_quant_description(_Model(sd), str(tmp_path), quant_config)
    out = json.loads((tmp_path / "parameters_type_map.json").read_text())
    assert out["model_quant_type"] == "W8A8"
    assert out["version"] == "1.0.0"
    assert out["a.weight"] == "W8A8"  # int8 .weight → 量化类型
    assert out["c.bias"] == "W8A8"  # int32 .bias → 量化类型
    assert out["b.weight"] == "FLOAT"  # float .weight → FLOAT
    assert out["d.scale"] == "FLOAT"  # 非 .weight/.bias → FLOAT


def test_generate_quant_description_no_config_all_float(loader_cls, tmp_path):
    sd = {"a.weight": torch.ones(2, dtype=torch.int8)}
    loader_cls.generate_quant_description(_Model(sd), str(tmp_path), None)
    out = json.loads((tmp_path / "parameters_type_map.json").read_text())
    assert out["model_quant_type"] == "FLOAT"
    assert out["a.weight"] == "FLOAT"  # quant_config None → quantize_type=FLOAT


def test_save_model_single_part(loader_cls, tmp_path):
    """无视 max_size，永远只写一个 part 文件（rank=0, part=0）。"""
    sd = {"w": torch.ones(2, 2)}
    loader_cls.save_model(_Model(sd), str(tmp_path), max_size=1)  # max_size 被忽略
    files = sorted(p.name for p in tmp_path.glob("*.safetensors"))
    assert files == ["model-rank-0-part-0.safetensors"]


# --------------------------------------------------------------------------- #
# 4. patch_distributed.py — all_gather 模拟
# --------------------------------------------------------------------------- #
@pytest.fixture
def patch_dist_mod():
    reg = ModRegistry()

    class AscendDeviceType:
        _310P = object()

    reg.define(
        "vllm_ascend.utils",
        AscendDeviceType=AscendDeviceType,
        get_ascend_device_type=lambda: object(),  # != _310P → 模块尾守卫不触发，import 不污染全局
    )
    mod = _load("patch_distributed.py")
    yield mod
    reg.cleanup()


@pytest.fixture
def dist_sandbox():
    """保存/恢复 torch.distributed 全局符号，单进程模拟 all_gather。"""
    d = torch.distributed
    c10d = d.distributed_c10d
    saved = {
        "broadcast": d.broadcast,
        "all_reduce": d.all_reduce,
        "get_rank": d.get_rank,
        "get_world_size": d.get_world_size,
        "all_gather": d.all_gather,
        "c_broadcast": c10d.broadcast,
        "c_all_reduce": c10d.all_reduce,
    }
    state = {"world_size": 1, "rank": 0, "gathered": None}

    def fake_all_gather(tensor_list, tensor, group=None):
        src = state["gathered"] if state["gathered"] is not None else [tensor] * len(tensor_list)
        for i in range(len(tensor_list)):
            tensor_list[i].copy_(src[i])

    d.get_rank = lambda group=None: state["rank"]
    d.get_world_size = lambda group=None: state["world_size"]
    d.all_gather = fake_all_gather
    yield d, state
    for k, fn in saved.items():
        if k.startswith("c_"):
            setattr(c10d, k[2:], fn)
        else:
            setattr(d, k, fn)


def test_all_reduce_int64_sum(patch_dist_mod, dist_sandbox):
    d, state = dist_sandbox
    patch_dist_mod.communication_adaptation_310p()
    state["world_size"] = 2
    state["gathered"] = [torch.tensor([1, 2], dtype=torch.int64), torch.tensor([10, 20], dtype=torch.int64)]
    out = torch.distributed.all_reduce(torch.tensor([0, 0], dtype=torch.int64))
    assert list(out) == [11, 22]  # all_gather 后 stack().sum(0)


def test_all_reduce_int64_max(patch_dist_mod, dist_sandbox):
    d, state = dist_sandbox
    patch_dist_mod.communication_adaptation_310p()
    state["world_size"] = 2
    state["gathered"] = [torch.tensor([1, 90], dtype=torch.int64), torch.tensor([7, 8], dtype=torch.int64)]
    out = torch.distributed.all_reduce(torch.tensor([0, 0], dtype=torch.int64), op=torch.distributed.ReduceOp.MAX)
    assert list(out) == [7, 90]


def test_all_reduce_non_int64_passthrough(patch_dist_mod, dist_sandbox):
    d, state = dist_sandbox
    calls = {}

    def orig(tensor, op, group, async_op):
        calls["hit"] = True
        return "ORIG"

    d.all_reduce = orig
    patch_dist_mod.communication_adaptation_310p()
    out = torch.distributed.all_reduce(torch.tensor([1.0, 2.0], dtype=torch.float32))
    assert out == "ORIG" and calls.get("hit")  # 非 int64 → 原生 fn


def test_broadcast_cpu_passthrough_uses_group_src(patch_dist_mod, dist_sandbox):
    d, state = dist_sandbox
    seen = {}

    def orig(tensor, src=0, group=None, async_op=False):
        seen["src"] = src
        return "ORIG"

    d.broadcast = orig
    patch_dist_mod.communication_adaptation_310p()
    cpu_t = torch.tensor([1, 2], dtype=torch.int64)
    out = torch.distributed.broadcast(cpu_t, group_src=3)  # group_src 优先于 src
    assert out == "ORIG" and seen["src"] == 3  # CPU 张量走原生 fn，root=group_src


# --------------------------------------------------------------------------- #
# 5. utils.py — is_310p 总开关
# --------------------------------------------------------------------------- #
@pytest.fixture
def utils_mod():
    mod = _load("utils.py")
    yield mod


def _set_build_info(reg, **attrs):
    reg.define("vllm_ascend", __name__="vllm_ascend")
    reg.define("vllm_ascend._build_info", **attrs)


def test_is_310p_true_from_soc_version(utils_mod):
    reg = ModRegistry()
    _set_build_info(reg, __soc_version__="Ascend310P3")
    utils_mod._ascend_device_type = None
    assert utils_mod.is_310p() is True
    assert utils_mod.get_ascend_device_type() is utils_mod.AscendDeviceType._310P
    reg.cleanup()


def test_is_310p_false_for_910(utils_mod):
    reg = ModRegistry()
    _set_build_info(reg, __soc_version__="ASCEND910B1")
    utils_mod._ascend_device_type = None
    assert utils_mod.is_310p() is False
    reg.cleanup()


def test_device_type_from_explicit_field(utils_mod):
    reg = ModRegistry()
    _set_build_info(reg, __device_type__="_310P")
    utils_mod._ascend_device_type = None
    assert utils_mod.get_ascend_device_type() is utils_mod.AscendDeviceType._310P
    reg.cleanup()


def test_acl_format_constant(utils_mod):
    assert utils_mod.ACL_FORMAT_FRACTAL_NZ == 29
    assert utils_mod.SOC_VERSION_INFERENCE_SERIES == ["Ascend310P3"]


# --------------------------------------------------------------------------- #
# 6. platform.py — worker_cls + backend_map_310 横切分流
# --------------------------------------------------------------------------- #
@pytest.fixture
def platform_mod():
    reg = ModRegistry()

    class AscendDeviceType:
        _310P = "310P"
        A2 = "A2"

    reg.define("vllm.attention.backends.registry", AttentionBackendEnum=types.SimpleNamespace(FLASH_ATTN="fa"))
    reg.define(
        "vllm_ascend.utils",
        AscendDeviceType=AscendDeviceType,
        get_ascend_device_type=lambda: AscendDeviceType.A2,
        is_310p=lambda: False,
    )
    mod = _load("platform.py")
    mod._AscendDeviceType = AscendDeviceType
    yield mod
    reg.cleanup()


def _attn_cfg(use_mla=False, use_sparse=False, use_compress=False):
    return types.SimpleNamespace(use_mla=use_mla, use_sparse=use_sparse, use_compress=use_compress)


def test_backend_map_310_only_non_mla(platform_mod):
    platform_mod.is_310p = lambda: True
    got = platform_mod.get_attn_backend_cls(None, None, _attn_cfg(use_mla=False, use_sparse=False))
    assert got == "vllm_ascend._310p.attention.attention_v1.AscendAttentionBackend310"


def test_backend_map_310_mla_falls_back_not_supported(platform_mod):
    """MLA/SFA 在 backend_map_310 被注释掉——key (True,False) 落回 (False,False) 默认。"""
    platform_mod.is_310p = lambda: True
    got = platform_mod.get_attn_backend_cls(None, None, _attn_cfg(use_mla=True, use_sparse=False))
    assert got == "vllm_ascend._310p.attention.attention_v1.AscendAttentionBackend310"


def test_non_310p_uses_mainline_backend(platform_mod):
    platform_mod.is_310p = lambda: False
    got = platform_mod.get_attn_backend_cls(None, None, _attn_cfg(use_mla=True, use_sparse=False))
    assert got == "vllm_ascend.attention.mla_v1.AscendMLABackend"


def _make_configs():
    parallel = types.SimpleNamespace(worker_cls="auto", all2all_backend=None)
    compilation = types.SimpleNamespace(custom_ops=None)
    vllm_cfg = types.SimpleNamespace(
        compilation_config=types.SimpleNamespace(pass_config=types.SimpleNamespace(enable_sp=True))
    )
    return vllm_cfg, parallel, compilation


def test_worker_cls_310p_entry_point(platform_mod):
    AscendDeviceType = platform_mod._AscendDeviceType
    platform_mod.is_310p = lambda: True
    platform_mod.get_ascend_device_type = lambda: AscendDeviceType._310P
    vllm_cfg, parallel, compilation = _make_configs()
    platform_mod.select_worker_cls_and_custom_ops(vllm_cfg, parallel, compilation, ascend_config=None)
    assert parallel.worker_cls == "vllm_ascend._310p.worker_310p.NPUWorker310"
    assert compilation.custom_ops is None  # 310P 不启用 custom_ops=['all']


def test_worker_cls_non_310p(platform_mod):
    AscendDeviceType = platform_mod._AscendDeviceType
    platform_mod.is_310p = lambda: False
    platform_mod.get_ascend_device_type = lambda: AscendDeviceType.A2
    vllm_cfg, parallel, compilation = _make_configs()
    platform_mod.select_worker_cls_and_custom_ops(vllm_cfg, parallel, compilation, ascend_config=None)
    assert parallel.worker_cls == "vllm_ascend.worker.worker.NPUWorker"
    assert compilation.custom_ops == ["all"]


# --------------------------------------------------------------------------- #
# 7. 重型 runner 子类：源码级结构断言（host 无法实例化运行时）
# --------------------------------------------------------------------------- #
def _classes(filename):
    tree = ast.parse((IMPL_DIR / filename).read_text())
    return {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}


def _bases(node):
    out = []
    for b in node.bases:
        if isinstance(b, ast.Name):
            out.append(b.id)
        elif isinstance(b, ast.Attribute):
            out.append(b.attr)
    return out


def _methods(node):
    return {n.name for n in node.body if isinstance(n, ast.FunctionDef)}


def test_model_runner_310_inheritance_and_overrides():
    cls = _classes("model_runner_310p.py")["NPUModelRunner310"]
    assert _bases(cls) == ["NPUModelRunner"]  # 再继承昇腾主栈一层
    m = _methods(cls)
    # 四大主线覆写点都在子类定义（非继承）
    for name in [
        "__init__",
        "_update_states",
        "_build_attn_state",
        "_prepare_inputs",
        "_init_kv_zero_meta",
        "initialize_kv_cache_tensors",
        "_allocate_kv_cache_tensors",
    ]:
        assert name in m, name


def test_model_runner_310_constants():
    src = (IMPL_DIR / "model_runner_310p.py").read_text()
    assert "_ATTENTION_BLOCK_SIZE_LIMIT = 128 * 128" in src
    assert "self._acl_format = ACL_FORMAT_FRACTAL_NZ" in src
    # 受限硬件能力边界：三处 raise
    assert src.count("is not supported for 310P.") == 3


def test_npu_input_batch_310_only_swaps_block_table():
    cls = _classes("npu_input_batch.py")["NPUInputBatch310"]
    assert _bases(cls) == ["NPUInputBatch"]
    assert _methods(cls) == {"__init__"}  # 唯一改动在 __init__
    src = (IMPL_DIR / "npu_input_batch.py").read_text()
    assert "self.block_table = MultiGroupBlockTable(" in src


def test_worker_310_init_device_swaps_runner():
    cls = _classes("worker_310p.py")["NPUWorker310"]
    assert _bases(cls) == ["NPUWorker"]
    m = _methods(cls)
    assert "init_device" in m and "save_sharded_state" in m
    src = (IMPL_DIR / "worker_310p.py").read_text()
    assert "self.model_runner = NPUModelRunner310(" in src
