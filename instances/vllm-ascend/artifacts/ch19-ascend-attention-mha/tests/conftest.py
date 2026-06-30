"""ch19 测试脚手架：host 无 NPU/CANN，在 sys.modules 桩掉 torch_npu 与 vllm/vllm_ascend 的重运行时
依赖，再把（已减法的）implementation/ 模块按**规范模块名**注册进去，让它们彼此 import 解析到精简版。

可在 host 验证、与真仓一致的纯 Python 控制流：
  (1) AscendAttentionState 五态；
  (2) split_decodes_and_prefills 在已重排 batch 上找 decode|prefill 分界；
  (3) build 装配 slot_mapping/block_table/seq_lens/attn_state → AscendMetadata；
  (4) forward_impl 按 attn_state 选 paged vs fused；_get_fia_params 按五态整理 KV；
  (5) reshape_and_cache → DeviceOperator → torch_npu._npu_reshape_and_cache；
  (6) get_impl_cls/get_builder_cls 按 enable_cp 分流（f7）；workspace 预取节拍。
真实 torch_npu 算子由「记录调用」替身承接——只验入参/分流，不真算（昇腾才有内核）。
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


def _load(filename, modname):
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


class NpuRecorder:
    """记录每个 torch_npu 算子调用的 (name, kwargs)，验证分流与入参。算子不真算。"""

    def __init__(self):
        self.calls = []

    def _record(self, name, **kw):
        self.calls.append((name, kw))

    def names(self):
        return [c[0] for c in self.calls]

    def last(self, name):
        for n, kw in reversed(self.calls):
            if n == name:
                return kw
        raise KeyError(name)

    # --- 写 KV ---
    def _npu_reshape_and_cache(self, **kw):
        self._record("_npu_reshape_and_cache", **kw)

    # --- decode 分页注意力 ---
    def _npu_paged_attention(self, **kw):
        self._record("_npu_paged_attention", **kw)
        return None

    def _npu_paged_attention_get_workspace(self, **kw):
        self._record("_npu_paged_attention_get_workspace", **kw)
        return torch.zeros(1)

    # --- prefill/混批 融合注意力 ---
    def npu_fused_infer_attention_score(self, query=None, **kw):
        self._record("npu_fused_infer_attention_score", query=query, **kw)
        flat = torch.zeros(query.shape[0], query.shape[1] * query.shape[2])
        return flat, None

    def npu_fused_infer_attention_score_v2(self, *a, **kw):
        self._record("npu_fused_infer_attention_score_v2", **kw)
        q = a[0]
        return torch.zeros(q.shape[0], q.shape[1] * q.shape[2]), None


def _make_config(knobs):
    """构造一个可控的 vllm_config 替身（speculative / cudagraph / parallel）。"""
    return types.SimpleNamespace(
        speculative_config=knobs.speculative_config,
        compilation_config=types.SimpleNamespace(cudagraph_mode=knobs.cudagraph_mode),
        parallel_config=types.SimpleNamespace(
            prefill_context_parallel_size=knobs.prefill_cp_size,
            decode_context_parallel_size=knobs.decode_cp_size,
        ),
        model_config=types.SimpleNamespace(max_model_len=2048, runner_type="generate"),
        scheduler_config=types.SimpleNamespace(enable_chunked_prefill=True),
        quant_config=None,
        kv_transfer_config=None,
    )


@pytest.fixture
def env():
    stubs = _Stubs()
    rec = NpuRecorder()

    knobs = types.SimpleNamespace(
        use_v2_model_runner=True,
        capturing=False,
        speculative_config=None,
        cudagraph_mode="FULL_DECODE_ONLY",
        pa_shape_list=[1, 2, 4, 8],
        device_type="A2",
        prefill_cp_size=1,
        decode_cp_size=1,
    )

    # ---- torch_npu：记录调用替身 ---- #
    sys.modules["torch_npu"] = rec
    stubs.added.append("torch_npu")

    # ---- vllm.envs ---- #
    stubs.mod("vllm")

    class _EnvsModule(types.ModuleType):
        def __getattr__(self, name):
            if name == "VLLM_USE_V2_MODEL_RUNNER":
                return knobs.use_v2_model_runner
            raise AttributeError(name)

    envs = _EnvsModule("vllm.envs")
    sys.modules["vllm.envs"] = envs
    setattr(sys.modules["vllm"], "envs", envs)
    stubs.added.append("vllm.envs")

    # ---- vllm.config ---- #
    cfg = stubs.mod("vllm.config")
    cfg.VllmConfig = type("VllmConfig", (), {})
    cfg.get_current_vllm_config = lambda: knobs.vllm_config
    compilation = stubs.mod("vllm.config.compilation")
    compilation.CUDAGraphMode = types.SimpleNamespace(FULL_DECODE_ONLY="FULL_DECODE_ONLY")

    # ---- vllm.utils.math_utils.cdiv ---- #
    mu = stubs.mod("vllm.utils.math_utils")
    mu.cdiv = lambda a, b: -(-a // b)

    # ---- vllm.v1.attention.backend ---- #
    backend = stubs.mod("vllm.v1.attention.backend")
    backend.AttentionBackend = type("AttentionBackend", (), {})
    backend.AttentionImpl = type("AttentionImpl", (), {})
    backend.AttentionLayer = type("AttentionLayer", (), {})
    backend.AttentionCGSupport = types.SimpleNamespace(ALWAYS="ALWAYS")
    backend.AttentionType = types.SimpleNamespace(ENCODER_DECODER="encoder_decoder", ENCODER_ONLY="encoder_only")

    class _AMB:
        def __class_getitem__(cls, item):
            return cls

        def __init__(self, *a, **k):
            pass

    backend.AttentionMetadataBuilder = _AMB

    # ---- vllm.v1.attention.backends.registry ---- #
    registry = stubs.mod("vllm.v1.attention.backends.registry")
    registry.AttentionBackendEnum = types.SimpleNamespace(CUSTOM="CUSTOM")
    registry.register_backend = lambda *a, **k: (lambda cls: cls)

    # ---- vllm.v1.kv_cache_interface ---- #
    kvci = stubs.mod("vllm.v1.kv_cache_interface")
    kvci.AttentionSpec = type("AttentionSpec", (), {})

    # ---- vllm_ascend package-level stubs ---- #
    stubs.mod("vllm_ascend")
    stubs.mod("vllm_ascend.attention")
    stubs.mod("vllm_ascend.device")
    uutil = stubs.mod("vllm_ascend.utils")
    uutil.singleton = lambda cls: cls  # 测试用 identity（不影响 mask 正确性）
    uutil.AscendDeviceType = types.SimpleNamespace(A5="A5", A2="A2")
    uutil.get_ascend_device_type = lambda: getattr(uutil.AscendDeviceType, knobs.device_type)
    uutil.get_ascend_config = lambda: types.SimpleNamespace(pa_shape_list=knobs.pa_shape_list)

    plat = stubs.mod("vllm_ascend.platform")
    plat.ModelConfig = type("ModelConfig", (), {})

    afc = stubs.mod("vllm_ascend.ascend_forward_context")
    afc._EXTRA_CTX = types.SimpleNamespace(capturing=False)

    acl = stubs.mod("vllm_ascend.compilation.acl_graph")
    _graph_params = types.SimpleNamespace(workspaces={})
    acl.get_graph_params = lambda: _graph_params
    acl.update_graph_params_workspaces = lambda n, ws: _graph_params.workspaces.__setitem__(n, ws)

    # CP 版 impl/builder（f7 真分支的延迟 import 目标）
    cp = stubs.mod("vllm_ascend.attention.context_parallel.attention_cp")
    cp.AscendAttentionCPImpl = type("AscendAttentionCPImpl", (), {})
    cp.AscendAttentionCPMetadataBuilder = type("AscendAttentionCPMetadataBuilder", (), {})

    knobs.vllm_config = _make_config(knobs)

    # ---- 按依赖顺序加载精简版 ---- #
    device_op = _load("device_op.py", "vllm_ascend.device.device_op")
    attention_mask = _load("attention_mask.py", "vllm_ascend.attention.attention_mask")
    utils = _load("utils.py", "vllm_ascend.attention.utils")
    attention_v1 = _load("attention_v1.py", "vllm_ascend.attention.attention_v1")

    afc._EXTRA_CTX.capturing = knobs.capturing

    mods = types.SimpleNamespace(
        device_op=device_op,
        attention_mask=attention_mask,
        utils=utils,
        attention_v1=attention_v1,
        rec=rec,
        graph_params=_graph_params,
        forward_ctx=afc._EXTRA_CTX,
    )
    try:
        yield mods, knobs
    finally:
        utils.enable_cp.cache_clear()
        stubs.cleanup()
        for n in (
            "vllm.envs",
            "vllm.config",
            "vllm.config.compilation",
            "vllm.utils.math_utils",
            "vllm.v1.attention.backend",
            "vllm.v1.attention.backends.registry",
            "vllm.v1.kv_cache_interface",
            "vllm_ascend.device.device_op",
            "vllm_ascend.attention.attention_mask",
            "vllm_ascend.attention.utils",
            "vllm_ascend.attention.attention_v1",
            "vllm_ascend.ascend_forward_context",
            "vllm_ascend.compilation.acl_graph",
            "vllm_ascend.platform",
            "vllm_ascend.utils",
        ):
            sys.modules.pop(n, None)


def make_common_metadata(mods, *, query_lens, attn_state, num_blocks=8, block_size=128, causal=True):
    """构造一个 AscendCommonAttentionMetadata（已重排：decode 段在前）作 build 输入。"""
    utils = mods.utils
    num_reqs = len(query_lens)
    num_tokens = int(sum(query_lens))
    qsl = torch.tensor([0] + list(torch.tensor(query_lens).cumsum(0).tolist()), dtype=torch.int64)
    seq_lens = torch.tensor([max(int(q), 1) for q in query_lens], dtype=torch.int64)
    block_table = torch.arange(num_reqs * 4, dtype=torch.int32).reshape(num_reqs, 4)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
    return utils.AscendCommonAttentionMetadata(
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=int(max(query_lens)),
        query_start_loc_cpu=qsl,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        _seq_lens_cpu=seq_lens,
        attn_state=attn_state,
        causal=causal,
    )
