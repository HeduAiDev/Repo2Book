"""ch21 测试脚手架：host 无 NPU/CANN，在 sys.modules 桩掉 torch_npu / vllm / vllm_ascend 的重运行时
依赖，并把 torch.ops._C_ascend / torch.ops.vllm 换成「记录调用」替身，再把（已减法的）
implementation/{abstract,device_op,sfa_v1,dsa_v1}.py 按规范模块名注册进去，让它们互相解析到精简版。

可在 host 验证、与真仓一致的纯 Python·形状级控制流：
  (1) 后端契约：AscendSFABackend/AscendDSABackend 的 get_name(HACK)/get_builder_cls/get_impl_cls；
      继承关系 SFA→MLA 基类、DSA→DSAAttentionImpl/AttentionMetadataBuilder（建在 MLA 之上 vs 自起一套）。
  (2) DeviceOperator 门面多态：get_device_adaptor 按 AscendDeviceType 选 Base/A5；reshape_and_cache 派发。
  (3) SFA 两段式：indexer_select_post_process→npu_lightning_indexer(sparse_count=2048,sparse_mode=3)；
      _execute_sparse_flash_attention_process→npu_sparse_flash_attention(sparse_indices=topk,sparse_mode=3)。
  (4) DSA Lightning Indexer：build_prefill/decode_metadata→npu_quant_lightning_indexer_metadata
      (sparse_count=index_topk=512,sparse_mode=3)；_indexer_qli→npu_quant_lightning_indexer(sparse_count=512)。
真实稀疏算子由记录替身承接——只验派发/入参/稀疏度数值，不真算（昇腾才有内核）。
"""
import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"

# 设备代际持有器：测试可改写以验证门面多态派发。
DEVICE_TYPE = {"value": "A2"}


def _load(filename, modname):
    spec = importlib.util.spec_from_file_location(modname, IMPL_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


class _Recorder:
    """记录 (name, args, kwargs)。"""

    def __init__(self):
        self.calls = []

    def names(self):
        return [c[0] for c in self.calls]

    def last(self, name):
        for n, a, kw in reversed(self.calls):
            if n == name:
                return a, kw
        raise KeyError(name)

    def count(self, name):
        return sum(1 for n, _, _ in self.calls if n == name)


class _CAscend(_Recorder):
    """torch.ops._C_ascend 替身：按算子名返回与真实签名一致的元数（不真算）。"""

    def __getattr__(self, name):
        def f(*a, **kw):
            self.calls.append((name, a, kw))
            if "metadata" in name:  # *_metadata：返回单个元数据张量
                return torch.zeros(1024, dtype=torch.int32)
            if name == "npu_quant_lightning_indexer":
                return torch.zeros(1, 512, dtype=torch.int32), None
            if name == "npu_lightning_indexer":
                return torch.zeros(1, 2048, dtype=torch.int32), None
            if name == "npu_sparse_flash_attention":
                return torch.zeros(1, 1, 8), None, None
            if name == "npu_sparse_attn_sharedkv":
                return (torch.zeros(1, 1, 8),)
            if name == "compressor":
                return torch.zeros(4, 1, 8)
            if name == "inplace_partial_rotary_mul":
                return None
            return (torch.zeros(1),)

        return f


class _VllmOps(_Recorder):
    def maybe_all_gather_and_maybe_unpad(self, hidden_states, need_gather):
        self.calls.append(("maybe_all_gather_and_maybe_unpad", (need_gather,), {}))
        return hidden_states


class _TorchNpu(_Recorder):
    """torch_npu 替身。"""

    def __getattr__(self, name):
        def f(*a, **kw):
            self.calls.append((name, a, kw))
            if name == "npu_dynamic_quant":
                x = a[0]
                return x.to(torch.int8) if hasattr(x, "to") else torch.zeros(1, dtype=torch.int8), torch.ones(
                    x.shape[0] if hasattr(x, "shape") else 1
                )
            if name == "npu_transpose_batchmatmul":
                return torch.zeros(a[0].shape[0], a[0].shape[1] if a[0].dim() > 1 else 1, 8)
            if name in ("npu_rotary_mul", "npu_interleave_rope"):
                return a[0]
            return None

        return f


@pytest.fixture
def env():
    saved_modules = dict(sys.modules)
    saved_ops_c = getattr(torch.ops, "_C_ascend", None)
    saved_ops_vllm = getattr(torch.ops, "vllm", None)

    rec_npu = _TorchNpu()
    rec_c = _CAscend()
    rec_vllm = _VllmOps()
    torch.ops._C_ascend = rec_c
    torch.ops.vllm = rec_vllm
    sys.modules["torch_npu"] = rec_npu

    added = []

    def mod(dotted):
        parts = dotted.split(".")
        for i in range(len(parts)):
            name = ".".join(parts[: i + 1])
            if name not in sys.modules:
                m = types.ModuleType(name)
                sys.modules[name] = m
                added.append(name)
                if i > 0:
                    setattr(sys.modules[".".join(parts[:i])], parts[i], m)
        return sys.modules[dotted]

    # ---- vllm.envs ---- #
    mod("vllm")

    class _Envs(types.ModuleType):
        def __getattr__(self, n):
            if n == "VLLM_USE_V2_MODEL_RUNNER":
                return False  # 让 get_name() 走 ASCEND_SFA/ASCEND_DSA 分支
            raise AttributeError(n)

    sys.modules["vllm.envs"] = _Envs("vllm.envs")
    setattr(sys.modules["vllm"], "envs", sys.modules["vllm.envs"])

    # ---- vllm.config ---- #
    cfg = mod("vllm.config")
    cfg.VllmConfig = type("VllmConfig", (), {})
    _vllm_config = types.SimpleNamespace(kv_transfer_config=None)
    cfg.get_current_vllm_config = lambda: _vllm_config

    # ---- vllm.distributed ---- #
    dist = mod("vllm.distributed")
    dist.get_tensor_model_parallel_world_size = lambda: 1
    dist.get_tp_group = lambda: types.SimpleNamespace(rank_in_group=0)

    # ---- vllm.forward_context ---- #
    fc = mod("vllm.forward_context")
    fc.get_forward_context = lambda: types.SimpleNamespace(num_tokens=4)

    # ---- vllm.model_executor.layers.attention.mla_attention：基类 MLACommonMetadataBuilder ---- #
    mla = mod("vllm.model_executor.layers.attention.mla_attention")

    class _MLACommonMetadataBuilder:
        def __class_getitem__(cls, item):
            return cls

        def __init__(self, kv_cache_spec, layer_names, vllm_config, device, metadata_cls, supports_dcp_with_varlen):
            self.kv_cache_spec = kv_cache_spec
            self.vllm_config = vllm_config
            self.model_config = vllm_config.model_config

    mla.MLACommonMetadataBuilder = _MLACommonMetadataBuilder

    # ---- vllm.v1.attention.backend：基类们 ---- #
    backend = mod("vllm.v1.attention.backend")
    backend.AttentionBackend = type("AttentionBackend", (), {})
    backend.MLAAttentionImpl = type("MLAAttentionImpl", (), {})
    backend.AttentionLayer = type("AttentionLayer", (), {})

    class _Generic:
        def __class_getitem__(cls, item):
            return cls

    backend.AttentionImpl = type("AttentionImpl", (_Generic,), {})
    backend.AttentionMetadataBuilder = type("AttentionMetadataBuilder", (_Generic,), {})
    backend.AttentionCGSupport = types.SimpleNamespace(UNIFORM_BATCH="UNIFORM_BATCH")

    mod("vllm.v1.kv_cache_interface")
    sys.modules["vllm.v1.kv_cache_interface"].AttentionSpec = type("AttentionSpec", (), {})
    sys.modules["vllm.v1.kv_cache_interface"].MLAAttentionSpec = type("MLAAttentionSpec", (), {})

    # ---- vllm_ascend.* ---- #
    ac = mod("vllm_ascend.ascend_config")
    ac.get_ascend_config = lambda: types.SimpleNamespace()

    av = mod("vllm_ascend.attention.attention_v1")
    av.AscendAttentionState = types.SimpleNamespace(ChunkedPrefill="ChunkedPrefill", DecodeOnly="DecodeOnly")

    util = mod("vllm_ascend.utils")
    util.AscendDeviceType = types.SimpleNamespace(A2="A2", A3="A3", A5="A5")
    util.get_ascend_device_type = lambda: getattr(util.AscendDeviceType, DEVICE_TYPE["value"])

    au = mod("vllm_ascend.attention.utils")
    au.AscendCommonAttentionMetadata = type("AscendCommonAttentionMetadata", (), {})

    def _split(common, decode_threshold=1):
        # 简化：按 query_lens 判 decode/prefill（精简版只需可观察拆分语义）
        return common.num_decodes, common.num_prefills, common.num_decode_tokens, common.num_prefill_tokens

    au.split_decodes_and_prefills = _split

    rope = mod("vllm_ascend.ops.rope_dsv4")

    def _cos_sin(positions, *a, **k):
        n = positions.shape[0] if hasattr(positions, "shape") else 4
        return torch.zeros(n, 2), torch.zeros(n, 2)

    rope.get_cos_and_sin_dsa = _cos_sin

    # ---- 加载精简版（顺序：abstract/device_op 先于 sfa/dsa）---- #
    mod("vllm_ascend.attention")
    mod("vllm_ascend.device")
    abstract = _load("abstract.py", "vllm_ascend.attention.abstract")
    device_op = _load("device_op.py", "vllm_ascend.device.device_op")
    sfa = _load("sfa_v1.py", "vllm_ascend.attention.sfa_v1")
    dsa = _load("dsa_v1.py", "vllm_ascend.attention.dsa_v1")

    bundle = types.SimpleNamespace(
        abstract=abstract, device_op=device_op, sfa=sfa, dsa=dsa,
        rec_npu=rec_npu, rec_c=rec_c, rec_vllm=rec_vllm, backend=backend,
    )
    yield bundle

    # cleanup
    DEVICE_TYPE["value"] = "A2"
    if saved_ops_c is not None:
        torch.ops._C_ascend = saved_ops_c
    if saved_ops_vllm is not None:
        torch.ops.vllm = saved_ops_vllm
    sys.modules.clear()
    sys.modules.update(saved_modules)
