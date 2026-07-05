"""ch20 测试脚手架：host 无 NPU/CANN，在 sys.modules 桩掉 torch_npu 与 vllm/vllm_ascend 的重运行时
依赖，再把（已减法的）implementation/mla_v1.py 按规范模块名注册进去，让它解析到精简版。

可在 host 验证、与真仓一致的纯 Python·形状级控制流：
  (1) 权重吸收 absorb：process_weights_after_loading 拆 kv_b_proj→W_UK/W_UV/W_UK_T 的形状代数；
      _q_proj_and_k_up_proj 的 transpose/bmm → ql_nope；_v_up_proj 的 latent→V。
  (2) 三段 metadata：build 用 split_decodes_and_prefills 切 decode/prefill，装配 prefill/decode 段；
      build_chunked_metadata 的 num_chunks/max_context_chunk 分块游标。
  (3) 前向派发：_mla_preprocess 一把 fused_qkv_a_proj 拆 q_c/kv_no_split，按 has_decode/has_prefill 双路；
      forward 按 decode/prefill 把结果写进 o_proj_input 不同切片再 o_proj。
  (4) chunked-context：_compute_prefill_context 逐 chunk 算注意力 + npu_attention_update 在线合并。
  (5) exec_kv_decode/prefill：npu_kv_rmsnorm_rope_cache 的 is_output_kv 差异与返回值选择。
真实 torch_npu 算子由「记录调用」替身承接——只验入参/分流/形状，不真算（昇腾才有内核）。
"""
import importlib.util
import sys
import types
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import pytest
import torch
from pathlib import Path

IMPL_DIR = Path(__file__).resolve().parent.parent / "implementation"


def _load(filename, modname):
    spec = importlib.util.spec_from_file_location(modname, IMPL_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
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
    """记录每个 torch_npu 算子调用的 (name, args, kwargs)，验证分流/入参/形状。算子不真算。"""

    def __init__(self):
        self.calls = []

    def _record(self, name, args=(), kw=None):
        self.calls.append((name, args, kw or {}))

    def names(self):
        return [c[0] for c in self.calls]

    def last(self, name):
        for n, a, kw in reversed(self.calls):
            if n == name:
                return a, kw
        raise KeyError(name)

    # --- 权重排布 cast：identity，保留真实张量让 .T/view/split/permute 的形状代数可跑 ---
    def npu_format_cast(self, tensor, fmt):
        self._record("npu_format_cast", (tensor, fmt))
        return tensor

    # --- _v_up_proj：(N,B,L) x (N,L,V) --perm_y(1,0,2)--> (B,N,V)（真算 bmm 保证形状精确）---
    def npu_transpose_batchmatmul(self, x, weight, perm_y=None):
        self._record("npu_transpose_batchmatmul", (x, weight), {"perm_y": perm_y})
        out = torch.bmm(x, weight)  # (N, B, V)
        if perm_y is not None:
            out = out.permute(*perm_y)  # (B, N, V)
        return out

    # --- RMSNorm+RoPE+写cache 三合一：返回 4 个可区分张量（填值 1/2/3/4，形状取自 KV cache），
    #     decode 取[0:2]（写进 cache 的隐向量）、prefill 取[2:4]（未量化输出 KV）---
    def npu_kv_rmsnorm_rope_cache(self, kv_no_split, weight, cos, sin, slots, c1, c0, **kw):
        self._record("npu_kv_rmsnorm_rope_cache", (kv_no_split,), kw)
        B = kv_no_split.shape[0]
        N, L = c0.shape[2], c0.shape[3]  # kv_cache[0]: (..., num_kv_heads, kv_lora_rank)
        R = c1.shape[3]  # kv_cache[1]: (..., qk_rope_head_dim)
        return (
            torch.full((B, N, R), 1.0),  # k_pe_cache
            torch.full((B, N, L), 2.0),  # k_nope_cache（缓存隐向量 kv_c）
            torch.full((B, N, R), 3.0),  # k_pe_out
            torch.full((B, N, L), 4.0),  # k_nope_out（供 prefill 显式解压）
        )

    # --- prefill/chunked 注意力（TND）：返回 (out, lse) ---
    def npu_fused_infer_attention_score(self, query, key, value, **kw):
        self._record("npu_fused_infer_attention_score", (query, key, value), kw)
        num_heads = kw.get("num_heads", 1)
        t = query.shape[0]
        out = torch.zeros(t, num_heads, value.shape[-1])
        lse = torch.zeros(num_heads, t)
        return out, lse

    # --- decode 注意力（K=V=隐向量）：返回 (out, None) ---
    def npu_fused_infer_attention_score_v2(self, q_nope, k_nope, k_nope2, **kw):
        self._record("npu_fused_infer_attention_score_v2", (q_nope, k_nope, k_nope2), kw)
        n = kw.get("num_query_heads", 1)
        b = q_nope.shape[0]
        # 输出 latent 布局 (N, B, 1, L)，喂 _v_up_proj 的 view(N,-1,L)
        return torch.zeros(n, b, 1, k_nope.shape[-1]), None

    # --- 在线 softmax 合并：把 lse_list/out_list 合并成 output_final ---
    def npu_attention_update(self, lse_list, out_list, dim):
        self._record("npu_attention_update", (lse_list, out_list, dim))
        return out_list[0], None

    # --- q_pe 加 RoPE：identity（形状不变）---
    def npu_interleave_rope(self, x, cos, sin):
        self._record("npu_interleave_rope", (x,))
        return x


class _Linear:
    """投影层替身：callable → [out]，out = zeros(B, out_dim)，记录调用名。"""

    def __init__(self, name, out_dim, rec, quant_method=None):
        self.name = name
        self.out_dim = out_dim
        self.rec = rec
        self.quant_method = quant_method
        self.weight = types.SimpleNamespace(data=torch.zeros(out_dim, out_dim))

    def __call__(self, x, **kw):
        self.rec._record(f"proj:{self.name}", (x,), kw)
        return [torch.zeros(x.shape[0], self.out_dim)]


@pytest.fixture
def env():
    stubs = _Stubs()
    rec = NpuRecorder()

    knobs = types.SimpleNamespace(
        use_v2_model_runner=True,
        prefill_cp_size=1,
        decode_cp_size=1,
        enable_kv_nz=False,
        chunked_prefill_enabled=False,
        chunked_prefill_workspace_size=512,
        block_size=8,
        head_size=16,
        rope_dim=2,
        speculative_config=None,
    )

    # 让 NPU-only 的张量方法在 host 上变 no-op（pin_memory/.npu()），并使 zeros(pin_memory=True) 可跑
    orig_npu = getattr(torch.Tensor, "npu", None)
    orig_pin = torch.Tensor.pin_memory
    orig_zeros = torch.zeros
    torch.Tensor.npu = lambda self, *a, **k: self
    torch.Tensor.pin_memory = lambda self, *a, **k: self

    def _zeros(*a, **k):
        k.pop("pin_memory", None)
        return orig_zeros(*a, **k)

    torch.zeros = _zeros

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

    # ---- vllm.model_executor.layers.attention.mla_attention：基类 MLACommonMetadataBuilder ---- #
    mla_attn = stubs.mod("vllm.model_executor.layers.attention.mla_attention")

    class _MLACommonMetadataBuilder:
        def __class_getitem__(cls, item):
            return cls

        def __init__(self, kv_cache_spec, layer_names, vllm_config, device, metadata_cls, supports_dcp_with_varlen):
            # 基类设置 build/__init__ 实际读到的字段
            self.model_config = vllm_config.model_config
            self.device = device
            self.metadata_cls = metadata_cls
            self.chunked_prefill_workspace_size = knobs.chunked_prefill_workspace_size
            self.chunked_prefill_workspace = torch.zeros(knobs.chunked_prefill_workspace_size)

    mla_attn.MLACommonMetadataBuilder = _MLACommonMetadataBuilder

    # ---- vllm.model_executor.layers.linear：UnquantizedLinearMethod ---- #
    linear = stubs.mod("vllm.model_executor.layers.linear")
    linear.UnquantizedLinearMethod = type("UnquantizedLinearMethod", (), {})

    # ---- vllm.utils.math_utils ---- #
    mu = stubs.mod("vllm.utils.math_utils")
    mu.cdiv = lambda a, b: -(-a // b)
    mu.round_down = lambda x, n: (x // n) * n

    # ---- vllm.v1.attention.backend ---- #
    backend = stubs.mod("vllm.v1.attention.backend")
    backend.AttentionBackend = type("AttentionBackend", (), {})
    backend.MLAAttentionImpl = type("MLAAttentionImpl", (), {})
    backend.AttentionCGSupport = types.SimpleNamespace(UNIFORM_BATCH="UNIFORM_BATCH")

    # ---- vllm.v1.kv_cache_interface ---- #
    kvci = stubs.mod("vllm.v1.kv_cache_interface")
    kvci.AttentionSpec = type("AttentionSpec", (), {})
    kvci.MLAAttentionSpec = type("MLAAttentionSpec", (), {})

    # ---- vllm_ascend.* package-level stubs ---- #
    stubs.mod("vllm_ascend")
    stubs.mod("vllm_ascend.attention")
    stubs.mod("vllm_ascend.device")
    stubs.mod("vllm_ascend.ops")

    ac = stubs.mod("vllm_ascend.ascend_config")
    ac.get_ascend_config = lambda: types.SimpleNamespace(enable_kv_nz=knobs.enable_kv_nz)

    afc = stubs.mod("vllm_ascend.ascend_forward_context")
    afc._EXTRA_CTX = types.SimpleNamespace(num_tokens=0)

    amask = stubs.mod("vllm_ascend.attention.attention_mask")

    class _AttentionMaskBuilder:
        def __init__(self, device):
            self.device = device

        def get_splitfuse_attn_mask(self):
            return torch.zeros(4, 4)

    amask.AttentionMaskBuilder = _AttentionMaskBuilder

    av1 = stubs.mod("vllm_ascend.attention.attention_v1")
    import enum

    class _AscendAttentionState(enum.Enum):
        PrefillNoCache = 0
        PrefillCacheHit = 1
        DecodeOnly = 2
        ChunkedPrefill = 3
        SpecDecoding = 4

    av1.AscendAttentionState = _AscendAttentionState

    # ---- vllm_ascend.attention.utils：AscendCommonAttentionMetadata / split / enable_cp ---- #
    autil = stubs.mod("vllm_ascend.attention.utils")

    @dataclass
    class AscendCommonAttentionMetadata:
        num_reqs: int = 0
        num_actual_tokens: int = 0
        num_input_tokens: int = 0
        max_query_len: int = 0
        graph_pad_size: int = 0
        query_start_loc: torch.Tensor = None
        query_start_loc_cpu: torch.Tensor = None
        positions: torch.Tensor = None
        block_table_tensor: torch.Tensor = None
        slot_mapping: torch.Tensor = None
        seq_lens: torch.Tensor = None
        seq_lens_cpu: torch.Tensor = None
        _seq_lens_cpu: torch.Tensor = None
        attn_state: Any = None

    def split_decodes_and_prefills(common_attn_metadata, decode_threshold=1):
        max_query_len = common_attn_metadata.max_query_len
        num_reqs = common_attn_metadata.num_reqs
        num_tokens = common_attn_metadata.num_actual_tokens
        query_start_loc = common_attn_metadata.query_start_loc_cpu
        if max_query_len <= decode_threshold:
            return num_reqs, 0, num_tokens, 0
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        is_prefill = query_lens > decode_threshold
        if not torch.any(is_prefill):
            return num_reqs, 0, num_tokens, 0
        first_prefill = is_prefill.int().argmax(dim=-1).item()
        num_decodes = first_prefill
        num_prefills = num_reqs - num_decodes
        num_decode_tokens = query_start_loc[first_prefill].item()
        num_prefill_tokens = num_tokens - num_decode_tokens
        return (num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens)

    @lru_cache(maxsize=1)
    def enable_cp():
        cfg = knobs.vllm_config.parallel_config
        return cfg.prefill_context_parallel_size > 1 or cfg.decode_context_parallel_size > 1

    autil.AscendCommonAttentionMetadata = AscendCommonAttentionMetadata
    autil.split_decodes_and_prefills = split_decodes_and_prefills
    autil.enable_cp = enable_cp
    autil.ascend_chunked_prefill_workspace_size = lambda vc: knobs.chunked_prefill_workspace_size

    # ---- vllm_ascend.device.device_op：DeviceOperator ---- #
    dop = stubs.mod("vllm_ascend.device.device_op")

    class _DeviceOperator:
        @staticmethod
        def kv_cache_load(cache_kv_c, cache_k_pe, block_table, ctx_len, start, key=None, value=None):
            rec._record("kv_cache_load", (), {})

    dop.DeviceOperator = _DeviceOperator

    # ---- vllm_ascend.ops.rotary_embedding：get_cos_and_sin_mla ---- #
    rope = stubs.mod("vllm_ascend.ops.rotary_embedding")

    def get_cos_and_sin_mla(positions, use_cache=False):
        n = positions.shape[0]
        return torch.zeros(n, knobs.rope_dim), torch.zeros(n, knobs.rope_dim)

    rope.get_cos_and_sin_mla = get_cos_and_sin_mla

    # ---- vllm_ascend.utils ---- #
    uutil = stubs.mod("vllm_ascend.utils")
    uutil.ACL_FORMAT_FRACTAL_ND = 2
    uutil.maybe_trans_nz = lambda w: w  # identity（真仓转 FRACTAL_NZ(29) 喂 cube）

    # CP 版 impl/builder（f-收口真分支的延迟 import 目标）
    cp = stubs.mod("vllm_ascend.attention.context_parallel.mla_cp")
    cp.AscendMlaCPImpl = type("AscendMlaCPImpl", (), {})
    cp.AscendMlaCPMetadataBuilder = type("AscendMlaCPMetadataBuilder", (), {})

    # ---- vllm_config 替身 ---- #
    knobs.vllm_config = types.SimpleNamespace(
        speculative_config=knobs.speculative_config,
        scheduler_config=types.SimpleNamespace(enable_chunked_prefill=knobs.chunked_prefill_enabled),
        cache_config=types.SimpleNamespace(block_size=knobs.block_size),
        parallel_config=types.SimpleNamespace(
            prefill_context_parallel_size=knobs.prefill_cp_size,
            decode_context_parallel_size=knobs.decode_cp_size,
        ),
        model_config=types.SimpleNamespace(
            max_model_len=2048,
            get_head_size=lambda: knobs.head_size,
            hf_text_config=types.SimpleNamespace(qk_rope_head_dim=knobs.rope_dim),
        ),
    )

    # ---- 加载精简版 ---- #
    mla_v1 = _load("mla_v1.py", "vllm_ascend.attention.mla_v1")

    mods = types.SimpleNamespace(
        mla_v1=mla_v1,
        rec=rec,
        state=_AscendAttentionState,
        UnquantizedLinearMethod=linear.UnquantizedLinearMethod,
        common_meta_cls=AscendCommonAttentionMetadata,
        _Linear=_Linear,
    )
    try:
        yield mods, knobs
    finally:
        enable_cp.cache_clear()
        if orig_npu is not None:
            torch.Tensor.npu = orig_npu
        else:
            try:
                del torch.Tensor.npu
            except AttributeError:
                pass
        torch.Tensor.pin_memory = orig_pin
        torch.zeros = orig_zeros
        stubs.cleanup()
        sys.modules.pop("vllm.envs", None)
        sys.modules.pop("vllm_ascend.attention.mla_v1", None)


def make_mla_impl(mods, knobs, *, num_heads=2, num_kv_heads=1, kv_lora_rank=4,
                  qk_nope_head_dim=3, qk_rope_head_dim=2, v_head_dim=3, q_lora_rank=5):
    """构造 AscendMLAImpl，投影层用 _Linear 替身。"""
    rec = mods.rec
    qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
    kv_b_proj = mods._Linear("kv_b_proj", num_heads * (qk_nope_head_dim + v_head_dim), rec,
                             quant_method=mods.UnquantizedLinearMethod())
    # kv_b_proj.weight.data 形状 = (N*(P+V), L)，process_weights 取 .T
    kv_b_proj.weight = types.SimpleNamespace(
        data=torch.randn(num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank)
    )
    kwargs = dict(
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        rotary_emb=None,
        fused_qkv_a_proj=mods._Linear("fused_qkv_a_proj", q_lora_rank + kv_lora_rank + qk_rope_head_dim, rec),
        q_b_proj=mods._Linear("q_b_proj", num_heads * qk_head_dim, rec),
        kv_b_proj=kv_b_proj,
        o_proj=mods._Linear("o_proj", num_heads * v_head_dim, rec),
        kv_a_proj_with_mqa=None,
        kv_a_layernorm=types.SimpleNamespace(weight=torch.zeros(kv_lora_rank), variance_epsilon=1e-6),
        q_a_layernorm=(lambda x: x),
    )
    impl = mods.mla_v1.AscendMLAImpl(
        num_heads=num_heads, head_size=16, scale=0.125, num_kv_heads=num_kv_heads,
        alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
        logits_soft_cap=None, attn_type="decoder", kv_sharing_target_layer_name=None,
        **kwargs,
    )
    return impl


def make_builder(mods, knobs):
    return mods.mla_v1.AscendMLAMetadataBuilder(
        kv_cache_spec=None,
        layer_names=["layer.0"],
        vllm_config=knobs.vllm_config,
        device=torch.device("cpu"),
    )


def make_common_metadata(mods, *, query_lens, attn_state, num_blocks=8):
    num_reqs = len(query_lens)
    num_tokens = int(sum(query_lens))
    qsl = torch.tensor([0] + torch.tensor(query_lens).cumsum(0).tolist(), dtype=torch.int64)
    seq_lens = torch.tensor([max(int(q), 1) for q in query_lens], dtype=torch.int64)
    block_table = torch.arange(num_reqs * 4, dtype=torch.int32).reshape(num_reqs, 4)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64)
    positions = torch.arange(num_tokens, dtype=torch.int64)
    return mods.common_meta_cls(
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        num_input_tokens=num_tokens,
        max_query_len=int(max(query_lens)),
        graph_pad_size=0,
        query_start_loc=qsl,
        query_start_loc_cpu=qsl,
        positions=positions,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        _seq_lens_cpu=seq_lens,
        attn_state=attn_state,
    )
