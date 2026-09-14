# ch26《DeepSeek 索引器 NSA→DSA》精简版测试 —— TDD（先于实现写就）
#
# 目标代码仓真实行为（vLLM v0.27.1, 6e448d0ea）的可观察复现：
#   - 打分数学 I_{t,s}=Σ_j w_j·ReLU(q_j·k_s)（DSA Eq.(1)）经 FP8 量化路径仍复现
#   - prefill/decode 两路打分 + top-k 选块 = 暴力 top-k（因果边界/降序/tie 取小）
#   - topk_indices_buffer 的 -1 哨兵 / 行=本拍 query token / 纯副作用
#   - 双预算切块 / decode 展开（flatten 变长注释例 / native 2D）/ 压缩坐标
#   - V4：compressor 数学（softmax 门控压缩→RMSNorm→RoPE→FP8）/ 短上下文全选 /
#     一核双源（SWA ∪ top-k）== NSA 三支路回归的数值证明
#   - index 空间换算 / FP8 wk 权重融合加载 / 接线顺序（indexer 先于 mla_attn）
#
# 运行：cd instances/vllm/artifacts-v3/ch26-deepseek-indexer-nsa-dsa
#       python -m pytest tests/ -q
import sys
from pathlib import Path

import torch

_IMPL = Path(__file__).resolve().parents[1] / "implementation"
sys.path.insert(0, str(_IMPL))

from transformers import DeepseekV2Config  # noqa: E402

FP8_MAX = 448.0
DEV = torch.device("cpu")
RADIX_WS = 1024 * 1024


# ─────────────────────────── 配置工厂 ───────────────────────────
def make_v32_hf_config(**over):
    """DSV3.2 形状的 HF config（index_* 头表 = 独立小头的字面证据）。"""
    cfg = DeepseekV2Config(
        hidden_size=7168,
        num_hidden_layers=61,
        num_attention_heads=128,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=1e-6,
        max_position_embeddings=163840,
        rope_parameters={"rope_type": "default"},
        # DSA 独立小头头表
        index_topk=2048,
        index_n_heads=64,
        index_head_dim=128,
        indexer_rope_interleave=True,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def make_small_hf_config(**over):
    """数值测试用小配置（几何不变：head_dim=128 / rope=64 / quant_block=128）。"""
    cfg = make_v32_hf_config(
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=8,
        q_lora_rank=32,
        index_topk=8,
        index_n_heads=2,
    )
    cfg.max_position_embeddings = 4096
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def make_v4_hf_config(**over):
    """V4 形状 HF config 载体（SimpleNamespace——head_dim 等字段在
    transformers 的 DeepseekV2Config 里是派生属性会被覆盖，V4 真实
    config 类不在本 transformers 版本内）。"""
    from types import SimpleNamespace

    cfg = SimpleNamespace(
        hidden_size=64, num_hidden_layers=3, num_attention_heads=8,
        q_lora_rank=32, kv_lora_rank=512, qk_nope_head_dim=448,
        qk_rope_head_dim=64, v_head_dim=512, head_dim=512,
        o_lora_rank=16, o_groups=2, sliding_window=128,
        rms_norm_eps=1e-6, max_position_embeddings=4096,
        rope_parameters={"rope_type": "default"},
        index_topk=8, index_n_heads=2, index_head_dim=128,
        indexer_rope_interleave=True, compress_ratios=[1, 4, 128],
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def make_vllm_config(hf_config, max_model_len=None, block_size=64,
                     max_num_batched_tokens=4096, max_num_seqs=8,
                     cache_dtype="auto", spec_tokens=None,
                     backend="FLASHMLA_SPARSE"):
    from vllm.config import (AttentionConfig, CacheConfig, CompilationConfig,
                             ModelConfig, ParallelConfig, SchedulerConfig,
                             set_current_vllm_config)

    class _Spec:
        num_speculative_tokens = spec_tokens or 0

    class _VllmCfg:
        def __init__(self):
            self.model_config = ModelConfig(
                hf_config, max_model_len or hf_config.max_position_embeddings)
            self.cache_config = CacheConfig(block_size=block_size,
                                            cache_dtype=cache_dtype)
            self.scheduler_config = SchedulerConfig(
                max_num_batched_tokens=max_num_batched_tokens,
                max_num_seqs=max_num_seqs)
            self.parallel_config = ParallelConfig()
            self.compilation_config = CompilationConfig()
            self.attention_config = AttentionConfig(backend=backend)
            self.quant_config = None
            self.speculative_config = _Spec() if spec_tokens else None
            self.kv_transfer_config = None

    cfg = _VllmCfg()
    set_current_vllm_config(cfg)
    return cfg


# ─────────────────────── 独立参考数学（测试自有真相源） ───────────────────────
def ref_group_quant_ue8m0(x: torch.Tensor):
    """per-token-group FP8 量化参考（128 一组、ue8m0 scale）——fused 核同式。"""
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2(torch.clamp(amax, min=1e-10) / FP8_MAX)))
    q = torch.clamp(x / scale, -FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q, scale.squeeze(-1).float()


def ref_dequant(q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return q.float() * scale.float().unsqueeze(-1)


def ref_interleave_rope(x: torch.Tensor, pos: torch.Tensor, base=10000.0):
    """GPT-J interleaved RoPE 参考（偶奇对 (x0,x1)：x0'=x0c-x1s, x1'=x1c+x0s）。"""
    half = x.shape[-1] // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, dtype=torch.float32) / half))
    ang = pos.float().unsqueeze(-1) * inv_freq          # [T, half]
    cos, sin = ang.cos(), ang.sin()
    while cos.dim() < x.dim():                          # 广播到头维
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
    x0, x1 = x[..., 0::2].float(), x[..., 1::2].float()
    r0 = x0 * cos - x1 * sin
    r1 = x1 * cos + x0 * sin
    out = torch.empty_like(x)
    out[..., 0::2] = r0
    out[..., 1::2] = r1
    return out


def ref_dsa_logits(q_deq: torch.Tensor, k_deq: torch.Tensor,
                   weights: torch.Tensor) -> torch.Tensor:
    """I_{t,s} = Σ_j w_{t,j}·ReLU(q_{t,j}·k_s) —— DSA Eq.(1) 参考实现。"""
    dots = torch.einsum("mhd,nd->mhn", q_deq, k_deq)
    return torch.einsum("mhn,mh->mn", torch.relu(dots), weights)


def ref_topk_desc(vals, k: int, base: int = 0):
    """降序 top-k，tie 取小 index（与 CUDA 核插入排序 tie-break 同构）。"""
    order = sorted(range(len(vals)), key=lambda i: (-vals[i], i))
    sel = order[:k]
    return [base + i for i in sel] + [-1] * max(0, k - len(sel))


def ref_sparse_attn(q, kv, indices, lengths, scale, head_dim_v=512):
    """只对选中条目算注意力（O(Lk)）的参考：q [T,H,D], kv [N,1,D]。"""
    T, H, D = q.shape
    out = torch.zeros(T, H, head_dim_v)
    for t in range(T):
        ids = [i for i in indices[t].tolist()[: lengths[t]] if i >= 0]
        if not ids:
            continue
        keys = kv[ids, 0, :].float()
        scores = (q[t].float() @ keys.T) * scale
        probs = torch.softmax(scores, dim=-1)
        out[t] = probs @ keys[:, :head_dim_v]
    return out


def make_block_table(seq_lens, block_size, first_block=0):
    """分页块表：请求 b 第 i 块 → 物理块号（顺序分配）。"""
    max_blocks = max(1, max((s + block_size - 1) // block_size for s in seq_lens))
    bt = torch.zeros(len(seq_lens), max_blocks, dtype=torch.int32)
    nxt = first_block
    for b, s in enumerate(seq_lens):
        for i in range((s + block_size - 1) // block_size):
            bt[b, i] = nxt
            nxt += 1
    return bt, nxt


class _CommonMeta:
    """CommonAttentionMetadata 测试替身（字段面 = builder 消费字段）。"""

    def __init__(self, num_reqs, num_actual_tokens, query_start_loc, seq_lens,
                 max_query_len, max_seq_len, block_table, slot_mapping):
        self.num_reqs = num_reqs
        self.num_actual_tokens = num_actual_tokens
        self.query_start_loc = query_start_loc.to(DEV)
        self.query_start_loc_cpu = query_start_loc
        self.seq_lens = seq_lens.to(DEV)
        self.max_query_len = max_query_len
        self.max_seq_len = max_seq_len
        self.block_table_tensor = block_table.to(DEV)
        self.slot_mapping = slot_mapping.to(DEV)
        self.seq_lens_cpu_upper_bound = seq_lens.clone()
        self.dcp_local_seq_lens = None
        self.is_prefilling = None

    def token_to_req_indices(self, buffer):
        import numpy as np
        starts = np.asarray(self.query_start_loc_cpu, dtype=np.int32)
        lens = np.diff(starts)
        return torch.from_numpy(np.repeat(
            np.arange(lens.shape[0], dtype=np.int32), lens)).to(buffer.device)


class _Recorder(torch.nn.Module):
    def __init__(self, sink):
        super().__init__()
        self._sink = sink

    def __call__(self, *args):
        self._sink["args"] = args
        return args[-1]


class _OrderRecorder(torch.nn.Module):
    def __init__(self, calls, name, ret=None):
        super().__init__()
        self.calls, self.name, self.ret = calls, name, ret

    def __call__(self, *a, **k):
        self.calls.append(self.name)
        return self.ret


class _IdNorm(torch.nn.Module):
    def __call__(self, x):
        return x


class _IdLinear(torch.nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(o, i) * 0.02)

    def __call__(self, x):
        return torch.nn.functional.linear(x, self.weight), None


class _ParamStub:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, param, tensor, shard_id):
        self.loaded = tensor.clone()


class _LayerStub:
    pass


def _cos_sin_cache(max_pos, rope_dim, base=10000.0):
    half = rope_dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half).float() / half))
    pos = torch.arange(max_pos).float()
    ang = torch.outer(pos, inv)
    return torch.cat([ang.cos(), ang.sin()], dim=-1)


# ═══════════════════════ A. 装配幕（站 1-4） ═══════════════════════
class TestAssembly:
    def test_model_entry_allocates_shared_topk_buffer(self):
        """站 1/m03：is_v32=hasattr(index_topk) → 全模型共享 buffer 分配。"""
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model

        hf = make_small_hf_config(num_hidden_layers=2, index_topk=2048)
        vllm_config = make_vllm_config(hf, max_model_len=1024,
                                       max_num_batched_tokens=512)
        model = DeepseekV2Model(vllm_config=vllm_config, prefix="model")
        assert model.is_v32 is True
        buf = model.topk_indices_buffer
        assert buf.shape == (512, 2048)
        assert buf.dtype == torch.int32
        # 同一块 buffer 传给每个 decoder layer（无所有权封装——裸共享）
        assert model.layers[0].self_attn.topk_indices_buffer is buf
        assert model.layers[1].self_attn.topk_indices_buffer is buf

    def test_non_v32_no_buffer(self):
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model

        hf = make_small_hf_config(num_hidden_layers=1)
        del hf.index_topk  # 旧 DSV3 无 index_* 头表
        vllm_config = make_vllm_config(hf, max_model_len=1024)
        model = DeepseekV2Model(vllm_config=vllm_config, prefix="model")
        assert model.is_v32 is False
        assert model.topk_indices_buffer is None

    def test_indexer_independent_small_head(self):
        """站 3/m01：独立小头——头表全来自 config.index_*，权重几何 [out, in]。"""
        from vllm.model_executor.models.deepseek_v2 import Indexer

        hf = make_small_hf_config()
        vllm_config = make_vllm_config(hf, max_model_len=256)
        buf = torch.zeros(64, hf.index_topk, dtype=torch.int32)
        idx = Indexer(vllm_config, hf, hf.hidden_size, hf.q_lora_rank,
                      None, vllm_config.cache_config, buf,
                      "model.layers.0.self_attn.indexer")
        assert idx.topk_tokens == 8
        assert idx.n_head == 2
        assert idx.head_dim == 128
        assert idx.rope_dim == 64
        assert idx.softmax_scale == 128 ** -0.5
        assert idx.n_head_scale == 2 ** -0.5
        # wq_b：q_lora_rank(32) → head_dim*n_head(256)，复制不切 TP
        assert idx.wq_b.weight.shape == (256, 32)
        # wk_weights_proj 一枪 GEMM：hidden → [head_dim, n_head] 两片
        assert idx.wk_weights_proj.weight.shape == (128 + 2, 64)
        assert idx.k_norm.eps == 1e-6
        # 132B/条：128B fp8 + 4B fp32 scale
        assert idx.k_cache.head_dim == 128 + 128 // 128 * 4
        # workspace = max_model_len × 40（魔数账）
        assert idx.max_total_seq_len == 256 * 40

    def _make_mla_attention(self, hf, vllm_config, lid=0):
        from vllm.model_executor.models.deepseek_v2 import DeepseekV2MLAAttention

        buf = torch.zeros(64, hf.index_topk, dtype=torch.int32)
        return DeepseekV2MLAAttention(
            vllm_config, hf, hf.hidden_size, hf.num_attention_heads,
            hf.qk_nope_head_dim, hf.qk_rope_head_dim, hf.v_head_dim,
            hf.q_lora_rank, hf.kv_lora_rank,
            max_position_embeddings=hf.max_position_embeddings,
            cache_config=vllm_config.cache_config,
            prefix=f"model.layers.{lid}.self_attn",
            topk_indices_buffer=buf)

    def test_skip_topk_pattern(self):
        """站 2/m08：freq/pattern/offset 三旋钮 + MTP 层恒建。"""
        built, skipped = [], []
        for lid in range(6):  # 4 backbone + 2 MTP
            hf_l = make_small_hf_config(num_hidden_layers=4)
            hf_l.index_topk_freq = 2          # 每两层一层建
            hf_l.index_skip_topk_offset = 2   # 前两层恒建
            cfg_l = make_vllm_config(hf_l, max_model_len=256)
            attn = self._make_mla_attention(hf_l, cfg_l, lid)
            (built if attn.indexer is not None else skipped).append(lid)
            if lid >= 4:  # MTP 层恒建且 skip_topk=False
                assert attn.indexer is not None
                assert attn.mla_attn.skip_topk is False
            elif lid == 2:
                assert attn.indexer is None           # skip 层不建 indexer
                assert attn.mla_attn.skip_topk is True  # 运行时跳过打分
        assert built == [0, 1, 3, 4, 5]
        assert skipped == [2]

    def test_pattern_list_mode(self):
        results = []
        for lid in range(4):
            hf_l = make_small_hf_config(num_hidden_layers=4)
            hf_l.index_topk_pattern = ["D", "S", "D", "S"]
            cfg_l = make_vllm_config(hf_l, max_model_len=256)
            attn = self._make_mla_attention(hf_l, cfg_l, lid)
            results.append(attn.indexer is not None)
        assert results == [True, False, True, False]

    def test_v32_indexer_cache_spec(self):
        """站 4/m02：spec 自报 MLAAttentionSpec(num_kv_heads=1, head_size=132)。"""
        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV32IndexerBackend, DeepseekV4IndexerBackend)
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        hf = make_small_hf_config()
        vllm_config = make_vllm_config(hf)
        cache = DeepseekV32IndexerCache(
            head_dim=132, dtype=torch.uint8, prefix="t.k_cache",
            cache_config=vllm_config.cache_config)
        spec = cache.get_kv_cache_spec(vllm_config)
        assert isinstance(spec, MLAAttentionSpec)
        assert spec.num_kv_heads == 1        # 只有一根向量，无 K+V 之分
        assert spec.head_size == 132
        assert spec.dtype == torch.uint8
        assert cache.get_attn_backend() is DeepseekV32IndexerBackend
        # 后端族：V3.2 块 64 / V4 子类块 256；num_kv_heads==1 断言 + 三维 shape
        assert DeepseekV32IndexerBackend.get_kv_cache_shape(10, 64, 1, 132) == (10, 64, 132)
        assert DeepseekV4IndexerBackend.get_kv_cache_shape(10, 256, 1, 132) == (10, 256, 132)
        # 跨层 KV 布局不支持：identity 排列即『不兼容』信号
        assert DeepseekV32IndexerBackend.get_kv_cache_stride_order(True) == (0, 1, 2, 3)

    def test_duplicate_layer_name_rejected(self):
        from vllm.model_executor.models.deepseek_v2 import DeepseekV32IndexerCache

        hf = make_small_hf_config()
        vllm_config = make_vllm_config(hf)
        DeepseekV32IndexerCache(head_dim=132, dtype=torch.uint8,
                                prefix="dup.k_cache",
                                cache_config=vllm_config.cache_config)
        try:
            DeepseekV32IndexerCache(head_dim=132, dtype=torch.uint8,
                                    prefix="dup.k_cache",
                                    cache_config=vllm_config.cache_config)
            assert False, "duplicate prefix must raise"
        except ValueError:
            pass

    def test_get_max_prefill_buffer_size_magic_40(self):
        from vllm.v1.attention.backends.mla.indexer import (
            get_max_prefill_buffer_size)

        hf = make_v32_hf_config()
        vllm_config = make_vllm_config(hf, max_model_len=163840)
        assert get_max_prefill_buffer_size(vllm_config) == 163840 * 40

    def test_v4_layer_types_and_c4a_only_indexer(self):
        """站 10/13/m10/m11：compress_ratios 逐层表；只有 C4A 建 indexer；
        压缩坐标系 //4；FP8 132B 同布局；skip_k_cache_insert=True。"""
        from vllm.models.deepseek_v4.attention import (
            DeepseekV4Attention, DeepseekV4Indexer, DeepseekV4IndexerCache)
        from vllm.v1.attention.backends.mla.indexer import (
            get_max_prefill_buffer_size)

        class _Concrete(DeepseekV4Attention):
            @classmethod
            def get_padded_num_q_heads(cls, num_heads):
                return num_heads

            def forward_mqa(self, q, kv, positions, output):
                raise NotImplementedError

            def _o_proj(self, o, positions):
                raise NotImplementedError

        ratios = [1, 4, 128]
        kinds = []
        layers = {}
        for i, r in enumerate(ratios):
            hf = make_v4_hf_config()
            hf.compress_ratios = [r, r, r]
            vllm_config = make_vllm_config(hf, max_model_len=1000,
                                           max_num_batched_tokens=64,
                                           cache_dtype="fp8_ds_mla")
            buf = torch.zeros(64, hf.index_topk, dtype=torch.int32)
            layer = _Concrete(
                vllm_config=vllm_config,
                prefix=f"model.layers.{i}.self_attn",
                topk_indices_buffer=buf, aux_stream_list=None,
                eager_scratch_pool=None)
            kinds.append(layer.indexer is not None)
            layers[r] = layer
            if r != 4:
                assert layer.indexer is None  # 只有 C4A 建 indexer
                # MTP 护栏：layer_id >= num_hidden_layers → 恒 1
        assert kinds == [False, True, False]
        c4a = layers[4].indexer
        assert isinstance(c4a, DeepseekV4Indexer)
        assert c4a.compress_ratio == 4
        # 压缩坐标系：max_model_len//4 一处换算、全链路跟着换
        assert c4a.max_model_len == 1000 // 4
        vc4a = make_vllm_config(make_v4_hf_config(), max_model_len=1000,
                                max_num_batched_tokens=64,
                                cache_dtype="fp8_ds_mla")
        assert c4a.max_total_seq_len == (
            get_max_prefill_buffer_size(vc4a) // 4)
        # FP8 布局与 V3.2 同款 132B；插入归 compressor（skip_k_cache_insert）
        assert c4a.k_cache.head_dim == 132
        assert c4a.k_cache.compress_ratio == 4
        assert c4a.indexer_op.skip_k_cache_insert is True
        assert c4a.compressor is not None
        assert layers[1].swa_cache_layer.block_size == 64  # SWA 支路并存


# ═══════════ B. V3.2 一拍：打分→选块→落 buffer（站 7-11） ═══════════
class TestV32Scoring:
    def _make_indexer(self, seed=0):
        from vllm.model_executor.models.deepseek_v2 import Indexer

        torch.manual_seed(seed)
        hf = make_small_hf_config()
        vllm_config = make_vllm_config(hf, max_model_len=256)
        buf = torch.zeros(256, hf.index_topk, dtype=torch.int32)
        idx = Indexer(vllm_config, hf, hf.hidden_size, hf.q_lora_rank,
                      None, vllm_config.cache_config, buf,
                      "model.layers.0.self_attn.indexer")
        return idx, hf

    def test_indexer_forward_quant_rope_and_scale_fold(self):
        """站 7/m01：q FP8 量化（ue8m0）+ 全部 scale 折进 weights（标量搬出核）。"""
        from vllm.model_executor.layers.rotary_embedding import get_rope

        idx, hf = self._make_indexer(seed=3)
        T = 5
        hidden = torch.randn(T, hf.hidden_size)
        qr = torch.randn(T, hf.q_lora_rank)
        positions = torch.tensor([0, 1, 2, 3, 4])
        # indexer 专属 RoPE：is_neox 取反（interleave）
        rope = get_rope(64, max_position=64,
                        rope_parameters={"rope_type": "default"},
                        is_neox_style=False)

        captured = {}
        idx.indexer_op = _Recorder(captured)
        ret = idx(hidden, qr, positions, rope)
        assert ret is captured["args"][-1]  # op 返回值即透传

        q_fp8, k, weights = captured["args"][1:4]
        assert q_fp8.dtype == torch.float8_e4m3fn
        assert q_fp8.shape == (T, hf.index_n_heads, 128)
        # 参考：wq_b 上投 → 切 rope/nope → interleave RoPE 只打 rope 段
        q_ref = idx.wq_b(qr)[0].view(T, hf.index_n_heads, 128)
        q_pe = ref_interleave_rope(q_ref[..., :64], positions)
        q_ref_rot = torch.cat([q_pe, q_ref[..., 64:]], dim=-1).reshape(-1, 128)
        q_fp8_ref, q_scale_ref = ref_group_quant_ue8m0(q_ref_rot)
        assert torch.equal(q_fp8.reshape(-1).view(torch.uint8),
                           q_fp8_ref.reshape(-1).view(torch.uint8))
        # weights = raw_w · q_scale · softmax_scale · n_head_scale
        kw = idx.wk_weights_proj(hidden)[0]
        w_raw = kw[:, 128:]
        w_ref = (w_raw.view(T, hf.index_n_heads)
                 * q_scale_ref.view(T, hf.index_n_heads)
                 * idx.softmax_scale * idx.n_head_scale)
        assert torch.allclose(weights.view(T, hf.index_n_heads), w_ref,
                              rtol=1e-5, atol=1e-6)
        # k：wk 片 + LayerNorm(eps=1e-6) + rope 段 interleave
        k_ref = idx.k_norm(kw[:, :128])
        k_pe = ref_interleave_rope(k_ref[:, :64], positions)
        k_ref = torch.cat([k_pe, k_ref[:, 64:]], dim=-1)
        assert torch.allclose(k, k_ref, rtol=1e-4, atol=1e-5)

    def test_dsa_logits_quant_path_matches_float_reference(self):
        """m01：量化路径的 I_{t,:} == float 参考（Eq.(1)）——fp8 容差内。"""
        from vllm.utils.deep_gemm import fp8_fp4_mqa_logits

        torch.manual_seed(7)
        M, N, H, D = 4, 16, 2, 128
        q = torch.randn(M, H, D)
        k = torch.randn(N, D)
        weights = torch.randn(M, H)
        q_fp8, q_scale = ref_group_quant_ue8m0(q.reshape(-1, D))
        q_fp8 = q_fp8.view(M, H, D)
        k_fp8, k_scale = ref_group_quant_ue8m0(k)
        cu_ks = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
        cu_ke = torch.tensor([8, 12, 16, 16], dtype=torch.int32)
        # 核的契约：q_scale 已折进 weights（fused_indexer_q_rope_quant 的折叠）
        w_fold = weights * q_scale.view(M, H)
        logits = fp8_fp4_mqa_logits(
            (q_fp8, None),
            (k_fp8.view(-1, D), k_scale.view(-1, 1).view(torch.uint8).view(
                torch.float32).view(-1, 1)),
            w_fold, cu_ks, cu_ke, clean_logits=False)
        ref = ref_dsa_logits(q_fp8.float(), ref_dequant(k_fp8, k_scale), w_fold)
        assert torch.allclose(logits, ref, rtol=5e-2, atol=5e-2)

    def test_k_quant_and_cache_132b_layout_roundtrip(self):
        """m02/m16：K 量化+缓存插入融合；132B/条布局；gather 往返。"""
        from vllm import _custom_ops as ops

        torch.manual_seed(11)
        block_size, T = 64, 10
        kv_cache = torch.zeros(2, block_size, 132, dtype=torch.uint8)
        k = torch.randn(T, 128)
        slot_mapping = torch.arange(T, dtype=torch.int64)
        ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping, 128, "ue8m0")
        # 布局核验：前 128B fp8 值 + 尾 4B fp32 scale
        row0 = kv_cache[0, 0]
        scale0 = row0[128:132].view(torch.float32).item()
        assert scale0 > 0
        k_deq = row0[:128].view(torch.float8_e4m3fn).float() * scale0
        assert torch.allclose(k_deq, k[0], rtol=1e-1, atol=1e-1)
        # 参考数学同式（ue8m0 幂次 scale）
        _, s_ref = ref_group_quant_ue8m0(k[0:1])
        assert scale0 == s_ref.item()
        # gather 往返：分页物理 slot → 连续 workspace
        k_quant = torch.zeros(T, 128, dtype=torch.uint8)
        k_scale = torch.zeros(T, 4, dtype=torch.uint8)
        bt, _ = make_block_table([T], block_size)
        ops.cp_gather_indexer_k_quant_cache(
            kv_cache, k_quant, k_scale, bt,
            torch.tensor([0, T], dtype=torch.int32))  # cu_seq_lens（累积界）
        assert torch.equal(k_quant[0], row0[:128])
        assert torch.equal(k_scale[0], row0[128:132])

    def _run_op_prefill(self, seed=1, seq_lens=(12, 10), query_lens=(9, 7),
                        topk=8):
        """构造 2 请求（各带 3 token 历史）prefill 一拍：builder 真路径+op 真入口。"""
        from vllm.forward_context import ForwardContext, set_forward_context
        from vllm.model_executor.layers.sparse_attn_indexer import (
            sparse_attn_indexer)
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV32IndexerMetadataBuilder)
        from vllm.v1.kv_cache_interface import MLAAttentionSpec
        from vllm import _custom_ops as ops

        torch.manual_seed(seed)
        hf = make_small_hf_config(index_topk=topk)
        vllm_config = make_vllm_config(hf, max_model_len=64,
                                       max_num_batched_tokens=64)
        block_size = 64
        n_tok = sum(query_lens)
        T_hist = sum(s - q for s, q in zip(seq_lens, query_lens))
        spec = MLAAttentionSpec(block_size=block_size, num_kv_heads=1,
                                head_size=132, dtype=torch.uint8)
        builder = DeepseekV32IndexerMetadataBuilder(
            kv_cache_spec=spec, layer_names=["t.k_cache"],
            vllm_config=vllm_config, device=DEV, block_table_width=4)
        bt, n_blocks = make_block_table(list(seq_lens), block_size)
        qsl = torch.tensor([0] + list(torch.cumsum(
            torch.tensor(query_lens), 0)), dtype=torch.int32)
        # slot_mapping：本拍 token 落各自请求的历史之后（block 内位置）
        slots = []
        for b, (S, Q) in enumerate(zip(seq_lens, query_lens)):
            base = int(bt[b, 0].item()) * block_size
            slots.extend(range(base + S - Q, base + S))
        common = _CommonMeta(num_reqs=len(seq_lens), num_actual_tokens=n_tok,
                             query_start_loc=qsl, seq_lens=torch.tensor(seq_lens),
                             max_query_len=max(query_lens),
                             max_seq_len=max(seq_lens), block_table=bt,
                             slot_mapping=torch.tensor(slots, dtype=torch.int64))
        indexer_meta = builder.build(0, common)
        # 打分原料：历史 key 先入缓存，本拍 k 由 op 插入
        num_blocks = max(2, n_blocks)
        kv_cache = torch.zeros(num_blocks, block_size, 132, dtype=torch.uint8)
        k_all = torch.randn(sum(seq_lens), 128)
        hist_slots = []
        for b, (S, Q) in enumerate(zip(seq_lens, query_lens)):
            base = int(bt[b, 0].item()) * block_size
            hist_slots.extend(range(base, base + S - Q))
        hist_rows = [pos for b, (S, Q) in enumerate(zip(seq_lens, query_lens))
                     for pos in range(S - Q)]  # 请求序的历史行
        hist_idx = 0
        flat_hist = []
        for b, (S, Q) in enumerate(zip(seq_lens, query_lens)):
            flat_hist.extend(range(hist_idx, hist_idx + S - Q))
            hist_idx += S - Q
        ops.indexer_k_quant_and_cache(
            k_all[flat_hist], kv_cache,
            torch.tensor(hist_slots, dtype=torch.int64), 128, "ue8m0")
        # 本拍 query + 当前 token 的 indexer k（op 会插入缓存）
        q = torch.randn(n_tok, hf.index_n_heads, 128)
        q_fp8, q_scale = ref_group_quant_ue8m0(q.reshape(-1, 128))
        q_fp8 = q_fp8.view(n_tok, hf.index_n_heads, 128)
        weights = (torch.randn(n_tok, hf.index_n_heads)
                   * q_scale.view(n_tok, hf.index_n_heads)
                   * 128 ** -0.5 * hf.index_n_heads ** -0.5)
        k_cur = []
        for b, (S, Q) in enumerate(zip(seq_lens, query_lens)):
            k_cur.append(k_all[sum(seq_lens[:b]) + S - Q: sum(seq_lens[:b]) + S])
        k_cur = torch.cat(k_cur)
        buf = torch.full((64, topk), -7, dtype=torch.int32)  # 预清由 op 做
        ctx = ForwardContext(no_compile_layers={}, attn_metadata={
            "t.k_cache": indexer_meta}, slot_mapping={})
        with set_forward_context(ctx):
            out = sparse_attn_indexer(
                hidden_states=torch.zeros(n_tok, hf.hidden_size),
                k_cache_prefix="t.k_cache", kv_cache=kv_cache,
                q_quant=q_fp8, q_scale=None, k=k_cur,
                weights=weights, quant_block_size=128, scale_fmt="ue8m0",
                topk_tokens=topk, head_dim=128, max_model_len=64,
                total_seq_lens=sum(seq_lens), topk_indices_buffer=buf,
                skip_k_cache_insert=False, use_pcp=False,
                dense_mha_metadata_layer_name="")
        assert out is buf
        # 参考用 k_full：请求序全量（历史+本拍），op 插入后从缓存反读
        k_full = []
        for b, (S, Q) in enumerate(zip(seq_lens, query_lens)):
            base = int(bt[b, 0].item()) * block_size
            for pos in range(S):
                row = kv_cache.reshape(-1, 132)[base + pos]
                k_full.append(row[:128].view(torch.float8_e4m3fn).float()
                              * row[128:].view(torch.float32).item())
        return (buf, weights, q_fp8, q_scale, torch.stack(k_full),
                seq_lens, query_lens, topk, hf)

    def test_prefill_topk_matches_bruteforce_with_causality(self):
        """站 9/m05：buffer 行 == 暴力因果 top-k（降序/tie 取小/相对窗口起点）。"""
        (buf, weights, q_fp8, q_scale, k_deq, seq_lens, query_lens,
         topk, hf) = self._run_op_prefill(seed=5)
        q_deq = q_fp8.float()  # scale 已折 weights
        logits_full = ref_dsa_logits(q_deq, k_deq, weights)
        tok = 0
        for b, (S, Q) in enumerate(zip(seq_lens, query_lens)):
            start_pos = S - Q
            for o in range(Q):
                causal = start_pos + 1 + o
                # 请求内逻辑位：窗口 = 本请求前 causal 个 token（gathered 坐标
                # 减去本请求基址 = 请求内位置）
                base = sum(seq_lens[:b])
                ref = ref_topk_desc(
                    logits_full[tok, base:base + causal].tolist(), topk)
                got = buf[tok, :topk].tolist()
                assert got == ref, f"req{b} tok{o}: {got} != {ref}"
                tok += 1
        # 无效位 -1 哨兵（req1 首 token 只有 4 个历史 → 后 4 位 -1）
        assert (buf[9, 4:] == -1).all()

    def test_prefill_short_context_all_minus_one(self):
        """m03：local_total_seq_lens==0 → 整行 -1（topk_indices.fill_(-1)）。"""
        from vllm.forward_context import ForwardContext, set_forward_context
        from vllm.model_executor.layers.sparse_attn_indexer import (
            sparse_attn_indexer)
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV32IndexerMetadata, DeepseekV32IndexerPrefillChunkMetadata,
            DeepseekV32IndexerPrefillMetadata)

        hf = make_small_hf_config(index_topk=4)
        vllm_config = make_vllm_config(hf, max_model_len=64)
        chunk = DeepseekV32IndexerPrefillChunkMetadata(
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            cu_seqlen_ks=torch.zeros(2, dtype=torch.int32),
            cu_seqlen_ke=torch.zeros(2, dtype=torch.int32),
            cu_seq_lens=torch.zeros(2, dtype=torch.int32),
            token_to_seq=torch.zeros(0, dtype=torch.int32),
            total_seq_lens=0, token_start=0, token_end=2, num_reqs=1,
            local_cu_seq_lens=torch.zeros(2, dtype=torch.int32),
            local_total_seq_lens=0, max_local_total_seq_lens=0)
        meta = DeepseekV32IndexerMetadata(
            seq_lens=torch.tensor([0]), max_seq_len=0,
            slot_mapping=torch.zeros(2, dtype=torch.int64),
            num_decodes=0, num_decode_tokens=0, num_prefills=1,
            num_prefill_tokens=2,
            prefill=DeepseekV32IndexerPrefillMetadata(chunks=[chunk]))
        buf = torch.full((8, 4), -7, dtype=torch.int32)
        ctx = ForwardContext(no_compile_layers={},
                             attn_metadata={"t.k_cache": meta}, slot_mapping={})
        with set_forward_context(ctx):
            sparse_attn_indexer(
                hidden_states=torch.zeros(2, 8), k_cache_prefix="t.k_cache",
                kv_cache=torch.zeros(1, 64, 132, dtype=torch.uint8),
                q_quant=torch.zeros(2, 2, 128, dtype=torch.float8_e4m3fn),
                q_scale=None, k=None,
                weights=torch.ones(2, 2), quant_block_size=128,
                scale_fmt="ue8m0", topk_tokens=4, head_dim=128,
                max_model_len=64, total_seq_lens=0, topk_indices_buffer=buf,
                skip_k_cache_insert=True, use_pcp=False,
                dense_mha_metadata_layer_name="")
        assert (buf[:2] == -1).all()

    def test_decode_paged_scoring_native_2d_spec(self):
        """站 10/m06：decode 免 gather 直接 paged 打分；native 2D 逐 token 上下文。"""
        from vllm.forward_context import ForwardContext, set_forward_context
        from vllm.model_executor.layers.sparse_attn_indexer import (
            sparse_attn_indexer)
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV32IndexerMetadataBuilder)
        from vllm.v1.kv_cache_interface import MLAAttentionSpec
        from vllm import _custom_ops as ops

        torch.manual_seed(13)
        hf = make_small_hf_config(index_topk=6)
        vllm_config = make_vllm_config(hf, max_model_len=64,
                                       max_num_batched_tokens=64, spec_tokens=1)
        block_size = 64
        B, next_n, L = 2, 2, 9   # spec 窗口 2：每拍 2 token
        seq_lens = [L, L]
        spec = MLAAttentionSpec(block_size=block_size, num_kv_heads=1,
                                head_size=132, dtype=torch.uint8)
        builder = DeepseekV32IndexerMetadataBuilder(
            kv_cache_spec=spec, layer_names=["t.k_cache"],
            vllm_config=vllm_config, device=DEV, block_table_width=4)
        qsl = torch.tensor([0, next_n, 2 * next_n], dtype=torch.int32)
        common = _CommonMeta(num_reqs=B, num_actual_tokens=B * next_n,
                             query_start_loc=qsl,
                             seq_lens=torch.tensor(seq_lens),
                             max_query_len=next_n, max_seq_len=L,
                             block_table=make_block_table(seq_lens, block_size)[0],
                             slot_mapping=torch.arange(B * next_n, dtype=torch.int64))
        meta = builder.build(0, common)
        assert meta.decode is not None
        # 2D seq_lens：[b, j] = L - next_n + j + 1
        assert meta.decode.seq_lens.shape == (B, next_n)
        assert meta.decode.seq_lens.tolist() == [[8, 9], [8, 9]]
        assert meta.decode.requires_padding is False
        # 历史量化 key 已入缓存（两请求各一块；本拍 2 token 也算入 L 内）
        kv_cache = torch.zeros(2, block_size, 132, dtype=torch.uint8)
        k_hist = torch.randn(L, 128)
        ops.indexer_k_quant_and_cache(
            k_hist, kv_cache, torch.arange(L, dtype=torch.int64), 128, "ue8m0")
        ops.indexer_k_quant_and_cache(  # req1 的块 1 同内容（ref 共享 k_deq）
            k_hist, kv_cache,
            torch.arange(block_size, block_size + L, dtype=torch.int64),
            128, "ue8m0")
        q = torch.randn(B * next_n, hf.index_n_heads, 128)
        q_fp8, q_scale = ref_group_quant_ue8m0(q.reshape(-1, 128))
        q_fp8 = q_fp8.view(B * next_n, hf.index_n_heads, 128)
        weights = (torch.randn(B * next_n, hf.index_n_heads)
                   * q_scale.view(B * next_n, hf.index_n_heads))
        buf = torch.zeros(64, hf.index_topk, dtype=torch.int32)
        ctx = ForwardContext(no_compile_layers={},
                             attn_metadata={"t.k_cache": meta}, slot_mapping={})
        with set_forward_context(ctx):
            sparse_attn_indexer(
                hidden_states=torch.zeros(B * next_n, 8),
                k_cache_prefix="t.k_cache", kv_cache=kv_cache,
                q_quant=q_fp8, q_scale=None, k=torch.randn(B * next_n, 128),
                weights=weights, quant_block_size=128, scale_fmt="ue8m0",
                topk_tokens=hf.index_topk, head_dim=128, max_model_len=64,
                total_seq_lens=L, topk_indices_buffer=buf,
                skip_k_cache_insert=True, use_pcp=False,
                dense_mha_metadata_layer_name="")
        flat = kv_cache.reshape(-1, 132)
        k_deq = torch.stack([
            flat[s, :128].view(torch.float8_e4m3fn).float()
            * flat[s, 128:].view(torch.float32).item() for s in range(L)])
        logits_full = ref_dsa_logits(q_fp8.float(), k_deq, weights)
        for b in range(B):
            for j in range(next_n):
                row = b * next_n + j
                bound = L - next_n + j + 1
                ref = ref_topk_desc(logits_full[row, :bound].tolist(),
                                    hf.index_topk)
                assert buf[row, :hf.index_topk].tolist() == ref

    def test_decode_flatten_variable_lens(self):
        """flatten 变长路：注释原例 [8,9,10,7,9,10,11,12] + requires_padding。"""
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV32IndexerMetadataBuilder)
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        hf = make_small_hf_config()
        vllm_config = make_vllm_config(hf, max_model_len=64,
                                       max_num_batched_tokens=64, spec_tokens=3)
        spec = MLAAttentionSpec(block_size=64, num_kv_heads=1, head_size=132,
                                dtype=torch.uint8)
        builder = DeepseekV32IndexerMetadataBuilder(
            kv_cache_spec=spec, layer_names=["t.k_cache"],
            vllm_config=vllm_config, device=DEV, block_table_width=4)
        seq_lens = [10, 7, 12]
        decode_lens = [3, 1, 4]  # 变长
        qsl = torch.tensor([0] + list(torch.cumsum(
            torch.tensor(decode_lens), 0)), dtype=torch.int32)
        common = _CommonMeta(
            num_reqs=3, num_actual_tokens=sum(decode_lens), query_start_loc=qsl,
            seq_lens=torch.tensor(seq_lens), max_query_len=4, max_seq_len=12,
            block_table=make_block_table(seq_lens, 64)[0],
            slot_mapping=torch.arange(8, dtype=torch.int64))
        meta = builder.build(0, common)
        assert meta.decode is not None
        # flatten 展平后每行 decode_len==1 → 无需 pack（requires_padding=False）
        assert meta.decode.requires_padding is False
        # seq_lens = 逐 token 上下文长（注释原例）
        assert meta.decode.decode_lens.tolist() == [1] * 8
        assert meta.decode.seq_lens.view(-1).tolist() == [8, 9, 10, 7, 9, 10, 11, 12]

    def test_decode_flatten_uniform_kernel_math(self):
        """均匀 decode_lens 走 _prepare_uniform_decode_kernel 路。"""
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV32IndexerMetadataBuilder)
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        hf = make_small_hf_config()
        vllm_config = make_vllm_config(hf, max_model_len=64,
                                       max_num_batched_tokens=64, spec_tokens=2)
        spec = MLAAttentionSpec(block_size=64, num_kv_heads=1, head_size=132,
                                dtype=torch.uint8)
        builder = DeepseekV32IndexerMetadataBuilder(
            kv_cache_spec=spec, layer_names=["t.k_cache"],
            vllm_config=vllm_config, device=DEV, block_table_width=4)
        seq_lens = [10, 7]
        decode_lens = [3, 3]
        qsl = torch.tensor([0, 3, 6], dtype=torch.int32)
        common = _CommonMeta(
            num_reqs=2, num_actual_tokens=6, query_start_loc=qsl,
            seq_lens=torch.tensor(seq_lens), max_query_len=3, max_seq_len=10,
            block_table=make_block_table(seq_lens, 64)[0],
            slot_mapping=torch.arange(6, dtype=torch.int64))
        meta = builder.build(0, common)
        # per-token: seq_len - max_decode_len + local_idx + 1
        assert meta.decode.seq_lens.view(-1).tolist() == [8, 9, 10, 5, 6, 7]

    def test_three_topk_kernels_agree(self):
        """三核分派（cooperative/persistent/per_row）语义一致。"""
        from vllm import _custom_ops as ops

        torch.manual_seed(17)
        N, rows, K = 64, 3, 8
        logits = torch.randn(rows, N)
        seq_lens = torch.tensor([64, 40, 7], dtype=torch.int32)
        outs = []
        for call in (
            lambda o: ops.top_k_per_row_decode(
                logits, 1, seq_lens, o, rows, logits.stride(0),
                logits.stride(1), K),
            lambda o: torch.ops._C.cooperative_topk(
                logits, seq_lens, o, torch.empty(RADIX_WS, dtype=torch.uint8),
                K, N),
            lambda o: torch.ops._C.persistent_topk(
                logits, seq_lens, o, torch.empty(RADIX_WS, dtype=torch.uint8),
                K, N),
        ):
            buf = torch.full((rows, K), -1, dtype=torch.int32)
            call(buf)
            outs.append(buf)
        assert torch.equal(outs[0], outs[1])
        assert torch.equal(outs[0], outs[2])
        # tie-break：等值时小 index 在前；越界位不入选
        tie = torch.tensor([[5.0, 5.0, 1.0], [9.0, 2.0, 2.0]])
        buf = torch.full((2, 2), -1, dtype=torch.int32)
        ops.top_k_per_row_decode(
            tie, 1, torch.tensor([2, 3], dtype=torch.int32), buf, 2,
            tie.stride(0), tie.stride(1), 2)
        assert buf.tolist() == [[0, 1], [0, 1]]  # 行1 tie：2@1 先于 2@2

    def test_topk_decode_1d_self_bounds(self):
        """1D seq_lens：核内自算 rowEnd = seq_len - next_n + j + 1。"""
        from vllm import _custom_ops as ops

        logits = torch.tensor([[9.0, 9.0, 1.0, 9.0]])  # 越界位 9 也不得入选
        seq_lens = torch.tensor([2], dtype=torch.int32)
        buf = torch.full((1, 2), -1, dtype=torch.int32)
        ops.top_k_per_row_decode(logits, 2, seq_lens, buf, 1,
                                 logits.stride(0), logits.stride(1), 2)
        # rowEnd = 2 - 2 + 0 + 1 = 1：只有 pos0 有效 → 第二位 -1 哨兵
        assert buf.tolist() == [[0, -1]]


# ═══════════ C. 双预算切块 + builder（站 5/m04） ═══════════
class TestChunking:
    def test_double_budget_chunking(self):
        from vllm.v1.attention.backends.mla.indexer import (
            split_indexer_prefill_chunks)

        # N 约束 + M·N·4 字节约束
        seq = torch.tensor([100, 100, 100])
        qry = torch.tensor([10, 10, 10])
        # workspace=N 上限 250：前两个请求一块，第三个因 N 超限切开
        chunks = split_indexer_prefill_chunks(seq, qry, 250, 10 ** 9)
        assert chunks == [(slice(0, 2), slice(0, 20)), (slice(2, 3), slice(0, 10))]
        # logits 预算 M·N ≤ 500：每请求 (M=10,N=100) 的 10*100=1000>500
        # → 逐请求子切，max_q = 500//100 = 5 → (0,5)+(5,10) 六片
        chunks = split_indexer_prefill_chunks(seq, qry, 250, 4 * 500)
        assert chunks == [
            (slice(0, 1), slice(0, 5)), (slice(0, 1), slice(5, 10)),
            (slice(1, 2), slice(0, 5)), (slice(1, 2), slice(5, 10)),
            (slice(2, 3), slice(0, 5)), (slice(2, 3), slice(5, 10))]

    def test_single_request_over_budget_subchunk(self):
        from vllm.v1.attention.backends.mla.indexer import (
            split_indexer_prefill_chunks)

        seq = torch.tensor([1000])
        qry = torch.tensor([40])
        chunks = split_indexer_prefill_chunks(seq, qry, 100, 4 * 2500)
        # max_q = 2500//1000 = 2 → 40 行切成 20 片
        assert all(c[0] == slice(0, 1) for c in chunks)
        assert [c[1] for c in chunks[:3]] == [slice(0, 2), slice(2, 4), slice(4, 6)]
        assert len(chunks) == 20

    def test_prefill_chunk_metadata_causal_bounds(self):
        from vllm.v1.attention.backends.mla.indexer import (
            build_prefill_chunk_metadata)

        seq_lens = torch.tensor([6, 4], dtype=torch.int32)
        qsl = torch.tensor([0, 4, 7], dtype=torch.int32)  # query 4+3
        block_table, _ = make_block_table([6, 4], 64)
        m = build_prefill_chunk_metadata(
            0, 2, query_start_loc=qsl.to(DEV),
            query_start_loc_cpu=qsl,
            uncompressed_seq_lens=seq_lens.to(DEV),
            compressed_seq_lens=seq_lens.to(DEV),
            compressed_seq_lens_cpu=seq_lens,
            block_table=block_table.to(DEV), compress_ratio=1)
        assert m is not None
        # cu_seq_lens：请求 K 行基址 [0, 6, 10]
        assert m.cu_seq_lens.tolist() == [0, 6, 10]
        # cu_seqlen_ks：请求内每行 K 起点 = 本请求基址
        assert m.cu_seqlen_ks.tolist() == [0] * 4 + [6] * 3
        # cu_seqlen_ke = ks + 因果长（start_pos+1+offset）
        # req0: start_pos=2 → [3,4,5,6]; req1: start_pos=1 → [8,9,10]
        assert m.cu_seqlen_ke.tolist() == [3, 4, 5, 6, 8, 9, 10]
        assert m.token_to_seq.tolist() == [0] * 6 + [1] * 4
        assert (m.token_start, m.token_end) == (0, 7)

    def test_builder_decode_compress_ratio_coordinates(self):
        """m11：compress_ratio>1 → seq_lens//4 + 压缩 slot（(pos+1)%4==0）。"""
        from vllm.v1.attention.backends.mla.indexer import (
            DeepseekV32IndexerMetadataBuilder)
        from vllm.v1.kv_cache_interface import MLAAttentionSpec

        hf = make_small_hf_config()
        vllm_config = make_vllm_config(hf, max_model_len=256)
        spec = MLAAttentionSpec(block_size=64, num_kv_heads=1, head_size=132,
                                dtype=torch.uint8, compress_ratio=4)
        builder = DeepseekV32IndexerMetadataBuilder(
            kv_cache_spec=spec, layer_names=["t.k_cache"],
            vllm_config=vllm_config, device=DEV, block_table_width=8)
        seq_lens = [16, 8]  # 末 token pos=15/7（(pos+1)%4==0 的边界 token）
        qsl = torch.tensor([0, 1, 2], dtype=torch.int32)
        common = _CommonMeta(
            num_reqs=2, num_actual_tokens=2, query_start_loc=qsl,
            seq_lens=torch.tensor(seq_lens), max_query_len=1, max_seq_len=16,
            block_table=make_block_table(seq_lens, 64)[0],
            slot_mapping=torch.tensor([15, 7], dtype=torch.int64))
        meta = builder.build(0, common)
        # decode 上下文长进压缩坐标：16//4=4；8//4=2
        assert meta.decode.seq_lens.view(-1).tolist() == [4, 2]
        # slot 15/7 → 压缩 slot 3/17（(pos+1)%4==0 才有效；压缩缓存以
        # storage_block=16 分页：slot = 物理块×16 + 压缩位）
        assert meta.slot_mapping.tolist() == [3, 1 * 16 + 1]


# ═══════════ D. V4：压缩机/短上下文/双源消费（站 13-14） ═══════════
class TestV4:
    def _make_compressor(self, compress_ratio=4, seed=0):
        from vllm.models.deepseek_v4.compressor import DeepseekCompressor

        torch.manual_seed(seed)
        hf = make_v4_hf_config()
        vllm_config = make_vllm_config(hf, max_model_len=256)
        comp = DeepseekCompressor(
            vllm_config=vllm_config, compress_ratio=compress_ratio,
            hidden_size=hf.hidden_size, head_dim=128, rotate=True,
            prefix="model.layers.1.self_attn.indexer.compressor",
            k_cache_prefix="model.layers.1.self_attn.indexer.k_cache")
        return comp, hf, vllm_config

    def test_compressor_assembly(self):
        comp, hf, _ = self._make_compressor()
        assert comp.compress_ratio == 4
        assert comp.overlap is True and comp.coff == 2  # 1 + (ratio==4)
        # 一枪 GEMM：hidden(64) → [coff·head_dim + coff·head_dim]（KV+gate 两半）
        assert comp.fused_wkv_wgate.weight.shape == (256 + 256, 64)
        assert comp.fused_wkv_wgate.output_sizes == [256, 256]
        # 状态缓存：kv_state 前半 + score_state 后半；窗宽 coff·ratio=8
        assert comp.state_cache.state_dim == 2 * 2 * 128
        assert comp.state_cache.sliding_window == 8
        assert comp.state_cache.block_size == 4
        # ape：[compress_ratio, coff·head_dim]
        assert comp.ape.shape == (4, 256)

    def test_save_partial_states_layout(self):
        """m11：kv 前半 / score+ape（ape[pos % ratio]）后半写进分页 state_cache。"""
        from vllm.models.deepseek_v4.common.ops import save_partial_states

        block_size, state_width = 4, 256
        state_cache = torch.zeros(2, block_size, 2 * state_width)
        T = 3
        kv = torch.randn(T, state_width)
        score = torch.randn(T, state_width)
        ape = torch.randn(4, state_width)
        positions = torch.tensor([3, 4, 5])  # ape 行 = pos % 4
        slot_mapping = torch.tensor([3, 4, 5], dtype=torch.int64)
        save_partial_states(kv=kv, score=score, ape=ape, positions=positions,
                            state_cache=state_cache, slot_mapping=slot_mapping,
                            block_size=block_size, state_width=state_width,
                            compress_ratio=4, pdl_kwargs={})
        flat = state_cache.reshape(-1, 2 * state_width)
        assert torch.allclose(flat[3, :state_width], kv[0], atol=1e-5)
        assert torch.allclose(flat[3, state_width:], score[0] + ape[3], atol=1e-5)
        assert torch.allclose(flat[5, state_width:], score[2] + ape[1], atol=1e-5)

    def test_compress_math_matches_reference(self):
        """compress→RMSNorm→RoPE→FP8 == 独立参考（Eq.(9)-(17) 落地）。"""
        from vllm.models.deepseek_v4.common.ops import (
            compress_norm_rope_store_triton)

        torch.manual_seed(19)
        ratio, head_dim, rope_dim, window = 4, 128, 64, 8
        block_size = 4
        state_width = 256
        state_cache = torch.zeros(2, block_size, 2 * state_width)
        kv_state = torch.randn(window, head_dim)
        score_state = torch.randn(window, head_dim)
        # 打包布局：[kv 头0 | kv 头1 | score 头0 | score 头1]（coff=2）；
        # 两头填同值——参考只对 overlap 第二半窗（头1）求和
        packed = state_cache.reshape(-1, 2 * state_width)
        packed[:window, 0:128] = kv_state
        packed[:window, 128:256] = kv_state
        packed[:window, 256:384] = score_state
        packed[:window, 384:512] = score_state
        block_table = torch.tensor([[0, 1]], dtype=torch.int32)
        kv_cache = torch.zeros(2, block_size, 132, dtype=torch.uint8)
        norm_w = torch.randn(head_dim)
        cos_sin = _cos_sin_cache(64, rope_dim)
        slot_mapping = torch.tensor([7], dtype=torch.int64)  # (pos+1)%4==0
        compress_norm_rope_store_triton(
            state_cache=state_cache, num_actual=1,
            token_to_req_indices=torch.tensor([0], dtype=torch.int32),
            positions=torch.tensor([7]), slot_mapping=slot_mapping,
            block_table=block_table, block_size=block_size,
            state_width=state_width, cos_sin_cache=cos_sin,
            kv_cache=kv_cache,
            k_cache_metadata=_KMeta(slot_mapping), pdl_kwargs={},
            head_dim=head_dim, rope_head_dim=rope_dim, compress_ratio=ratio,
            overlap=True, use_fp4_cache=False, rms_norm_weight=norm_w,
            rms_norm_eps=1e-6, quant_block=128, token_stride=128, scale_dim=4)
        # 参考：窗内 8 项（前 4 读头0、后 4 读头1——两头填同值 ⇒ 逐 t 用
        # kv_state[t]/score_state[t]），softmax 过 8 项
        s = torch.softmax(score_state, dim=0)
        compressed = (kv_state * s).sum(0)
        var = (compressed ** 2).sum() / head_dim
        normed = compressed * torch.rsqrt(var + 1e-6) * norm_w
        pos_c = (7 // 4) * 4
        ref = normed.clone()
        ref[64:] = ref_interleave_rope(normed[64:].unsqueeze(0),
                                       torch.tensor([pos_c])).squeeze(0)
        # 核在 amax 前做 bf16 round-trip（quant_input 位）——参考同型
        q_ref, s_ref = ref_group_quant_ue8m0(
            ref.to(torch.bfloat16).float().unsqueeze(0))
        got_q = kv_cache.reshape(-1, 132)[7, :128]
        got_s = kv_cache.reshape(-1, 132)[7, 128:].view(torch.float32).item()
        assert got_s == s_ref.item()
        assert torch.equal(got_q.view(torch.float8_e4m3fn).view(torch.uint8),
                           q_ref.view(torch.uint8).squeeze(0))

    def test_short_context_fill_all_selected(self):
        """m12：candidates≤topk → 直填 0..n-1、其余 -1。"""
        import importlib

        att = importlib.import_module("vllm.models.deepseek_v4.attention")
        buf = torch.full((3, 8), -7, dtype=torch.int32)
        positions = torch.tensor([7, 15, 20])  # 压缩后 2/4/5 个 candidates
        att._fill_short_context_topk_indices[(3,)](
            buf, positions, TOP_K=8, COMPRESS_RATIO=4,
            PADDED_TOP_K=8, num_warps=1)
        assert buf[0].tolist() == [0, 1] + [-1] * 6
        assert buf[1].tolist() == [0, 1, 2, 3] + [-1] * 4
        assert buf[2].tolist() == [0, 1, 2, 3, 4, -1, -1, -1]

    def test_compute_global_topk_indices_and_lens(self):
        """m07/V4：局部压缩 index → 全局物理 slot + 有效计数。"""
        from vllm.models.deepseek_v4.common.ops import (
            compute_global_topk_indices_and_lens)

        block_table = torch.tensor([[5, 6], [9, 0]], dtype=torch.int32)
        block_size = 64
        topk = torch.tensor([[0, 70, -1], [3, 66, 1]], dtype=torch.int32)
        t2r = torch.tensor([0, 1], dtype=torch.int32)
        is_valid = torch.tensor([1, 0], dtype=torch.int32)  # 第二 token 是 pad
        gi, lens = compute_global_topk_indices_and_lens(
            topk, t2r, block_table, block_size, is_valid)
        assert gi[0].tolist() == [5 * 64, 6 * 64 + 6, -1]
        assert gi[1].tolist() == [9 * 64 + 3, 0 * 64 + 2, 9 * 64 + 1]
        assert lens.tolist() == [2, 0]  # -1 不计数；pad token 计数清零

    def test_combine_topk_swa_union(self):
        """m13：combined = top-k 段(M·b 起) + SWA 滑窗段；len=两段和。"""
        from vllm.models.deepseek_v4.common.ops import combine_topk_swa_indices

        qsl = torch.tensor([0, 2, 4], dtype=torch.int32)
        seq_lens = torch.tensor([10, 7], dtype=torch.int32)
        gather_lens = torch.tensor([4, 3], dtype=torch.int32)
        topk = torch.tensor([[3, -1], [0, 1], [2, -1], [5, -1]], dtype=torch.int32)
        M, N, top_k, window, ratio = 2, 5, 2, 4, 4
        ci, cl = combine_topk_swa_indices(
            topk, qsl, seq_lens, gather_lens, window, ratio, top_k, M, N)
        # token0（req0 首 token pos=start_pos=8）：topk_len=min(9//4,2)=2；swa_len=4
        assert cl[0].item() == 2 + 4
        assert ci[0, :2].tolist() == [3 + 0 * M, -1]
        # gather_start=10-4=6；swa 段 = N + pos - swa_len + 1 - gather_start
        base0 = 0 * M + N + 8 - 4 + 1 - 6
        assert ci[0, 2:6].tolist() == [base0, base0 + 1, base0 + 2, base0 + 3]
        # token2（req1 首 token pos=5）：topk_len=min(6//4,2)=1
        assert cl[2].item() == 1 + 4
        assert ci[2, 0].item() == 2 + 1 * M  # topk[2,0]=2 + M·b

    def test_v4_decode_dual_source_attention(self):
        """站 14/m13：一核双源 out == softmax over union(SWA, top-k)——NSA 回归证明。"""
        from vllm.forward_context import ForwardContext, set_forward_context
        from vllm.models.deepseek_v4.attention import DeepseekV4Attention
        from vllm.models.deepseek_v4.nvidia.flashmla import (
            DeepseekV4FlashMLAAttention)

        class _Concrete(DeepseekV4FlashMLAAttention):
            pass  # get_padded_num_q_heads/_o_proj 沿用父类实现

        torch.manual_seed(23)
        hf = make_v4_hf_config()
        hf.sliding_window = 8
        vllm_config = make_vllm_config(hf, max_model_len=64,
                                       max_num_batched_tokens=64,
                                       cache_dtype="fp8_ds_mla")
        buf = torch.full((64, hf.index_topk), -1, dtype=torch.int32)  # -1 哨兵
        layer = _Concrete(
            vllm_config=vllm_config, prefix="model.layers.1.self_attn",
            topk_indices_buffer=buf, aux_stream_list=None,
            eager_scratch_pool=None)
        assert layer.indexer is not None
        assert layer.compress_ratio == 4

        B = 2
        num_decode_tokens = B
        H = layer.n_local_heads
        D = 512
        q = torch.randn(num_decode_tokens, H, D) * 0.5
        positions = torch.tensor([15, 11])
        L = (positions + 1).tolist()
        # SWA 缓存：滑窗位（请求内绝对 slot）
        swa_cache = torch.randn(4, 64, D)
        layer.swa_cache_layer.kv_cache = swa_cache
        kv_cache = torch.randn(4, 64, D)
        layer.kv_cache = kv_cache  # 压缩 KV 池（runner 侧回填位）
        comp_lens = [(l + 3) // 4 for l in L]
        bt_compressed, _ = make_block_table(comp_lens, 64)
        # buffer：请求内压缩坐标 top-k（C4A 局部 index）
        topk = torch.tensor([[3, 1, -1], [2, 0, -1]], dtype=torch.int32)
        buf[:B, :3] = topk
        swa_ids, swa_lens = self._swa_window_slots(positions, hf.sliding_window)
        swa_meta = _SwaMeta(num_decodes=B, num_decode_tokens=B, num_prefills=0,
                            num_prefill_tokens=0, block_size=64,
                            is_valid_token=torch.ones(B, dtype=torch.int32),
                            token_to_req_indices=torch.tensor(
                                [0, 1], dtype=torch.int32),
                            decode_swa_indices=swa_ids,
                            decode_swa_lens=swa_lens)
        swa_meta.tile_sched_c4a = object()
        attn_meta = _V4Meta(block_size=256, block_table=bt_compressed)
        out = torch.zeros(num_decode_tokens, H, D)
        ctx = ForwardContext(no_compile_layers={}, attn_metadata={
            layer.prefix: attn_meta,
            layer.swa_cache_layer.prefix: swa_meta}, slot_mapping={})
        with set_forward_context(ctx):
            layer.forward_mqa(q=q, kv=kv_cache, positions=positions, output=out)

        # 参考：union(SWA 行, 压缩 top-k 行) 上的 softmax 注意力
        for b in range(B):
            ids = swa_ids[b][: swa_lens[b]].tolist()
            comp_ids = [t for t in topk[b].tolist() if t >= 0]
            phys = [bt_compressed[b, t // 64].item() * 64 + t % 64
                    for t in comp_ids]
            keys = torch.cat([swa_cache.reshape(-1, D)[ids],
                              kv_cache.reshape(-1, D)[phys]])
            scores = (q[b] @ keys.T) * layer.scale
            probs = torch.softmax(scores, dim=-1)
            ref = probs @ keys
            assert torch.allclose(out[b], ref, rtol=1e-4, atol=1e-4), \
                f"req{b} dual-source mismatch"

    @staticmethod
    def _swa_window_slots(positions, window):
        ids, lens = [], []
        for p in positions.tolist():
            swa_len = min(p + 1, window)
            start = p - swa_len + 1
            ids.append(torch.arange(start, p + 1, dtype=torch.int32))
            lens.append(swa_len)
        max_len = max(lens)
        pad = torch.full((len(ids), max_len), -1, dtype=torch.int32)
        for i, row in enumerate(ids):
            pad[i, : len(row)] = row
        return pad, torch.tensor(lens, dtype=torch.int32)


class _KMeta:
    def __init__(self, slot_mapping):
        self.slot_mapping = slot_mapping


class _SwaMeta:
    def __init__(self, **kw):
        self.tile_sched_swaonly = None
        self.tile_sched_c4a = None
        self.tile_sched_c128a = None
        for k, v in kw.items():
            setattr(self, k, v)


class _V4Meta:
    def __init__(self, **kw):
        self.c128a_global_decode_topk_indices = None
        self.c128a_decode_topk_lens = None
        self.c128a_prefill_topk_indices = None
        for k, v in kw.items():
            setattr(self, k, v)


# ═══════════ E. 换算 / 加载 / 接线（站 6/8/12） ═══════════
class TestPlumbing:
    def test_triton_convert_req_index_to_global(self):
        """m07：out = block_table[req, idx//B]*B + idx%B；-1 直通；越界 -1。"""
        from vllm.v1.attention.backends.mla.sparse_utils import (
            triton_convert_req_index_to_global_index)

        req_id = torch.tensor([0, 0], dtype=torch.int32)
        bt = torch.tensor([[5, 7]], dtype=torch.int32)
        tok = torch.tensor([[70, -1], [130, 3]], dtype=torch.int32)
        out = triton_convert_req_index_to_global_index(
            req_id, bt, tok, BLOCK_SIZE=64, NUM_TOPK_TOKENS=2, BLOCK_N=1)
        assert out.tolist() == [[7 * 64 + 6, -1], [-1, 5 * 64 + 3]]
        out2, counts = triton_convert_req_index_to_global_index(
            req_id, bt, tok, BLOCK_SIZE=64, NUM_TOPK_TOKENS=2, BLOCK_N=1,
            return_valid_counts=True)
        assert counts.tolist() == [1, 1]

    def test_v32_consume_forward_mqa_selective_attention(self):
        """站 12/m14：V3.2 消费 = 只对选中 latent 条目真算（O(Lk)）。"""
        from vllm.v1.attention.backends.mla.flashmla_sparse import (
            FlashMLASparseImpl)

        torch.manual_seed(29)
        T, H, D, Dv = 3, 2, 576, 512
        scale = 0.1
        hf = make_small_hf_config()
        make_vllm_config(hf, max_model_len=64, max_num_batched_tokens=64)
        N = 16
        kv_cache = torch.randn(N, 1, D)
        topk = 4
        buf = torch.zeros(64, topk, dtype=torch.int32)
        ids = torch.tensor([[3, 1, 0, 2], [7, 5, 4, -1], [15, 0, 9, 8]],
                           dtype=torch.int32)
        buf[:T, :topk] = ids
        impl = FlashMLASparseImpl(
            num_heads=H, head_size=D, scale=scale, num_kv_heads=1,
            alibi_slopes=None, sliding_window=None, kv_cache_dtype="auto",
            logits_soft_cap=None, attn_type="decoder",
            kv_sharing_target_layer_name=None,
            topk_indices_buffer=buf,
            q_lora_rank=512, kv_lora_rank=512, qk_nope_head_dim=512,
            qk_rope_head_dim=64, qk_head_dim=576, v_head_dim=128,
            kv_b_proj=None)
        q = torch.randn(T, H, D)
        meta = _SparseMeta(
            req_id_per_token=torch.tensor([0, 0, 0], dtype=torch.int32),
            block_table=torch.zeros(1, 1, dtype=torch.int32),
            block_size=64)
        attn_out, lse = impl.forward_mqa(
            q, kv_cache, meta, layer=_LayerStub())
        lengths = [4, 3, 4]
        ref = ref_sparse_attn(q, kv_cache, ids, lengths, scale, head_dim_v=Dv)
        assert torch.allclose(attn_out, ref, rtol=1e-4, atol=1e-4)
        assert lse is None

    def test_try_load_fp8_indexer_wk_fuses_on_second_arrival(self):
        """m01 可信度：FP8 wk + BF16 weights_proj 的融合加载（两段缓冲）。"""
        from vllm.model_executor.models.deepseek_v2 import _try_load_fp8_indexer_wk

        prefix = "model.layers.0.self_attn.indexer"
        w = torch.randn(128, 64).to(torch.float8_e4m3fn)
        scale = torch.rand(128) + 0.5  # per-channel（每出通道一个 scale）
        buf, loaded = {}, set()
        params = {f"{prefix}.wk_weights_proj.weight": _ParamStub()}
        # 非 indexer.wk 名字：不处理
        assert _try_load_fp8_indexer_wk(
            "model.layers.0.mlp.gate_proj.weight", torch.randn(2, 2),
            buf, params, loaded, []) is False
        # weight 先到：缓冲等待
        assert _try_load_fp8_indexer_wk(
            f"{prefix}.wk.weight", w, buf, params, loaded, []) is True
        assert buf == {prefix: {"weight": w}}
        assert loaded == set()
        # scale 到齐：反量化融合进 wk_weights_proj（per-channel scale 组形）
        assert _try_load_fp8_indexer_wk(
            f"{prefix}.wk.weight_scale_inv", scale, buf, params, loaded,
            []) is True
        assert buf == {}
        assert loaded == {f"{prefix}.wk_weights_proj.weight"}
        got = params[f"{prefix}.wk_weights_proj.weight"].loaded
        assert got.shape == (128, 64) and got.dtype == torch.bfloat16
        assert torch.allclose(got.float(), (w.float() * scale.unsqueeze(-1)),
                              rtol=1e-2)

    def test_try_load_fp8_indexer_wk_skips_pp_missing(self):
        from vllm.model_executor.models.deepseek_v2 import _try_load_fp8_indexer_wk

        prefix = "model.layers.0.self_attn.indexer"
        res = _try_load_fp8_indexer_wk(
            f"{prefix}.wk.weight",
            torch.randn(4, 4).to(torch.float8_e4m3fn),
            {}, {"x": _ParamStub()}, set(), [f"{prefix}.wk."])
        assert res is True

    def test_mla_wrapper_calls_indexer_before_mla_attn(self):
        """站 6：indexer 纯副作用接线——先于 mla_attn、返回值无人消费。"""
        from vllm.model_executor.layers.mla import (MLAModules,
                                                    MultiHeadLatentAttentionWrapper)

        hf = make_small_hf_config()

        class _Rope2(torch.nn.Module):
            def __call__(self, pos, q, k):
                return q, k

        calls = []
        rope = _Rope2()
        indexer = _OrderRecorder(calls, "indexer")
        indexer.topk_tokens = 8  # wrapper __init__ 的 hasattr 断言位
        indexer.topk_indices_buffer = torch.zeros(8, 8, dtype=torch.int32)
        mla_attn = _OrderRecorder(calls, "mla_attn", ret=torch.zeros(3, 8))
        wrapper = MultiHeadLatentAttentionWrapper(
            hidden_size=hf.hidden_size, num_heads=2, scale=0.1,
            qk_nope_head_dim=hf.qk_nope_head_dim,
            qk_rope_head_dim=hf.qk_rope_head_dim, v_head_dim=hf.v_head_dim,
            q_lora_rank=hf.q_lora_rank, kv_lora_rank=hf.kv_lora_rank,
            mla_modules=MLAModules(
                kv_a_layernorm=_IdNorm(), kv_b_proj=_IdLinear(512, 8),
                rotary_emb=rope, o_proj=_IdLinear(8, hf.hidden_size),
                fused_qkv_a_proj=_IdLinear(hf.hidden_size,
                                           hf.q_lora_rank + 512 + 64),
                kv_a_proj_with_mqa=None,
                q_a_layernorm=_IdNorm(),
                q_b_proj=_IdLinear(hf.q_lora_rank, 2 * 192),
                q_proj=None, indexer=indexer, is_sparse=True,
                topk_indices_buffer=torch.zeros(8, 8, dtype=torch.int32),
                indexer_rotary_emb=rope),
        )
        wrapper.mla_attn = mla_attn
        T = 3
        out = wrapper(torch.arange(T), torch.randn(T, hf.hidden_size))
        assert calls == ["indexer", "mla_attn"]  # indexer 先于稀疏 MLA
        assert out.shape == (3, hf.hidden_size)  # o_proj 收尾
        # skip_topk=True：indexer 不被调用
        wrapper.skip_topk = True
        calls.clear()
        wrapper(torch.arange(T), torch.randn(T, hf.hidden_size))
        assert calls == ["mla_attn"]

    def test_maybe_execute_in_parallel_sequential_join_order(self):
        """站 13：aux_stream=None 退化为顺序执行，join 保证 K 写先于打分。"""
        from vllm.utils.multi_stream_utils import maybe_execute_in_parallel

        order = []
        ev0, ev1 = torch.cuda.Event(), torch.cuda.Event()
        r0, r1 = maybe_execute_in_parallel(
            lambda: (order.append("wq_b"), "q")[1],
            lambda: (order.append("compressor"), "k")[1],
            ev0, ev1, None)
        assert (r0, r1) == ("q", "k")
        assert order == ["wq_b", "compressor"]


class _SparseMeta:
    def __init__(self, **kw):
        self.fp8_extra_metadata = None
        self.fp8_use_mixed_batch = False
        for k, v in kw.items():
            setattr(self, k, v)
