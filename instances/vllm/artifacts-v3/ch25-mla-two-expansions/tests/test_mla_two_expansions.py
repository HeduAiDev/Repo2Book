# ch25《MLA 的两种展开》测试电池 —— TDD：先测真实 vLLM v0.27.1（6e448d0ea）
# 的可观察行为，精简版实现到通过为止。全部 host 可跑（纯单元，不依赖真 vllm
# 包安装、无 CUDA 上下文——CUDA kernel 面（FlashMLA / merge_attn_states /
# concat_and_cache_mla / gather_cache）以 HOST SEAM 镜像承载精确数学；
# 后端选择按 ch21 域的真实注入位（MLAAttention.__init__ 的 attn_backend 参数
# + register_backend(CUSTOM) 覆盖表）以参考后端注入）。
#
# 行为基准 = 真实源码（instances/vllm/source，v0.27.1 现核行号）：
#   - deepseek_v2.py:L1226-L1238（选型三岔）/L1006-L1066（投影装配）
#   - mla.py:L150-L226（外层低秩链 + 解耦 RoPE）/L110-L127（Wrapper 装配）
#   - mla_attention.py:L687-L992（forward_impl 混批分流）/L994-L1100（吸收重排）
#     /L2581-L2666（forward_mha）/L2301-L2422（分块上下文）/L1140-L1152（spec 自报）
#   - backends/utils.py:L564-L635（split）/L665-L742（reorder 四区）
#   - kv_cache_interface.py:L388-L468（MLAAttentionSpec 字节账）
#   - kv_cache_utils.py:L1781-L1852（hybrid 组化四级）
#   - gpu_model_runner.py:L1115-L1138/L7220-L7238/L7800-L7837（runner 三薄层）
#   - flashmla.py:L118-L122（阈值 128）/L266-L346（forward_mqa 调用面）
#   - deepseek_v4/attention.py:L210-L213（逐层 compress_ratio）/L655-L674（spec）
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass, replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "implementation"))

from vllm.config import (  # noqa: E402
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import get_tensor_model_parallel_world_size  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402
from vllm.model_executor.layers.attention import MLAAttention  # noqa: E402
from vllm.model_executor.layers.mla import (  # noqa: E402
    MLAModules,
    MultiHeadLatentAttentionWrapper,
)
from vllm.model_executor.models.deepseek_v2 import (  # noqa: E402
    DeepseekV2DecoderLayer,
    DeepseekV2MLAAttention,
)
from vllm.model_executor.layers.attention.mla_attention import (  # noqa: E402
    MLADims,
    MLACommonBackend,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    build_mla_chunked_context_metadata,
    get_mla_dims,
)
from vllm.models.deepseek_v4.attention import DeepseekV4Attention  # noqa: E402
from vllm.third_party.flashmla.flash_mla_interface import (  # noqa: E402
    flash_mla_with_kvcache,
)
from vllm.v1.attention.backends.mla.flashmla import (  # noqa: E402
    FlashMLABackend,
    FlashMLAMetadataBuilder,
)
from vllm.v1.attention.backends.mla.prefill.base import MLAPrefillBackend  # noqa: E402
from vllm.v1.attention.backends.mla.prefill.registry import (  # noqa: E402
    MLAPrefillBackendEnum,
    register_mla_prefill_backend,
)
from vllm.v1.attention.backends.registry import (  # noqa: E402
    AttentionBackendEnum,
    register_backend,
)
from vllm.v1.attention.backends.utils import (  # noqa: E402
    reorder_batch_to_split_decodes_and_prefills,
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.merge_attn_states import merge_attn_states  # noqa: E402
from vllm.v1.core.kv_cache_utils import get_kv_cache_groups  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.attention.backend import CommonAttentionMetadata  # noqa: E402

torch.manual_seed(2525)

# ─── 维度档：数值档（小 N/P/V，但 head_size=Lkv+R=576 与 DSV3 同实） ────────
MINI = dict(
    hidden_size=128,
    q_lora_rank=64,
    num_heads=4,
    qk_nope_head_dim=16,
    qk_rope_head_dim=64,
    kv_lora_rank=512,
    v_head_dim=16,
)
# 形状账档（更接近 DSV3 比例：N=8, P=64, V=64, H=512, Lq=192）
SHAPEY = dict(MINI, hidden_size=512, q_lora_rank=192, num_heads=8,
              qk_nope_head_dim=64, v_head_dim=64)

BLOCK = 64  # FlashMLA 家族的 kernel block size（flashmla.py:L58-L59）


def make_hf_cfg(dims, model_type="deepseek_v3", rope_theta=10000.0):
    cfg = SimpleNamespace(
        model_type=model_type,
        hidden_size=dims["hidden_size"],
        num_attention_heads=dims["num_heads"],
        q_lora_rank=dims["q_lora_rank"],
        kv_lora_rank=dims["kv_lora_rank"],
        qk_nope_head_dim=dims["qk_nope_head_dim"],
        qk_rope_head_dim=dims["qk_rope_head_dim"],
        v_head_dim=dims["v_head_dim"],
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        rope_parameters={"rope_type": "default", "rope_theta": rope_theta},
        num_hidden_layers=4,
        n_routed_experts=None,
        first_k_dense_replace=3,
        moe_layer_freq=1,
    )
    return cfg


def make_vllm_config(hf_cfg=None, dims=None, max_model_len=64, max_num_seqs=8,
                     block_size=BLOCK, cache_dtype="auto", compress_ratios=None):
    from vllm.config import (
        AttentionConfig, CacheConfig, CompilationConfig, ModelConfig,
        ParallelConfig, SchedulerConfig,
    )

    if hf_cfg is None:
        hf_cfg = make_hf_cfg(dims or MINI)
    vllm_config = VllmConfig()
    vllm_config.model_config = ModelConfig(
        hf_config=hf_cfg, dtype=torch.float32, max_model_len=max_model_len)
    vllm_config.cache_config = CacheConfig(
        cache_dtype=cache_dtype, block_size=block_size,
        calculate_kv_scales=False, kv_cache_dtype_skip_layers=None,
    )
    vllm_config.scheduler_config = SchedulerConfig(max_num_seqs=max_num_seqs)
    vllm_config.parallel_config = ParallelConfig()
    vllm_config.compilation_config = CompilationConfig()
    vllm_config.attention_config = AttentionConfig()
    vllm_config.speculative_config = None
    vllm_config.kv_transfer_config = None
    return vllm_config


# ═══ 参考后端（真实注入位承载，数学 = 文件头伪码的精确注意力） ═══════════


@pytest.fixture(autouse=True)
def _sm90_platform(monkeypatch):
    """host 平台的算力代占位：SM90（FlashMLA 家族的属地）。真实选择器的
    显式 mla_prefill_backend 支在 device_capability 为 None 时不会走到
    （L95-L100 的回退支）——host 测试以 SM90 走真实显式选择分支
    （validate_configuration → get_class → CUSTOM 覆盖表）。"""
    from vllm.platforms import current_platform
    from vllm.platforms.interface import DeviceCapability

    monkeypatch.setattr(
        type(current_platform), "get_device_capability",
        classmethod(lambda cls: DeviceCapability(9, 0)),
    )


class RefPrefillBackend(MLAPrefillBackend):
    """参考 prefill 后端：varlen 精确注意力（new-token 段 causal / 上下文段
    非 causal，按 cu_seq_lens 分请求），LSE 以 [N, T] 返回——FA 序约定。"""

    # host 数值档用 float32（真实家族仅 fp16/bf16——参考后端放宽声明）
    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]

    @staticmethod
    def get_name() -> str:
        return "REF_PREFILL"

    def _attend(self, q, k, v, causal, cu):
        # q [Tq, N, D] / k [Tk, N, D] / v [Tk, N, V] / cu: 每请求 key 行边界
        N, D = q.shape[1], q.shape[2]
        V = v.shape[2]
        out = torch.zeros_like(q[..., :V])
        lse = torch.full((N, q.shape[0]), -float("inf"))
        for r in range(len(cu) - 1):
            qs, qe = cu[r], cu[r + 1]  # 本请求的 query 行（q/k 同段）
            ks = cu[r] if causal else 0
            ke = qe if causal else cu[r + 1]
            qq = q[qs:qe]  # [tq, N, D]
            kk, vv = k[ks:ke], v[ks:ke]  # [tk, N, D/V]
            s = torch.einsum("tnd,snd->nts", qq, kk) * self.scale
            tq = qe - qs
            if causal:
                mask = torch.triu(torch.ones(tq, tq, dtype=torch.bool), 1)
                s.masked_fill_(mask.unsqueeze(0), -float("inf"))
            p = torch.softmax(s, dim=-1)  # [N, tq, tk]
            out[qs:qe] = torch.einsum("nts,snv->tnv", p, vv)
            lse[:, qs:qe] = torch.logsumexp(s, dim=-1)
        return out, lse

    def run_prefill_new_tokens(self, q, k, v, return_softmax_lse,
                               out=None, output_scale=None):
        md = self._prefill_metadata
        cu = md.query_start_loc.tolist()
        o, lse = self._attend(q, k, v, causal=True, cu=cu)
        return (o, lse) if return_softmax_lse else o

    def run_prefill_context_chunk(self, chunk_idx, q, k, v):
        # 上下文块：非 causal——每个请求的**全部** query 行对该请求的
        # chunk 行（cu 是 chunk 的 per-request key 边界，不是 query 边界）
        md = self._prefill_metadata
        cu = md.chunked_context.cu_seq_lens[chunk_idx].tolist()
        qsl = md.query_start_loc.tolist()
        N, V = q.shape[1], v.shape[2]
        out = torch.zeros(q.shape[0], N, V)
        lse = torch.full((N, q.shape[0]), -float("inf"))
        for r in range(len(qsl) - 1):
            qq = q[qsl[r]:qsl[r + 1]]  # 本请求全部 query 行
            kk, vv = k[cu[r]:cu[r + 1]], v[cu[r]:cu[r + 1]]  # 本请求 chunk 行
            if kk.shape[0] == 0:
                continue  # 空上下文行：lse=-inf / out=0（mask_empty_context 语义）
            s = torch.einsum("tnd,snd->nts", qq, kk) * self.scale
            p = torch.softmax(s, dim=-1)
            out[qsl[r]:qsl[r + 1]] = torch.einsum("nts,snv->tnv", p, vv)
            lse[:, qsl[r]:qsl[r + 1]] = torch.logsumexp(s, dim=-1)
        return out, lse


# 参考后端经真实注册表机制登记（CUSTOM 槽位——第三方注册流）
register_mla_prefill_backend(
    MLAPrefillBackendEnum.CUSTOM,
    f"{__name__}.RefPrefillBackend",
)


class RefFlashMLAImplBase:
    """参考 decode impl 基座：FlashMLAImpl.forward_mqa 的 host 形态——
    同一 flash_mla_with_kvcache 调用面（flashmla.py:L332-L342），仅去掉
    spec-decode reshape（delete[4]）与 fp8 双 kernel 分派（delete[6]）。"""

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        assert kv_c_and_k_pe_cache.numel() > 0
        assert attn_metadata.decode is not None
        if type(q) is tuple:
            q = torch.cat(q, dim=-1)
        num_decodes = attn_metadata.num_decodes
        # SUBTRACTED(spec-decode reshape)：均匀段直接 view 成 [B, sq, N, D]
        sq = q.shape[0] // num_decodes
        q = q.view(num_decodes, sq, *q.shape[1:])
        o, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=kv_c_and_k_pe_cache.unsqueeze(-2),  # Add head dim of 1
            block_table=attn_metadata.decode.block_table,
            cache_seqlens=attn_metadata.decode.seq_lens,
            head_dim_v=self.kv_lora_rank,
            tile_scheduler_metadata=attn_metadata.decode.scheduler_metadata,
            softmax_scale=self.scale,
            causal=True,
            is_fp8_kvcache=False,
        )
        return o.reshape(-1, *o.shape[2:]), lse


from vllm.model_executor.layers.attention.mla_attention import MLACommonImpl  # noqa: E402


class RefFlashMLAImpl(RefFlashMLAImplBase, MLACommonImpl):
    pass


class RefFlashMLABackend(MLACommonBackend):
    @staticmethod
    def get_name() -> str:
        return "REF_FLASHMLA"

    @staticmethod
    def get_builder_cls():
        return FlashMLAMetadataBuilder

    @staticmethod
    def get_impl_cls():
        return RefFlashMLAImpl


@register_backend(AttentionBackendEnum.CUSTOM)
class RefRegisteredBackend(RefFlashMLABackend):
    """经真实 register_backend 覆盖表登记（registry.py 的第三方注册流）——
    selector 的显式后端路径（attention_config.backend="CUSTOM"）解析它。"""

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"


def enable_ref_backends(vllm_config):
    """把参考后端接进两个真实选择轴：decode 家族（attention_config.backend
    → register_backend 覆盖表）与 prefill 家族（mla_prefill_backend →
    MLAPrefillBackendEnum.CUSTOM 注册槽）。"""
    vllm_config.attention_config.backend = "CUSTOM"
    vllm_config.attention_config.mla_prefill_backend = MLAPrefillBackendEnum.CUSTOM
    return vllm_config


# ═══ 装配与 metadata 工具 ═══════════════════════════════════════════════


def build_mla_layer(vllm_config, prefix="model.layers.0.self_attn", dims=MINI):
    hf_cfg = vllm_config.model_config.hf_config
    with set_current_vllm_config(vllm_config):
        layer = DeepseekV2MLAAttention(
            vllm_config=vllm_config,
            config=hf_cfg,
            hidden_size=dims["hidden_size"],
            num_heads=dims["num_heads"],
            qk_nope_head_dim=dims["qk_nope_head_dim"],
            qk_rope_head_dim=dims["qk_rope_head_dim"],
            v_head_dim=dims["v_head_dim"],
            q_lora_rank=dims["q_lora_rank"],
            kv_lora_rank=dims["kv_lora_rank"],
            max_position_embeddings=hf_cfg.max_position_embeddings,
            cache_config=vllm_config.cache_config,
            quant_config=None,
            prefix=prefix,
        )
    inner = layer.mla_attn.mla_attn  # MLAAttention 插座
    with set_current_vllm_config(vllm_config):
        inner.process_weights_after_loading(torch.float32)  # 吸收重排（站 4）
    return layer, inner


def make_common(query_lens, seq_lens, num_blocks=64, block=BLOCK):
    """构造 CommonAttentionMetadata：批已按 decode→…→prefill 排好。"""
    n = len(query_lens)
    qsl = torch.zeros(n + 1, dtype=torch.int32)
    qsl[1:] = torch.cumsum(torch.tensor(query_lens, dtype=torch.int32), 0)
    seq = torch.tensor(seq_lens, dtype=torch.int32)
    blocks_per = [(s + block - 1) // block for s in seq]
    max_blocks = max(1, max(blocks_per))
    bt = torch.full((n, max_blocks), -1, dtype=torch.int32)
    b = 0
    for i, nb in enumerate(blocks_per):
        bt[i, :nb] = torch.arange(b, b + nb)
        b += nb
    num_tokens = int(qsl[-1])
    slots = torch.zeros(num_tokens, dtype=torch.int64)
    off = 0
    for i, (q, s) in enumerate(zip(query_lens, seq_lens)):
        for j in range(q):
            pos = s - q + j
            slots[off + j] = int(bt[i, pos // block]) * block + pos % block
        off += q
    return (
        CommonAttentionMetadata(
            query_start_loc=qsl,
            query_start_loc_cpu=qsl.clone(),
            seq_lens=seq,
            num_reqs=n,
            num_actual_tokens=num_tokens,
            max_query_len=max(query_lens),
            max_seq_len=int(seq.max()),
            block_table_tensor=bt,
            slot_mapping=slots,
            seq_lens_cpu_upper_bound=seq.clone(),
        ),
        slots,
    )


def build_metadata(builder, query_lens, seq_lens):
    common, slots = make_common(query_lens, seq_lens)
    md = builder.build(0, common)
    return md, common, slots


# ═══ 数值 oracle：上投影 MHA（文件头 Compute Friendly 伪码的精确实现） ═══
# (per-request oracle inlined in each test)


# ═══════════════════════════════════════════════════════════════════════
# 幕一：装配（站 1-4）
# ═══════════════════════════════════════════════════════════════════════


class TestAssembly:
    def test_trichotomy_use_mla(self):
        # deepseek_v2.py:L1226-L1238：非 deepseek(MHA) 且 use_mla → MLA 层
        vllm_config = enable_ref_backends(make_vllm_config(dims=MINI))
        with set_current_vllm_config(vllm_config):
            dl = DeepseekV2DecoderLayer(
                vllm_config, "model.layers.0",
                config=vllm_config.model_config.hf_config,
            )
        assert type(dl.self_attn) is DeepseekV2MLAAttention
        assert dl.use_mha is False

    def test_trichotomy_old_deepseek_mha(self):
        # model_type=="deepseek"（老 Deepseek）→ use_mha → DeepseekAttention
        vllm_config = enable_ref_backends(make_vllm_config(dims=MINI))
        vllm_config.model_config.hf_config.model_type = "deepseek"
        with set_current_vllm_config(vllm_config):
            dl = DeepseekV2DecoderLayer(
                vllm_config, "model.layers.0",
                config=vllm_config.model_config.hf_config,
            )
        assert dl.use_mha is True

    def test_projection_bricks_shapes(self):
        # L1006-L1066：六块积木 + 融合下投影的 [out, in] 朝向契约
        vllm_config = enable_ref_backends(make_vllm_config(dims=SHAPEY))
        layer, inner = build_mla_layer(vllm_config, dims=SHAPEY)
        d = SHAPEY
        H, Lq, N, P, R, Lkv, V = (d["hidden_size"], d["q_lora_rank"],
                                  d["num_heads"], d["qk_nope_head_dim"],
                                  d["qk_rope_head_dim"], d["kv_lora_rank"],
                                  d["v_head_dim"])
        w_fused = layer.fused_qkv_a_proj.weight
        assert w_fused.shape == (Lq + Lkv + R, H)  # [out, in]
        assert layer.kv_b_proj.weight.shape == (N * (P + V), Lkv)
        assert layer.q_b_proj.weight.shape == (N * (P + R), Lq)
        assert layer.o_proj.weight.shape == (H, N * V)
        assert layer.q_a_layernorm.weight.shape == (Lq,)
        assert layer.kv_a_layernorm.weight.shape == (Lkv,)
        assert layer.mla_attn.mla_attn is inner
        # kv_b_proj 引用被一并交进插座（吸收重排要靠它拿权重，站 3）
        assert inner.kv_b_proj is layer.kv_b_proj

    def test_no_q_lora_branch(self):
        # L1017-L1024 / L1043-L1050：q_lora_rank=None 时 kv_a_proj_with_mqa
        #（DSV2-Lite 形态；真实代码此时不定义 fused_qkv_a_proj 属性——
        #   MLAModules 的三元组按需短路取 None）
        dims = dict(MINI, q_lora_rank=None)
        hf = make_hf_cfg(dims)
        hf.q_lora_rank = None
        vllm_config = enable_ref_backends(make_vllm_config(hf_cfg=hf, dims=dims))
        layer, inner = build_mla_layer(vllm_config, dims=dims)
        assert not hasattr(layer, "fused_qkv_a_proj")
        assert layer.kv_a_proj_with_mqa is not None
        assert layer.kv_a_proj_with_mqa.weight.shape == (
            dims["kv_lora_rank"] + dims["qk_rope_head_dim"], dims["hidden_size"])
        assert not hasattr(layer, "q_b_proj") and layer.q_proj is not None

    def test_wrapper_is_pluggable_layer_with_modules(self):
        # mla.py:L35 注册位 + MLAModules 契约字段
        vllm_config = enable_ref_backends(make_vllm_config(dims=MINI))
        layer, _ = build_mla_layer(vllm_config)
        wrapper = layer.mla_attn
        assert isinstance(wrapper, MultiHeadLatentAttentionWrapper)
        for f in ("kv_a_layernorm", "kv_b_proj", "rotary_emb", "o_proj",
                  "fused_qkv_a_proj", "q_a_layernorm", "q_b_proj"):
            assert hasattr(wrapper, f)

    def test_absorption_reweight_layout(self):
        # process_weights_after_loading（L994-L1100）：kv_b_proj .T 拆
        # W_UK/W_UV → W_UV(N,L,V) 与 W_UK_T(N,P,L) 两份 bmm 副本
        vllm_config = enable_ref_backends(make_vllm_config(dims=MINI))
        layer, inner = build_mla_layer(vllm_config)
        d = MINI
        N, P, Lkv, V = (d["num_heads"], d["qk_nope_head_dim"],
                        d["kv_lora_rank"], d["v_head_dim"])
        assert inner.W_UK_T.shape == (N, P, Lkv)
        assert inner.W_UV.shape == (N, Lkv, V)
        # 副本与原权重逐元素一致（重排不改变数值，只改变布局）：
        w = inner.kv_b_proj.weight  # [N*(P+V), Lkv]
        w3 = w.T.view(Lkv, N, P + V)
        W_UK, W_UV = w3.split([P, V], dim=-1)  # 各 [Lkv, N, P/V]
        assert torch.equal(inner.W_UK_T, W_UK.permute(1, 2, 0))
        assert torch.equal(inner.W_UV, W_UV.transpose(0, 1))
        # bmm 形状账：q_nope(N,B,P)×W_UK_T(N,P,L)→(N,B,L)；v_up (N,B,L)×(N,L,V)
        B = 3
        q_nope = torch.randn(N, B, P)
        assert torch.bmm(q_nope, inner.W_UK_T).shape == (N, B, Lkv)
        assert torch.bmm(torch.bmm(q_nope, inner.W_UK_T),
                         inner.W_UV).shape == (N, B, V)

    def test_mla_backend_family_assert(self):
        # mla_attention.py:L423-L426：插座断言后端 is_mla —— 非 MLA 家族拒绝
        class NotMLABackend(MLACommonBackend):
            @classmethod
            def is_mla(cls):
                return False

        with pytest.raises(AssertionError, match="must be an MLA backend"):
            MLAAttention(
                num_heads=MINI["num_heads"], scale=0.1,
                qk_nope_head_dim=MINI["qk_nope_head_dim"],
                qk_rope_head_dim=MINI["qk_rope_head_dim"],
                v_head_dim=MINI["v_head_dim"],
                q_lora_rank=MINI["q_lora_rank"],
                kv_lora_rank=MINI["kv_lora_rank"],
                kv_b_proj=None,
                cache_config=None, prefix="x.attn",
                attn_backend=NotMLABackend,
            )


# ═══════════════════════════════════════════════════════════════════════
# 幕二：外层前向低秩链（站 9 前半）+ spec/组化（站 5-6）
# ═══════════════════════════════════════════════════════════════════════


class TestOuterForward:
    def _probe(self, vllm_config, dims, T=5):
        layer, inner = build_mla_layer(vllm_config, dims=dims)
        captured = {}

        class Recorder(torch.nn.Module):
            def forward(self, q, kv_c_normed, k_pe, output_shape=None,
                        q_dcp_replicated=None):
                captured["q"], captured["kv_c_normed"], captured["k_pe"] = (
                    q, kv_c_normed, k_pe)
                return torch.zeros(*output_shape)

        layer.mla_attn.mla_attn = Recorder()
        h = torch.randn(T, dims["hidden_size"])
        pos = torch.arange(T)
        out = layer(pos, h, None)
        assert out.shape == (T, dims["hidden_size"])
        # 形状账：q[Sq,N,P+R] / kv_c_normed[Sq,Lkv] / k_pe[Sq,1,R]
        N, P, R, Lkv = (dims["num_heads"], dims["qk_nope_head_dim"],
                        dims["qk_rope_head_dim"], dims["kv_lora_rank"])
        assert captured["q"].shape == (T, N, P + R)
        assert captured["kv_c_normed"].shape == (T, Lkv)
        assert captured["k_pe"].shape == (T, 1, R)
        return layer, captured, h, pos

    def test_lowrank_chain_and_decoupled_rope(self):
        # mla.py:L150-L226：q 低秩链 / KV 压缩 / RoPE 只打 rope 段（m02）
        vllm_config = enable_ref_backends(make_vllm_config(dims=MINI))
        dims = MINI
        layer, captured, h, pos = self._probe(vllm_config, dims)
        w = layer.fused_qkv_a_proj.weight  # [Lq+Lkv+R, H]
        Lq, Lkv, R, N, P = (dims["q_lora_rank"], dims["kv_lora_rank"],
                            dims["qk_rope_head_dim"], dims["num_heads"],
                            dims["qk_nope_head_dim"])
        qkv = h @ w.T
        q_c, kv_lora = qkv.split([Lq, Lkv + R], dim=-1)
        q_c = layer.q_a_layernorm(q_c)
        q_manual = (q_c @ layer.q_b_proj.weight.T).view(-1, N, P + R)
        # nope 段不旋转（低秩可吸收的数学前提——逐位相等）
        assert torch.equal(captured["q"][..., :P], q_manual[..., :P])
        # rope 段被旋转（positions>0 处必然不同）
        assert not torch.allclose(captured["q"][1:, :, P:], q_manual[1:, :, P:])
        # kv_a_layernorm 只打 kv_c、不打 k_pe；k_pe 的 rope 段同样被旋转
        #（潜向量 = norm(kv_c) ⊕ roped(k_pe)——cache 写腿的 576 维构成）
        kv_c, k_pe_raw = kv_lora.split([Lkv, R], dim=-1)
        assert torch.allclose(captured["kv_c_normed"],
                              layer.kv_a_layernorm(kv_c), atol=1e-6)
        _, k_pe_rot = layer.mla_attn.rotary_emb(
            pos, k_pe_raw.unsqueeze(1), k_pe_raw.unsqueeze(1))
        assert torch.allclose(captured["k_pe"], k_pe_rot, atol=1e-6)
        assert not torch.allclose(captured["k_pe"][1:], k_pe_raw[1:].unsqueeze(1),
                                  atol=1e-6)


class TestSpecAndGrouping:
    def test_get_kv_cache_spec_self_report(self):
        # mla_attention.py:L1140-L1152：MLAAttentionSpec(num_kv_heads=1,
        # head_size=576)；L1370-L1377 cache shape 无 K/V 拆维
        vllm_config = enable_ref_backends(make_vllm_config(dims=MINI))
        _, inner = build_mla_layer(vllm_config)
        with set_current_vllm_config(vllm_config):
            spec = inner.get_kv_cache_spec(vllm_config)
        assert isinstance(spec, MLAAttentionSpec)
        assert spec.num_kv_heads == 1
        assert spec.head_size == 576
        assert spec.block_size == BLOCK
        assert MLACommonBackend.get_kv_cache_shape(100, BLOCK, 1, 576) == (
            100, BLOCK, 576)
        assert MLACommonBackend.get_supported_head_sizes() == [320, 576]
        assert MLACommonBackend.is_mla() is True

    def test_mla_attention_spec_byte_accounts(self):
        # kv_cache_interface.py:L403-L426：584B/656B 特账 + storage÷compress
        bf16 = MLAAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=576,
                                dtype=torch.bfloat16)
        assert bf16.real_page_size_bytes == BLOCK * 576 * 2
        v4 = MLAAttentionSpec(block_size=256, num_kv_heads=1, head_size=512,
                              dtype=torch.uint8, cache_dtype_str="fp8_ds_mla",
                              compress_ratio=4, alignment=576,
                              model_version="deepseek_v4")
        assert v4.storage_block_size == 64  # 256 // 4
        assert v4.real_page_size_bytes == 64 * 584
        assert v4.page_size_padded == math.ceil(64 * 584 / 576) * 576
        v32 = MLAAttentionSpec(block_size=256, num_kv_heads=1, head_size=576,
                               dtype=torch.uint8, cache_dtype_str="fp8_ds_mla")
        assert v32.real_page_size_bytes == 256 * 656
        # merge 断言组内四字段全同（L429-L468）
        same = MLAAttentionSpec(block_size=64, num_kv_heads=1, head_size=576,
                                dtype=torch.bfloat16)
        assert MLAAttentionSpec.merge([bf16, same]).head_size == 576
        mixed = replace(v4, compress_ratio=1)
        with pytest.raises(AssertionError):
            MLAAttentionSpec.merge([v4, mixed])
        with pytest.raises(AssertionError):
            MLAAttentionSpec.merge([v4, replace(v4, model_version=None)])

    def test_get_mla_dims_dual_format(self):
        # L1499-L1530：DSV4 统一 head_dim=512 记法 vs DSV2/V3 分离字段
        v3cfg = SimpleNamespace(kv_lora_rank=512, q_lora_rank=1536,
                                qk_nope_head_dim=128, qk_rope_head_dim=64,
                                v_head_dim=128)
        dims = get_mla_dims(SimpleNamespace(hf_text_config=v3cfg))
        assert dims == MLADims(q_lora_rank=1536, kv_lora_rank=512,
                               qk_nope_head_dim=128, qk_rope_head_dim=64,
                               v_head_dim=128)
        v4cfg = SimpleNamespace(compress_ratios=[1], head_dim=512,
                                qk_rope_head_dim=64, q_lora_rank=1536)
        dims4 = get_mla_dims(SimpleNamespace(hf_text_config=v4cfg))
        assert dims4.kv_lora_rank == 512
        assert dims4.qk_nope_head_dim == 512 - 64 == 448
        assert dims4.v_head_dim == 512

    def test_get_kv_cache_groups_uniform(self):
        # 四级分流第一级：全同 spec → 一组
        spec = MLAAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=576,
                                dtype=torch.bfloat16)
        layers = {f"model.layers.{i}.self_attn.attn": spec for i in range(61)}
        vllm_config = make_vllm_config(dims=MINI)
        groups = get_kv_cache_groups(vllm_config, dict(layers))
        assert len(groups) == 1
        assert len(groups[0].layer_names) == 61
        assert isinstance(groups[0].kv_cache_spec, MLAAttentionSpec)

    def test_get_kv_cache_groups_dsv4_grouped(self):
        # 四级分流第三级：DSV4 grouped（MLA + 不同页大小的 SWA-MLA 多组）
        mla = MLAAttentionSpec(block_size=256, num_kv_heads=1, head_size=512,
                               dtype=torch.uint8, cache_dtype_str="fp8_ds_mla",
                               compress_ratio=4, alignment=576,
                               model_version="deepseek_v4")
        # storage 64×584 → 与 SWA 的 bf16 页（64×576×2=73728B）不同页大小
        swa = SlidingWindowMLASpec(block_size=64, num_kv_heads=1, head_size=512,
                                   dtype=torch.bfloat16, sliding_window=4096,
                                   model_version="deepseek_v4")
        layers = {f"c4a.{i}": mla for i in range(3)}
        layers.update({f"swa.{i}": swa for i in range(3)})
        vllm_config = make_vllm_config(dims=MINI)
        groups = get_kv_cache_groups(vllm_config, dict(layers))
        assert len(groups) == 2  # 全 MLA 一组 + SWA 一组
        assert all(n.startswith("c4a") for n in groups[0].layer_names)
        assert all(n.startswith("swa") for n in groups[1].layer_names)
        assert isinstance(groups[0].kv_cache_spec, UniformTypeKVCacheSpecs)

    def test_get_kv_cache_groups_uniform_page_size(self):
        # 四级分流第四级：一般混合（full + swa 同页大小）→ 页归一多组
        #（2 full + 4 swa → group_size=2 → 3 组各 2 层）
        full = FullAttentionSpec(block_size=BLOCK, num_kv_heads=8, head_size=128,
                                 dtype=torch.bfloat16)
        swa = SlidingWindowSpec(block_size=BLOCK, num_kv_heads=8, head_size=128,
                                dtype=torch.bfloat16, sliding_window=512)
        assert full.page_size_bytes == swa.page_size_bytes  # 页大小可归一
        layers = {f"full.{i}": full for i in range(2)}
        layers.update({f"swa.{i}": swa for i in range(4)})
        vllm_config = make_vllm_config(dims=MINI)
        groups = get_kv_cache_groups(vllm_config, dict(layers))
        assert len(groups) == 3
        assert all(len(g.layer_names) == 2 for g in groups)
        kinds = ["full" if n.startswith("full") else "swa"
                 for g in groups for n in g.layer_names[:1]]
        assert kinds.count("full") == 1 and kinds.count("swa") == 2

    def test_attention_free_returns_empty(self):
        vllm_config = make_vllm_config(dims=MINI)
        assert get_kv_cache_groups(vllm_config, {}) == []


# ═══════════════════════════════════════════════════════════════════════
# 幕三：每拍前置（站 7-8）——批重排与边界计数
# ═══════════════════════════════════════════════════════════════════════


class TestReorderAndSplit:
    def test_split_decodes_and_prefills_boundary(self):
        # utils.py:L564-L635：排好序的批上找边界（decode→…→prefill）
        qsl = torch.tensor([0, 1, 2, 3, 5, 13, 77], dtype=torch.int32)
        common = SimpleNamespace(
            max_query_len=64, num_reqs=6, num_actual_tokens=77,
            query_start_loc_cpu=qsl,
        )
        nd, np_, ndt, npt = split_decodes_and_prefills(common, decode_threshold=2)
        # query_lens=[1,1,1,2,8,64]、阈值 2 → 前 4 请求（5 token）是 decode
        assert (nd, np_, ndt, npt) == (4, 2, 5, 72)
        # 全 decode 早退（L599-L604）
        common2 = SimpleNamespace(max_query_len=1, num_reqs=3,
                                  num_actual_tokens=3,
                                  query_start_loc_cpu=torch.tensor([0, 1, 2, 3],
                                                                   dtype=torch.int32))
        assert split_decodes_and_prefills(common2, decode_threshold=1) == (3, 0, 3, 0)
        # 首请求即 prefill → 无 decode（L606-L609）
        common3 = SimpleNamespace(max_query_len=64, num_reqs=2,
                                  num_actual_tokens=65,
                                  query_start_loc_cpu=torch.tensor([0, 64, 65],
                                                                   dtype=torch.int32))
        assert split_decodes_and_prefills(common3, decode_threshold=2) == (0, 2, 0, 65)

    def test_reorder_batch_four_regions(self):
        # utils.py:L665-L742：互斥四区 + 换位成 decode→short→long→prefill
        import numpy as np

        class FakeBatch:
            """真实 InputBatch 消费面：req_ids 列表 + numpy 的
            num_computed_tokens_cpu / num_prompt_tokens + swap_states。"""

            def __init__(self, ids, computed, prompt):
                self.req_ids = list(ids)
                self.num_computed_tokens_cpu = np.array(computed)
                self.num_prompt_tokens = np.array(prompt)
                self.log = []

            def swap_states(self, a, b):
                self.log.append((a, b))
                lst = self.req_ids
                lst[a], lst[b] = lst[b], lst[a]
                for attr in ("num_computed_tokens_cpu", "num_prompt_tokens"):
                    arr = getattr(self, attr)
                    arr[a], arr[b] = arr[b], arr[a]

        # 五请求乱序：pref(首块) long(续灌) short(尾段) decode decode
        batch = FakeBatch(
            ids=["p0", "long", "short", "d0", "d1"],
            computed=[0, 8, 15, 10, 31],
            prompt=[8, 20, 16, 10, 31],
        )
        sched = SimpleNamespace(num_scheduled_tokens={
            "p0": 8, "long": 12, "short": 1, "d0": 1, "d1": 1})
        changed = reorder_batch_to_split_decodes_and_prefills(
            batch, sched, decode_threshold=2)
        assert changed is True
        # 期望序：d0 d1 short long p0（decode→short_extend→long_extend→prefill）
        ids = batch.req_ids
        assert ids.index("d0") < ids.index("d1") < ids.index("short") \
            < ids.index("long") < ids.index("p0")
        assert ids[0] in ("d0", "d1") and ids[-1] == "p0"
        # 已排好 → False（不动）
        batch2 = FakeBatch(ids=["d0", "d1", "p0"], computed=[10, 31, 0],
                           prompt=[10, 31, 8])
        sched2 = SimpleNamespace(num_scheduled_tokens={"d0": 1, "d1": 1, "p0": 8})
        assert reorder_batch_to_split_decodes_and_prefills(
            batch2, sched2, decode_threshold=2) is False
        assert batch2.log == []

    def test_flashmla_threshold_128(self):
        # flashmla.py:L121：后端自报阈值 128（'process small prefills with
        # decode pathway'）
        assert FlashMLAMetadataBuilder.reorder_batch_threshold == 128

    def test_runner_threshold_takes_min(self):
        # gpu_model_runner.py:L7220-L7238：取全部组 builder 声明的最小值
        from vllm.v1.worker.gpu_model_runner import GPUModelRunnerSlice

        def _runner_with(thresholds):
            runner = object.__new__(GPUModelRunnerSlice)
            runner._attn_group_iterator = lambda: [
                SimpleNamespace(get_metadata_builder=lambda t=t: SimpleNamespace(
                    reorder_batch_threshold=t))
                for t in thresholds
            ]
            return runner

        r1 = _runner_with([128, 512])
        r1.calculate_reorder_batch_threshold()
        assert r1.reorder_batch_threshold == 128
        r2 = _runner_with([128, None])
        r2.calculate_reorder_batch_threshold()
        assert r2.reorder_batch_threshold == 128
        r3 = _runner_with([])
        r3.calculate_reorder_batch_threshold()
        assert r3.reorder_batch_threshold is None

    def test_may_reorder_batch(self):
        # gpu_model_runner.py:L1115-L1138：零组早退；有阈值才重排
        from vllm.v1.worker.gpu_model_runner import GPUModelRunnerSlice

        runner = object.__new__(GPUModelRunnerSlice)
        runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[])
        runner.reorder_batch_threshold = 128
        runner.input_batch = SimpleNamespace()
        runner.input_batch.touched = False

        class Sched:
            pass

        runner._may_reorder_batch(Sched())  # 零组 → 原样返回

        class RecordingBatch:
            def __init__(self):
                self.calls = []

        def fake_reorder(batch, sched, decode_threshold):
            batch.calls.append(decode_threshold)
            return True

        import vllm.v1.worker.gpu_model_runner as gmr
        orig = gmr.reorder_batch_to_split_decodes_and_prefills
        gmr.reorder_batch_to_split_decodes_and_prefills = fake_reorder
        try:
            runner.kv_cache_config = SimpleNamespace(kv_cache_groups=[object()])
            runner.input_batch = RecordingBatch()
            runner._may_reorder_batch(Sched())
            assert runner.input_batch.calls == [128]
        finally:
            gmr.reorder_batch_to_split_decodes_and_prefills = orig

    def test_runner_collects_specs(self):
        # gpu_model_runner.py:L7800-L7837：遍历 static_forward_context 逐层收
        from vllm.v1.worker.gpu_model_runner import GPUModelRunnerSlice

        vllm_config = enable_ref_backends(make_vllm_config(dims=MINI))
        with set_current_vllm_config(vllm_config):
            _, inner0 = build_mla_layer(vllm_config, prefix="m.l0.self_attn")
            _, inner1 = build_mla_layer(vllm_config, prefix="m.l1.self_attn")

        class NullLayer:  # get_kv_cache_spec 返回 None 的层被跳过（DSV4 SWA 段）
            def get_kv_cache_spec(self, vllm_config):
                return None

            def get_attn_backend(self):
                return None

        vllm_config.compilation_config.static_forward_context["m.swa"] = NullLayer()
        runner = object.__new__(GPUModelRunnerSlice)
        runner.vllm_config = vllm_config
        runner.shared_kv_cache_layers = {}
        specs = runner.get_kv_cache_spec()
        assert set(specs) == {"m.l0.self_attn.attn", "m.l1.self_attn.attn"}
        assert all(s.num_kv_heads == 1 and s.head_size == 576
                   for s in specs.values())


# ═══════════════════════════════════════════════════════════════════════
# 幕四：一拍前向——两种展开的数学等价 + 混批分流（站 9-13）
# ═══════════════════════════════════════════════════════════════════════


def _make_builder(vllm_config, layer_name, dims=MINI, max_model_len=64,
                  max_num_seqs=8):
    spec = MLAAttentionSpec(block_size=BLOCK, num_kv_heads=1, head_size=576,
                            dtype=torch.float32)
    with set_current_vllm_config(vllm_config):
        builder = FlashMLAMetadataBuilder(
            spec, [layer_name], vllm_config, torch.device("cpu"))
    return builder


class TestWorkspaceAndChunking:
    def test_workspace_size_formula(self):
        # L1803-L1831：min(max(8×max_len, 4×seqs×block), 64k) 且 ≥1 页/请求
        vc = make_vllm_config(dims=MINI, max_model_len=1000, max_num_seqs=4,
                              block_size=16)
        assert MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
            vc) == max(min(max(8000, 256), 65536), 64) == 8000
        vc2 = make_vllm_config(dims=MINI, max_model_len=100_000, max_num_seqs=4,
                               block_size=16)
        assert MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
            vc2) == 65536
        vc3 = make_vllm_config(dims=MINI, max_model_len=2, max_num_seqs=32,
                               block_size=16)
        # 8×2=16 太小 → 4×32×16=2048 顶住（且 ≥ 1 页/请求 = 512）
        assert MLACommonMetadataBuilder.determine_chunked_prefill_workspace_size(
            vc3) == 2048

    def test_chunked_context_metadata(self):
        # build_mla_chunked_context_metadata：workspace 切片账本
        context_lens = torch.tensor([100, 50], dtype=torch.int32)
        qsl_cpu = torch.tensor([0, 7, 12], dtype=torch.int32)
        workspace = torch.zeros(256, 576)
        md = build_mla_chunked_context_metadata(
            context_lens_cpu=context_lens,
            prefill_query_start_loc_cpu=qsl_cpu,
            num_prefills=2,
            chunked_prefill_workspace=workspace,
            chunked_prefill_workspace_size=256,
            block_size=16,
            align_chunk_to_block=True,
            device=torch.device("cpu"),
            dcp_world_size=1,
            dcp_local_block_size=16,
            dcp_virtual_block_size=16,
        )
        assert md is not None
        # workspace 256 均摊给 2 个有上下文的 prefill → 每请求 128 → round_down
        # 到 block 16 → 128；chunks = cdiv(100, 128) = 1
        assert len(md.seq_tot) == 1
        assert md.seq_tot[0] == 150
        assert md.chunk_total_token[0].item() == 150
        # 2 个 prefill 都有上下文 → 有上下文前缀 = qsl[2] = 12
        assert md.prefill_tokens_with_context == 12
        assert torch.equal(md.seq_lens, torch.tensor([[100, 50]]))
        # token_to_seq 行 0 = 请求 0×100 + 请求 1×50
        tts = md.token_to_seq[0]
        assert tts.shape[0] == 150
        assert tts[:100].eq(0).all() and tts[100:].eq(1).all()
        # 无上下文 → None
        md0 = build_mla_chunked_context_metadata(
            context_lens_cpu=torch.tensor([0, 0], dtype=torch.int32),
            prefill_query_start_loc_cpu=qsl_cpu, num_prefills=2,
            chunked_prefill_workspace=workspace,
            chunked_prefill_workspace_size=256, block_size=16,
            align_chunk_to_block=True, device=torch.device("cpu"),
            dcp_world_size=1, dcp_local_block_size=16, dcp_virtual_block_size=16)
        assert md0 is None

    def test_builder_build_mixed_counts(self):
        # L2005-L2063：builder 数出 num_decodes/num_decode_tokens（已排好批）
        vllm_config = enable_ref_backends(make_vllm_config(dims=MINI))
        _, inner = build_mla_layer(vllm_config, prefix="m.l0.self_attn")
        builder = _make_builder(vllm_config, "m.l0.self_attn.attn")
        # FlashMLA 的 query_len_support=UNIFORM → require_uniform=True：
        # query_lens=[1,1,1,2,8]，2-token 短 extend 为保 decode 段均匀被划成
        # prefill（is_prefill = lens != lens[0]）——decodes=3、prefills=2
        md, common, slots = build_metadata(builder, [1, 1, 1, 2, 8],
                                           [10, 5, 3, 6, 8])
        assert md.num_decodes == 3
        assert md.num_prefills == 2
        assert md.num_decode_tokens == 3
        assert md.num_prefills + md.num_decodes == md.num_reqs
        assert md.decode is not None and md.prefill is not None
        assert md.prefill.chunked_context is not None  # prefill 有历史上下文
        # 纯 decode 批：prefill=None
        md2, _, _ = build_metadata(builder, [1, 1], [7, 3])
        assert md2.num_prefills == 0 and md2.prefill is None
        assert md2.num_decodes == 2 and md2.num_decode_tokens == 2


class TestTwoExpansionsEquiv:
    """两种展开的数学等价：同一注意力问题，MHA 上投影腿 == MQA 吸收腿。"""

    def _setup(self, dims=MINI, max_model_len=18, max_num_seqs=1, block=BLOCK):
        vllm_config = enable_ref_backends(
            make_vllm_config(dims=dims, max_model_len=max_model_len,
                             max_num_seqs=max_num_seqs, block_size=block))
        layer, inner = build_mla_layer(vllm_config, prefix="m.l0.self_attn",
                                       dims=dims)
        spec = MLAAttentionSpec(block_size=block, num_kv_heads=1, head_size=576,
                                dtype=torch.float32)
        with set_current_vllm_config(vllm_config):
            builder = FlashMLAMetadataBuilder(
                spec, ["m.l0.self_attn.attn"], vllm_config, torch.device("cpu"))
        num_blocks = 32
        inner.kv_cache = torch.zeros(num_blocks, block, 576)
        return vllm_config, layer, inner, builder, dims, block

    def _run(self, vllm_config, layer, builder, query_lens, seq_lens, h_new,
             positions, block=None):
        common, slots = make_common(query_lens, seq_lens, block=block or BLOCK)
        md = builder.build(0, common)
        layer_name = "m.l0.self_attn.attn"
        with set_forward_context({layer_name: md}, vllm_config,
                                 slot_mapping={layer_name: slots}):
            out = layer(positions, h_new, None)
        return out, md, common

    def test_decode_mqa_leg_equals_upprojected_mha(self):
        # 站 13：MQA 吸收腿（bmm→576 维 MQA→_v_up_proj）与上投影 MHA 等价
        dims = MINI
        vllm_config, layer, inner, builder, dims, blk = self._setup(
            dims, max_model_len=64, max_num_seqs=8)
        N, P, R, Lkv, V, H = (dims["num_heads"], dims["qk_nope_head_dim"],
                              dims["qk_rope_head_dim"], dims["kv_lora_rank"],
                              dims["v_head_dim"], dims["hidden_size"])
        hist_lens = [7, 3, 12]
        # ① 历史经真实管线写入 cache（prefill 段、无上下文）
        q_hist = [s for s in hist_lens]
        h_hist = torch.randn(sum(q_hist), H)
        pos_hist = torch.cat([torch.arange(s) for s in hist_lens])
        out_hist, md0, common0 = self._run(vllm_config, layer, builder,
                                           q_hist, hist_lens, h_hist, pos_hist)
        # ② decode 步：每请求 1 个新 token（MQA 腿）
        seq2 = [s + 1 for s in hist_lens]
        h_new = torch.randn(3, H)
        pos_new = torch.tensor(hist_lens)
        out, md1, common1 = self._run(vllm_config, layer, builder,
                                      [1, 1, 1], seq2, h_new, pos_new)
        assert md1.num_decode_tokens == 3 and md1.num_prefills == 0
        # ③ oracle：上投影 MHA（文件头 Compute Friendly 伪码）——重建 decode
        # 步的 q（已 rope）后逐请求对照
        cache = inner.kv_cache.view(-1, 576)
        w_fused = layer.fused_qkv_a_proj.weight
        Lq = dims["q_lora_rank"]
        qkv = h_new @ w_fused.T
        q_c = layer.q_a_layernorm(qkv[:, :Lq])
        q = (q_c @ layer.q_b_proj.weight.T).view(3, N, P + R)
        q_rot, _ = layer.mla_attn.rotary_emb(pos_new, q[..., P:].clone(),
                                             torch.zeros(3, 1, R))
        q_full = torch.cat([q[..., :P], q_rot], dim=-1)
        # 逐请求 oracle
        scale = (P + R) ** -0.5
        ref = torch.zeros(3, N, V)
        bt = common1.block_table_tensor
        for r, s in enumerate(seq2):
            rows = []
            for pos in range(s):
                blk = int(bt[r, pos // BLOCK])
                rows.append(blk * BLOCK + pos % BLOCK)
            lat = cache[torch.tensor(rows)]  # [s, 576]
            w = inner.kv_b_proj.weight  # [N*(P+V), Lkv]
            kv_nope = (lat[:, :Lkv] @ w.T).view(s, N, P + V)
            k_nope, v = kv_nope.split([P, V], dim=-1)
            k = torch.cat([k_nope, lat[:, Lkv:].unsqueeze(1).expand(-1, N, -1)],
                          dim=-1)
            s_mat = torch.einsum("tnd,snd->nts", q_full[r:r + 1], k) * scale
            p = torch.softmax(s_mat, dim=-1)
            ref[r] = torch.einsum("nts,snv->tnv", p, v).squeeze(0)
        ref_out = layer.o_proj(ref.reshape(3, N * V))[0]
        torch.testing.assert_close(out, ref_out, rtol=2e-4, atol=2e-4)

    def test_prefill_mha_chunked_context_equals_full(self):
        # 站 12：chunked prefill 分块 + LSE merge == 全量上投影 MHA
        dims = MINI
        # block=16/max_model_len=18/max_num_seqs=1 → workspace=144 < context=150
        # → 2 块；M=150 > 阈值 128 → 走 MHA 腿（≤128 的小 chunk 会按
        # 'process small prefills with decode pathway' 划进 MQA 段）
        vllm_config, layer, inner, builder, dims, blk = self._setup(
            dims, max_model_len=18, max_num_seqs=1, block=16)
        N, P, R, Lkv, V, H = (dims["num_heads"], dims["qk_nope_head_dim"],
                              dims["qk_rope_head_dim"], dims["kv_lora_rank"],
                              dims["v_head_dim"], dims["hidden_size"])
        C, M = 150, 150
        # ① 历史 prefill（首块，无上下文）写入 cache
        h_hist = torch.randn(C, H)
        pos_hist = torch.arange(C)
        self._run(vllm_config, layer, builder, [C], [C], h_hist, pos_hist,
                  block=blk)
        # ② 第二块 prefill（有 150 上下文 + 150 新 token）→ MHA 腿 + 分块 merge
        h_new = torch.randn(M, H)
        pos_new = torch.arange(C, C + M)
        out, md, common = self._run(vllm_config, layer, builder, [M], [C + M],
                                    h_new, pos_new, block=blk)
        assert md.num_decode_tokens == 0 and md.num_prefills == 1
        assert md.prefill.chunked_context is not None
        assert len(md.prefill.chunked_context.seq_tot) == 2  # 150/144 → 2 块
        # ③ oracle：全量（C+M token）上投影 MHA、块内 causal
        cache = inner.kv_cache.view(-1, 576)
        bt = common.block_table_tensor
        rows = [int(bt[0, pos // blk]) * blk + pos % blk for pos in range(C + M)]
        lat = cache[torch.tensor(rows)]
        w = inner.kv_b_proj.weight
        kv_nope = (lat[:, :Lkv] @ w.T).view(C + M, N, P + V)
        k_nope, v = kv_nope.split([P, V], dim=-1)
        k = torch.cat([k_nope, lat[:, Lkv:].unsqueeze(1).expand(-1, N, -1)], dim=-1)
        Lq = dims["q_lora_rank"]
        q_c = layer.q_a_layernorm(
            (h_new @ layer.fused_qkv_a_proj.weight.T)[:, :Lq])
        q = (q_c @ layer.q_b_proj.weight.T).view(M, N, P + R)
        q_rot, _ = layer.mla_attn.rotary_emb(pos_new, q[..., P:].clone(),
                                             torch.zeros(M, 1, R))
        q_full = torch.cat([q[..., :P], q_rot], dim=-1)
        s = torch.einsum("tnd,snd->nts", q_full, k) * (P + R) ** -0.5
        pos_q = torch.arange(C, C + M)
        mask = torch.arange(C + M).unsqueeze(0) > pos_q.unsqueeze(1)
        s.masked_fill_(mask.unsqueeze(0), -float("inf"))
        p = torch.softmax(s, dim=-1)
        ref = torch.einsum("nts,snv->tnv", p, v)
        ref_out = layer.o_proj(ref.reshape(M, N * V))[0]
        torch.testing.assert_close(out, ref_out, rtol=2e-4, atol=2e-4)

    def test_mixed_batch_two_legs_one_forward(self):
        # 站 11：同一前向内 decode 走 MQA、prefill 走 MHA，token 维切一刀
        dims = MINI
        vllm_config, layer, inner, builder, dims, blk = self._setup(
            dims, max_model_len=64, max_num_seqs=8)
        N, P, R, Lkv, V, H = (dims["num_heads"], dims["qk_nope_head_dim"],
                              dims["qk_rope_head_dim"], dims["kv_lora_rank"],
                              dims["v_head_dim"], dims["hidden_size"])
        # ① 两条历史：一条 10（将 decode）、一条 30（将继续 prefill 8）
        hist = [10, 30]
        h_hist = torch.randn(sum(hist), H)
        pos_hist = torch.cat([torch.arange(s) for s in hist])
        self._run(vllm_config, layer, builder, hist, hist, h_hist, pos_hist)
        # ② 混批：decode(1) + prefill-chunk(140, 上下文 30)——chunk > 阈值
        # 128 才走 MHA 腿（≤128 会按 decode pathway 划进 MQA 段）
        seq2 = [11, 170]
        ql = [1, 140]
        h_new = torch.randn(141, H)
        pos_new = torch.tensor([10] + list(range(30, 170)))
        out, md, common = self._run(vllm_config, layer, builder, ql, seq2,
                                    h_new, pos_new)
        assert md.num_decode_tokens == 1
        # num_mha_tokens = 141 - 1 = 140（forward_impl 的切刀）
        assert md.num_actual_tokens - md.num_decode_tokens == 140
        # oracle：逐请求上投影 MHA
        cache = inner.kv_cache.view(-1, 576)
        scale = (P + R) ** -0.5
        Lq = dims["q_lora_rank"]
        q_c = layer.q_a_layernorm(
            (h_new @ layer.fused_qkv_a_proj.weight.T)[:, :Lq])
        q = (q_c @ layer.q_b_proj.weight.T).view(141, N, P + R)
        q_rot, _ = layer.mla_attn.rotary_emb(pos_new, q[..., P:].clone(),
                                             torch.zeros(141, 1, R))
        q_full = torch.cat([q[..., :P], q_rot], dim=-1)
        ref = torch.zeros(141, N, V)
        bt = common.block_table_tensor
        for r, (qlen, s) in enumerate(zip(ql, seq2)):
            rows = [int(bt[r, pos // BLOCK]) * BLOCK + pos % BLOCK
                    for pos in range(s)]
            lat = cache[torch.tensor(rows)]
            w = inner.kv_b_proj.weight
            kv_nope = (lat[:, :Lkv] @ w.T).view(s, N, P + V)
            k_nope, v = kv_nope.split([P, V], dim=-1)
            k = torch.cat([k_nope, lat[:, Lkv:].unsqueeze(1).expand(-1, N, -1)],
                          dim=-1)
            lo, hi = (0, 1) if r == 0 else (1, 141)
            s_mat = torch.einsum("tnd,snd->nts", q_full[lo:hi], k) * scale
            if r == 1:  # prefill 段块内 causal
                pm = torch.arange(30, 170)
                mask = torch.arange(170).unsqueeze(0) > pm.unsqueeze(1)
                s_mat.masked_fill_(mask.unsqueeze(0), -float("inf"))
            p = torch.softmax(s_mat, dim=-1)
            ref[lo:hi] = torch.einsum("nts,snv->tnv", p, v)
        ref_out = layer.o_proj(ref.reshape(141, N * V))[0]
        torch.testing.assert_close(out, ref_out, rtol=2e-4, atol=2e-4)

    def test_forward_impl_token_dim_split(self):
        # L771-L772 + L812-L949：token 维前缀 MQA / 后缀 MHA 的切片账
        dims = MINI
        vllm_config, layer, inner, builder, dims, blk = self._setup(
            dims, max_model_len=64, max_num_seqs=8)
        seen = {}

        class ProbeImpl(RefFlashMLAImpl):
            def forward_mqa(self, q, cache, md, lay):
                seen["mqa_rows"] = q[0].shape[0] if type(q) is tuple else q.shape[0]
                return super().forward_mqa(q, cache, md, lay)

            def forward_mha(self, q, kv_c_normed, k_pe, cache, md, k_scale,
                            output, output_scale=None):
                seen["mha_rows"] = q.shape[0]
                seen["mha_offset_ok"] = output.shape[0] == q.shape[0]

        with set_current_vllm_config(vllm_config):
            inner.impl = ProbeImpl(
                num_heads=inner.num_heads, head_size=inner.head_size,
                scale=inner.scale, num_kv_heads=1, alibi_slopes=None,
                sliding_window=None, kv_cache_dtype=inner.kv_cache_dtype,
                logits_soft_cap=None, attn_type="decoder",
                kv_sharing_target_layer_name=None,
                q_lora_rank=inner.q_lora_rank, kv_lora_rank=inner.kv_lora_rank,
                qk_nope_head_dim=inner.qk_nope_head_dim,
                qk_rope_head_dim=inner.qk_rope_head_dim,
                qk_head_dim=inner.qk_head_dim, v_head_dim=inner.v_head_dim,
                kv_b_proj=inner.kv_b_proj)
        hist = [5]
        H = dims["hidden_size"]
        h_hist = torch.randn(5, H)
        self._run(vllm_config, layer, builder, [5], [5], h_hist,
                  torch.arange(5))
        h_new = torch.randn(141, H)
        out, md, common = self._run(
            vllm_config, layer, builder, [1, 140], [6, 140], h_new,
            torch.tensor([5] + list(range(140))))
        # 批序：decode 请求 1 token 在前、fresh prefill 140 token 在后
        assert seen["mqa_rows"] == 1
        assert seen["mha_rows"] == 140


class TestMergeAttnStates:
    def test_lse_merge_identity(self):
        # merge_attn_states：分块 LSE 合并 == 整块 softmax（精确恒等式）
        torch.manual_seed(7)
        T, N, V = 4, 2, 16
        q = torch.randn(T, N, 32)
        k1, v1 = torch.randn(5, N, 32), torch.randn(5, N, V)
        k2, v2 = torch.randn(3, N, 32), torch.randn(3, N, V)
        scale = 32 ** -0.5
        s1 = torch.einsum("tnd,snd->nts", q, k1) * scale
        s2 = torch.einsum("tnd,snd->nts", q, k2) * scale
        o1 = torch.einsum("nts,snv->tnv", torch.softmax(s1, -1), v1)
        o2 = torch.einsum("nts,snv->tnv", torch.softmax(s2, -1), v2)
        l1, l2 = torch.logsumexp(s1, -1), torch.logsumexp(s2, -1)
        out = torch.zeros(T, N, V)
        merge_attn_states(output=out, prefix_output=o1, prefix_lse=l1,
                          suffix_output=o2, suffix_lse=l2)
        # 整块参考
        k = torch.cat([k1, k2]); v = torch.cat([v1, v2])
        s = torch.einsum("tnd,snd->nts", q, k) * scale
        ref = torch.einsum("nts,snv->tnv", torch.softmax(s, -1), v)
        torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    def test_prefill_tokens_with_context_gate(self):
        # prefill_tokens_with_context：无上下文行直接取 suffix
        T, N, V = 3, 1, 8
        o1 = torch.full((T, N, V), 7.0)
        l1 = torch.zeros(N, T)
        o2 = torch.full((T, N, V), 3.0)
        l2 = torch.zeros(N, T)
        out = torch.zeros(T, N, V)
        merge_attn_states(output=out, prefix_output=o1, prefix_lse=l1,
                          suffix_output=o2, suffix_lse=l2,
                          prefill_tokens_with_context=1)
        assert out[0].min() == 5.0  # 有上下文行：等权合并 (7+3)/2
        assert torch.equal(out[1:], o2[1:])  # 无上下文行：suffix 直通


class TestFlashMLAKernelSeam:
    def test_flash_mla_with_kvcache_reference(self):
        # seam kernel == 文件头 Data-Movement 伪码（吸收后的 576 维 MQA）
        B, N, Lkv, R = 2, 3, 512, 64
        S = [5, 9]
        sq = 1
        cache = torch.randn(4, BLOCK, 576)
        bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
        seqlens = torch.tensor(S, dtype=torch.int32)
        ql = torch.randn(B, sq, N, Lkv)
        qpe = torch.randn(B, sq, N, R)
        q = torch.cat([ql, qpe], dim=-1)
        scale = 0.05

        class _Sched:
            pass

        o, lse = flash_mla_with_kvcache(
            q=q, k_cache=cache.unsqueeze(-2), block_table=bt,
            cache_seqlens=seqlens, head_dim_v=Lkv,
            tile_scheduler_metadata=_Sched(), softmax_scale=scale,
            causal=True, is_fp8_kvcache=False)
        assert o.shape == (B, sq, N, Lkv)
        for b in range(B):
            keys = torch.stack([cache[bt[b, j // BLOCK], j % BLOCK]
                                for j in range(S[b])])  # [S, 576]
            s = (torch.einsum("nd,sd->ns", q[b, 0], keys) * scale)
            p = torch.softmax(s, dim=-1)
            ref = p @ keys[:, :Lkv]
            torch.testing.assert_close(o[b, 0], ref, rtol=1e-5, atol=1e-5)


# ═══════════════════════════════════════════════════════════════════════
# 幕五：DSV4 第三代（m12/m15）
# ═══════════════════════════════════════════════════════════════════════


class TestDeepseekV4:
    def _make(self, compress_ratios, cache_dtype="auto"):
        hf = SimpleNamespace(
            model_type="deepseek_v4", hidden_size=512,
            num_attention_heads=8, q_lora_rank=192, o_lora_rank=128,
            head_dim=512, qk_rope_head_dim=64, o_groups=4,
            sliding_window=4096, rms_norm_eps=1e-6,
            max_position_embeddings=512,
            num_hidden_layers=len(compress_ratios),
            compress_ratios=list(compress_ratios),
            index_topk=2048,
        )
        vllm_config = enable_ref_backends(
            make_vllm_config(hf_cfg=hf, cache_dtype=cache_dtype))
        return vllm_config, hf

    def _build_attention(self, vllm_config, hf, layer_id, fp8_layout=None):
        # 平台子类未接入（ch23 立的硬隔离布局）——以最小平台子类承载
        # get_padded_num_q_heads/forward_mqa/_o_proj 三个抽象位（真实注入位：
        # nvidia/amd/xpu 子类由平台分发装配）；use_fp8_ds_mla_layout 是
        # 平台子类的 ClassVar 开关（FlashMLA=True / FlashInfer 连续布局=False）
        class _NvidiaLike(DeepseekV4Attention):
            if fp8_layout is None:
                use_fp8_ds_mla_layout = not vllm_config.cache_config.cache_dtype == "auto"
            else:
                use_fp8_ds_mla_layout = fp8_layout

            @classmethod
            def get_padded_num_q_heads(cls, num_heads):
                return num_heads

            def forward_mqa(self, q, kv, positions, output):
                raise NotImplementedError

            def _o_proj(self, o, positions):
                raise NotImplementedError

        with set_current_vllm_config(vllm_config):
            attn = _NvidiaLike(vllm_config, f"model.layers.{layer_id}.self_attn")
        return attn

    def test_compress_ratio_per_layer_and_mtp(self):
        # attention.py:L207-L213：逐层 compress_ratios[layer_id]，MTP 恒 1
        ratios = [1, 4, 128, 4, 1]
        vllm_config, hf = self._make(ratios)
        for i, want in enumerate(ratios):
            attn = self._build_attention(vllm_config, hf, i)
            assert attn.compress_ratio == want, f"layer {i}"
        # MTP 层（layer_id >= num_hidden_layers）恒 1
        attn = self._build_attention(vllm_config, hf, len(ratios))
        assert attn.compress_ratio == 1
        # ratio=0 的护栏：max(1, 0) = 1（NOTE(zyongye) 注释的语义）
        v2, hf2 = self._make([0, 4])
        a0 = self._build_attention(v2, hf2, 0)
        assert a0.compress_ratio == 1

    def test_get_kv_cache_spec_dsv4(self):
        # attention.py:L655-L674：compress_ratio>1 → MLAAttentionSpec 特账；
        # ≤1 → None（SWA 段由子缓存另报）
        vllm_config, hf = self._make([1, 4])
        with set_current_vllm_config(vllm_config):
            a_swa = self._build_attention(vllm_config, hf, 0)
            a_c4 = self._build_attention(vllm_config, hf, 1)
            assert a_swa.get_kv_cache_spec(vllm_config) is None
            spec = a_c4.get_kv_cache_spec(vllm_config)
        assert isinstance(spec, MLAAttentionSpec)
        assert spec.compress_ratio == 4
        assert spec.head_size == 512
        assert spec.model_version == "deepseek_v4"
        assert spec.alignment == 512  # 非 fp8_ds_mla → 自然元素页
        assert spec.storage_block_size == BLOCK // 4
        # fp8_ds_mla 布局：uint8 + 576B 对齐
        v4fp8, hf3 = self._make([4], cache_dtype="fp8_ds_mla")
        a4 = self._build_attention(v4fp8, hf3, 0, fp8_layout=True)
        spec8 = a4.get_kv_cache_spec(v4fp8)
        assert spec8.dtype == torch.uint8
        assert spec8.alignment == 576
        assert spec8.real_page_size_bytes == (BLOCK // 4) * 584

    def test_swa_subcache_self_report(self):
        # sparse_swa.py:L87-L102：SWA 子缓存自报 SlidingWindowMLASpec
        from vllm.v1.attention.backends.mla.sparse_swa import DeepseekV4SWACache

        vllm_config, hf = self._make([1, 4])
        with set_current_vllm_config(vllm_config):
            swa = DeepseekV4SWACache(
                head_dim=512, window_size=4096, dtype=torch.bfloat16,
                prefix="model.layers.0.self_attn.swa_cache",
                cache_config=vllm_config.cache_config)
            spec = swa.get_kv_cache_spec(vllm_config)
        assert isinstance(spec, SlidingWindowMLASpec)
        assert spec.sliding_window == 4096
        assert spec.num_kv_heads == 1
        assert spec.block_size == 64  # 与 C4A [256/4] 共享物理张量的页对齐
        assert spec.real_page_size_bytes == 64 * 512 * 2
        # 以独立 prefix 注册进 static_forward_context（一层多子缓存实证）
        assert vllm_config.compilation_config.static_forward_context[
            "model.layers.0.self_attn.swa_cache"] is swa

    def test_o_side_lowrank_bmm(self):
        # attention.py:L244-L262：wo_a.is_bmm 组化 + wo_b 回投
        vllm_config, hf = self._make([4])
        attn = self._build_attention(vllm_config, hf, 0)
        assert attn.wo_a.is_bmm is True
        assert attn.wo_a.bmm_batch_size == attn.n_local_groups
        # wo_a: n_heads*head_dim/n_groups → n_groups*o_lora_rank；wo_b → hidden
        assert attn.wo_a.weight.shape == (hf.o_groups * hf.o_lora_rank,
                                          hf.num_attention_heads * hf.head_dim
                                          // hf.o_groups)
        assert attn.wo_b.weight.shape == (hf.hidden_size,
                                          hf.o_groups * hf.o_lora_rank)


# ═══════════════════════════════════════════════════════════════════════
# 幕六：prefill 家族注册表（m10）与算子注册面
# ═══════════════════════════════════════════════════════════════════════


class TestPrefillRegistry:
    def test_enum_members(self):
        # registry.py:L34-L57：五成员 + CUSTOM 槽
        names = {m.name for m in MLAPrefillBackendEnum}
        assert names == {"FLASH_ATTN", "FLASHINFER", "TRTLLM_RAGGED",
                         "TOKENSPEED_MLA", "ROCM_AITER_FA", "CUSTOM"}
        assert MLAPrefillBackendEnum.FLASH_ATTN.get_path().endswith(
            "FlashAttnPrefillBackend")

    def test_custom_registration_and_clone(self):
        # CUSTOM 槽注册流 + clone()（base.py:L142-L151）
        assert MLAPrefillBackendEnum.CUSTOM.is_overridden()
        cls = MLAPrefillBackendEnum.CUSTOM.get_class()
        assert cls is RefPrefillBackend
        vllm_config = make_vllm_config(dims=MINI)
        inst = cls(num_heads=4, scale=0.1, kv_lora_rank=512,
                   qk_nope_head_dim=16, qk_rope_head_dim=64, v_head_dim=16,
                   vllm_config=vllm_config)
        twin = inst.clone()
        assert type(twin) is RefPrefillBackend and twin.scale == 0.1
        MLAPrefillBackendEnum.CUSTOM.clear_override()

    def test_backend_registry_custom(self):
        # decode 家族的注册表覆盖机制（registry.py 第三方注册流）
        from vllm.v1.attention.backends.registry import _ATTN_OVERRIDES

        assert AttentionBackendEnum.CUSTOM in _ATTN_OVERRIDES
        assert _ATTN_OVERRIDES[AttentionBackendEnum.CUSTOM].endswith(
            "RefRegisteredBackend")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
