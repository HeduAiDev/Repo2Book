# ch23《模型定义层拼装术》测试电池 —— TDD：先测真实 vLLM v0.27.1
# (6e448d0ea) 的可观察行为，精简版实现到通过为止。全部 host 可跑（纯单元，
# 不依赖真 vllm 包安装、无 CUDA 上下文——分布式/平台/编译面以 HOST SEAM
# 单进程退化承载；Attention 的 impl/attn_backend 按 ch21 域的真实注入位
# 以参考后端注入，数学 = ch20 已立的精确注意力）。
#
# 行为基准 = 真实源码（instances/vllm/source，v0.27.1 现核行号）：
#   - registry.py:L92-L95/L1439-L1445/L1017-L1019（新旧布局/惰性 importlib）
#   - llama.py:L79-L540（五类拼装 + TP 切头 + 两级 load_weights）
#   - linear.py:L1222-L1400（QKV 分片算术）/L829-L834（Merged）/L1717-L1737（Row）
#   - logits_processor.py:L137-L153（logits 三步）/L84-L96（gather 语义）
#   - attention.py:L443-L446（static_forward_context 自注册）/L732-L772
#   - gpu_model_runner.py:L2232-L2240/L4483-L4485（采样位策略归 runner）
from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig as HFLlamaConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "implementation"))

import vllm.distributed as dist  # noqa: E402
from vllm.config import (  # noqa: E402
    CacheConfig,
    CompilationConfig,
    DeviceConfig,
    LoadConfig,
    ModelConfig,
    ParallelConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.distributed import divide  # noqa: E402
from vllm.forward_context import set_forward_context  # noqa: E402
from vllm.model_executor.layers.attention import Attention  # noqa: E402
from vllm.model_executor.layers.attention.attention import (  # noqa: E402
    get_attention_context,
    unified_kv_cache_update,
)
from vllm.model_executor.layers.layernorm import RMSNorm  # noqa: E402
from vllm.model_executor.layers.linear import (  # noqa: E402
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor  # noqa: E402
from vllm.model_executor.layers.rotary_embedding import get_rope  # noqa: E402
from vllm.model_executor.layers.vocab_parallel_embedding import (  # noqa: E402
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader import (  # noqa: E402
    get_model,
    get_model_loader,
)
from vllm.model_executor.model_loader.utils import (  # noqa: E402
    get_model_architecture,
    initialize_model,
)
from vllm.model_executor.models.llama import (  # noqa: E402
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaMLP,
    LlamaModel,
)
from vllm.model_executor.models.registry import (  # noqa: E402
    ModelRegistry,
    _resolve_module_name,
)
from vllm.model_executor.models.utils import (  # noqa: E402
    AutoWeightsLoader,
    PPMissingLayer,
    WeightsMapper,
    make_empty_intermediate_tensors_factory,
    make_layers,
)
from vllm.sequence import IntermediateTensors  # noqa: E402
from vllm.v1.attention.backend import AttentionType  # noqa: E402
from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # noqa: E402


# ────────────────────────── 测试基础设施 ──────────────────────────


@pytest.fixture(autouse=True)
def _reset_tp_state():
    """每个测试回到单 rank 退化态（真实源码的默认分布式态）。"""
    dist._TP_STATE.update(rank=0, size=1, pp_rank=0, pp_size=1)
    yield
    dist._TP_STATE.update(rank=0, size=1, pp_rank=0, pp_size=1)


def make_hf_config(**over):
    # 注意：head_dim 等 LlamaConfig 派生量在构造时按 num_attention_heads
    # 算定——测试变体必须走构造参数（不是事后 setattr）
    kwargs = dict(
        vocab_size=100,  # 非 64 倍数 → 词表 padding 面可见
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    kwargs.update(over)
    cfg = HFLlamaConfig(**kwargs)
    # 真实 HF config.json 的 architectures 字段（registry 查表入口）
    cfg.architectures = ["LlamaForCausalLM"]
    return cfg


def make_vllm_config(hf_config=None, load_format="auto"):
    hf_config = hf_config or make_hf_config()
    return VllmConfig(
        model_config=ModelConfig(hf_config=hf_config, dtype=torch.float32),
        cache_config=CacheConfig(),
        load_config=LoadConfig(load_format=load_format),
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(),
        parallel_config=ParallelConfig(),
    )


# 参考注意力后端：实现真实 AttentionImpl 接口（v1/attention/backend.py
# AttentionImplBase 属性面 + unified_kv_cache_update 要求的
# do_kv_cache_update + impl.forward(output=...) 签名）。数学 = ch20 已立的
# 精确注意力：KV 按 slot_mapping 写入 cache、按请求窗读出、GQA 广播、causal。
@dataclass
class RefAttnMetadata:
    query_start_loc: torch.Tensor
    seq_lens: torch.Tensor


class RefAttentionImpl:
    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads=None,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
        **extra,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads

    def process_weights_after_loading(self, act_dtype):
        pass

    def do_kv_cache_update(self, attn_layer, key, value, kv_cache, slot_mapping):
        # kv_cache: [max_slots, 2, num_kv_heads, head_dim]（测试装配形态）
        for t, slot in enumerate(slot_mapping.tolist()):
            if slot >= 0:
                kv_cache[slot, 0] = key[t]
                kv_cache[slot, 1] = value[t]

    def forward(
        self,
        attn_layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
    ):
        qsl = attn_metadata.query_start_loc.tolist()
        seq_lens = attn_metadata.seq_lens.tolist()
        _, hq, hd = query.shape
        hk = self.num_kv_heads
        rep = hq // hk
        out = torch.zeros_like(query)
        for r in range(len(qsl) - 1):
            qs, qe = qsl[r], qsl[r + 1]
            total = seq_lens[r]
            K = kv_cache[:total, 0].reshape(total, hk, hd)
            V = kv_cache[:total, 1].reshape(total, hk, hd)
            for h in range(hq):
                kv_h = h // rep
                q = query[qs:qe, h].double()
                k = K[:, kv_h].double()
                v = V[:, kv_h].double()
                scores = (q @ k.T) * self.scale
                causal = torch.arange(total).unsqueeze(0) <= torch.arange(qs, qe).unsqueeze(1)
                scores = scores.masked_fill(~causal, float("-inf"))
                p = torch.softmax(scores, dim=-1)
                out[qs:qe, h] = (p @ v).to(query.dtype)
        if output is not None:
            output.copy_(out)
            return None
        return out


class RefAttentionBackend:
    forward_includes_kv_cache_update = False

    @staticmethod
    def get_name():
        return "REFERENCE"

    @staticmethod
    def get_impl_cls():
        return RefAttentionImpl


def inject_ref_backend(model, vllm_config):
    """把参考后端绑进模型每个 Attention 插座——真实注入位（attention.py
    L251 的 attn_backend 参数 + L421-L433 的 impl 构造产物；后端选择归 ch21，
    host 上由测试注入）。"""
    for name, m in model.named_modules():
        if isinstance(m, Attention):
            m.attn_backend = RefAttentionBackend
            m.impl = RefAttentionImpl(
                m.num_heads,
                m.head_size,
                m.head_size**-0.5,
                num_kv_heads=m.num_kv_heads,
                kv_cache_dtype="auto",
                attn_type=m.attn_type,
            )


def build_model(vllm_config):
    """经真实 initialize_model 建空壳（新式签名校验 + set_current_vllm_config），
    再注入参考注意力后端。"""
    model = initialize_model(vllm_config)
    inject_ref_backend(model, vllm_config)
    return model


def make_kv_cache(num_layers, max_slots, num_kv_heads, head_dim, dtype=torch.float32):
    return [
        torch.zeros(max_slots, 2, num_kv_heads, head_dim, dtype=dtype)
        for _ in range(num_layers)
    ]


def gen_ckpt_weights(hf_config, seed=0, dtype=torch.float32):
    """HF 命名（q_proj/k_proj/v_proj/gate_proj/up_proj）的确定性 checkpoint。"""
    g = torch.Generator().manual_seed(seed)
    H, I, V = hf_config.hidden_size, hf_config.intermediate_size, hf_config.vocab_size
    L = hf_config.num_hidden_layers
    hd = H // hf_config.num_attention_heads
    nkvh = hf_config.num_key_value_heads
    weights = {
        "model.embed_tokens.weight": torch.randn(V, H, generator=g, dtype=dtype),
        "lm_head.weight": torch.randn(V, H, generator=g, dtype=dtype),
        "model.norm.weight": torch.randn(H, generator=g, dtype=dtype) * 0.1 + 1.0,
    }
    for i in range(L):
        p = f"model.layers.{i}"
        weights[f"{p}.self_attn.q_proj.weight"] = torch.randn(
            hf_config.num_attention_heads * hd, H, generator=g, dtype=dtype
        ) * 0.05
        weights[f"{p}.self_attn.k_proj.weight"] = torch.randn(
            nkvh * hd, H, generator=g, dtype=dtype
        ) * 0.05
        weights[f"{p}.self_attn.v_proj.weight"] = torch.randn(
            nkvh * hd, H, generator=g, dtype=dtype
        ) * 0.05
        weights[f"{p}.self_attn.o_proj.weight"] = torch.randn(
            H, hf_config.num_attention_heads * hd, generator=g, dtype=dtype
        ) * 0.05
        weights[f"{p}.mlp.gate_proj.weight"] = torch.randn(I, H, generator=g, dtype=dtype) * 0.05
        weights[f"{p}.mlp.up_proj.weight"] = torch.randn(I, H, generator=g, dtype=dtype) * 0.05
        weights[f"{p}.mlp.down_proj.weight"] = torch.randn(H, I, generator=g, dtype=dtype) * 0.05
        weights[f"{p}.input_layernorm.weight"] = torch.randn(H, generator=g, dtype=dtype) * 0.1 + 1.0
        weights[f"{p}.post_attention_layernorm.weight"] = (
            torch.randn(H, generator=g, dtype=dtype) * 0.1 + 1.0
        )
    return weights


def load_into_model(model, weights):
    return model.load_weights(iter(list(weights.items())))


def bind_kv_caches(vllm_config, num_layers, max_slots, num_kv_heads, head_dim):
    """给每个 Attention 插座绑定真实 KV cache（占位 → bind_kv_cache 替换）。"""
    kvs = make_kv_cache(num_layers, max_slots, num_kv_heads, head_dim)
    for layer_name, attn in vllm_config.compilation_config.static_forward_context.items():
        li = int(layer_name.split(".")[2])
        attn.bind_kv_cache(kvs[li])
    return kvs


# ────────────────────── m8/m9：registry 与新旧布局 ──────────────────────


class TestRegistry:
    def test_resolve_module_name_two_layouts(self):
        # registry.py:L1439-L1445：旧扁平拼前缀；vllm. 开头原样返回（新布局）
        assert _resolve_module_name("llama") == "vllm.model_executor.models.llama"
        assert (
            _resolve_module_name("vllm.models.deepseek_v4") == "vllm.models.deepseek_v4"
        )

    def test_entry_table_two_layouts_coexist(self):
        # registry.py:L92-L95：同一张表里两代布局并存
        assert (
            ModelRegistry.models["DeepseekV32ForCausalLM"].module_name
            == "vllm.model_executor.models.deepseek_v2"
        )
        assert (
            ModelRegistry.models["DeepseekV32ForCausalLM"].class_name
            == "DeepseekV3ForCausalLM"
        )
        dsv4 = ModelRegistry.models["DeepseekV4ForCausalLM"]
        assert dsv4.module_name == "vllm.models.deepseek_v4"
        assert dsv4.class_name == "DeepseekV4ForCausalLM"
        # 同款全限定条目：DSparkDraftModel（L617）/ DeepSeekV4MTPModel（L643）
        assert ModelRegistry.models["DSparkDraftModel"].module_name == "vllm.models.deepseek_v4"
        assert ModelRegistry.models["DeepSeekV4MTPModel"].class_name == "DeepSeekV4MTP"

    def test_lazy_load_model_cls_resolves_llama(self):
        # registry.py:L1017-L1019：importlib.import_module → getattr
        vllm_config = make_vllm_config()
        model_cls, arch = ModelRegistry.resolve_model_cls(
            ["LlamaForCausalLM"], model_config=vllm_config.model_config
        )
        assert arch == "LlamaForCausalLM"
        assert model_cls is LlamaForCausalLM

    def test_unknown_arch_raises(self):
        vllm_config = make_vllm_config()
        with pytest.raises(ValueError):
            ModelRegistry.resolve_model_cls(
                ["NoSuchModelForCausalLM"], model_config=vllm_config.model_config
            )

    def test_get_model_architecture_caches(self):
        # model_loader/utils.py:L239-L255：hash 键缓存，二次调用同结果
        vllm_config = make_vllm_config()
        a = get_model_architecture(vllm_config.model_config)
        b = get_model_architecture(vllm_config.model_config)
        assert a == b
        assert a[0] is LlamaForCausalLM


# ────────────────── m2/m7：TP 切头数学与分片 weight_loader ──────────────────


class TestTPShardMath:
    def test_llama_attention_head_math_tp1(self):
        hf = make_hf_config()
        vllm_config = make_vllm_config(hf)
        with set_current_vllm_config(vllm_config):
            attn = LlamaAttention(
                config=hf,
                hidden_size=64,
                num_heads=4,
                num_kv_heads=2,
                cache_config=vllm_config.cache_config,
                prefix="model.layers.0.self_attn",
            )
        # llama.py:L138-L159 的切头推导（tp=1 退化）
        assert attn.num_heads == 4
        assert attn.num_kv_heads == 2
        assert attn.head_dim == 16
        assert attn.q_size == 64
        assert attn.kv_size == 32
        assert attn.scaling == 16**-0.5
        # qkv_proj 融合权重三段 [64, 32, 32] 共 128 行（weight.shape==[out,in]）
        assert attn.qkv_proj.weight.shape == (128, 64)

    def test_llama_attention_head_math_tp2_partition(self):
        # theory worked example：8q/2kv/tp=2 走 partition 分支
        hf = make_hf_config(num_attention_heads=8, num_key_value_heads=2)
        dist._TP_STATE.update(rank=1, size=2)
        vllm_config = make_vllm_config(hf)
        with set_current_vllm_config(vllm_config):
            attn = LlamaAttention(
                config=hf,
                hidden_size=64,
                num_heads=8,
                num_kv_heads=2,
                cache_config=vllm_config.cache_config,
                prefix="model.layers.0.self_attn",
            )
        assert attn.num_heads == 4  # 8 // 2
        assert attn.num_kv_heads == 1  # 2 // 2（partition 分支）
        assert attn.q_size == 4 * 8  # head_dim = 64//8 = 8
        assert attn.kv_size == 8
        # 融合权重每 rank = q 32 + k 8 + v 8 = 48 行
        assert attn.qkv_proj.weight.shape == (48, 64)

    def test_llama_attention_tp_assert(self):
        # llama.py:L142：total_num_heads % tp 断言（快速失败）
        hf = make_hf_config(num_attention_heads=4)
        dist._TP_STATE.update(rank=0, size=3)
        vllm_config = make_vllm_config(hf)
        with set_current_vllm_config(vllm_config), pytest.raises(AssertionError):
            LlamaAttention(
                config=hf,
                hidden_size=64,
                num_heads=4,
                num_kv_heads=2,
                cache_config=vllm_config.cache_config,
                prefix="model.layers.0.self_attn",
            )

    def _qkv(self, total_q, total_kv, hidden=64, head_size=16):
        dist._TP_STATE.update(rank=1, size=2)
        return QKVParallelLinear(
            hidden_size=hidden,
            head_size=head_size,
            total_num_heads=total_q,
            total_num_kv_heads=total_kv,
            bias=False,
            prefix="qkv",
        )

    def test_qkv_loader_partition(self):
        # linear.py:L1317-L1333 + L1375-L1383：8q/2kv/tp=2，rank1 装载
        layer = self._qkv(8, 2)
        assert layer.num_heads == 4
        assert layer.num_kv_heads == 1
        assert layer.num_kv_head_replicas == 1
        q_global = torch.arange(128 * 64, dtype=torch.float32).reshape(128, 64) / 128
        k_global = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64) / 32 + 100
        v_global = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64) / 32 + 200
        layer.weight_loader(layer.weight, q_global, loaded_shard_id="q")
        layer.weight_loader(layer.weight, k_global, loaded_shard_id="k")
        layer.weight_loader(layer.weight, v_global, loaded_shard_id="v")
        w = layer.weight.data
        # q 段 [0:64)：tp_rank=1 → 全局 q 第 [64:128) 行
        assert torch.equal(w[:64], q_global[64:128])
        # k 段 [64:80)：shard_rank = tp_rank // replicas = 1 → 全局 k 第 [16:32) 行
        assert torch.equal(w[64:80], k_global[16:32])
        # v 段 [80:96)：同上
        assert torch.equal(w[80:96], v_global[16:32])

    def test_qkv_loader_replicate(self):
        # GQA 1-KV-头例（8q/1kv/tp=2）：replicas=2，两 rank 的 k/v 都取全局第 0 段
        layer = self._qkv(8, 1)
        assert layer.num_kv_head_replicas == 2
        assert layer.num_kv_heads == 1
        k_global = torch.arange(16 * 64, dtype=torch.float32).reshape(16, 64) / 16 + 50
        layer.weight_loader(layer.weight, k_global, loaded_shard_id="k")
        w = layer.weight.data
        # shard_rank = tp_rank(1) // replicas(2) = 0 → 全局 k 第 [0:16) 行（复制语义）
        assert torch.equal(w[64:80], k_global[0:16])

    def test_merged_loader_offsets(self):
        # linear.py:L829-L834：gate 段在前、up 段在后，各 //tp
        dist._TP_STATE.update(rank=1, size=2)
        layer = MergedColumnParallelLinear(
            input_size=64, output_sizes=[32, 32], bias=False, prefix="gate_up"
        )
        assert layer.output_sizes == [32, 32]
        gate = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64) / 32 + 1
        up = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64) / 32 + 2
        layer.weight_loader(layer.weight, gate, loaded_shard_id=0)
        layer.weight_loader(layer.weight, up, loaded_shard_id=1)
        w = layer.weight.data
        assert torch.equal(w[:16], gate[16:32])  # shard_offset=0, start=16
        assert torch.equal(w[16:32], up[16:32])  # shard_offset=32//2=16, start=16

    def test_row_loader_input_dim_slice(self):
        # linear.py:L1717-L1737：沿 input_dim 行切（权重 [out,in]，input_dim=1）
        dist._TP_STATE.update(rank=1, size=2)
        layer = RowParallelLinear(input_size=64, output_size=32, bias=False, prefix="down")
        global_w = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64) / 64
        layer.weight_loader(layer.weight, global_w)
        # shard_size=32（本秩 input 行数），start_idx=tp_rank*32=32 → 全局第 [32:64) 列
        assert torch.equal(layer.weight.data, global_w[:, 32:64])

    def test_qkv_loader_fused_on_disk(self):
        # linear.py:L1236-L1265 + L1309-L1313（delete[7] 豁免保留）：Phi-3 类
        # 已融合 checkpoint（shard_id=None）——按全局段位 narrow 出 q/k/v 三段
        # 再递归回单段装载路径（tp=2, rank1, 8q/2kv，head_size=16：
        # q 段 shard_size=64 start=64、k/v 段 shard_size=16 start=16）
        layer = self._qkv(8, 2)
        assert layer.weight.shape == (96, 64)
        q_g = torch.arange(128 * 64, dtype=torch.float32).reshape(128, 64) / 128
        k_g = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64) / 32 + 100
        v_g = torch.arange(32 * 64, dtype=torch.float32).reshape(32, 64) / 32 + 200
        fused = torch.cat([q_g, k_g, v_g], dim=0)
        layer.weight_loader(layer.weight, fused, loaded_shard_id=None)
        w = layer.weight.data
        assert torch.equal(w[:64], q_g[64:128])
        assert torch.equal(w[64:80], k_g[16:32])  # 全局 [128+16 : 128+32) = [144:160)
        assert torch.equal(w[80:96], v_g[16:32])  # 全局 [160+16 : 160+32) = [176:192)

    def test_qkv_loader_output_dim_none_warns_and_copies(self):
        # linear.py:L1385-L1397 尾部：无 output_dim 的参数走警告支（assume
        # same for all partitions），整拷不 narrow
        layer = self._qkv(8, 2)
        plain = torch.nn.Parameter(torch.zeros(16, 64))
        lw = torch.randn(16, 64)
        layer.weight_loader(plain, lw, loaded_shard_id="q")
        assert torch.equal(plain.data, lw)

    def test_merged_loader_fused_on_disk(self):
        # linear.py:L759-L826 豁免骨架：gate/up 已融合（shard_id=None）——
        # shard_offsets 循环按全局段位 narrow 再递归；tp=1 时整体直拷等价
        layer = MergedColumnParallelLinear(
            input_size=64, output_sizes=[32, 32], bias=False, prefix="gate_up"
        )
        fused = torch.cat(
            [torch.full((32, 64), 1.0), torch.full((32, 64), 2.0)], dim=0
        )
        layer.weight_loader(layer.weight, fused, loaded_shard_id=None)
        assert torch.equal(layer.weight.data, fused)

    def test_merged_validate_shard_id_tuple(self):
        # linear.py:L729-L744：tuple 多段 shard_id 支（fused 分支的消费位）
        layer = MergedColumnParallelLinear(
            input_size=64, output_sizes=[32, 32], bias=False, prefix="gate_up"
        )
        assert layer.validate_shard_id(None) is True
        assert layer.validate_shard_id((0, 1)) is True
        with pytest.raises(ValueError, match="between 0 and"):
            layer.validate_shard_id((2,))
        # 非连续多段（需三段表才能构造）：(0,2) 跳过 1
        layer3 = MergedColumnParallelLinear(
            input_size=64, output_sizes=[16, 16, 16], bias=False, prefix="gu3"
        )
        with pytest.raises(ValueError, match="consecutive"):
            layer3.validate_shard_id((0, 2))

    def test_divide(self):
        assert divide(8, 2) == 4
        with pytest.raises(ValueError):
            divide(7, 2)


# ────────────────────── m6：权重名映射与两级 load_weights ──────────────────────


class TestWeightMapping:
    def test_hf_to_vllm_mapper_stacked(self):
        # llama.py:L345-L354：q/k/v→qkv_proj+shard_id、gate/up→gate_up_proj+0/1
        mapper = LlamaModel.hf_to_vllm_mapper
        assert mapper._map_name_with_shard("model.layers.0.self_attn.q_proj.weight") == (
            "model.layers.0.self_attn.qkv_proj.weight",
            "q",
        )
        assert mapper._map_name_with_shard("model.layers.0.self_attn.k_proj.weight")[1] == "k"
        assert mapper._map_name_with_shard("model.layers.0.self_attn.v_proj.weight")[1] == "v"
        assert mapper._map_name_with_shard("model.layers.0.mlp.gate_proj.weight") == (
            "model.layers.0.mlp.gate_up_proj.weight",
            0,
        )
        assert mapper._map_name_with_shard("model.layers.0.mlp.up_proj.weight") == (
            "model.layers.0.mlp.gate_up_proj.weight",
            1,
        )
        # 直通名不受影响
        assert mapper._map_name_with_shard("model.embed_tokens.weight") == (
            "model.embed_tokens.weight",
            None,
        )

    def test_mapper_apply_stamps_shard_id(self):
        # WeightsMapper.apply：shard_id 挂到张量属性上（weight_loader 的定位键）
        mapper = LlamaModel.hf_to_vllm_mapper
        w = torch.zeros(4, 4)
        out = list(mapper.apply([("x.q_proj.weight", w)]))
        assert out[0][0] == "x.qkv_proj.weight"
        assert getattr(out[0][1], "shard_id", None) == "q"

    def test_load_weights_end_to_end(self):
        hf = make_hf_config()
        vllm_config = make_vllm_config(hf)
        model = build_model(vllm_config)
        ckpt = gen_ckpt_weights(hf)
        loaded = load_into_model(model, ckpt)
        # 词表 padding：embed 权重被 pad 到 64 倍数（128），checkpoint 100 行进 [0:100]
        assert model.model.embed_tokens.weight.shape == (128, 64)
        assert torch.equal(
            model.model.embed_tokens.weight.data[:100], ckpt["model.embed_tokens.weight"]
        )
        assert torch.all(model.model.embed_tokens.weight.data[100:] == 0)
        # 融合段落位
        qkv = model.model.layers[0].self_attn.qkv_proj.weight.data
        assert torch.equal(qkv[:64], ckpt["model.layers.0.self_attn.q_proj.weight"])
        assert torch.equal(qkv[64:96], ckpt["model.layers.0.self_attn.k_proj.weight"])
        assert torch.equal(qkv[96:128], ckpt["model.layers.0.self_attn.v_proj.weight"])
        mlp = model.model.layers[0].mlp
        assert torch.equal(
            mlp.gate_up_proj.weight.data[:128], ckpt["model.layers.0.mlp.gate_proj.weight"]
        )
        assert torch.equal(
            mlp.gate_up_proj.weight.data[128:], ckpt["model.layers.0.mlp.up_proj.weight"]
        )
        assert torch.equal(mlp.down_proj.weight.data, ckpt["model.layers.0.mlp.down_proj.weight"])
        # lm_head 同样 pad 到 128
        assert model.lm_head.weight.shape == (128, 64)
        assert torch.equal(model.lm_head.weight.data[:100], ckpt["lm_head.weight"])
        # 返回已装载名集合（AutoWeightsLoader 递归分发 + 子模块委派）
        assert any("qkv_proj" in n for n in loaded)
        assert any("embed_tokens" in n for n in loaded)

    def test_load_weights_tied(self):
        hf = make_hf_config(tie_word_embeddings=True)
        vllm_config = make_vllm_config(hf)
        model = build_model(vllm_config)
        # llama.py:L491-L492：tie 时 lm_head 复用 embed_tokens 权重
        assert model.lm_head.weight is model.model.embed_tokens.weight
        ckpt = gen_ckpt_weights(hf)
        ckpt.pop("lm_head.weight")
        loaded = load_into_model(model, ckpt)
        # skip_prefixes=['lm_head.']：checkpoint 无 lm_head 也不报错
        assert any("embed_tokens" in n for n in loaded)


# ────────────────────── m1/m3/m4/m5：前向与出口契约 ──────────────────────


def _fwd_setup(seed=7, T=5):
    hf = make_hf_config()
    vllm_config = make_vllm_config(hf)
    model = build_model(vllm_config)
    ckpt = gen_ckpt_weights(hf)
    load_into_model(model, ckpt)
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, hf.vocab_size, (T,), generator=g)
    positions = torch.arange(T, dtype=torch.long)
    qsl = torch.tensor([0, T], dtype=torch.int32)
    seq_lens = torch.tensor([T], dtype=torch.int32)
    meta = {
        name: RefAttnMetadata(qsl, seq_lens)
        for name in vllm_config.compilation_config.static_forward_context
    }
    hd = hf.hidden_size // hf.num_attention_heads
    return hf, vllm_config, model, ckpt, input_ids, positions, meta, T, hd


class TestForwardContract:
    def test_forward_returns_hidden_states_not_logits(self):
        # llama.py:L516-L526：forward 只出 hidden_states，不是 logits
        hf, vllm_config, model, _, input_ids, positions, meta, T, hd = _fwd_setup()
        sm = {name: torch.arange(T) for name in meta}
        with set_forward_context(meta, vllm_config, slot_mapping=sm):
            bind_kv_caches(vllm_config, hf.num_hidden_layers, T,
                           hf.num_key_value_heads, hd)
            out = model(input_ids=input_ids, positions=positions, intermediate_tensors=None)
        assert out.shape == (T, hf.hidden_size)
        assert not isinstance(out, IntermediateTensors)

    def test_forward_matches_reference_implementation(self):
        # 王冠数值测试：整模型前向 == 测试侧手写的 HF 式参考实现
        hf, vllm_config, model, ckpt, input_ids, positions, meta, T, hd = _fwd_setup()
        sm = {name: torch.arange(T) for name in meta}
        with set_forward_context(meta, vllm_config, slot_mapping=sm):
            bind_kv_caches(vllm_config, hf.num_hidden_layers, T,
                           hf.num_key_value_heads, hd)
            hidden = model(input_ids=input_ids, positions=positions, intermediate_tensors=None)
        ref = reference_forward(hf, ckpt, input_ids, positions)
        assert torch.allclose(hidden.double(), ref, atol=2e-4, rtol=1e-4)

    def test_compute_logits_vocab_cut(self):
        # logits_processor.py:L150-L152：裁掉词表 padding
        hf, vllm_config, model, _, _, _, _, _, _ = _fwd_setup()
        h = torch.randn(3, hf.hidden_size, dtype=torch.float32)
        logits = model.compute_logits(h)
        assert logits.shape == (3, 100)  # 128 裁回 org_vocab 100
        ref = h.double() @ model.lm_head.weight.double()[:100].T
        assert torch.allclose(logits.double(), ref, atol=1e-5)

    def test_compute_logits_tp1_no_gather(self):
        # _get_logits：tp=1 时不 gather（lm_head.tp_size>1 判置假），直接裁 padding
        with set_current_vllm_config(make_vllm_config()):
            lp = LogitsProcessor(vocab_size=100)
            assert lp.org_vocab_size == 100
            lm_head = ParallelLMHead(100, 8)
            h = torch.randn(2, 8)
            out = lp._get_logits(h, lm_head, None)
        assert out.shape == (2, 100)

    def test_runner_sampling_position_policy(self):
        # gpu_model_runner.py:L2232-L2240 + L4483-L4485：策略归 runner
        hf, vllm_config, model, ckpt, _, positions, _, T, hd = _fwd_setup(seed=3, T=8)
        # 两请求：req0 5 token、req1 3 token → 采样位 [4, 7]
        qsl = torch.tensor([0, 5, 8], dtype=torch.int32)
        seq_lens = torch.tensor([5, 3], dtype=torch.int32)
        meta = {
            name: RefAttnMetadata(qsl, seq_lens)
            for name in vllm_config.compilation_config.static_forward_context
        }
        g = torch.Generator().manual_seed(3)
        input_ids = torch.randint(0, hf.vocab_size, (T,), generator=g)
        positions = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], dtype=torch.long)
        sm = {name: torch.arange(T) for name in meta}
        bind_kv_caches(vllm_config, hf.num_hidden_layers, T, hf.num_key_value_heads, hd)
        runner = GPUModelRunner(
            vllm_config=vllm_config,
            model=model,
            query_start_loc=qsl,
            attn_metadata=meta,
            slot_mapping=sm,
        )
        logits = runner.execute_model(input_ids=input_ids, positions=positions)
        # decode 批每请求只物化 1 行 logits → [2, 100]
        assert logits.shape == (2, 100)
        logits_indices = qsl[1:] - 1
        assert logits_indices.tolist() == [4, 7]
        # 数值对照：hidden[logits_indices] 过 compute_logits
        with set_forward_context(meta, vllm_config, slot_mapping=sm):
            hidden = model(input_ids=input_ids, positions=positions, intermediate_tensors=None)
            ref = model.compute_logits(hidden[logits_indices])
        assert torch.allclose(logits, ref, atol=1e-5)


# HF 式参考前向（测试侧独立实现，双精度）
def reference_forward(hf, w, input_ids, positions):
    H = hf.hidden_size
    hd = H // hf.num_attention_heads
    nkvh = hf.num_key_value_heads
    rep = hf.num_attention_heads // nkvh
    eps = hf.rms_norm_eps
    base = getattr(hf, "rope_theta", 10000.0)

    def rms(x, weight):
        v = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(v + eps) * weight

    inv = 1.0 / (base ** (torch.arange(0, hd, 2, dtype=torch.float64) / hd))
    freqs = positions.double()[:, None] * inv[None, :]
    cos, sin = freqs.cos().unsqueeze(1), freqs.sin().unsqueeze(1)  # [T,1,hd/2]

    def rope(x):  # neox 式
        x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    h = F.embedding(input_ids, w["model.embed_tokens.weight"].double())
    residual = None
    for i in range(hf.num_hidden_layers):
        p = f"model.layers.{i}"
        if residual is None:
            residual = h
            h = rms(h, w[f"{p}.input_layernorm.weight"].double())
        else:
            s = h + residual
            residual = s
            h = rms(s, w[f"{p}.input_layernorm.weight"].double())
        qkv = h @ w[f"{p}.self_attn.q_proj.weight"].double().T
        k_all = h @ w[f"{p}.self_attn.k_proj.weight"].double().T
        v_all = h @ w[f"{p}.self_attn.v_proj.weight"].double().T
        T = h.shape[0]
        q = rope(qkv.reshape(T, hf.num_attention_heads, hd))
        k = rope(k_all.reshape(T, nkvh, hd))
        v = v_all.reshape(T, nkvh, hd)
        heads = []
        for hq_ in range(hf.num_attention_heads):
            kh = hq_ // rep
            scores = q[:, hq_] @ k[:, kh].T * (hd**-0.5)
            causal = torch.arange(T)[:, None] >= torch.arange(T)[None, :]
            p_attn = torch.softmax(scores.masked_fill(~causal, float("-inf")), dim=-1)
            heads.append(p_attn @ v[:, kh])
        attn_out = torch.cat(heads, dim=-1) @ w[f"{p}.self_attn.o_proj.weight"].double().T
        s = attn_out + residual
        residual = s
        h = rms(s, w[f"{p}.post_attention_layernorm.weight"].double())
        gate_up = h @ torch.cat(
            [w[f"{p}.mlp.gate_proj.weight"], w[f"{p}.mlp.up_proj.weight"]], dim=0
        ).double().T
        d = gate_up.shape[-1] // 2
        act = F.silu(gate_up[..., :d]) * gate_up[..., d:]
        h = act @ w[f"{p}.mlp.down_proj.weight"].double().T
    final = rms(h + residual, w["model.norm.weight"].double())
    return final


# ────────────────────── m1：RMSNorm 融合 add-norm ──────────────────────


class TestRMSNorm:
    def test_single_arg(self):
        # layernorm.py forward_native → ir.ops.rms_norm 数学
        with set_current_vllm_config(make_vllm_config()):
            norm = RMSNorm(8, eps=1e-6)
            x = torch.randn(4, 8)
            out = norm(x)
        v = x.double().pow(2).mean(-1, keepdim=True)
        ref = x.double() * torch.rsqrt(v + 1e-6) * 1.0
        assert torch.allclose(out.double(), ref, atol=1e-6)

    def test_fused_add_returns_pair(self):
        # RMSNorm(x, residual) 一个 kernel 同时完成加残差+归一化，返回二元组
        with set_current_vllm_config(make_vllm_config()):
            norm = RMSNorm(8, eps=1e-6)
            norm.weight.data = torch.randn(8) * 0.1 + 1
            x = torch.randn(4, 8)
            r = torch.randn(4, 8)
            out, new_residual = norm(x, r)
        s = x.double() + r.double()
        v = s.pow(2).mean(-1, keepdim=True)
        ref = s * torch.rsqrt(v + 1e-6) * norm.weight.data.double()
        assert torch.allclose(out.double(), ref, atol=1e-5)
        # 返回的 residual 是未归一化的和（层间一等公民）
        assert torch.allclose(new_residual.double(), s, atol=1e-6)

    def test_decoder_layer_first_layer_special_case(self):
        # llama.py:L317-L319：首层 residual=None 特判。权重先经真实
        # load_weights 装载（未初始化的 torch.empty 权重数值无界，会淹没
        # 1e-6 级的残差断言——线性权重的 create_weights 本就是 torch.empty）
        hf = make_hf_config()
        vllm_config = make_vllm_config(hf)
        model = build_model(vllm_config)
        load_into_model(model, gen_ckpt_weights(hf))
        layer = model.model.layers[0]
        x = torch.randn(2, hf.hidden_size)
        positions = torch.arange(2)
        qsl = torch.tensor([0, 2], dtype=torch.int32)
        seq_lens = torch.tensor([2], dtype=torch.int32)
        meta = {
            n: RefAttnMetadata(qsl, seq_lens)
            for n in vllm_config.compilation_config.static_forward_context
        }
        sm = {n: torch.arange(2) for n in meta}
        captured = {}
        layer.self_attn.register_forward_hook(
            lambda m, inp, out: captured.__setitem__("attn_out", out)
        )
        with set_forward_context(meta, vllm_config, slot_mapping=sm):
            bind_kv_caches(vllm_config, hf.num_hidden_layers, 2,
                           hf.num_key_value_heads,
                           hf.hidden_size // hf.num_attention_heads)
            h, residual = layer(positions, x, None)
        # 首层特判：residual 从 x 起步；post_attention 融合 add-norm 后
        # residual = attn_out + x（未归一化的和——层间一等公民）
        assert torch.allclose(residual - captured["attn_out"], x, atol=1e-6)
        assert not torch.equal(h, x)  # hidden 已 norm+attn+mlp


# ────────────────────── m3/m11：Attention 插座契约 ──────────────────────


class TestAttentionSocket:
    def test_forward_signature_has_no_metadata(self):
        # Part VI hook 的源码实证：签名里没有 attn_metadata/kv_cache/slot_mapping
        params = inspect.signature(Attention.forward).parameters
        assert "attn_metadata" not in params
        assert "kv_cache" not in params
        assert "slot_mapping" not in params

    def test_self_registration_into_static_forward_context(self):
        # attention.py:L443-L446：__init__ 尾部自注册；重复 layer_name 即 raise
        vllm_config = make_vllm_config()
        with set_current_vllm_config(vllm_config):
            attn = Attention(
                4,
                16,
                16**-0.5,
                num_kv_heads=2,
                cache_config=vllm_config.cache_config,
                prefix="model.layers.9.self_attn.attn",
                attn_backend=RefAttentionBackend,
            )
        assert (
            vllm_config.compilation_config.static_forward_context[
                "model.layers.9.self_attn.attn"
            ]
            is attn
        )
        with set_current_vllm_config(vllm_config), pytest.raises(
            ValueError, match="Duplicate layer name"
        ):
            Attention(
                4,
                16,
                16**-0.5,
                num_kv_heads=2,
                cache_config=vllm_config.cache_config,
                prefix="model.layers.9.self_attn.attn",
                attn_backend=RefAttentionBackend,
            )
        # 占位 kv_cache 等 bind_kv_cache 替换（attention.py:L460-L463 + AttentionLayerBase）
        assert attn.kv_cache.numel() == 0
        real = torch.zeros(4, 2, 2, 16)
        attn.bind_kv_cache(real)
        assert attn.kv_cache is real

    def test_get_attention_context_spec_decode_list(self):
        # attention.py:L757-L762：list[dict] 时 [0] 为 base 模型
        vllm_config = make_vllm_config()
        meta0 = {"layer.a": "base"}
        meta1 = {"layer.a": "draft"}
        slot = {"layer.a": torch.zeros(2, dtype=torch.long)}

        class FakeLayer:
            kv_cache = torch.zeros(1)

        vllm_config.compilation_config.static_forward_context["layer.a"] = FakeLayer()
        with set_forward_context([meta0, meta1], vllm_config, slot_mapping=slot):
            attn_metadata, attn_layer, kv_cache, layer_slot = get_attention_context("layer.a")
        assert attn_metadata == "base"
        assert layer_slot.shape == (2,)

    def test_dummy_dep_preserves_ordering(self):
        # unified_kv_cache_update 返回空张量 dummy（torch.compile 数据依赖的载体）
        vllm_config = make_vllm_config()
        layer_name = "model.layers.0.self_attn.attn"
        impl = RefAttentionImpl(4, 16, 0.25, num_kv_heads=2)

        class Layer:
            kv_cache = torch.zeros(8, 2, 2, 16)

        Layer.impl = impl  # class-body 内 `impl = impl` 会 NameError，外置赋值

        vllm_config.compilation_config.static_forward_context[layer_name] = Layer()
        k = torch.randn(3, 2, 16)
        v = torch.randn(3, 2, 16)
        slots = torch.tensor([0, 1, 2])
        with set_forward_context(
            {layer_name: RefAttnMetadata(torch.tensor([0, 3]), torch.tensor([3]))},
            vllm_config,
            slot_mapping={layer_name: slots},
        ):
            dummy = unified_kv_cache_update(k, v, layer_name)
        assert dummy.numel() == 0
        assert torch.equal(Layer.kv_cache[0, 0], k[0])


# ────────────────────── m10/m12：装载编排与词表分片 ──────────────────────


class TestLoadOrchestration:
    def test_get_model_loader_table(self):
        # model_loader/__init__.py：load_format 查表；default 与 dummy 两项（delete[9]）
        assert (
            type(get_model_loader(LoadConfig(load_format="auto"))).__name__
            == "DefaultModelLoader"
        )
        assert (
            type(get_model_loader(LoadConfig(load_format="dummy"))).__name__
            == "DummyModelLoader"
        )
        with pytest.raises(ValueError, match="not supported"):
            get_model_loader(LoadConfig(load_format="tensorizer"))

    def test_base_loader_load_model_orchestration(self):
        # base_loader.py:L40-L82：dtype/device 上下文 → 建空壳 → load_weights →
        # process_weights_after_loading → eval()——四段主干
        hf = make_hf_config()
        vllm_config = make_vllm_config(hf)
        ckpt = gen_ckpt_weights(hf)

        from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

        class SeededLoader(DefaultModelLoader):
            # SUBTRACTED get_all_weights 的 host 测试替身（checkpoint IO 面）
            def get_all_weights(self, model_config, model):
                for name, tensor in ckpt.items():
                    yield name, tensor

        loader = SeededLoader(LoadConfig(load_format="auto"))
        model = loader.load_model(
            vllm_config=vllm_config, model_config=vllm_config.model_config
        )
        inject_ref_backend(model, vllm_config)
        assert isinstance(model, LlamaForCausalLM)
        assert model.training is False  # eval()
        assert torch.equal(
            model.model.layers[1].self_attn.qkv_proj.weight.data[:64],
            ckpt["model.layers.1.self_attn.q_proj.weight"],
        )

    def test_get_model_unified_entry(self):
        # model_loader/__init__.py get_model：统一入口（dummy 装载免 checkpoint）
        vllm_config = make_vllm_config(load_format="dummy")
        model = get_model(vllm_config=vllm_config)
        assert isinstance(model, LlamaForCausalLM)
        assert model.training is False

    def test_gpu_model_runner_load_model_line(self):
        # gpu_model_runner.py:L5303-L5326：load_model → get_model_loader →
        # loader.load_model(vllm_config, model_config)——模型层由此进厂
        vllm_config = make_vllm_config(load_format="dummy")
        runner = GPUModelRunner(vllm_config=vllm_config)
        assert runner.model is None
        runner.load_model()
        assert isinstance(runner.model, LlamaForCausalLM)

    def test_load_model_keeps_eagle3_aux_call_and_dead_eplb_block(self):
        # gpu_model_runner.py:L5362 + L5376-L5390（delete[10] 明示原样保留）：
        # ① _setup_eagle3_aux_hidden_state_outputs() 无条件调用（默认
        # use_aux_hidden_state_outputs=False → no-op 守卫 L5483-L5484 首行即
        # return）；② EPLB enable 条件块整块含头——self._moe_model 维持
        # __init__ L551 的 None，条件第一项恒假、块体不进
        vllm_config = make_vllm_config(load_format="dummy")
        runner = GPUModelRunner(vllm_config=vllm_config)
        assert runner._moe_model is None  # __init__ L551
        assert runner.eplb_state is None  # __init__ L550
        assert runner.parallel_config.enable_eplb is False
        calls = []
        runner._setup_eagle3_aux_hidden_state_outputs = lambda: calls.append(1)
        runner.load_model()
        assert calls == [1]  # L5362 调用确实发生（真实方法体是 no-op）
        assert runner._moe_model is None  # EPLB 块恒假不进，_moe_model 不被赋值

    def test_load_model_noop_guard_returns_early(self):
        # gpu_model_runner.py:L5482-L5484：no-op 守卫——默认 False 首行即 return
        vllm_config = make_vllm_config(load_format="dummy")
        runner = GPUModelRunner(vllm_config=vllm_config)
        assert runner.use_aux_hidden_state_outputs is False  # __init__ L578
        runner._setup_eagle3_aux_hidden_state_outputs()  # 不 raise 即通过

    def test_execute_model_aux_unpack_and_broadcast_head(self):
        # gpu_model_runner.py:L4459-L4465（delete[11] 明示保留）：aux 解包段
        # ——use_aux=True 时解二元组取 hidden_states；False 时直用（常路）；
        # L4467 if not self.broadcast_pp_output: 头下主路径切片两行 L4484-L4485
        hf = make_hf_config()
        vllm_config = make_vllm_config(hf)
        model = build_model(vllm_config)
        load_into_model(model, gen_ckpt_weights(hf))
        T = 4
        hidden = torch.randn(T, hf.hidden_size)
        qsl = torch.tensor([0, T], dtype=torch.int32)

        class AuxStubModel:
            # EAGLE3 形状的输出：(hidden_states, aux_hidden_states) 二元组
            def __call__(self, input_ids=None, positions=None,
                         intermediate_tensors=None, inputs_embeds=None, **kw):
                return (hidden, ["aux_layer0", "aux_layer1"])

            def compute_logits(self, h):
                return model.compute_logits(h)

        runner = GPUModelRunner(
            vllm_config=vllm_config,
            model=AuxStubModel(),
            query_start_loc=qsl,
        )
        assert runner.broadcast_pp_output is False
        # EAGLE3 场景：旗标为真时模型返回二元组（真实不变量，__init__ L578
        # 默认 False、spec config 置 True——此处直置模拟该场景）
        runner.use_aux_hidden_state_outputs = True
        logits = runner.execute_model(input_ids=torch.zeros(T, dtype=torch.long),
                                      positions=torch.arange(T))
        # 解包取第一元 → 采样位切片 → compute_logits：[1, vocab]
        assert logits.shape == (1, hf.vocab_size)
        assert torch.allclose(logits, model.compute_logits(hidden[[T - 1]]))
        # 常路（use_aux=False）：model_output 直用（非元组模型）
        runner2 = GPUModelRunner(
            vllm_config=vllm_config, model=model, query_start_loc=qsl
        )
        assert runner2.use_aux_hidden_state_outputs is False

    def test_parallel_lm_head_is_vocab_embedding(self):
        # vocab_parallel_embedding.py:L520：ParallelLMHead = VocabParallelEmbedding 子类
        assert issubclass(ParallelLMHead, VocabParallelEmbedding)
        emb = VocabParallelEmbedding(100, 64)
        # pad_vocab_size：100 → 128（64 倍数）
        assert emb.num_embeddings_padded == 128
        head = ParallelLMHead(100, 64)
        assert head.weight.shape == (128, 64)

    def test_lm_head_forward_raises(self):
        # 真实行为：LMHead 权重只在 sampler/compute_logits 里用，forward 即 raise
        head = ParallelLMHead(100, 64)
        with pytest.raises(RuntimeError, match="sampler"):
            head(torch.zeros(2, dtype=torch.long))


class TestPluggableAndNewStyle:
    def test_logits_processor_pluggable_registered(self):
        # logits_processor.py:L22：@PluggableLayer.register("logits_processor")
        #   ——in-tree 注册面（op_registry；OOT 位 register_oot 另立）
        from vllm.model_executor.custom_op import op_registry

        assert op_registry["logits_processor"] is LogitsProcessor

    def test_initialize_model_new_style_signature(self):
        # utils.py:L57-L64：inspect 校验 (vllm_config, prefix) →
        # set_current_vllm_config 上下文内构造
        vllm_config = make_vllm_config()
        model = initialize_model(vllm_config)
        assert isinstance(model, LlamaForCausalLM)
        names = dict(model.named_children())
        assert "model" in names and "lm_head" in names

    def test_make_layers_pp_sentinel(self):
        # models/utils.py:L798-L826：get_pp_indices 均分 + PPMissingLayer 占空段
        dist._TP_STATE.update(rank=0, size=1, pp_rank=1, pp_size=2)
        with set_current_vllm_config(make_vllm_config()):
            start, end, layers = make_layers(4, lambda prefix: RMSNorm(4), "model.layers")
        assert (start, end) == (2, 4)
        assert isinstance(layers[0], PPMissingLayer)
        assert isinstance(layers[1], PPMissingLayer)
        assert not isinstance(layers[2], PPMissingLayer)
        assert not isinstance(layers[3], PPMissingLayer)
        # 参数名稳定：空段仍占名
        names = [n for n, _ in layers.named_children()]
        assert names == ["0", "1", "2", "3"]

    def test_make_empty_intermediate_tensors(self):
        # models/utils.py:L866-L877：PP 空中间张量钩子（hidden_states/residual）
        f = make_empty_intermediate_tensors_factory(["hidden_states", "residual"], 64)
        t = f(3, torch.float32, torch.device("cpu"))
        assert isinstance(t, IntermediateTensors)
        assert t["hidden_states"].shape == (3, 64)
        assert torch.all(t["residual"] == 0)

    def test_intermediate_tensors_getitem(self):
        t = IntermediateTensors({"a": torch.zeros(2, 2), "b": torch.ones(2, 2)})
        assert t["a"].shape == (2, 2)
        sliced = t[0:1]
        assert sliced["b"].shape == (1, 2)


# ────────────────────── m13：v0.27 新面（叙事锚） ──────────────────────


class TestV027NewFaces:
    def test_head_dtype_default_none(self):
        # logits_processor.py:L57-L61：head_dtype 默认 None（fp32 head 属可选配置）
        with set_current_vllm_config(make_vllm_config()):
            lp = LogitsProcessor(vocab_size=100)
        assert lp.head_dtype is None

    def test_apply_head_main_path(self):
        # _apply_head 主路径（head_dtype None → quant_method.apply）
        with set_current_vllm_config(make_vllm_config()):
            lp = LogitsProcessor(vocab_size=100)
            lm_head = ParallelLMHead(100, 8)
            h = torch.randn(2, 8)
            out = lp._apply_head(lm_head, h, None)
            ref = h @ lm_head.weight.T
        assert torch.allclose(out, ref, atol=1e-5)

    def test_gather_logits_identity_tp1(self):
        # _gather_logits：tp=1 时 gather 通道退化恒等（GroupCoordinator.gather
        # L758-L761 world_size==1 bypass）
        with set_current_vllm_config(make_vllm_config()):
            lp = LogitsProcessor(vocab_size=100)
            x = torch.randn(2, 100)
            out = lp._gather_logits(x)
        assert out is x

    def test_gather_logits_all_gather_path(self):
        # use_all_gather=True（TPU/XLA 平台）→ all-gather 通道（tp=1 恒等）
        with set_current_vllm_config(make_vllm_config()):
            lp = LogitsProcessor(vocab_size=100)
            lp.use_all_gather = True
            x = torch.randn(2, 100)
            out = lp._gather_logits(x)
        assert torch.equal(out, x)

    def test_rotary_emb_blackbox(self):
        # get_rope 工厂产物：neox 式旋转（forward 五行之一）
        with set_current_vllm_config(make_vllm_config()):
            rope = get_rope(16, max_position=32, is_neox_style=True)
            q = torch.randn(3, 16)
            k = torch.randn(3, 16)
            positions = torch.tensor([0, 1, 5], dtype=torch.long)
            q2, k2 = rope(positions, q.clone(), k.clone())
        inv = 1.0 / (10000.0 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16))
        freqs = positions.float()[:, None] * inv[None, :]
        cos, sin = freqs.cos(), freqs.sin()
        ref_q = torch.cat(
            [q[:, :8] * cos - q[:, 8:] * sin, q[:, 8:] * cos + q[:, :8] * sin], dim=-1
        )
        assert torch.allclose(q2, ref_q, atol=1e-5)

    def test_llama_mlp_silu_only(self):
        with pytest.raises(ValueError, match="Only silu"):
            LlamaMLP(hidden_size=8, intermediate_size=16, hidden_act="gelu")

    def test_autoweights_loader_class_face(self):
        assert AutoWeightsLoader is not None
        assert hasattr(WeightsMapper, "apply")
