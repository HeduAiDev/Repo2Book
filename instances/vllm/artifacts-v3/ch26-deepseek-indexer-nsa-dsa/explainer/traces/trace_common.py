# ch26 explainer 驱动脚本公共装配件 —— 复用 tests/ 的真实注入件与注册流。
# 运行环境：host CPython 3.11 + torch CPU float32（无 CUDA/vLLM 安装）；
# 精简版 implementation/ 的 HOST SEAM 镜像承载 CUDA kernel 的精确数学
# （impl-notes §Seam B1）。trace 数值是『kernel 数学』的 host 实证。
import json
import sys
from pathlib import Path

import torch

_IMPL = Path(__file__).resolve().parents[2] / "implementation"
sys.path.insert(0, str(_IMPL))

from transformers import DeepseekV2Config  # noqa: E402

FP8_MAX = 448.0
DEV = torch.device("cpu")
RADIX_WS = 1024 * 1024
OUT_DIR = Path(__file__).resolve().parent


def dump(name: str, doc: dict) -> None:
    """落盘 trace JSON（LF 行尾，CLAUDE.md 坑 8）。"""
    (OUT_DIR / name).open("w", encoding="utf-8", newline="\n").write(
        json.dumps(doc, ensure_ascii=False, indent=1))


# ─────────────────────────── 配置工厂（与 tests/ 同款） ───────────────────────────
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
    """V4 形状 HF config 载体（SimpleNamespace——V4 真实 config 类不在本
    transformers 版本内，字段面与 tests/ 同款）。"""
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


# ─────────────────────── 独立参考数学（与 tests/ 同款） ───────────────────────
def ref_group_quant_ue8m0(x: torch.Tensor):
    """per-token-group FP8 量化参考（128 一组、ue8m0 幂次 scale）。"""
    amax = x.abs().amax(dim=-1, keepdim=True)
    scale = torch.exp2(torch.ceil(torch.log2(
        torch.clamp(amax, min=1e-10) / FP8_MAX)))
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


def ref_dsa_logits(q_deq, k_deq, weights):
    """I_{t,s} = Σ_j w_{t,j}·ReLU(q_{t,j}·k_s) —— DSA Eq.(1) 参考。"""
    dots = torch.einsum("mhd,nd->mhn", q_deq, k_deq)
    return torch.einsum("mhn,mh->mn", torch.relu(dots), weights)


def ref_topk_desc(vals, k: int, base: int = 0):
    """降序 top-k，tie 取小 index（与 CUDA 核插入排序 tie-break 同构）。"""
    order = sorted(range(len(vals)), key=lambda i: (-vals[i], i))
    sel = order[:k]
    return [base + i for i in sel] + [-1] * max(0, k - len(sel))


def make_block_table(seq_lens, block_size, first_block=0):
    """分页块表：请求 b 第 i 块 → 物理块号（first_block 起顺序分配；
    first_block>0 用来制造物理位≠逻辑位的非平凡分页）。"""
    max_blocks = max(1, max((s + block_size - 1) // block_size
                            for s in seq_lens))
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


class _ParamStub:
    def __init__(self):
        self.loaded = None

    def weight_loader(self, param, tensor, shard_id):
        self.loaded = tensor.clone()


def _cos_sin_cache(max_pos, rope_dim, base=10000.0):
    half = rope_dim // 2
    inv = 1.0 / (base ** (torch.arange(0, half).float() / half))
    pos = torch.arange(max_pos).float()
    ang = torch.outer(pos, inv)
    return torch.cat([ang.cos(), ang.sin()], dim=-1)
