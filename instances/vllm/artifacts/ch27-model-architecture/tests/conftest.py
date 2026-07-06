"""ch25 tests — 共享 fixture：构造一个最小但形状自洽的 DeepSeek-V4 fake config + vllm_config，
让精简版的模块能在 host 上实例化、被结构性巡查。

这些值参照真实 DeepSeek-V4 hf_config 的字段名与量级关系（小型化以便 host 跑），不杜撰新字段。
"""
import sys
import types
from pathlib import Path

import pytest

# 让 `import implementation.xxx` 可用（章目录在 tests/ 的上一级）。
CH_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CH_DIR))


def make_hf_config():
    head_dim = 64
    return types.SimpleNamespace(
        hidden_size=128,
        num_attention_heads=4,
        num_hidden_layers=2,
        head_dim=head_dim,
        qk_rope_head_dim=16,          # nope = head_dim - rope = 48
        q_lora_rank=32,
        o_lora_rank=24,
        o_groups=2,
        compress_ratios=[1, 1],       # dense（无稀疏 SWA）
        sliding_window=0,
        rms_norm_eps=1e-6,
        max_position_embeddings=256,
        vocab_size=1000,
        # MoE
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        n_shared_experts=1,
        swiglu_limit=None,
        norm_topk_prob=True,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=2.0,
        num_hash_layers=0,
        topk_method="noaux_tc",
        hidden_act="silu",
        expert_dtype="fp8",
        # hc 超连接
        hc_mult=2,
        hc_sinkhorn_iters=3,
        hc_eps=1e-3,
        # rope（精简版 get_rope 只读 head_dim/max_position）
        rope_parameters={"rope_type": "default"},
        rope_scaling=None,
        rope_theta=10000.0,
        compress_rope_theta=10000.0,
        index_topk=8,
        quantization_config={"scale_fmt": "e8m0"},
        # MTP
        num_nextn_predict_layers=1,
    )


def make_vllm_config(hf_config=None):
    hf_config = hf_config or make_hf_config()
    return types.SimpleNamespace(
        model_config=types.SimpleNamespace(hf_config=hf_config, dtype=__import__("torch").float32),
        quant_config=None,
        cache_config=None,
        parallel_config=types.SimpleNamespace(enable_expert_parallel=False),
        kernel_config=types.SimpleNamespace(moe_backend="fused_moe"),
        scheduler_config=types.SimpleNamespace(max_num_batched_tokens=16),
        speculative_config=types.SimpleNamespace(
            draft_model_config=types.SimpleNamespace(hf_config=hf_config)
        ),
        compilation_config=types.SimpleNamespace(static_forward_context={}),
    )


@pytest.fixture
def hf_config():
    return make_hf_config()


@pytest.fixture
def vllm_config(hf_config):
    return make_vllm_config(hf_config)
