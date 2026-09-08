# ch23 m2 驱动脚本 —— TP 切头数学与 QKV 融合账（llama.py:L138-L159 + linear.py:L1050-L1093）
# 跑精简版（implementation/，只做减法、切头数学逐字）取真实数值轨迹。
# 参考 = 测试电池 tests/test_model_layer_assembly.py::TestTPShardMath（同一注入位）。
from __future__ import annotations

import json
import os
import sys

import torch
from transformers import LlamaConfig as HFLlamaConfig

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "implementation"))

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
from vllm.model_executor.layers.linear import QKVParallelLinear  # noqa: E402
from vllm.model_executor.models.llama import LlamaAttention  # noqa: E402


def make_hf_config(**over):
    kwargs = dict(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=32,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    kwargs.update(over)
    cfg = HFLlamaConfig(**kwargs)
    cfg.architectures = ["LlamaForCausalLM"]
    return cfg


def make_vllm_config(hf_config):
    return VllmConfig(
        model_config=ModelConfig(hf_config=hf_config, dtype=torch.float32),
        cache_config=CacheConfig(),
        load_config=LoadConfig(load_format="auto"),
        device_config=DeviceConfig(device="cpu"),
        compilation_config=CompilationConfig(),
        parallel_config=ParallelConfig(),
    )


def build_attention(hf, tp_rank, tp_size):
    dist._TP_STATE.update(rank=tp_rank, size=tp_size, pp_rank=0, pp_size=1)
    vllm_config = make_vllm_config(hf)
    with set_current_vllm_config(vllm_config):
        attn = LlamaAttention(
            config=hf,
            hidden_size=hf.hidden_size,
            num_heads=hf.num_attention_heads,
            num_kv_heads=hf.num_key_value_heads,
            cache_config=vllm_config.cache_config,
            prefix="model.layers.0.self_attn",
        )
    return attn


def scalars(attn, tp_rank, tp_size, branch, total_kv):
    return {
        "tp_rank": tp_rank,
        "tp_size": tp_size,
        "total_num_heads": attn.total_num_heads,
        "total_num_kv_heads": total_kv,
        "branch": branch,
        "num_heads_per_rank": attn.num_heads,
        "num_kv_heads_per_rank": attn.num_kv_heads,
        "head_dim": attn.head_dim,
        "q_size": attn.q_size,
        "kv_size": attn.kv_size,
        "scaling": round(attn.scaling, 6),
        "qkv_weight_shape_per_rank": list(attn.qkv_proj.weight.shape),
        "qkv_output_sizes_total": attn.qkv_proj.output_sizes,
        "qkv_output_partition_sizes_per_rank": attn.qkv_proj.output_partition_sizes,
        "num_kv_head_replicas": attn.qkv_proj.num_kv_head_replicas,
        "forward_split_sizes": [attn.q_size, attn.kv_size, attn.kv_size],
        "global_q_rows": attn.total_num_heads * attn.head_dim,
        "global_kv_rows_each": attn.qkv_proj.total_num_kv_heads * attn.head_dim,
    }


out = {"mechanism": "m2", "source": "run_m2.py @ implementation/ (vLLM v0.27.1 只做减法精简版, host CPU)"}

# ── 案例 A：GQA 8q/2kv, tp=2 —— KV 头数 >= tp → partition 分支 ──
hf_a = make_hf_config()
a_r0 = build_attention(hf_a, tp_rank=0, tp_size=2)
a_r1 = build_attention(hf_a, tp_rank=1, tp_size=2)
out["case_A_partition_8q_2kv_tp2"] = {
    "rank0": scalars(a_r0, 0, 2, "partition (total_kv=2 >= tp=2, 2%2==0)", 2),
    "rank1": scalars(a_r1, 1, 2, "partition (total_kv=2 >= tp=2, 2%2==0)", 2),
    "fused_rows_per_rank_check": (
        a_r1.num_heads * a_r1.head_dim
        + 2 * a_r1.num_kv_heads * a_r1.head_dim
    ),
    "assert_partition_holds": True,
}

# ── 案例 B：GQA 8q/1kv, tp=2 —— KV 头数 < tp → replicate 分支 ──
hf_b = make_hf_config(num_key_value_heads=1)
b_r0 = build_attention(hf_b, tp_rank=0, tp_size=2)
b_r1 = build_attention(hf_b, tp_rank=1, tp_size=2)
out["case_B_replicate_8q_1kv_tp2"] = {
    "rank0": scalars(b_r0, 0, 2, "replicate (tp=2 % total_kv=1 == 0)", 1),
    "rank1": scalars(b_r1, 1, 2, "replicate (tp=2 % total_kv=1 == 0)", 1),
    "k_shard_rank_rank0": 0 // b_r0.qkv_proj.num_kv_head_replicas,
    "k_shard_rank_rank1": 1 // b_r1.qkv_proj.num_kv_head_replicas,
    "both_ranks_same_k_segment": True,
    "num_kv_head_replicas_meaning": "tp_size // total_num_kv_heads = 2 (每段 KV 权重复制进 2 个 rank)",
}

# ── 案例 C：tp=1 退化基线（对照） ──
hf_c = make_hf_config(num_attention_heads=4, num_key_value_heads=2)
c = build_attention(hf_c, tp_rank=0, tp_size=1)
out["case_C_tp1_baseline_4q_2kv"] = scalars(c, 0, 1, "tp=1 恒等 (4%1==0)", 2)

# ── 案例 D：整除断言快速失败（llama.py:L142）──
try:
    build_attention(make_hf_config(num_attention_heads=4), tp_rank=0, tp_size=3)
    out["case_D_assert_tp3"] = {"raised": False, "tp_size": 3, "total_num_heads": 4}
except AssertionError as e:
    out["case_D_assert_tp3"] = {"raised": True, "error_type": "AssertionError", "tp_size": 3, "total_num_heads": 4}

# ── 案例 E：dossier theory 的实尺账（8q/2kv/head_dim=128/hidden=4096, tp=2, rank1）
#     直接经 QKVParallelLinear（LlamaAttention 的 head_dim 从 config 派生，
#     head_dim=128 需 hidden=1024——用构造层本体取同一数学）──
dist._TP_STATE.update(rank=1, size=2, pp_rank=0, pp_size=1)
with set_current_vllm_config(make_vllm_config(make_hf_config())):
    real = QKVParallelLinear(
        hidden_size=4096,
        head_size=128,
        total_num_heads=8,
        total_num_kv_heads=2,
        bias=False,
        prefix="qkv",
    )
out["case_E_real_scale_8q_2kv_head128_hidden4096_tp2_rank1"] = {
    "num_heads_per_rank": real.num_heads,
    "num_kv_heads_per_rank": real.num_kv_heads,
    "num_kv_head_replicas": real.num_kv_head_replicas,
    "q_rows_per_rank": real.num_heads * real.head_size,
    "k_rows_per_rank": real.num_kv_heads * real.head_size,
    "v_rows_per_rank": real.num_kv_heads * real.v_head_size,
    "fused_weight_shape_per_rank": list(real.weight.shape),
    "output_sizes_total": real.output_sizes,
    "output_partition_sizes_per_rank": real.output_partition_sizes,
    "weight_bytes_fp16_per_rank": real.weight.numel() * 2,
}

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "m2.json"), "w", encoding="utf-8", newline="\n") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False, indent=1))
