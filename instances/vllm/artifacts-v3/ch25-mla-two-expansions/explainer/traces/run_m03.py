# ch25 m03 —— MHA 展开腿（prefill 段）：kv_b_proj 上投影成完整 K/V 做标准注意力
# （vllm/model_executor/layers/attention/mla_attention.py:L2581-L2666 forward_mha）
# 场景：一条 fresh prefill（首块、无历史上下文）M=140 > 阈值 128 → 全批走 MHA 腿。
# 记录上投影张量形状账 + 与「上投影 MHA 参照」（文件头 Compute Friendly 伪码
# L44-L118 逐式）数值对照。
from __future__ import annotations

import torch

from trace_common import MINI, dump, fmt, oracle_mha, run, setup

vllm_config, layer, inner, builder = setup(seed=31)
dims = MINI
H, Lq, N, P, R, Lkv, V = (dims["hidden_size"], dims["q_lora_rank"],
                          dims["num_heads"], dims["qk_nope_head_dim"],
                          dims["qk_rope_head_dim"], dims["kv_lora_rank"],
                          dims["v_head_dim"])

M = 140  # 新 token 数：> FlashMLA 阈值 128 → MHA 腿（≤128 会划进 MQA 段）
torch.manual_seed(31)
h_new = torch.randn(M, H)
pos_new = torch.arange(M)

# ── 在 MHA 腿入口（impl.forward_mha）与 prefill 后端处插探针 ──
rec = {}


def fwd_mha_probe(q, kv_c_normed, k_pe, kv_cache, attn_metadata, k_scale,
                  output, output_scale=None):
    rec["entry_q_shape"] = list(q.shape)
    rec["entry_kv_c_normed_shape"] = list(kv_c_normed.shape)
    rec["entry_k_pe_shape"] = list(k_pe.shape)
    # 与 forward_mha 同式重放上投影（kv_b_proj → view → split → concat）
    kv_nope = inner.kv_b_proj(kv_c_normed)[0].view(
        -1, inner.num_heads, inner.qk_nope_head_dim + inner.v_head_dim)
    k_nope, v = kv_nope.split([inner.qk_nope_head_dim, inner.v_head_dim],
                              dim=-1)
    k = torch.cat([k_nope, k_pe.expand(-1, N, -1)], dim=-1)
    rec["up_projected_kv_nope_shape"] = list(kv_nope.shape)
    rec["k_nope_shape"] = list(k_nope.shape)
    rec["v_shape"] = list(v.shape)
    rec["k_concat_shape"] = list(k.shape)
    return orig_fwd_mha(q, kv_c_normed, k_pe, kv_cache, attn_metadata,
                        k_scale, output, output_scale)


orig_fwd_mha = inner.impl.forward_mha
inner.impl.forward_mha = fwd_mha_probe

pb = builder._prefill_backend  # builder 装配的 prefill 后端 clone（真实注入位）
orig_run = pb.run_prefill_new_tokens


def run_probe(q, k, v, return_softmax_lse, out=None, output_scale=None):
    rec["backend_q_shape"] = list(q.shape)
    rec["backend_k_shape"] = list(k.shape)
    rec["backend_v_shape"] = list(v.shape)
    return orig_run(q, k, v, return_softmax_lse, out, output_scale)


pb.run_prefill_new_tokens = run_probe

out, md, common = run(layer, builder, vllm_config, [M], [M], h_new, pos_new)

# ── 参照：上投影 MHA（文件头伪码）──
ref_out, _ = oracle_mha(layer, inner, common, h_new, pos_new, [M], [M])
max_diff = (out - ref_out).abs().max().item()

# 上投影 vs 潜向量的元素账（MINI 实测 + DSV3 同式）
up_elems_per_token_mini = N * (P + R) + N * V          # K 全展开 + V
latent_per_token = Lkv + R
up_total_mini = M * up_elems_per_token_mini
latent_total = M * latent_per_token
# DSV3 同式（维度取 traces/m02.json dsv3_shapes：N=128/P=128/R=64/V=128）
up_per_token_dsv3 = 128 * (128 + 64) + 128 * 128       # 24576 + 16384 = 40960

trace = {
    "mechanism": "ch25-m03",
    "source": "run_m03.py @ implementation/（vLLM v0.27.1 只做减法精简版, host CPU, torch float32）",
    "code_anchor": "vllm/model_executor/layers/attention/mla_attention.py:L2581-L2666（forward_mha；上投影在 L2608-L2612）",
    "params": {
        "M_new_tokens": M,
        "flashmla_reorder_threshold": 128,
        "note_threshold": "M=140 > 128 → MHA 腿；≤128 的 chunk 会按 'process small prefills with decode pathway'（flashmla.py:L121）划进 MQA 段",
        "num_heads": N, "qk_nope_head_dim": P, "qk_rope_head_dim": R,
        "kv_lora_rank": Lkv, "v_head_dim": V, "hidden_size": H,
    },
    "metadata_counts": {
        "num_decode_tokens": int(md.num_decode_tokens),   # 0
        "num_prefills": int(md.num_prefills),             # 1
        "chunked_context_is_none": md.prefill.chunked_context is None,
    },
    "shapes": {k: rec[k] for k in sorted(rec)},
    "up_projection_account": {
        "kv_b_proj_weight_shape_mini": list(inner.kv_b_proj.weight.shape),
        "kv_b_proj_out_width_mini": N * (P + V),           # 128? → 4×32
        "up_projected_elements_per_token_mini": up_elems_per_token_mini,
        "latent_elements_per_token": latent_per_token,
        "up_projected_total_mini": up_total_mini,
        "latent_total": latent_total,
        "up_per_token_dsv3": up_per_token_dsv3,
        "dsv3_note": "DSV3 每上投影 1 个 token 的 K/V = 128×(128+64) + 128×128 = 40960 元素 vs 潜向量 576（71 倍）——prefill 算力富余时愿意付这笔算力换标准 MHA kernel",
        "kv_b_proj_weight_shape_dsv3": [32768, 512],
    },
    "verification": {
        "vs_upprojected_mha_oracle_max_abs_diff": fmt(max_diff, 6),
        "oracle": "文件头 Compute Friendly 伪码（mla_attention.py:L44-L118）逐式：潜向量 @ W_UK/W_UV 上投影 → 标准 causal MHA",
    },
    "outcome": {
        "out_shape": list(out.shape),
        "leg_taken": "MHA（num_mha_tokens=140, num_mqa_tokens=0）",
    },
}

assert md.num_decode_tokens == 0 and md.num_prefills == 1
assert md.prefill.chunked_context is None
assert rec["k_concat_shape"] == [M, N, P + R]
assert max_diff < 1e-4
dump("m03.json", trace)
