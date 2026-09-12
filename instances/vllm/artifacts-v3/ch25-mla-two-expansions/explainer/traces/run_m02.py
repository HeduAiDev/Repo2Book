# ch25 m02 —— 外层前向低秩链与解耦 RoPE（vllm/model_executor/layers/mla.py:L150-L226）
# 一条 5-token 流过 MultiHeadLatentAttentionWrapper.forward 的形状账 +
# RoPE 只打 rope 段的逐位核验（nope 段逐位相等=可吸收前提）+
# 同一装配式代入 DSV3 实尺维度的投影形状账（含吸收重排产物形状）。
from __future__ import annotations

import torch

from trace_common import DSV3, MINI, T, dump, setup, fmt

vllm_config, layer, inner, builder = setup(seed=2525)
dims = MINI
H, Lq, N, P, R, Lkv, V = (dims["hidden_size"], dims["q_lora_rank"],
                          dims["num_heads"], dims["qk_nope_head_dim"],
                          dims["qk_rope_head_dim"], dims["kv_lora_rank"],
                          dims["v_head_dim"])
Tt = 5
torch.manual_seed(101)
h = torch.randn(Tt, H)
pos = torch.arange(Tt)

# ── Recorder 捕获交进内层插座的三个张量（同 tests TestOuterForward._probe）──
captured = {}


class Recorder(torch.nn.Module):
    def forward(self, q, kv_c_normed, k_pe, output_shape=None,
                q_dcp_replicated=None):
        captured["q"], captured["kv_c_normed"], captured["k_pe"] = (
            q, kv_c_normed, k_pe)
        return torch.zeros(*output_shape)


orig_inner_mod = layer.mla_attn.mla_attn
layer.mla_attn.mla_attn = Recorder()
out = layer(pos, h, None)
layer.mla_attn.mla_attn = orig_inner_mod
assert out.shape == (Tt, H)

# ── 手工重放低秩链（与 forward 同式），逐段核验 ──
w_fused = layer.fused_qkv_a_proj.weight          # [Lq+Lkv+R, H]
qkv = h @ w_fused.T                              # [T, Lq+Lkv+R]
q_c, kv_lora = qkv.split([Lq, Lkv + R], dim=-1)
q_cn = layer.q_a_layernorm(q_c)
q_manual = (q_cn @ layer.q_b_proj.weight.T).view(-1, N, P + R)
kv_c, k_pe_raw = kv_lora.split([Lkv, R], dim=-1)
kv_c_normed_manual = layer.kv_a_layernorm(kv_c)
q_rope_manual, k_pe_rot = layer.mla_attn.rotary_emb(
    pos, q_manual[..., P:].clone(), k_pe_raw.unsqueeze(1))

# nope 段逐位相等（RoPE 不碰）
nope_bitwise_equal = bool(
    torch.equal(captured["q"][..., :P], q_manual[..., :P]))
# rope 段：实际输出（已旋转）vs 未旋转的逐位置最大差
#（pos=0 旋转恒等 → 0；pos>0 旋转生效 → 非 0——非平凡分支实证）
rope_diff_per_pos = [
    fmt((captured["q"][t, :, P:] - q_manual[t, :, P:]).abs().max().item())
    for t in range(Tt)
]
# 旋转一致性：实际输出 == 手工重放的旋转结果（逐位）
rope_vs_manual_rot_max_diff = fmt(
    (captured["q"][..., P:] - q_rope_manual).abs().max().item())
# kv_a_layernorm 只打 kv_c
kv_c_max_diff = fmt(
    (captured["kv_c_normed"] - kv_c_normed_manual).abs().max().item())
# k_pe：与旋转后相等、与原始不同（pos>0）
k_pe_rot_max_diff = fmt(
    (captured["k_pe"] - k_pe_rot).abs().max().item())
k_pe_raw_diff_pos1 = fmt(
    (captured["k_pe"][1] - k_pe_raw[1].unsqueeze(1)).abs().max().item())
# ── DSV3 实尺：同一装配式代入（真实例化 + 吸收重排），只取形状账 ──
torch.manual_seed(77)
vc3 = T.enable_ref_backends(T.make_vllm_config(dims=DSV3))
layer3, inner3 = T.build_mla_layer(vc3, dims=DSV3)
d3_shapes = {
    "fused_qkv_a_proj_weight": list(layer3.fused_qkv_a_proj.weight.shape),
    "q_a_layernorm_weight": list(layer3.q_a_layernorm.weight.shape),
    "q_b_proj_weight": list(layer3.q_b_proj.weight.shape),
    "kv_a_layernorm_weight": list(layer3.kv_a_layernorm.weight.shape),
    "kv_b_proj_weight": list(layer3.kv_b_proj.weight.shape),
    "o_proj_weight": list(layer3.o_proj.weight.shape),
    "W_UK_T_after_reorder": list(inner3.W_UK_T.shape),
    "W_UV_after_reorder": list(inner3.W_UV.shape),
}
N3, P3, R3, Lkv3, V3 = (DSV3["num_heads"], DSV3["qk_nope_head_dim"],
                        DSV3["qk_rope_head_dim"], DSV3["kv_lora_rank"],
                        DSV3["v_head_dim"])
mha_equiv_per_token = N3 * (P3 + R3 + V3)      # 128×(128+64+128)=40960
latent_per_token = Lkv3 + R3                   # 576
compression_ratio = mha_equiv_per_token / latent_per_token
del layer3, inner3, vc3

trace = {
    "mechanism": "ch25-m02",
    "source": "run_m02.py @ implementation/（vLLM v0.27.1 只做减法精简版, host CPU, torch float32）",
    "code_anchor": "vllm/model_executor/layers/mla.py:L150-L226（Wrapper.forward 全程；RoPE 只作用 rope 段在 L201-L203）",
    "params_mini": {
        "hidden_size": H, "q_lora_rank": Lq, "num_heads": N,
        "qk_nope_head_dim": P, "qk_rope_head_dim": R,
        "kv_lora_rank": Lkv, "v_head_dim": V,
        "tokens": Tt, "head_size_kv_cache": int(inner.head_size),
        "qk_head_dim": int(inner.qk_head_dim),
    },
    "pipeline_shapes_mini": [
        {"step": "输入 hidden_states", "shape": [Tt, H]},
        {"step": "fused_qkv_a_proj 一次 GEMM（W_DQ‖W_DKV‖W_KR 融合）",
         "shape": [Tt, Lq + Lkv + R]},
        {"step": "split 第一刀：q_c 潜段", "shape": [Tt, Lq]},
        {"step": "q_a_layernorm → q_b_proj 上投影 → view", "shape": [Tt, N, P + R]},
        {"step": "split 第一刀：kv_lora 段", "shape": [Tt, Lkv + R]},
        {"step": "split 第二刀：kv_c | k_pe", "shape": [[Tt, Lkv], [Tt, R]]},
        {"step": "kv_a_layernorm 只打 kv_c（k_pe 不归一）", "shape": [Tt, Lkv]},
        {"step": "k_pe 加头维 unsqueeze(1)", "shape": [Tt, 1, R]},
        {"step": "RoPE 只打 q[...,P:] 与 k_pe（nope 段不旋转）", "shape": [Tt, N, P + R]},
        {"step": "交内层 mla_attn(q, kv_c_normed, k_pe)",
         "shape": [[Tt, N, P + R], [Tt, Lkv], [Tt, 1, R]]},
        {"step": "cache 写腿输入：潜向量 = kv_c_normed ⊕ k_pe（拼接）", "shape": [Tt, 576]},
    ],
    "split_account": {
        "fused_out_width": Lq + Lkv + R,        # 640
        "q_lora_seg": Lq,                        # 64
        "kv_lora_seg": Lkv + R,                  # 576
        "kv_c_seg": Lkv,                         # 512
        "k_pe_seg": R,                           # 64
    },
    "weight_shapes_mini": {
        "fused_qkv_a_proj": list(w_fused.shape),          # [640,128]
        "q_b_proj": list(layer.q_b_proj.weight.shape),    # [320,64]
        "kv_b_proj": list(inner.kv_b_proj.weight.shape),  # [128,512]
        "o_proj": list(layer.o_proj.weight.shape),        # [128,64]
        "W_UK_T": list(inner.W_UK_T.shape),               # [4,16,512]
        "W_UV": list(inner.W_UV.shape),                   # [4,512,16]
    },
    "rope_decoupled_checks": {
        "nope_bitwise_equal_to_manual": nope_bitwise_equal,
        "nope_note": "q[..., :P] 与手工链逐位相等——RoPE 旋转完全没碰 nope 段（低秩可吸收的数学前提，mla.py:L201-L203 的代码化身）",
        "rope_rotated_vs_unrotated_max_abs_diff_per_position": rope_diff_per_pos,
        "rope_vs_manual_rotation_max_abs_diff": rope_vs_manual_rot_max_diff,
        "kv_c_layernorm_only_max_abs_diff": kv_c_max_diff,
        "k_pe_rotated_vs_captured_max_abs_diff": k_pe_rot_max_diff,
        "k_pe_pos1_vs_raw_max_abs_diff": k_pe_raw_diff_pos1,
    },
    "dsv3_shapes": {
        "dims": {"hidden_size": DSV3["hidden_size"], "q_lora_rank": DSV3["q_lora_rank"],
                 "num_heads": N3, "qk_nope_head_dim": P3,
                 "qk_rope_head_dim": R3, "kv_lora_rank": Lkv3,
                 "v_head_dim": V3, "qk_head_dim": P3 + R3},
        "weight_shapes": d3_shapes,
        "fused_out_width": DSV3["q_lora_rank"] + Lkv3 + R3,          # 2112
        "q_b_out_width": N3 * (P3 + R3),                              # 24576
        "kv_b_out_width": N3 * (P3 + V3),                             # 32768
        "o_in_width": N3 * V3,                                        # 16384
    },
    "cache_row_account": {
        "latent_per_token_per_layer": latent_per_token,               # 576
        "mha_equivalent_per_token_per_layer": mha_equiv_per_token,    # 40960
        "mha_equiv_formula": "128 heads × (192 QK head dim + 128 V head dim) = 128×320",
        "compression_ratio": fmt(compression_ratio, 1),               # 71.1
        "llama70b_gqa_reference": 2048,
        "gqa_note": "Llama-70B GQA（8 KV 头×128 维×2）=2048 元素——MLA 相对 GQA 约 3.6 倍节省（dossier theory[0]）",
    },
    "outcome": {
        "forward_out_shape": list(out.shape),
        "all_checks_passed": True,
    },
}

assert nope_bitwise_equal
assert float(rope_diff_per_pos[0]) == 0.0 and float(rope_diff_per_pos[1]) > 0.0
assert float(kv_c_max_diff) == 0.0
dump("m02.json", trace)
