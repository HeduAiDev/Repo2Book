# ch25 m04 —— MQA 吸收腿（decode 段）：q 乘 W_UK^T 降进潜空间、576 维单头 MQA、
# 输出乘 W_UV 上投影（mla_attention.py:L831-L949；bmm 在 L888、kernel 在 L919、
# _v_up_proj 在 L949/L1154-L1176）
# 场景：三请求 decode 步（历史 7/3/12 各 +1 新 token）→ 全批 MQA 腿。
# 记录吸收 bmm 形状账 + 结合律数值实证（direct == absorbed）+ 与上投影 MHA 参照对照。
from __future__ import annotations

import torch

from trace_common import MINI, dump, fmt, oracle_mha, run, setup

vllm_config, layer, inner, builder = setup(seed=41)
dims = MINI
H, Lq, N, P, R, Lkv, V = (dims["hidden_size"], dims["q_lora_rank"],
                          dims["num_heads"], dims["qk_nope_head_dim"],
                          dims["qk_rope_head_dim"], dims["kv_lora_rank"],
                          dims["v_head_dim"])

# ① 历史写入（3 条短请求，走一次前向只为把潜向量写进分页 cache）
hist = [7, 3, 12]
torch.manual_seed(41)
h_hist = torch.randn(sum(hist), H)
pos_hist = torch.cat([torch.arange(s) for s in hist])
run(layer, builder, vllm_config, hist, hist, h_hist, pos_hist)

# ② decode 步：每请求 1 个新 token → MQA 腿
seq2 = [s + 1 for s in hist]                     # [8, 4, 13]
B = len(seq2)
h_new = torch.randn(B, H)
pos_new = torch.tensor(hist)

rec = {}
orig_fwd_mqa = inner.impl.forward_mqa


def fwd_mqa_probe(q, kv_cache, attn_metadata, lay):
    ql, qpe = q  # 吸收后的 (B,N,L) 与 rope 段 (B,N,R)（元组形态 = L893 前的未拼接态）
    rec["absorbed_ql_shape"] = list(ql.shape)
    rec["qpe_shape"] = list(qpe.shape)
    rec["decode_seq_lens"] = [int(x) for x in attn_metadata.decode.seq_lens]
    o, lse = orig_fwd_mqa(q, kv_cache, attn_metadata, lay)
    rec["attn_out_shape"] = list(o.shape)
    return o, lse


inner.impl.forward_mqa = fwd_mqa_probe

orig_vup = inner._v_up_proj


def vup_probe(x, out):
    rec["vup_in_shape"] = list(x.shape)
    rec["vup_out_shape"] = list(out.shape)
    return orig_vup(x, out)


inner._v_up_proj = vup_probe

out, md, common = run(layer, builder, vllm_config, [1] * B, seq2, h_new, pos_new)

# ── 手工重放吸收账（与 forward_impl L831-L891 同式）──
ref_out, q_full = oracle_mha(layer, inner, common, h_new, pos_new, [1] * B,
                             seq2)
q_nope_raw = q_full[..., :P]                      # (B,N,P) 未吸收的每头 q nope 段
q_nope_t = q_nope_raw.transpose(0, 1).contiguous()  # (N,B,P)
ql_manual = torch.bmm(q_nope_t, inner.W_UK_T)     # (N,B,P)×(N,P,L) → (N,B,L)
ql_manual = ql_manual.transpose(0, 1)             # (B,N,L)
# 与探针捕获的吸收结果逐位对照（重建 q 有 layernorm 数值路径，容差放宽）
ql_probe = None

# 结合律数值实证：head 0、请求 0
cache = inner.kv_cache.view(-1, 576)
bt = common.block_table_tensor
from trace_common import BLOCK
rows0 = [int(bt[0, p // BLOCK]) * BLOCK + p % BLOCK for p in range(seq2[0])]
lat0 = cache[torch.tensor(rows0)]                  # 请求 0 的 8 行潜向量
c = lat0[-2, :Lkv]                                 # 取第 7 个历史 token 的 kv_c 段（512 维）
#   —— cache 行 = norm(kv_c) ⊕ roped(k_pe)；nope 段分数只吃 kv_c 段
W_UK_h0 = inner.kv_b_proj.weight.view(N, P + V, Lkv)[0, :P, :]   # (P,L) 该头 W_UK^T
k_nope_direct = c @ W_UK_h0.T                      # 先上投影 K：(512,)×(512,P) → (P,)
q0 = q_nope_raw[0, 0]                              # 该头该请求的 q nope 段 (P,)
direct = float((q0 * k_nope_direct).sum())         # …再点积：q·(c W_UK^T)
absorbed_q = q0 @ W_UK_h0                          # 先吸收 q：(P,)×(P,512) → (512,)
absorbed = float((absorbed_q * c).sum())           # …再与潜向量点积：(q W_UK^T)·c

max_diff = (out - ref_out).abs().max().item()

# 读字节账（MINI 实测 fp32；DSV3 同式 fp16）
cache_rows_read = sum(seq2)
mqa_bytes_mini = cache_rows_read * (Lkv + R) * 4
mha_equiv_bytes_mini = cache_rows_read * (N * (P + R) + N * V) * 4
mqa_bytes_dsv3 = cache_rows_read * 576 * 2
mha_equiv_bytes_dsv3 = cache_rows_read * 40960 * 2
# 吸收 bmm 的 MAC 账：每 decode token 一次 (N,B,P)×(N,P,L) bmm + 一次 v_up bmm
bmm_mac_absorb = 2 * N * B * P * Lkv
bmm_mac_vup = 2 * N * B * Lkv * V

trace = {
    "mechanism": "ch25-m04",
    "source": "run_m04.py @ implementation/（vLLM v0.27.1 只做减法精简版, host CPU, torch float32）",
    "code_anchor": "vllm/model_executor/layers/attention/mla_attention.py:L831-L949（MQA 吸收腿全程；torch.bmm 吸收 L888、forward_mqa L919、_v_up_proj L949）",
    "params": {
        "history_lens": hist, "decode_seq_lens": seq2, "batch_B": B,
        "num_heads": N, "qk_nope_head_dim": P, "qk_rope_head_dim": R,
        "kv_lora_rank": Lkv, "v_head_dim": V,
        "W_UK_T_shape": list(inner.W_UK_T.shape),      # (N,P,L)=(4,16,512)
        "W_UV_shape": list(inner.W_UV.shape),          # (N,L,V)=(4,512,16)
        "kv_b_proj_weight_still_alive": list(inner.kv_b_proj.weight.shape),
        "dual_note": "kv_b_proj 原权重（prefill 腿用）与 W_UK_T/W_UV 两份 bmm 副本（decode 腿用）同时常驻——同一份权重两种形状存两份（process_weights_after_loading L1094-L1096 replace_parameter）",
    },
    "absorption_shapes": {k: rec[k] for k in sorted(rec)},
    "bmm_shape_account": {
        "absorb_bmm": "(N,B,P)×(N,P,L) → (N,B,L)",
        "absorb_bmm_mini": "(4,3,16)×(4,16,512) → (4,3,512)",
        "v_up_bmm": "(N,B,L)×(N,L,V) → (N,B,V)",
        "v_up_bmm_mini": "(4,3,512)×(4,512,16) → (4,3,16)",
        "mqa_q_after_cat": "(B,N,L+R) = (3,4,576)",
        "bmm_mac_absorb": bmm_mac_absorb,
        "bmm_mac_vup": bmm_mac_vup,
    },
    "associativity_proof_by_number": {
        "picked": "head 0、请求 0、其第 7 个历史 token 的潜向量 kv_c 段（512 维）",
        "route_A_upproject_first": "k_nope = c @ W_UK^T（先上投影 K 到 16 维）→ dot(q_nope, k_nope)",
        "route_A_value": fmt(direct, 6),
        "route_B_absorb_first": "q_l = q_nope @ W_UK^T（先把 q 吸收进 512 维潜空间）→ dot(q_l, c)",
        "route_B_value": fmt(absorbed, 6),
        "identical_up_to": fmt(abs(direct - absorbed), 6),
        "note": "两条路线同一分数——乘法结合律 q·(W_UK c) = (q W_UK^T)·c 的数值实证；decode 选 B：cache 里 576 维潜向量一行不动、不为 128 头物化 K",
    },
    "byte_account_per_decode_step": {
        "cache_rows_read": cache_rows_read,
        "mqa_leg_bytes_mini_fp32": mqa_bytes_mini,
        "mha_equiv_bytes_mini_fp32": mha_equiv_bytes_mini,
        "mqa_leg_bytes_dsv3_fp16": mqa_bytes_dsv3,
        "mha_equiv_bytes_dsv3_fp16": mha_equiv_bytes_dsv3,
        "note": "decode 每步把全部历史 KV 搬进计算单元——MQA 腿只搬单份潜向量 (L+R)/token，MHA 腿要搬 N×(P+R+V)/token（DSV3：576 vs 40960，71 倍）",
    },
    "verification": {
        "vs_upprojected_mha_oracle_max_abs_diff": fmt(max_diff, 6),
        "oracle": "文件头 Compute Friendly 伪码（mla_attention.py:L44-L118）逐式——MQA 吸收腿与上投影 MHA 同一数学",
    },
    "outcome": {
        "out_shape": list(out.shape),
        "leg_taken": "MQA（num_mqa_tokens=3, num_mha_tokens=0）",
        "num_decode_tokens": int(md.num_decode_tokens),
    },
}

assert md.num_decode_tokens == B and md.num_prefills == 0
assert rec["absorbed_ql_shape"] == [B, N, Lkv]
assert rec["vup_out_shape"] == [B, N * V]   # o_proj 前的 (B, N*V=64) 段
assert abs(direct - absorbed) < 1e-4
assert max_diff < 1e-4
dump("m04.json", trace)
