"""ch31 explainer driver —— 跑论文忠实的参考实现(implementation/),把每个机制的教学数值
轨迹 dump 成 traces/<id>.json。每个 JSON 里的数字就是 explainer.json 逐轮表 / figure-spec
numbers 的唯一出处(lint_explainer 逐个核对)。

运行:python3 run_all.py   (host python3,纯 NumPy 控制流,无需 NPU/容器)
"""
import json
import sys
from pathlib import Path

# 让参考实现的模块名(mha_baseline / low_rank_mla / ...)可 import
IMPL = Path(__file__).resolve().parents[1].parent / "implementation"
sys.path.insert(0, str(IMPL))

import numpy as np

from low_rank_mla import (
    init_kv_compression_weights,
    init_q_compression_weights,
    kv_joint_compression,
    q_joint_compression,
    precompute_absorbed_query_weights,
    precompute_uv_head_slices,
    _head_slice,
    score_materialized_nope,
    score_absorbed_nope,
    attention_in_latent_space,
    latent_to_value,
    split_kv_heads,
)
from decoupled_rope import effective_middle_matrix
from mla_reference import MLAConfig, MLAReference, DecodeCache
from kv_cache_table import compare_kv_cache, deepseek_v2_numbers, toy_numbers
from numerics import softmax

OUT = Path(__file__).resolve().parent


def r(x, nd=4):
    """统一四舍五入到 nd 位小数——保证 trace 里的数字与表格字符串逐位一致。"""
    return round(float(x), nd)


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {name}")


# 统一玩具维度(小到可心算跟随)
D, N_H, D_H, D_C, D_C_Q, D_H_R = 6, 2, 4, 4, 4, 2
SEED = 0

# 统一输入序列 h_seq: T=3 个 token
rng = np.random.default_rng(42)
H_SEQ = rng.normal(scale=1.0, size=(3, D))
POSITIONS = [0, 1, 2]


# ============================================================ 1. 低秩 KV 联合压缩
def trace_compression():
    kv_w = init_kv_compression_weights(D, N_H, D_H, D_C, seed=SEED)
    c_kv_seq, k_c_seq, v_c_seq = kv_joint_compression(H_SEQ, kv_w)

    mha_per_tok = 2 * N_H * D_H          # 标准 MHA 每 token 每层缓存元素
    mla_per_tok = D_C + D_H_R            # MLA 只缓存 c_kv(d_c) + 解耦 key(d_h_r)
    ckv_only = D_C                       # 联合压缩本体:只缓存潜向量 d_c 维

    rows = []
    for t in range(3):
        rows.append({
            "token_t": t,
            "c_kv_first": r(c_kv_seq[t][0]),
            "c_kv_dim": D_C,
            "k_c_dim_materialized": N_H * D_H,        # 若物化 key 的维度
            "cache_mla_per_token": mla_per_tok,
            "cache_mha_per_token": mha_per_tok,
        })
    return {
        "config": {"d": D, "n_h": N_H, "d_h": D_H, "d_c": D_C, "d_h_r": D_H_R},
        "mha_per_token": mha_per_tok,
        "mla_per_token": mla_per_tok,
        "ckv_only_dim": ckv_only,
        "compression_ratio_ckv_vs_mha": r(mha_per_tok / mla_per_tok),
        "rows": rows,
    }


# ============================================================ 2. 权重吸收恒等式
def trace_absorption():
    kv_w = init_kv_compression_weights(D, N_H, D_H, D_C, seed=SEED)
    q_w = init_q_compression_weights(D, N_H, D_H, D_C_Q, seed=SEED + 1)
    c_kv_seq, k_c_seq, v_c_seq = kv_joint_compression(H_SEQ, kv_w)
    c_q_seq, q_c_seq = q_joint_compression(H_SEQ, q_w)
    k_c_heads, v_c_heads, q_c_heads = split_kv_heads(k_c_seq, v_c_seq, q_c_seq, N_H, D_H)
    w_tildes = precompute_absorbed_query_weights(q_w, kv_w, N_H, D_H)
    uv_heads = precompute_uv_head_slices(kv_w, N_H, D_H)

    # q 侧:物化打分 vs 吸收打分,逐 (t,j,head) 对
    head = 0
    pairs = [(1, 0), (1, 1), (2, 0), (2, 2)]
    rows = []
    max_q_diff = 0.0
    for (t, j) in pairs:
        mat = score_materialized_nope(q_c_heads[head][t], k_c_heads[head][j])
        ab = score_absorbed_nope(c_q_seq[t], c_kv_seq[j], w_tildes[head])
        diff = abs(mat - ab)
        max_q_diff = max(max_q_diff, diff)
        rows.append({
            "query_t": t, "key_j": j,
            "score_materialized": r(mat),
            "score_absorbed": r(ab),
            "abs_diff": r(diff, 10),
        })

    # o 侧:物化 v^C 加权求和 vs (潜空间加权 → 乘 W_UV) —— 取 t=2 head=0,权重用一段因果 softmax
    t = 2
    scores_row = softmax(np.array([0.3, 0.5, 0.2]))   # 已归一化的注意力权重(示意)
    materialized_o = scores_row @ v_c_heads[head]                       # 对物化 v^C 加权
    latent = attention_in_latent_space(scores_row, c_kv_seq)           # 潜空间加权
    absorbed_o = latent_to_value(latent, uv_heads[head])               # 乘 W_UV 还原
    o_diff = float(np.max(np.abs(materialized_o - absorbed_o)))

    return {
        "config": {"d_c": D_C, "d_c_q": D_C_Q, "d_h": D_H, "w_tilde_shape": [D_C, D_C_Q]},
        "rows": rows,
        "max_q_side_abs_diff": r(max_q_diff, 10),
        "o_side_first_materialized": r(materialized_o[0]),
        "o_side_first_absorbed": r(absorbed_o[0]),
        "o_side_max_abs_diff": r(o_diff, 10),
    }


# ============================================================ 3. 解耦 RoPE(为何不可吸收)
def trace_rope():
    kv_w = init_kv_compression_weights(D, N_H, D_H, D_C, seed=SEED)
    q_w = init_q_compression_weights(D, N_H, D_H, D_C_Q, seed=SEED + 1)
    w_tildes = precompute_absorbed_query_weights(q_w, kv_w, N_H, D_H)

    head = 0
    w_uq_i = _head_slice(q_w.W_UQ, head, D_H)    # (d_h, d_c_q)  充当 "W^Q" 角色(c_q -> q_nope)
    w_uk_i = _head_slice(kv_w.W_UK, head, D_H)   # (d_h, d_c)

    # 反证路线:若对 k^C 加 RoPE,打分中间夹 M(δ)=(W^Q)^T R_δ W^{UK},随 δ 变
    rows = []
    m00_by_delta = {}
    for delta in [0, 1, 2, 3]:
        M = effective_middle_matrix(w_uq_i, w_uk_i, float(delta))
        m00 = float(M[0, 0])
        fro = float(np.linalg.norm(M))
        m00_by_delta[delta] = r(m00)
        rows.append({
            "delta": delta,
            "R_delta_is_identity": (delta == 0),
            "M_delta_00": r(m00),
            "M_delta_fro": r(fro),
            "offline_precomputable": (delta == 0),   # 只有 δ=0(无 RoPE)才退化成静态矩阵
        })

    # 对照:解耦后主体走的静态矩阵 W~(与位置无关)—— 就是 δ=0 那一档,永远同一个值
    w_tilde_00 = float(w_tildes[head][0, 0])

    # 端到端:解耦 RoPE 下 decode 增量 == prefill 一次性(证明解耦保持正确)
    cfg = MLAConfig(d=D, n_h=N_H, d_h=D_H, d_c=D_C, d_c_q=D_C_Q, d_h_r=D_H_R)
    mla = MLAReference(cfg, seed=SEED)
    full = mla.forward_full(H_SEQ, POSITIONS)
    cache = DecodeCache()
    dec_rows = []
    max_e2e_diff = 0.0
    for t in range(3):
        u_t, cache = mla.decode_step(H_SEQ[t], POSITIONS[t], cache)
        d_ = float(np.max(np.abs(u_t - full[t])))
        max_e2e_diff = max(max_e2e_diff, d_)
        dec_rows.append({"step_t": t, "decode_vs_prefill_max_abs_diff": r(d_, 10)})

    return {
        "config": {"d_h": D_H, "d_h_r": D_H_R, "d_c": D_C},
        "middle_matrix_rows": rows,
        "m00_by_delta": m00_by_delta,
        "w_tilde_00_static": r(w_tilde_00),
        "delta1_minus_delta0_m00": r(m00_by_delta[1] - m00_by_delta[0]),
        "e2e_rows": dec_rows,
        "e2e_max_abs_diff": r(max_e2e_diff, 10),
    }


# ============================================================ 4. q 侧低秩
def trace_qside():
    q_w = init_q_compression_weights(D, N_H, D_H, D_C_Q, seed=SEED + 1)
    c_q_seq, q_c_seq = q_joint_compression(H_SEQ, q_w)
    rows = []
    for t in range(3):
        rows.append({
            "token_t": t,
            "c_q_first": r(c_q_seq[t][0]),
            "c_q_dim": D_C_Q,                 # 训练期激活:压到 d_c_q 维
            "q_c_dim_full": N_H * D_H,         # 上投影回满维 q
            "kv_cache_delta": 0,              # q 侧压缩不改变 KV cache
        })
    return {
        "config": {"d": D, "d_c_q": D_C_Q, "n_h_x_d_h": N_H * D_H},
        "activation_full_dim": N_H * D_H,
        "activation_compressed_dim": D_C_Q,
        "kv_cache_delta": 0,
        "rows": rows,
    }


# ============================================================ 5. KV cache 对比
def trace_cache_compare():
    # 每 token 每层(l=1),GQA 取 DeepSeek 67B 的 8 组做对照
    ds = compare_kv_cache(n_h=128, d_h=128, l=1, d_c=512, d_h_r=64, n_g=8)
    ds_full = deepseek_v2_numbers()          # l=60 全模型(n_g=1)
    toy = toy_numbers()
    rows = [
        {"mechanism": "MHA", "elems_per_token_layer": ds.mha},
        {"mechanism": "GQA-8", "elems_per_token_layer": ds.gqa},
        {"mechanism": "MQA", "elems_per_token_layer": ds.mqa},
        {"mechanism": "MLA", "elems_per_token_layer": ds.mla},
    ]
    return {
        "deepseek_v2_hparams": {"n_h": 128, "d_h": 128, "l": 60, "d_c": 512, "d_h_r": 64},
        "per_token_per_layer": rows,
        "mha_per_token_layer": ds.mha,
        "mla_per_token_layer": ds.mla,
        "gqa8_per_token_layer": ds.gqa,
        "mqa_per_token_layer": ds.mqa,
        "compression_ratio_mha_over_mla": r(ds.mla_compression_ratio_vs_mha, 2),
        "mla_equivalent_gqa_groups": r(ds.mla_equivalent_gqa_groups, 2),
        "full_model_mha": ds_full.mha,
        "full_model_mla": ds_full.mla,
        "toy_mha": toy.mha,
        "toy_mla": toy.mla,
        "toy_ratio": r(toy.mla_compression_ratio_vs_mha, 2),
    }


if __name__ == "__main__":
    dump("compression.json", trace_compression())
    dump("absorption.json", trace_absorption())
    dump("rope.json", trace_rope())
    dump("qside.json", trace_qside())
    dump("cache_compare.json", trace_cache_compare())
    print("all traces written to", OUT)
