"""ch32 explainer 驱动脚本 —— 跑 implementation/ 里论文忠实的参考实现,取真实数值轨迹。

每个 needs_worked_example 机制产出一份 traces/<id>.json:里面既含原始计算结果,也含
explainer.json 逐轮表要引用的 rows(逐字回填,保证每个数字都能在 trace 里找到)。

host 直跑:python3 run_all.py   (纯 NumPy 控制流,无需目标仓运行时)
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IMPL = os.path.abspath(os.path.join(HERE, "..", "..", "implementation"))
sys.path.insert(0, IMPL)

import standard_attention as sa
import nsa_framework as nf
import nsa_selection as ns
import lightning_indexer as li
import dsa_topk_selection as dts
import training_coadapt as tc
import cost_model as cm


def dump(name, obj):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"wrote {name}")


# ---------------------------------------------------------------------------
# 1. attn-quadratic-tax —— O(L^2) 注意力税:q.k 点积总数随 L 二次增长
# ---------------------------------------------------------------------------
def trace_quadratic_tax():
    Ls = [4, 8, 16, 32]
    base = sa.quadratic_dot_product_count(Ls[0])  # 10
    rows = []
    for L in Ls:
        cnt = sa.quadratic_dot_product_count(L)          # L(L+1)/2
        mult = round(cnt / base, 1)                       # 相对 L=4 的倍数
        rows.append([L, cnt, f"{mult}x"])
    # 每翻倍的增长倍数(近 4x → 二次)
    grow = [round(rows[i][1] / rows[i - 1][1], 2) for i in range(1, len(rows))]
    # 落地规模代入
    L_deploy = 131072
    deploy_cnt = sa.quadratic_dot_product_count(L_deploy)  # 8,590,000,128
    # 一个具体 L=4 因果注意力小样(展示"打分=softmax 加权")
    rng = np.random.default_rng(0)
    d_k = 4
    q_t = rng.normal(size=d_k)
    k_seq = rng.normal(size=(4, d_k))
    alpha = sa.causal_attention_scores(q_t, k_seq, d_k)
    dump("quadratic_tax.json", {
        "params": {"Ls": Ls, "d_k": d_k, "base_count_L4": base},
        "table_rows": rows,
        "growth_per_doubling": grow,
        "deploy": {"L": L_deploy, "dot_products": deploy_cnt},
        "concrete_L4_attention_weights": [round(float(a), 4) for a in alpha],
        "attention_weights_sum": round(float(alpha.sum()), 4),
    })


# ---------------------------------------------------------------------------
# 2. nsa-three-branch —— 三支路重映射 + N_t<<t 稀疏比 + 门控加权求和
# ---------------------------------------------------------------------------
def trace_three_branch():
    # 两种上下文长度,固定/近固定的每支路预算 → 稀疏比随 t 增大而下降
    cases = []
    for t, cmp_sz, slc_sz, win_sz in [(64, 4, 8, 4), (1024, 8, 16, 8)]:
        branch_k = {
            "cmp": np.zeros((cmp_sz, 2)),
            "slc": np.zeros((slc_sz, 2)),
            "win": np.zeros((win_sz, 2)),
        }
        n_t = nf.total_remapped_size(branch_k)
        ratio = round(nf.sparsity_ratio(n_t, t), 4)
        cases.append([t, cmp_sz, slc_sz, win_sz, n_t, ratio])
    # 门控加权求和小样:t=6, d_k=4, 三支路各留 2 条 KV
    rng = np.random.default_rng(1)
    d_k = 4
    q_t = rng.normal(size=d_k)
    branch_k = {c: rng.normal(size=(2, d_k)) for c in ("cmp", "slc", "win")}
    branch_v = {c: rng.normal(size=(2, d_k)) for c in ("cmp", "slc", "win")}
    gates = {"cmp": 0.2, "slc": 0.5, "win": 0.3}
    out = nf.gated_multi_branch_output(q_t, branch_k, branch_v, gates, d_k)
    dump("three_branch.json", {
        "table_rows": cases,
        "gate_sum": round(sum(gates.values()), 2),
        "gated_output": [round(float(x), 4) for x in out],
    })


# ---------------------------------------------------------------------------
# 3. nsa-importance-score —— 复用压缩注意力分数 → GQA 组内求和 → top-n 选块
# ---------------------------------------------------------------------------
def trace_importance_score():
    # 6 个压缩块,2 个 GQA 组内头。l'=l=d → p_slc=p_cmp(直接复用,零额外计算)
    rng = np.random.default_rng(2)
    q_t = rng.normal(size=3)
    # 每头对 6 个压缩块的压缩 key
    k_cmp_h0 = rng.normal(size=(6, 3))
    k_cmp_h1 = rng.normal(size=(6, 3))
    p_h0 = ns.compression_attention_scores(q_t, k_cmp_h0)
    p_h1 = ns.compression_attention_scores(q_t, k_cmp_h1)
    p_per_head = np.stack([p_h0, p_h1])          # (2, 6)
    p_group = ns.gqa_group_importance(p_per_head)  # (6,)
    n = 3
    sel = ns.topn_block_selection(p_group, n)     # top-3 块下标
    sel_set = set(sel.tolist())
    # 逐块表:块idx | 头0分 | 头1分 | 组内和 | 是否入 top-3
    ranks = np.argsort(-p_group, kind="stable")
    rank_of = {int(b): int(r + 1) for r, b in enumerate(ranks)}
    rows = []
    for b in range(6):
        rows.append([
            b,
            round(float(p_h0[b]), 4),
            round(float(p_h1[b]), 4),
            round(float(p_group[b]), 4),
            rank_of[b],
            1 if b in sel_set else 0,
        ])
    kept_mass = round(float(p_group[list(sel_set)].sum()), 4)
    total_mass = round(float(p_group.sum()), 4)
    dump("importance_score.json", {
        "n_blocks": 6, "n_heads": 2, "top_n": n,
        "selected_blocks": sorted(sel_set),
        "table_rows": rows,
        "kept_group_mass": kept_mass,
        "total_group_mass": total_mass,
        "kept_fraction_blocks": round(n / 6, 4),
    })


# ---------------------------------------------------------------------------
# 4. dsa-lightning-indexer —— I_{t,s}=Sum_j w_j ReLU(q_j.k_s),ReLU 清零负点积
# ---------------------------------------------------------------------------
def trace_lightning_indexer():
    # H^I=2 头, d^I=3, 前驱 t=4 个 token。刻意造一对负点积让 ReLU 清零
    H_I, d_I, t = 2, 3, 4
    q_t = np.array([[1.0, 0.0, -1.0],
                    [0.5, 1.0, 0.0]])            # (H^I, d^I)
    k_seq = np.array([[1.0, 0.0, 0.0],           # s=0
                      [0.0, 1.0, 0.0],           # s=1
                      [1.0, 1.0, 1.0],           # s=2
                      [-1.0, 0.0, 1.0]])         # s=3  → head0 点积 = -2 (负)
    w_t = np.array([1.0, 2.0])                   # (H^I,) 非负权重
    rows = []
    for s in range(t):
        dot0 = float(q_t[0] @ k_seq[s])
        dot1 = float(q_t[1] @ k_seq[s])
        r0 = max(dot0, 0.0)
        r1 = max(dot1, 0.0)
        score = li.indexer_score(q_t, k_seq[s], w_t)
        rows.append([s, round(dot0, 2), round(dot1, 2),
                     round(r0, 2), round(r1, 2), round(score, 2)])
    batch = li.indexer_scores_for_query(q_t, k_seq, w_t)
    # 每对打分的 MAC 成本 H^I*d^I,对照落地 64*128
    mac_toy = H_I * d_I
    mac_deploy = 64 * 128
    dump("lightning_indexer.json", {
        "params": {"H_I": H_I, "d_I": d_I, "t": t,
                   "w_t": w_t.tolist()},
        "table_rows": rows,
        "batch_scores": [round(float(x), 2) for x in batch],
        "mac_per_pair_toy": mac_toy,
        "mac_per_pair_deploy": mac_deploy,
    })


# ---------------------------------------------------------------------------
# 5. dsa-topk-selection —— top-k 选择 + 稀疏注意力;k=L 退化为稠密(不变量)
# ---------------------------------------------------------------------------
def trace_topk_selection():
    rng = np.random.default_rng(3)
    L, d_k = 8, 4
    q_t = rng.normal(size=d_k)
    k_seq = rng.normal(size=(L, d_k))
    v_seq = rng.normal(size=(L, d_k))
    index_scores = rng.normal(size=L)
    # dense 参照:全部 L 个前驱
    dense_out = sa.causal_attention_output(q_t, k_seq, v_seq, d_k)
    # round1: k=L → 稀疏路径应精确等于稠密
    out_full, idx_full = dts.indexer_then_sparse_attention(
        q_t, index_scores, k_seq, v_seq, L, d_k)
    max_abs_diff = float(np.max(np.abs(out_full - dense_out)))
    # round2: k=3 → 真稀疏
    out_k3, idx_k3 = dts.indexer_then_sparse_attention(
        q_t, index_scores, k_seq, v_seq, 3, d_k)
    rows = [
        [8, len(idx_full), 8, round(max_abs_diff, 6), "稠密(退化)"],
        [3, len(idx_k3), 8, round(float(8 / 3), 2), "稀疏"],
    ]
    # 落地规模的主注意力点积降幅
    L_deploy, k_deploy = 131072, 512
    reduction = L_deploy // k_deploy   # 256
    dump("topk_selection.json", {
        "params": {"L": L, "d_k": d_k},
        "index_scores": [round(float(x), 4) for x in index_scores],
        "topk3_indices": sorted(idx_k3.tolist()),
        "full_vs_dense_max_abs_diff": round(max_abs_diff, 6),
        "table_rows": rows,
        "deploy": {"L": L_deploy, "k": k_deploy, "dot_reduction": reduction},
    })


# ---------------------------------------------------------------------------
# 6. dsa-training-coadapt —— 对齐程度旋钮:KL 上升 ⟺ top-k 质量召回下降
# ---------------------------------------------------------------------------
def trace_training_coadapt():
    # 与 tests/test_training_coadapt.py 同构:peaky 的真实主注意力分布 p(Dirichlet over L=64),
    # top-k 取 k=8。对齐旋钮 alpha 从 0(完全对齐真实分布)扫到 1(完全随机),每点多次
    # 取平均,消掉随机噪声,得到干净的单调关系。
    L, k = 64, 8
    rng = np.random.default_rng(10)
    p = rng.dirichlet(np.ones(L))
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []
    n_trials = 400
    for a in alphas:
        kls, recalls = [], []
        for tr in range(n_trials):
            trial_rng = np.random.default_rng(1000 * tr + int(a * 100))
            logits = tc.simulate_indexer_logits(p, a, trial_rng)
            kls.append(tc.dense_warmup_kl(p, logits))
            recalls.append(tc.topk_mass_recall(p, logits, k))
        rows.append([a, round(float(np.mean(kls)), 3), round(float(np.mean(recalls)), 3)])
    # 完美对齐基例:KL(p||p)=0
    kl_self = tc.dense_warmup_kl(p, np.log(p + 1e-12))
    dump("training_coadapt.json", {
        "params": {"L": L, "k": k, "n_trials": n_trials, "alphas": alphas},
        "table_rows": rows,
        "kl_perfect_alignment": round(float(kl_self), 6),
        "note": "alpha=0 完全对齐真实分布, alpha=1 完全随机",
    })


# ---------------------------------------------------------------------------
# 7. dsa-cost-model —— O(L.d_idx + k.d) 加速账:主注意力 L/k 倍 + 端到端(含 indexer)
# ---------------------------------------------------------------------------
def trace_cost_model():
    rows = []
    detail = {}
    for k in (512, 2048):
        acc = cm.vllm_ascend_deployment_numbers(k=k)
        rows.append([
            k,
            acc.dense_main,
            acc.sparse_main,
            acc.indexer,
            f"{round(acc.main_only_speedup, 0):g}x",
            f"{round(acc.end_to_end_speedup, 2)}x",
        ])
        detail[str(k)] = {
            "dense_main": acc.dense_main,
            "sparse_main": acc.sparse_main,
            "indexer": acc.indexer,
            "main_only_speedup": round(acc.main_only_speedup, 2),
            "end_to_end_speedup": round(acc.end_to_end_speedup, 2),
        }
    # 整条 prefill 的 O(L^2) 标度(证明 indexer 仍是 O(L^2))
    L = 131072
    per_kv_dim = 128 * (512 + 64)
    prefill_dense = cm.prefill_total_main_cost_dense(L, per_kv_dim)
    prefill_sparse = cm.prefill_total_main_cost_sparse(L, 512, per_kv_dim)
    prefill_indexer = cm.prefill_total_indexer_cost(L, 64, 128)
    dump("cost_model.json", {
        "params": {"L": L, "per_kv_dim": per_kv_dim,
                   "indexer_heads": 64, "indexer_dim": 128},
        "table_rows": rows,
        "detail": detail,
        "prefill_scale": {
            "dense_main_O_L2": prefill_dense,
            "sparse_main_O_Lk": prefill_sparse,
            "indexer_O_L2": prefill_indexer,
        },
    })


if __name__ == "__main__":
    trace_quadratic_tax()
    trace_three_branch()
    trace_importance_score()
    trace_lightning_indexer()
    trace_topk_selection()
    trace_training_coadapt()
    trace_cost_model()
    print("all traces written")
