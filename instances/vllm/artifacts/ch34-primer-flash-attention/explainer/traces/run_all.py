#!/usr/bin/env python3
"""Explainer 驱动脚本 —— 跑 ch34 论文参考实现,取可示教的数值轨迹。

对每个 needs_worked_example 机制,用一组小而具体的参数跑精简版,把逐轮/逐步的关键标量
四舍五入到 4 位小数后落成 traces/<id>.json(trace_source="run" 的真相源)。

跑法(纯 CPU/NumPy,host 直接跑):
    cd instances/vllm/artifacts/ch34-primer-flash-attention/explainer/traces
    python3 run_all.py
"""
import json
import math
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
IMPL = HERE.parent.parent / "implementation"
sys.path.insert(0, str(IMPL))

from online_softmax import (  # noqa: E402
    safe_softmax,
    online_softmax_stats,
    online_softmax_merge,
    combine_blocks_via_merge,
)
from flash_attention import (  # noqa: E402
    standard_attention,
    flash_attention_forward,
    fa_block_sizes,
    hbm_accesses_standard,
    hbm_accesses_flash,
)
from lse_merge import attention_with_lse, merge_lse_states  # noqa: E402


def R(x, n=4):
    """四舍五入到 n 位小数,标量或 ndarray -> python 原生类型(便于 json 序列化)。"""
    if isinstance(x, np.ndarray):
        return [R(v, n) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [R(v, n) for v in x]
    return round(float(x), n)


def dump(name, obj):
    p = HERE / f"{name}.json"
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  wrote {p.name}")


# ==========================================================================
# 1) online-softmax-recurrence —— running (m, d) 单遍递推
# ==========================================================================
def trace_online_softmax_recurrence():
    x = np.array([1.0, 3.0, 2.0, 5.0])
    m = -np.inf
    d = 0.0
    steps = []
    for j, xj in enumerate(x, start=1):
        m_old = m
        m_new = max(m_old, float(xj))
        rescale = 0.0 if m_old == -np.inf else math.exp(m_old - m_new)
        d_new = d * (0.0 if m_old == -np.inf else math.exp(m_old - m_new)) + math.exp(float(xj) - m_new)
        steps.append({
            "j": j,
            "x_j": R(xj),
            "m_old": ("-inf" if m_old == -np.inf else R(m_old)),
            "m_new": R(m_new),
            "rescale_factor": ("n/a" if m_old == -np.inf else R(rescale)),
            "d_before": R(d),
            "d_new": R(d_new),
        })
        m, d = m_new, d_new
    # 交叉验证:online 末值 (m,d) 应与三遍 safe-softmax 的 (max, sum exp(x-max)) 恒等
    m_ref = float(x.max())
    d_ref = float(np.exp(x - m_ref).sum())
    y_online = np.exp(x - m) / d
    y_safe = safe_softmax(x)
    return {
        "params": {"x": R(x)},
        "steps": steps,
        "final": {"m": R(m), "d": R(d)},
        "reference_three_pass": {"m_V": R(m_ref), "d_V": R(d_ref)},
        "softmax_max_abs_diff_online_vs_safe": R(float(np.abs(y_online - y_safe).max()), 8),
    }


# ==========================================================================
# 2) online-softmax-merge-operator —— ⊕ 二元算子的结合律/乱序合并
# ==========================================================================
def trace_merge_operator():
    x = np.array([1.0, 3.0, 2.0, 5.0])
    A, B = x[:2], x[2:]
    sA = online_softmax_stats(A)          # 局部 (m,d) of block A=[1,3]
    sB = online_softmax_stats(B)          # 局部 (m,d) of block B=[2,5]
    fwd = online_softmax_merge(sA, sB)    # A ⊕ B
    rev = online_softmax_merge(sB, sA)    # B ⊕ A(交换律)
    single = online_softmax_stats(x)      # 一遍遍历整段
    via = combine_blocks_via_merge(x, 2)  # 分块局部再 ⊕ 归并
    m_ref = float(x.max())
    d_ref = float(np.exp(x - m_ref).sum())
    rows = [
        {"what": "block A=[1,3] local", "m": R(sA[0]), "d": R(sA[1])},
        {"what": "block B=[2,5] local", "m": R(sB[0]), "d": R(sB[1])},
        {"what": "A (+) B", "m": R(fwd[0]), "d": R(fwd[1])},
        {"what": "B (+) A (commute)", "m": R(rev[0]), "d": R(rev[1])},
        {"what": "single-pass whole", "m": R(single[0]), "d": R(single[1])},
        {"what": "three-pass safe ref", "m": R(m_ref), "d": R(d_ref)},
    ]
    return {
        "params": {"x": R(x), "block_size": 2},
        "rows": rows,
        "combine_blocks_via_merge": {"m": R(via[0]), "d": R(via[1])},
    }


# ==========================================================================
# 4) flashattention-tiling —— running (m_i, l_i, O_i) 2x2 分块递推
# ==========================================================================
def trace_tiling():
    # seq_len=4, d=2, 2x2 分块(block_r=2, block_c=2 => Tr=Tc=2)。追踪 query 行 0 的
    # 状态如何随两个 KV 列块演进,并对照标准 softmax(QK^T)V。
    Q = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]])
    K = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]])
    V = np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]])
    scale = 1.0 / math.sqrt(2)
    bc = 2  # KV 列块大小
    row = 0  # 追踪的 query 行

    # 手工复刻 Algorithm 1 内层对 query 行 `row` 的 running (m,l,O) 递推(外层遍历 KV 块)
    m_i = -np.inf
    l_i = 0.0
    O_i = np.zeros(2)
    steps = []
    n = Q.shape[0]
    for jb, kc0 in enumerate(range(0, n, bc), start=1):
        kc1 = min(kc0 + bc, n)
        Kj, Vj = K[kc0:kc1], V[kc0:kc1]
        s = (Q[row] @ Kj.T) * scale          # 局部 S_row,j (长度 bc,非 N)
        m_tilde = float(s.max())
        p_tilde = np.exp(s - m_tilde)
        l_tilde = float(p_tilde.sum())
        m_new = max(m_i, m_tilde)
        l_new = (0.0 if m_i == -np.inf else math.exp(m_i - m_new)) * l_i + math.exp(m_tilde - m_new) * l_tilde
        O_unnorm = (
            (0.0 if m_i == -np.inf else l_i * math.exp(m_i - m_new)) * O_i
            + math.exp(m_tilde - m_new) * (p_tilde @ Vj)
        )
        O_new = O_unnorm / l_new
        steps.append({
            "kv_block": jb,
            "s_local": R(s),
            "m_tilde": R(m_tilde),
            "l_tilde": R(l_tilde),
            "m_old": ("-inf" if m_i == -np.inf else R(m_i)),
            "m_new": R(m_new),
            "l_new": R(l_new),
            "O_new": R(O_new),
        })
        m_i, l_i, O_i = m_new, l_new, O_new

    O_ref = standard_attention(Q, K, V)[row]
    O_flash_full = flash_attention_forward(Q, K, V, block_size_r=2, block_size_c=2)[row]
    max_block = flash_attention_forward(
        Q, K, V, block_size_r=2, block_size_c=2, return_max_block_shape=True
    )
    return {
        "params": {"seq_len": n, "d": 2, "block_r": 2, "block_c": 2, "tracked_query_row": row},
        "steps": steps,
        "O_row_after_tiling": R(O_i),
        "O_row_standard_softmax": R(O_ref),
        "O_row_flash_full_impl": R(O_flash_full),
        "max_local_block_shape": list(max_block),
        "full_NxN_would_be": [n, n],
    }


# ==========================================================================
# 5) io-complexity-accounting —— HBM 访存元素计数 standard vs flash
# ==========================================================================
def trace_io_complexity():
    d = 64
    sram_M = 25600  # ~100KB / 4 bytes(fp32 元素数);典型 A100 片上 SRAM 量级
    bc, br = fa_block_sizes(sram_M, d)
    rows = []
    for n in (1024, 2048, 4096):
        std = hbm_accesses_standard(n, d)
        fla = hbm_accesses_flash(n, d, bc, br)
        rows.append({
            "N": n,
            "d": d,
            "block_c": bc,
            "standard_hbm_elems": std,
            "flash_hbm_elems": fla,
            "ratio_std_over_flash": R(std / fla, 2),
        })
    return {
        "params": {"d": d, "sram_M_elems": sram_M, "block_c": bc, "block_r": br},
        "rows": rows,
    }


# ==========================================================================
# 7) lse-merge —— cascade 前缀/后缀两段 (O,lse) 合并 == 一次性整体注意力
# ==========================================================================
def trace_lse_merge():
    # 2 个 query token,d=2。共享前缀 KV 长 2(所有 query 都能看,causal=False),
    # 私有后缀 KV 长 2(causal=True,各 query 只看到自己位置及之前)。
    d = 2
    scale = 1.0 / math.sqrt(d)
    Q = np.array([[1.0, 0.0], [0.0, 1.0]])
    K_pre = np.array([[1.0, 1.0], [1.0, -1.0]])   # 共享前缀 keys(位置 0,1)
    V_pre = np.array([[1.0, 0.0], [0.0, 1.0]])
    K_suf = np.array([[0.5, 0.5], [-1.0, 1.0]])   # 私有后缀 keys(位置 2,3)
    V_suf = np.array([[2.0, 0.0], [0.0, 2.0]])

    # 前缀段:causal=False(整段共享前缀可见)
    O_pre, lse_pre = attention_with_lse(Q, K_pre, V_pre, causal=False, scale=scale)
    # 后缀段:causal=True,query token t 位于全局位置 (prefix_len + t),query_offset 建模之
    O_suf, lse_suf = attention_with_lse(
        Q, K_suf, V_suf, causal=True, scale=scale, query_offset=0
    )
    O_merged, lse_merged = merge_lse_states(O_pre, lse_pre, O_suf, lse_suf)

    # 参考:对 [前缀 ; 后缀] 拼接后的完整 KV 一次性做注意力(前缀全可见 + 后缀 causal)。
    # 逐 query 行手工拼掩码:前缀 2 列恒可见,后缀 2 列按 causal(query_offset=0)。
    K_all = np.concatenate([K_pre, K_suf], axis=0)
    V_all = np.concatenate([V_pre, V_suf], axis=0)
    S = (Q @ K_all.T) * scale
    nq = Q.shape[0]
    # 掩码:前缀列(0,1)全部可见;后缀列(2,3)对 query t 仅当 (col-2) <= t 可见
    for t in range(nq):
        for c in range(2, 4):
            if (c - 2) > t:
                S[t, c] = -np.inf
    m = S.max(axis=-1, keepdims=True)
    P = np.exp(S - m)
    P = P / P.sum(axis=-1, keepdims=True)
    O_ref = P @ V_all

    rows = []
    for t in range(nq):
        rows.append({
            "token": t,
            "O_prefix": R(O_pre[t]),
            "lse_prefix": R(lse_pre[t]),
            "O_suffix": R(O_suf[t]),
            "lse_suffix": R(lse_suf[t]),
            "O_merged": R(O_merged[t]),
            "lse_merged": R(lse_merged[t]),
            "O_reference_oneshot": R(O_ref[t]),
        })
    return {
        "params": {"num_query_tokens": nq, "d": d, "prefix_kv_len": 2, "suffix_kv_len": 2},
        "rows": rows,
        "max_abs_diff_merged_vs_reference": R(float(np.abs(O_merged - O_ref).max()), 8),
    }


def main():
    print("== ch34 explainer traces ==")
    dump("online_softmax_recurrence", trace_online_softmax_recurrence())
    dump("online_softmax_merge_operator", trace_merge_operator())
    dump("flashattention_tiling", trace_tiling())
    dump("io_complexity_accounting", trace_io_complexity())
    dump("lse_merge", trace_lse_merge())
    print("done.")


if __name__ == "__main__":
    main()
