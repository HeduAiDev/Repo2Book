#!/usr/bin/env python3
"""Explainer 驱动脚本 —— chunked-prefill-row-independence 双路等价见证。

论点:因果注意力第 i 行的输出只依赖位置 ≤ i 的 KV,是绝对位置的纯函数,与 query 轴
被切成几块、每块何时算无关。故把一段 prefill 的 query 轴切成任意几块,每块只喂本块的
query、KV 为「累积到本块末尾的全量历史」、causal 掩码按绝对位置——逐块输出拼起来,与
一次性对整段做因果注意力逐元素恒等(浮点舍入内)。

双路:
  (a) 一次性:对 50-token 序列做整段 causal 注意力,得 O_full。
  (b) 分块:按 16/16/18 三块。第 c 块 query = Q[start:end];KV = K[0:end]/V[0:end]
      (累积全量);掩码按绝对位置(query 绝对行 i 只看 key 列 ≤ i)。逐块 O 拼接 = O_chunked。
断言两路逐元素接近,报告实测最大偏差量级。

纯 CPU/NumPy,host 直接跑:
    cd instances/vllm/artifacts/ch24-primer-flash-attention/explainer/traces
    python3 run_chunked_prefill.py
"""
import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent


def R(x, n=4):
    if isinstance(x, np.ndarray):
        return [R(v, n) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [R(v, n) for v in x]
    return round(float(x), n)


def causal_attention(Q, K, V, scale, q_abs_start=0):
    """对 query 行(绝对起始位置 q_abs_start)与全部给定 KV 做因果注意力。
    query 的第 t 行绝对位置 = q_abs_start + t,只可见 key 列 j ≤ 绝对位置。
    返回 O,形状 (len(Q), d)。float64 全程。"""
    S = (Q @ K.T) * scale                       # (nq, nk)
    nq, nk = S.shape
    for t in range(nq):
        i_abs = q_abs_start + t
        for j in range(nk):
            if j > i_abs:                       # 未来 key,屏蔽
                S[t, j] = -np.inf
    m = S.max(axis=-1, keepdims=True)
    P = np.exp(S - m)
    P = P / P.sum(axis=-1, keepdims=True)
    return P @ V


def main():
    rng = np.random.default_rng(0)
    N, d = 50, 8
    scale = 1.0 / np.sqrt(d)
    Q = rng.standard_normal((N, d))
    K = rng.standard_normal((N, d))
    V = rng.standard_normal((N, d))

    # ---- 路 (a):一次性整段因果注意力 ----
    O_full = causal_attention(Q, K, V, scale, q_abs_start=0)

    # ---- 路 (b):16/16/18 三块,每块 query 仅本块、KV 累积全量、causal 按绝对位置 ----
    chunk_sizes = [16, 16, 18]
    assert sum(chunk_sizes) == N
    O_chunked = np.empty_like(O_full)
    rows = []
    start = 0
    for c, sz in enumerate(chunk_sizes, start=1):
        end = start + sz
        Qc = Q[start:end]                       # 仅本块 query
        Kc, Vc = K[:end], V[:end]               # KV = 累积到本块末尾的全量历史
        Oc = causal_attention(Qc, Kc, Vc, scale, q_abs_start=start)
        O_chunked[start:end] = Oc
        block_max_dev = float(np.abs(Oc - O_full[start:end]).max())
        rows.append({
            "chunk": c,
            "query_abs_range": f"[{start}, {end - 1}]",
            "kv_cumulative_len": end,          # 可见的累积 KV 列数(= 绝对末位置+1)
            "causal": True,
            "block_max_abs_dev_vs_oneshot": R(block_max_dev, 18),
        })
        start = end

    # ---- 断言:两路逐元素接近 ----
    max_abs_dev = float(np.abs(O_chunked - O_full).max())
    close = bool(np.allclose(O_chunked, O_full, rtol=0, atol=1e-12))
    assert close, f"dual-path mismatch, max_abs_dev={max_abs_dev}"

    # 定点见证:抽一行(绝对位置 20,落在第 2 块)看两路输出前 3 维逐位相等
    probe_row = 20
    probe = {
        "row_abs": probe_row,
        "in_chunk": 2,
        "O_oneshot_head3": R(O_full[probe_row][:3]),
        "O_chunked_head3": R(O_chunked[probe_row][:3]),
    }

    out = {
        "params": {
            "N": N, "d": d, "scale": R(scale),
            "chunk_sizes": chunk_sizes,
            "seed": 0,
            "note": "50-token 随机序列,d=8。路(a)一次性 causal 注意力;路(b)按 16/16/18 分块,"
                    "每块 query 仅本块、KV 为累积全量、causal 按绝对位置,逐块输出拼接。",
        },
        "rows": rows,
        "probe_row": probe,
        "max_abs_dev_chunked_vs_oneshot": R(max_abs_dev, 18),
        "allclose_atol_1e-12": close,
    }
    p = HERE / "chunked_prefill_row_independence.json"
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {p.name}")
    print(f"  max_abs_dev = {max_abs_dev:.3e}  allclose(atol=1e-12) = {close}")
    for r in rows:
        print(f"  chunk {r['chunk']} q{r['query_abs_range']} kv_cum={r['kv_cumulative_len']} "
              f"dev={r['block_max_abs_dev_vs_oneshot']:.3e}")


if __name__ == "__main__":
    main()
