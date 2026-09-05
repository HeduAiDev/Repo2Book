# -*- coding: utf-8 -*-
"""ch20 推导审计 scratchpad — 从论文定义独立重推全章数值断言（不 import 参考实现）。

跑完即删（CLAUDE.md 规 6）。只 print，不落盘。
"""
import math
import numpy as np

FAIL = []

def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    if not cond:
        FAIL.append(name)
    print(f"[{status}] {name} {detail}")

def r4(v):
    return round(float(v), 4)

# ============ 0. 头部/§慢在搬运 的算术 ============
N8 = 8192
check("8K 表元素数", N8 * N8 == 67108864, f"= {N8*N8}")
check("8K 表 fp16 字节", N8 * N8 * 2 == 134217728, f"= {N8*N8*2} ~= 134.2MB")
check("往返 268MB -> 0.18ms @1.5TB/s", abs(2 * N8 * N8 * 2 / 1.5e12 * 1e3 - 0.1790) < 0.001,
      f"= {2*N8*N8*2/1.5e12*1e3:.4f} ms")
check("GPT-2 两表元素 2*1024^2 == 2097152", 2 * 1024**2 == 2097152)
check("GPT-2 两表 fp16 字节 4194304", 2 * 1024**2 * 2 == 4194304)
check("SRAM 合计 192KB*108 == 20736KB", 192 * 108 == 20736)
check("带宽比 19/2==9.5, 19/1.5==12.67", abs(19/2 - 9.5) < 1e-9 and abs(19/1.5 - 12.667) < 1e-3)
ratio_agg_40GiB = 40 * 2**30 / (20736 * 1024)
check("合计 SRAM vs 40GiB HBM ~= 2023 (正文「小两千倍以上」)", 2000 < ratio_agg_40GiB < 2100,
      f"= {ratio_agg_40GiB:.1f}")
ratio_persm = 40 * 2**30 / (192 * 1024)
check("单 SM SRAM vs 40GiB ~= 2.18e5 -> 5.34 个数量级 (图注「五个数量级以上」= 每 SM 口径)",
      2.1e5 < ratio_persm < 2.2e5, f"= {ratio_persm:.3g} -> {math.log10(ratio_persm):.2f} 个数量级")
check("算术强度 312/2.0=156, 312/1.5=208 (「一百多次」)", abs(312/2.0 - 156) < 1e-9 and abs(312/1.5 - 208) < 1e-9)
check("non-matmul 16x: 312/19.5 == 16.0", abs(312/19.5 - 16.0) < 1e-9)
check("FA-2 73% 峰值: 230/312 ~= 0.737", abs(230/312 - 0.737) < 1e-3)
check("softmax 每元素访存 4*8192=32768 -> 3*8192=24576", 4 * 8192 == 32768 and 3 * 8192 == 24576)

# ============ 1. m02: online-softmax 递推表 x=[1,3,2,5,4] ============
x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
rows = []
m_prev, d_prev = -np.inf, 0.0
for j, xj in enumerate(x, start=1):
    m_new = max(m_prev, xj)
    factor = None if m_prev == -np.inf else math.exp(m_prev - m_new)
    old_rescaled = d_prev * (factor if factor is not None else 0.0)
    new_term = math.exp(xj - m_new)
    d_new = old_rescaled + new_term
    rows.append((j, xj, m_prev, m_new, factor, old_rescaled, new_term, d_new))
    assert 1.0 <= d_new <= j
    m_prev, d_prev = m_new, d_new
exp_rows = [
    (1, 1.0, None, 0.0, 1.0, 1.0),
    (2, 3.0, 0.1353, 0.1353, 1.0, 1.1353),
    (3, 3.0, 1.0, 1.1353, 0.3679, 1.5032),
    (4, 5.0, 0.1353, 0.2034, 1.0, 1.2034),
    (5, 5.0, 1.0, 1.2034, 0.3679, 1.5713),
]
ok = True
for (j, xj, mp, mn, fac, oldr, newt, dn), (ej, emn, efac, eoldr, enewt, edn) in zip(rows, exp_rows):
    ok &= (r4(mn) == emn)
    ok &= (efac is None and fac is None) or (fac is not None and r4(fac) == efac)
    ok &= r4(oldr) == eoldr and r4(newt) == enewt and r4(dn) == edn
check("m02 表：五轮 (m_j, 折算, 旧账折算后, 新项, d_j)", ok, str([r4(r[7]) for r in rows]))
check("m02 末值 (m_V,d_V)=(5.0,1.5713)", r4(m_prev) == 5.0 and r4(d_prev) == 1.5713)
# 三版 y
y_safe = np.exp(x - x.max()) / np.exp(x - x.max()).sum()
y_naive = np.exp(x) / np.exp(x).sum()
y_online = np.array([math.exp(xi - m_prev) / d_prev for xi in x])
exp_y = [0.0117, 0.0861, 0.0317, 0.6364, 0.2341]
check("m02 三版 y == [0.0117,0.0861,0.0317,0.6364,0.2341]",
      all(r4(v) == e for v, e in zip(y_safe, exp_y))
      and np.allclose(y_naive, y_safe, atol=1e-12) and np.allclose(y_online, y_safe, atol=1e-12),
      str([r4(v) for v in y_safe]))
# 溢出对照
xo = np.array([1000.0, 1001.0])
y_o = np.exp(xo - xo.max()) / np.exp(xo - xo.max()).sum()
check("m02 溢出对照 x=[1000,1001] -> [0.2689, 0.7311]",
      r4(y_o[0]) == 0.2689 and r4(y_o[1]) == 0.7311, str([r4(v) for v in y_o]))
check("m02 naive e^1000 上溢 fp64", math.exp(1000) == math.inf if False else np.exp(np.float64(1000)) == np.inf)

# ============ 2. m03: ⊕ 四条路径 ============
def op(a, b):
    mi, di = a
    mj, dj = b
    mp = max(mi, mj)
    return (mp, di * math.exp(mi - mp) + dj * math.exp(mj - mp))

def summarize(xs):
    xs = np.asarray(xs, dtype=float)
    return (xs.max(), float(np.exp(xs - xs.max()).sum()))

b1, b2, b3 = summarize([1, 3]), summarize([2, 5]), summarize([4])
check("m03 三块局部态 (3.0,1.1353),(5.0,1.0498),(4.0,1.0)",
      (r4(b1[0]), r4(b1[1])) == (3.0, 1.1353) and (r4(b2[0]), r4(b2[1])) == (5.0, 1.0498)
      and (r4(b3[0]), r4(b3[1])) == (4.0, 1.0))
p_seq = op(op(b1, b2), b3)           # (b1⊕b2)⊕b3
p_out = op(op(b3, b2), b1)           # 乱序 (b3⊕b2)⊕b1
mid_seq = op(b1, b2)
mid_out = op(b3, b2)
p_par = op(summarize([1, 3, 2]), summarize([5, 4]))  # 换括号
check("m03 中间态: 顺序 (5.0,1.2034) / 乱序 (5.0,1.4177)",
      (r4(mid_seq[0]), r4(mid_seq[1])) == (5.0, 1.2034) and (r4(mid_out[0]), r4(mid_out[1])) == (5.0, 1.4177))
check("m03 三路径末态全 == (5.0, 1.5713) == 顺序单遍",
      all((r4(p[0]), r4(p[1])) == (5.0, 1.5713) for p in (p_seq, p_out, p_par))
      and (r4(m_prev), r4(d_prev)) == (5.0, 1.5713))
check("m03 换括号局部态 {1,3,2}=(3.0,1.5032) {5,4}=(5.0,1.3679)",
      (r4(summarize([1,3,2])[0]), r4(summarize([1,3,2])[1])) == (3.0, 1.5032)
      and (r4(summarize([5,4])[0]), r4(summarize([5,4])[1])) == (5.0, 1.3679))
# ⊕ == 多重集摘要的并集（换元）：随机 500 组
rng = np.random.default_rng(0)
ok_union = True
for _ in range(500):
    A = rng.normal(size=rng.integers(1, 8))
    B = rng.normal(size=rng.integers(1, 8))
    lhs = op(summarize(A), summarize(B))
    rhs = summarize(np.concatenate([A, B]))
    ok_union &= abs(lhs[0] - rhs[0]) < 1e-12 and abs(lhs[1] - rhs[1]) < 1e-9
check("m03 ⊕ == 多重集并集摘要（随机 500 组）", ok_union)
# 「一一对应」非单射反例（供 negotiable 依据）：{1,0} 与 {1,-1-ln2,-1-ln2} 同摘要
a = -math.log(2)  # x = 1 + ln(e^{-1}/2) = -ln2: 2e^{x-1} == e^{-1}
check("m03 反例：不同多重集可共享同一摘要（「一一对应」措辞过强）",
      abs(summarize([1.0, 0.0])[1] - summarize([1.0, a, a])[1]) < 1e-12 and summarize([1.0, 0.0])[0] == summarize([1.0, a, a])[0])

# ============ 3. m04: FA tiling 2x2（按论文 Alg.1 line 9-13 独立实现） ============
Q = np.array([[1.0, 0], [0, 1], [1, 1], [2, 0]])
K = np.array([[1.0, 0], [0, 1], [1, 1], [0, 2]])
V = np.array([[1.0, 2], [3, 4], [5, 6], [7, 8]])
Br = Bc = 2
T_r = T_c = 2
scale = 1.0
O = np.zeros((4, 2))
ell = np.zeros(4)
mrow = np.full(4, -np.inf)
trace = []
for j in range(T_c):
    Kj = K[j*Bc:(j+1)*Bc]
    Vj = V[j*Bc:(j+1)*Bc]
    for i in range(T_r):
        Qi = Q[i*Br:(i+1)*Br]
        Sij = scale * Qi @ Kj.T                    # B_r x B_c
        mt = Sij.max(axis=1)
        Pt = np.exp(Sij - mt[:, None])
        lt = Pt.sum(axis=1)
        for rr, grow in enumerate(range(i*Br, (i+1)*Br)):
            m_new = max(mrow[grow], mt[rr])
            fac = None if mrow[grow] == -np.inf else math.exp(mrow[grow] - m_new)
            ell_new = ell[grow] * (fac if fac is not None else 0.0) + math.exp(mt[rr] - m_new) * lt[rr]
            O[grow] = (ell[grow] * (fac if fac is not None else 0.0) * O[grow]
                       + math.exp(mt[rr] - m_new) * (Pt[rr] @ Vj)) / ell_new
            trace.append((j, i, grow, mrow[grow], m_new, fac, ell_new, O[grow].copy()))
            ell[grow], mrow[grow] = ell_new, m_new
# 期望表（章节 m04）
exp = [
    (0, 0, 0, None, 1.0, None, 1.3679, [1.5379, 2.5379]),
    (0, 0, 1, None, 1.0, None, 1.3679, [2.4621, 3.4621]),
    (0, 1, 2, None, 1.0, None, 2.0, [2.0, 3.0]),
    (0, 1, 3, None, 2.0, None, 1.1353, [1.2384, 2.2384]),
    (1, 0, 0, 1.0, 1.0, 1.0, 2.7358, [3.5379, 4.5379]),
    (1, 0, 1, 1.0, 2.0, 0.3679, 1.8711, [5.3864, 6.3864]),
    (1, 1, 2, 1.0, 2.0, 0.3679, 2.7358, [4.9242, 5.9242]),
    (1, 1, 3, 2.0, 2.0, 1.0, 2.2707, [3.2384, 4.2384]),
]
ok = True
for (j, i, g, mo, mn, fac, en, eo), t in zip(exp, trace):
    ok &= (t[0], t[1], t[2]) == (j, i, g)
    mo_is_inf = (mo is None)   # 首块旧 m = -inf（章节记 —）
    ok &= (mo_is_inf and t[3] == -np.inf) or (mo is not None and r4(t[3]) == mo)
    ok &= r4(t[4]) == mn
    ok &= (fac is None and t[5] is None) or (fac is not None and t[5] is not None and r4(t[5]) == fac)
    ok &= r4(t[6]) == en and [r4(v) for v in t[7]] == eo
check("m04 八步 (m 迁移/折算/ℓ/O) 全表", ok)
S_full = scale * Q @ K.T
check("m04 S 全表 == [[1,0,1,0],[0,1,1,2],[1,1,2,2],[2,0,2,0]]",
      np.allclose(S_full, np.array([[1,0,1,0],[0,1,1,2],[1,1,2,2],[2,0,2,0]], dtype=float)))
O_ref = np.exp(S_full - S_full.max(1, keepdims=True)) @ V / np.exp(S_full - S_full.max(1, keepdims=True)).sum(1, keepdims=True)
check("m04 终值 == softmax(QK^T)V（机器精度）", np.allclose(O, O_ref, atol=1e-12),
      f"max|diff| = {np.abs(O-O_ref).max():.2e}")
# 每步 == 只看已见 KV 的朴素版
ok_step = True
for (j, i, g, *_), orow in zip(exp, [t[7] for t in trace]):
    pass
idx = 0
for j in range(T_c):
    for i in range(T_r):
        Ssub = scale * Q @ K[: (j+1)*Bc].T
        Psub = np.exp(Ssub - Ssub.max(1, keepdims=True))
        Osub = Psub @ V[: (j+1)*Bc] / Psub.sum(1, keepdims=True)
        for grow in range(i*Br, (i+1)*Br):
            ok_step &= np.allclose(trace[idx][7], Osub[grow], atol=1e-12)
            idx += 1
check("m04 每步写回 == 只看已见 KV 的精确注意力", ok_step)
check("m04 片上块至多 2x2=4 元素、整表 16 从未创建", Br*Bc == 4 and 4*4 == 16)

# ============ 4. m05: IO 账 ============
N, d = 1024, 64
s1 = 2*N*d + N*N
s2 = 2*N*N
s3 = N*N + 2*N*d
std = s1 + s2 + s3
check("m05 标准三步 1179648+2097152+1179648 == 4456448",
      s1 == 1179648 and s2 == 2097152 and s3 == 1179648 and std == 4456448)
kv_once = 2*N*d
per_pass = 3*N*d + 4*N   # Q 载入 + O 读 + O 写 + (ℓ,m) 读写
check("m05 每趟 3Nd+4N == 200704", per_pass == 200704, f"= {per_pass}")
for Bc_, Tc_, tot_, rat_ in [(64, 16, 3342336, 1.3333), (128, 8, 1736704, 2.566), (256, 4, 933888, 4.7719)]:
    tot = kv_once + Tc_ * per_pass
    check(f"m05 FA Bc={Bc_}: Tc={Tc_} 总访问 {tot_}、比值 {rat_}",
          tot == tot_ and abs(round(std / tot, 4) - rat_) < 5e-4,
          f"tot={tot}, ratio={std/tot:.4f}")
    assert math.ceil(N / Bc_) == Tc_
check("m05 FA 额外统计量 (m,ℓ)=2N == 2048 元素 / 4096 字节", 2*N == 2048)
check("m05 M=100KB(fp16)=51200 元素时 d^2=4096 << M", 64**2 < 51200)

# ============ 5. m06: causal 整块跳过 ============
for NB, blk, vis, skip, ratio in [(8, 2, 10, 6, 1.6), (64, 8, 36, 28, 1.7778)]:
    T = NB // blk
    visited = sum(1 for bi in range(T) for bj in range(T) if bj <= bi)
    check(f"m06 N={NB} 块{blk}: 访 {vis} 跳 {skip} 比 {ratio}",
          visited == vis and T*T - visited == skip and abs(round(T*T/visited, 4) - ratio) < 5e-4)
# docstring 掩码例
def causal_mask(sq, sk):
    off = sk - sq
    return np.array([[1 if c <= r + off else 0 for c in range(sk)] for r in range(sq)])
check("掩码例 seqlen_q=2,seqlen_k=5 == [[1,1,1,1,0],[1,1,1,1,1]]",
      causal_mask(2, 5).tolist() == [[1,1,1,1,0],[1,1,1,1,1]])
check("掩码例 seqlen_q=5,seqlen_k=2 == 前三行全零+[[1,0],[1,1]]",
      causal_mask(5, 2).tolist() == [[0,0],[0,0],[0,0],[1,0],[1,1]])

# ============ 6. LSE 恒等式与小例 ============
def lse(v):
    v = np.asarray(v, dtype=float)
    return v.max() + np.log(np.exp(v - v.max()).sum())
check("LSE_A=log(1+1)=0.693", abs(lse([0, 0]) - 0.6931) < 5e-4, f"= {lse([0,0]):.4f}")
check("LSE_B=log(e^2+1)=2.127", abs(lse([2, 0]) - 2.1269) < 5e-4, f"= {lse([2,0]):.4f}")
la0, lb0 = lse([0, 0]), lse([2, 0])
merged = max(la0, lb0) + math.log(math.exp(la0 - max(la0, lb0)) + math.exp(lb0 - max(la0, lb0)))
check("LSE 合并 == log(2+8.389)=2.341 == log(1+1+e^2+1)",
      abs(merged - 2.3409) < 5e-4 and abs(lse([0,0,2,0]) - 2.3409) < 5e-4,
      f"merge={merged:.4f}, one-shot={lse([0,0,2,0]):.4f}")
rng = np.random.default_rng(1)
ok_id, ok_sq = True, True
for _ in range(500):
    a = rng.normal(size=6); b = rng.normal(size=5)
    ok_id &= abs(lse(np.concatenate([a, b])) - math.log(math.exp(lse(a)) + math.exp(lse(b)))) < 1e-10
    L = lse(a)
    ok_sq &= (a.max() <= L <= a.max() + math.log(len(a)))
check("LSE 合并恒等式（随机 500 组）", ok_id)
check("LSE 夹逼 max<=LSE<=max+log n（随机 500 组）", ok_sq)

# ============ 7. m07: cascade LSE 合并（从原始输入独立重算） ============
K_p = np.array([[1,0],[0,1],[1,1],[2,0]], float)
V_p = np.array([[1,2],[3,4],[5,6],[7,8]], float)
K_sA = np.array([[0,1],[1,0],[1,2]], float)
V_sA = np.array([[2,0],[0,3],[1,1]], float)
K_sB = np.array([[1,1],[0,2]], float)
V_sB = np.array([[4,2],[2,1]], float)
q_A = np.array([[1,1],[0,1]], float)
q_B = np.array([[2,0],[1,1]], float)

def attn_lse(Qq, Kk, Vv, causal=False, query_offset=None):
    """右下对齐: query 行 r 全局位 = r+offset; 保留 c <= r+offset."""
    S = Qq @ Kk.T
    if causal:
        assert query_offset is not None
        sk = len(Kk)
        off = query_offset if query_offset is not None else sk - len(Qq)
        S = np.where((np.arange(sk)[None, :] <= (off + np.arange(len(Qq))[:, None])), S, -np.inf)
    m = S.max(axis=1)
    P = np.exp(S - m[:, None])
    l = P.sum(axis=1)
    return P @ Vv / l[:, None], m + np.log(l)

def merge(Oa, la, Os, ls):
    M = max(la, ls)
    pa, sa = math.exp(la - M), math.exp(ls - M)
    out_se = pa + sa
    return (pa / out_se) * Oa + (sa / out_se) * Os, math.log(out_se) + M, pa, sa, out_se, pa/out_se, sa/out_se

O_pre, lse_pre = attn_lse(np.vstack([q_A, q_B]), K_p, V_p)
O_sA, lse_sA = attn_lse(q_A, K_sA, V_sA, causal=True, query_offset=1)
O_sB, lse_sB = attn_lse(q_B, K_sB, V_sB, causal=True, query_offset=0)
O_oneA, lse_oneA = attn_lse(q_A, np.vstack([K_p, K_sA]), np.vstack([V_p, V_sA]), causal=True, query_offset=5)
O_oneB, lse_oneB = attn_lse(q_B, np.vstack([K_p, K_sB]), np.vstack([V_p, V_sB]), causal=True, query_offset=4)

exp_m07 = [  # row, prefix_lse, suffix_lse, max, p_se, s_se, out_se, w_p, w_s, merged_lse, merged_O[0], one_lse
    ("A0", 3.0064, 1.6931, 3.0064, 1.0, 0.2689, 1.2689, 0.7881, 0.2119, 3.2446, 4.0925, 3.2446),
    ("A1", 2.0064, 2.4076, 2.4076, 0.6695, 1.0, 1.6695, 0.401, 0.599, 2.9201, 2.2957, 2.9201),
    ("B0", 4.2539, 2.0, 4.2539, 1.0, 0.105, 1.105, 0.905, 0.095, 4.3537, 5.9034, 4.3537),
    ("B1", 3.0064, 2.6931, 3.0064, 1.0, 0.7311, 1.7311, 0.5777, 0.4223, 3.5551, 4.1116, 3.5551),
]
pairs = [("A0", O_pre[0], lse_pre[0], O_sA[0], lse_sA[0], O_oneA[0], lse_oneA[0]),
         ("A1", O_pre[1], lse_pre[1], O_sA[1], lse_sA[1], O_oneA[1], lse_oneA[1]),
         ("B0", O_pre[2], lse_pre[2], O_sB[0], lse_sB[0], O_oneB[0], lse_oneB[0]),
         ("B1", O_pre[3], lse_pre[3], O_sB[1], lse_sB[1], O_oneB[1], lse_oneB[1])]
ok = True
for (name, la_e, ls_e, mx_e, pa_e, sa_e, se_e, wp_e, ws_e, lm_e, o0_e, lone_e), \
    (nm, Oa, la, Os, ls, O_one, l_one) in zip(exp_m07, pairs):
    Om, lm, pa, sa, se, wp, ws = merge(Oa, la, Os, ls)
    got = (r4(la), r4(ls), r4(max(la, ls)), r4(pa), r4(sa), r4(se), r4(wp), r4(ws), r4(lm), r4(Om[0]), r4(l_one))
    want = (la_e, ls_e, mx_e, pa_e, sa_e, se_e, wp_e, ws_e, lm_e, o0_e, lone_e)
    if got != want:
        ok = False
        print("   row", name, "got", got, "want", want)
    if not (np.allclose(Om, O_one, atol=1e-12) and abs(lm - l_one) < 1e-12 and abs(wp + ws - 1) < 1e-12):
        ok = False
        print("   row", name, "merge-vs-one-shot mismatch")
check("m07 四行全字段（lse/权重/合并 lse/O[0]）+ 合并==一次性", ok)
check("m07 A1 反超（2.0064 < 2.4076 -> 后缀权重 0.599）", exp_m07[1][1] < exp_m07[1][2] and exp_m07[1][8] == 0.599)

# ============ 8. FA-2 Alg.1 line 10 印刷勘误验证 ============
# 不变式: 未归一 O^(j) = sum_{k<=j} e^{S^(k)-m^(j)} V^(k)  =>  递推因子必为 e^{m^{j-1}-m^j}（正指数、无逆）
rng = np.random.default_rng(7)
S1 = rng.normal(size=(3, 4)) * 2
S2 = rng.normal(size=(3, 4)) * 2
V1 = rng.normal(size=(4, 2))
V2 = rng.normal(size=(4, 2))
m1 = S1.max(1)
m = np.maximum(m1, S2.max(1))
target = np.exp(S1 - m[:, None]) @ V1 + np.exp(S2 - m[:, None]) @ V2   # 不变式的右端
O1u = np.exp(S1 - m1[:, None]) @ V1                                     # 块1 未归一输出
corrected = np.exp(m1 - m)[:, None] * O1u + np.exp(S2 - m[:, None]) @ V2   # e^{m1-m} 折算（章节/参考实现）
printed = np.exp(m - m1)[:, None] * O1u + np.exp(S2 - m[:, None]) @ V2     # 论文印刷 diag(e^{m1-m})^{-1}
check("FA-2 勘误：正指数 e^{m^{j-1}-m^j} 满足不变式、印刷的 ^{-1} 破坏不变式",
      np.allclose(corrected, target) and not np.allclose(printed, target),
      f"|corrected-target|={np.abs(corrected-target).max():.1e}, |printed-target|={np.abs(printed-target).max():.1e}")

# ============ 9. 图注数字自洽 ============
check("fig paper-1: 16.8/2.2 ~= 7.6", abs(16.8/2.2 - 7.636) < 0.05 and round(16.8/2.2, 1) == 7.6)
check("fig paper-2: 75.2/66.6 多 13%", abs(75.2/66.6 - 1.129) < 0.001)
check("fig paper-2: 40.3/4.4 ~= 9.16 (「最多省 9x」)", abs(40.3/4.4 - 9.16) < 0.01)
check("fig paper-2: 41.7/7.3 ~= 5.7x", abs(41.7/7.3 - 5.71) < 0.01)
check("fig paper-4: 171/97 ~= 1.8x", abs(171/97 - 1.763) < 0.01)

# ============ 10. cascade 扫描账 ============
check("cascade 扫描账 13 vs 9、省 4、0.3077、一般式 P(R-1)",
      7+6 == 13 and 4+3+2 == 9 and 13-9 == 4 and abs((13-9)/13 - 0.3077) < 5e-4 and 4*(2-1) == 4)

# ============ 汇总 ============
print()
if FAIL:
    print("FAILED:", FAIL)
    raise SystemExit(1)
print("ALL CHECKS PASSED")
