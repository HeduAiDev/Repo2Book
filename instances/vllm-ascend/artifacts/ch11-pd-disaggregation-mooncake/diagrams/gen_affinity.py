#!/usr/bin/env python3
"""★ 亲和路由高潮：一条 P=400 的 prompt，命中 H=300、本地已算 C=100 → 只拉 H-C=200。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1180, 620
S = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
S.append('<defs>')
S.append('<marker id="arr" viewBox="0 0 10 7" refX="9" refY="3.5" markerWidth="8" markerHeight="6" orient="auto"><path d="M0,0 L10,3.5 L0,7 Z" fill="#475569"/></marker>')
S.append('</defs>')
S.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, rx=8, sw=1.6):
    S.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def txt(x, y, t, size=14, anchor="middle", weight="normal", fill="#1e293b", italic=False):
    st = ' font-style="italic"' if italic else ''
    S.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{size}" font-weight="{weight}" fill="{fill}"{st}>{esc(t)}</text>')


txt(W/2, 42, "★ KV 亲和路由：lookup 命中 → need_to_allocate → 跨节点字节只按缺口算", 21, "middle", "bold", "#0f172a")
txt(W/2, 70, "一条 P = 400 token 的 prompt：本地已算 C = 100，外部池命中 H = 300", 14, "middle", "normal", "#64748b")

# ── token 条 ──
bx0, by0, bw, bh = 90, 110, 1000, 64
scale = bw / 400.0  # token -> px
C, Hh, P = 100, 300, 400

# 段 1：[0,C] 本地已算
w1 = C * scale
box(bx0, by0, w1, bh, "#dcfce7", "#22c55e", 6, 1.8)
txt(bx0 + w1/2, by0 + 30, "本地已算", 13, "middle", "bold", "#15803d")
txt(bx0 + w1/2, by0 + 48, "0 → 100", 12, "middle", "normal", "#15803d")

# 段 2：[C,H] 池命中、从池加载（need_to_allocate）
w2 = (Hh - C) * scale
box(bx0 + w1, by0, w2, bh, "#ede9fe", "#8b5cf6", 6, 2.2)
txt(bx0 + w1 + w2/2, by0 + 28, "从外部 KV 池加载", 13, "middle", "bold", "#6d28d9")
txt(bx0 + w1 + w2/2, by0 + 46, "need_to_allocate = 200", 12, "middle", "bold", "#6d28d9")

# 段 3：[H,P] 本地重算
w3 = (P - Hh) * scale
box(bx0 + w1 + w2, by0, w3, bh, "#fef3c7", "#f59e0b", 6, 1.8)
txt(bx0 + w1 + w2 + w3/2, by0 + 30, "本地算", 13, "middle", "bold", "#b45309")
txt(bx0 + w1 + w2 + w3/2, by0 + 48, "300 → 400", 11.5, "middle", "normal", "#b45309")

# 刻度
for tk in (0, 100, 300, 400):
    x = bx0 + tk * scale
    S.append(f'<line x1="{x}" y1="{by0+bh}" x2="{x}" y2="{by0+bh+8}" stroke="#94a3b8" stroke-width="1.5"/>')
    txt(x, by0 + bh + 24, str(tk), 12, "middle", "normal", "#64748b")
txt(bx0 + bw + 30, by0 + bh + 24, "token", 12, "start", "normal", "#94a3b8")

# ── 左：lookup 调用 ──
box(90, 250, 470, 150, "#faf5ff", "#a855f7", 12, 2)
txt(325, 280, "① 先问池：lookup(token_len, block_hashes, group_ids)", 14, "middle", "bold", "#7e22ce")
txt(325, 308, "zmq REQ → 池的 lookup RPC → 返回命中 token 数 H", 12.5, "middle", "normal", "#7c3aed")
txt(325, 338, "H = 300（这条 prompt 的前 300 个 token 已在池里）", 13, "middle", "bold", "#6d28d9")
txt(325, 368, "若 H == request.num_tokens(=400) → 砍 1，留 1 个本地跑 forward", 11.5, "middle", "italic", "#7e22ce")
txt(325, 388, "（vLLM 不允许 0 个新 token 进 forward）", 11, "middle", "italic", "#9333ea")

# ── 右：need_to_allocate 算术 ──
box(620, 250, 470, 150, "#eff6ff", "#3b82f6", 12, 2)
txt(855, 280, "② 算缺口：只补 H − C，不重算/重传 P", 14, "middle", "bold", "#1d4ed8")
txt(855, 312, "need_to_allocate = max(0, H − C)", 13.5, "middle", "bold", "#1e40af")
txt(855, 338, "= max(0, 300 − 100) = 200", 13, "middle", "normal", "#1d4ed8")
txt(855, 368, "LoadSpec(vllm_cached=100, kvpool_cached=300)", 12, "middle", "normal", "#2563eb")
txt(855, 388, "vLLM 只分配这 200 个 token 的块，从池所在处加载", 11.5, "middle", "italic", "#1d4ed8")

# arrow between
S.append(f'<path d="M560,325 L616,325" stroke="#475569" stroke-width="2.2" fill="none" marker-end="url(#arr)"/>')

# ── 底：省了多少 ──
box(90, 440, 1000, 130, "#f0fdf4", "#16a34a", 12, 2)
txt(590, 470, "省下的传输 = 跨节点搬运量 ∝ 缺口(H−C)，而非整段 prompt P", 15, "middle", "bold", "#15803d")
txt(590, 500, "朴素 PD 分离：搬整段 400 token 的 KV", 13, "middle", "normal", "#475569")
txt(590, 524, "亲和路由：只搬 200 token 的 KV  →  传输量砍掉 (400−200)/400 = 50%", 13.5, "middle", "bold", "#166534")
txt(590, 552, "命中率越高，省得越多；前缀全命中时几乎不搬，只跑最后 1 个 token", 12, "middle", "italic", "#15803d")

S.append('</svg>')
open("affinity.svg", "w", encoding="utf-8").write("\n".join(S))
print("wrote affinity.svg")
