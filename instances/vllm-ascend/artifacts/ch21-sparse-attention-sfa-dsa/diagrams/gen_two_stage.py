#!/usr/bin/env python3
"""ch21 两段式稀疏注意力总览：轻量索引器选 top-k → 只对 top-k 算全精度注意力。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1340, 660
TXT = "#1e293b"
SUB = "#64748b"
IDX = "#7c3aed"   # 索引器 紫
ATT = "#1d4ed8"   # 注意力 蓝
KV = "#b45309"    # KV cache 橙
OUT = "#15803d"   # 输出 绿

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs><marker id="ar" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="9" markerHeight="7" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#475569"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{W/2}" y="40" text-anchor="middle" font-size="25" font-weight="bold" fill="{TXT}">两段式稀疏注意力：先挑出少数最相关的 KV，再只算这部分</text>')
L.append(f'<text x="{W/2}" y="68" text-anchor="middle" font-size="14.5" fill="{SUB}">把全长注意力的 O(L) 开销，压到只跟 top-k（DSA 512 / SFA 2048）相关</text>')


def box(cx, cy, w, h, fill, stroke, lines, fw="bold"):
    x = cx - w / 2
    L.append(f'<rect x="{x}" y="{cy}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="1.7"/>')
    n = len(lines)
    for i, (t, s) in enumerate(lines):
        ty = cy + h / 2 + (i - (n - 1) / 2) * (s + 5) + s * 0.34
        col = stroke if i == 0 else TXT
        L.append(f'<text x="{cx}" y="{ty}" text-anchor="middle" font-size="{s}" font-weight="{fw if i==0 else "normal"}" fill="{col}">{esc(t)}</text>')


def harrow(x1, x2, y, label=None, color="#475569"):
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="2.2" marker-end="url(#ar)"/>')
    if label:
        L.append(f'<text x="{(x1+x2)/2}" y="{y-9}" text-anchor="middle" font-size="12.5" fill="#475569">{esc(label)}</text>')


# 左：输入 query + 全长 KV cache
box(150, 150, 200, 56, "#eff6ff", ATT, [("query", 16), ("（当前 token 的查询）", 12)])
box(150, 300, 230, 74, "#fff7ed", KV, [("全长 KV cache", 15), ("L 个历史位置", 12.5), ("MLA 低秩 latent（回指 ch20）", 11.5)])

# 阶段一：轻量 Lightning Indexer
box(560, 120, 300, 130, "#faf5ff", IDX,
    [("阶段一 · Lightning Indexer", 16),
     ("低维投影打分", 13),
     ("（index_n_heads=64, head_dim=128）", 11.5),
     ("对 L 个 KV 各算一个", 12.5),
     ("「相关性代理分数」", 12.5)])
harrow(252, 408, 150, "q_li 投影 + RoPE", IDX)
harrow(267, 408, 300, "k_li / 压缩 KV", IDX)

# top-k 索引
box(560, 330, 300, 78, "#f5f3ff", IDX,
    [("选出 top-k 索引", 15),
     ("DSA：index_topk = 512", 12.5),
     ("SFA：sparse_count = 2048", 12.5)])
L.append(f'<line x1="560" y1="250" x2="560" y2="330" stroke="{IDX}" stroke-width="2.2" marker-end="url(#ar)"/>')
L.append(f'<text x="574" y="296" font-size="12.5" fill="#6d28d9">argsort 取前 k 个</text>')

# 阶段二：稀疏 flash 注意力
box(1010, 225, 290, 130, "#eff6ff", ATT,
    [("阶段二 · 稀疏注意力", 16),
     ("只对 top-k 个 KV", 13),
     ("算全精度注意力", 13),
     ("npu_sparse_flash_attention", 11.5),
     ("/ npu_sparse_attn_sharedkv", 11.5)])
harrow(712, 862, 150, None, ATT)            # query 直送阶段二
harrow(712, 862, 330, "sparse_indices = top-k", ATT)  # 索引喂内核
L.append(f'<text x="787" y="141" text-anchor="middle" font-size="12" fill="#1d4ed8">query（ql_nope, q_pe）</text>')

# 输出
box(1010, 470, 230, 56, "#f0fdf4", OUT, [("注意力输出", 15), ("→ o_proj", 12.5)])
L.append(f'<line x1="1010" y1="355" x2="1010" y2="470" stroke="{OUT}" stroke-width="2.2" marker-end="url(#ar)"/>')

# 底部复杂度对照
by = 580
L.append(f'<rect x="80" y="{by}" width="{W-160}" height="60" rx="10" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.4"/>')
L.append(f'<text x="110" y="{by+25}" font-size="14.5" font-weight="bold" fill="{TXT}">复杂度对照（单 query）：</text>')
L.append(f'<text x="110" y="{by+47}" font-size="13.5" fill="#475569">全长注意力 O(L·d) ——— 稀疏注意力 O(L·d_idx + k·d)：索引打分用极小 d_idx=128，全精度注意力被钉在常数 k 上；L ≫ k 时收益显著。</text>')

L.append('</svg>')
open("two_stage.svg", "w").write('\n'.join(L))
print("wrote two_stage.svg")
