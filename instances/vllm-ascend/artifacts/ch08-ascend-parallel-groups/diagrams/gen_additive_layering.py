#!/usr/bin/env python3
"""加法式扩展 vs patch 重绑：基座组在底，昇腾组在上叠加，共用一条工厂接缝。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1260, 660
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
         '<marker id="arr" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
         '<marker id="arrUp" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#0891b2"/></marker>'
         '</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{W/2}" y="44" font-family="sans-serif" font-size="25" font-weight="bold" fill="#0f172a" text-anchor="middle">加法式扩展：昇腾组叠在基座组之上，不替换、不重绑</text>')
L.append(f'<text x="{W/2}" y="74" font-family="sans-serif" font-size="14.5" fill="#64748b" text-anchor="middle">两层都调同一个基座工厂 init_model_parallel_group 造 GroupCoordinator——这是「复用」的接缝</text>')

# ---- 区域参数 ----
panel_x = 60
panel_w = 880
top_y = 110
top_h = 190
gap = 56
bot_y = top_y + top_h + gap
bot_h = 175

def group_boxes(x0, y0, total_w, items, col, label_col):
    """在一条横带里均匀排开若干小框。"""
    n = len(items)
    inner_pad = 18
    bw_gap = 14
    avail = total_w - 2 * inner_pad - (n - 1) * bw_gap
    bw = avail / n
    bh = 46
    by = y0
    for i, it in enumerate(items):
        bx = x0 + inner_pad + i * (bw + bw_gap)
        L.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="8" fill="white" stroke="{col}" stroke-width="2"/>')
        L.append(f'<text x="{bx+bw/2}" y="{by+28}" font-family="monospace" font-size="14" font-weight="bold" fill="{label_col}" text-anchor="middle">{esc(it)}</text>')
    return bw

# ---- 上层：昇腾 ----
L.append(f'<rect x="{panel_x}" y="{top_y}" width="{panel_w}" height="{top_h}" rx="14" fill="#eff6ff" stroke="#2563eb" stroke-width="2.5"/>')
L.append(f'<text x="{panel_x+22}" y="{top_y+30}" font-family="monospace" font-size="16" font-weight="bold" fill="#1d4ed8">init_ascend_model_parallel( )</text>')
L.append(f'<text x="{panel_x+22}" y="{top_y+52}" font-family="sans-serif" font-size="13" fill="#64748b">昇腾专属组：从同一张 all_ranks 网格沿不同维度切出</text>')
group_boxes(panel_x, top_y+72, panel_w, ["_MC2", "_OTP", "_LMTP", "_EMBED_TP", "_MLP_TP"], "#2563eb", "#1d4ed8")
group_boxes(panel_x, top_y+128, panel_w, ["_FLASHCOMM2_OTP", "_FLASHCOMM2_ODP", "_FC3_QUANT_X", "_DYNAMIC_EPLB"], "#7c3aed", "#6d28d9")

# ---- 下层：基座 ----
L.append(f'<rect x="{panel_x}" y="{bot_y}" width="{panel_w}" height="{bot_h}" rx="14" fill="#f0fdf4" stroke="#16a34a" stroke-width="2.5"/>')
L.append(f'<text x="{panel_x+22}" y="{bot_y+30}" font-family="monospace" font-size="16" font-weight="bold" fill="#15803d">initialize_model_parallel( )  （基座，先建好）</text>')
L.append(f'<text x="{panel_x+22}" y="{bot_y+52}" font-family="sans-serif" font-size="13" fill="#64748b">基座 TP/PP/DP/CP 组——昇腾一个都不动</text>')
group_boxes(panel_x, bot_y+72, panel_w, ["_TP", "_PP", "_DP", "_PCP", "_DCP"], "#16a34a", "#15803d")

# ---- 中缝接箭头 ----
seam_y = top_y + top_h + gap/2
L.append(f'<rect x="{panel_x+panel_w/2-200}" y="{seam_y-19}" width="400" height="38" rx="19" fill="#ecfeff" stroke="#0891b2" stroke-width="2"/>')
L.append(f'<text x="{panel_x+panel_w/2}" y="{seam_y+5}" font-family="monospace" font-size="14.5" font-weight="bold" fill="#0e7490" text-anchor="middle">init_model_parallel_group → GroupCoordinator</text>')
# 上层 → 接缝
L.append(f'<line x1="{panel_x+260}" y1="{top_y+top_h}" x2="{panel_x+260}" y2="{seam_y-19}" stroke="#0891b2" stroke-width="2" marker-end="url(#arrUp)"/>')
# 接缝 → 下层工厂
L.append(f'<line x1="{panel_x+260}" y1="{seam_y+19}" x2="{panel_x+260}" y2="{bot_y}" stroke="#0891b2" stroke-width="2" marker-end="url(#arrUp)"/>')

# ---- 右侧 worker 时序 ----
tx = panel_x + panel_w + 40
ty0 = top_y + 10
L.append(f'<text x="{tx+109}" y="{ty0-6}" font-family="sans-serif" font-size="14" font-weight="bold" fill="#0f172a" text-anchor="middle">worker 初始化时序</text>')
steps = [
    ("①", "init_distributed", "_environment(hccl)", "#64748b"),
    ("②", "ensure_model_", "parallel_initialized", "#16a34a"),
    ("③", "init_ascend_model", "_parallel（叠加）", "#2563eb"),
]
sw = 218
sy = ty0 + 24
sh = 96
for k, (no, l1, l2, col) in enumerate(steps):
    by = sy + k * (sh + 18)
    L.append(f'<rect x="{tx}" y="{by}" width="{sw}" height="{sh}" rx="10" fill="white" stroke="{col}" stroke-width="2"/>')
    L.append(f'<text x="{tx+24}" y="{by+36}" font-family="sans-serif" font-size="20" font-weight="bold" fill="{col}" text-anchor="middle">{esc(no)}</text>')
    L.append(f'<text x="{tx+44}" y="{by+30}" font-family="monospace" font-size="12" fill="#334155">{esc(l1)}</text>')
    L.append(f'<text x="{tx+44}" y="{by+52}" font-family="monospace" font-size="12" fill="#334155">{esc(l2)}</text>')
    if k < len(steps) - 1:
        L.append(f'<line x1="{tx+sw/2}" y1="{by+sh}" x2="{tx+sw/2}" y2="{by+sh+18}" stroke="#475569" stroke-width="2" marker-end="url(#arr)"/>')

L.append('</svg>')
open("additive_layering.svg", "w", encoding="utf-8").write('\n'.join(L))
print("wrote additive_layering.svg")
