#!/usr/bin/env python3
"""CP 归口：PCP/DCP 不增 world，只在已有 GPU 上换维度切（样例 C）。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1080, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
         '<marker id="arr" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{W/2}" y="42" font-family="sans-serif" font-size="25" font-weight="bold" fill="#0f172a" text-anchor="middle">上下文并行归口：样例 C（world=4, tp=2, pcp=2, dcp=2）</text>')
L.append(f'<text x="{W/2}" y="72" font-family="monospace" font-size="14.5" fill="#64748b" text-anchor="middle">all_ranks = arange(4).reshape(...,pcp=2,tp=2) = [[0,1],[2,3]]  （行=pcp，列=tp）</text>')

# 2x2 网格：行 = pcp，列 = tp
NR, NC = 2, 2
grid = [[0, 1], [2, 3]]
cell = 64
gw = NC * cell
gh = NR * cell


def draw(ox, oy, title, sub, groups, gcolors, note, note2):
    L.append(f'<text x="{ox+gw/2}" y="{oy-44}" font-family="sans-serif" font-size="17" font-weight="bold" fill="#0f172a" text-anchor="middle">{esc(title)}</text>')
    L.append(f'<text x="{ox+gw/2}" y="{oy-22}" font-family="monospace" font-size="12.5" fill="#64748b" text-anchor="middle">{esc(sub)}</text>')
    cg = {}
    for gi, g in enumerate(groups):
        for v in g:
            cg[v] = gi
    for r in range(NR):
        for c in range(NC):
            v = grid[r][c]
            x = ox + c * cell
            y = oy + r * cell
            col = gcolors[cg[v] % len(gcolors)]
            L.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="7" fill="{col[1]}" stroke="{col[0]}" stroke-width="2"/>')
            L.append(f'<text x="{x+cell/2}" y="{y+cell/2+8}" font-family="monospace" font-size="22" font-weight="bold" fill="{col[0]}" text-anchor="middle">{v}</text>')
    # 组描边
    pos = {grid[r][c]: (r, c) for r in range(NR) for c in range(NC)}
    for gi, g in enumerate(groups):
        col = gcolors[gi % len(gcolors)]
        rs = [pos[v][0] for v in g]
        cs = [pos[v][1] for v in g]
        x0 = ox + min(cs) * cell
        y0 = oy + min(rs) * cell
        x1 = ox + (max(cs) + 1) * cell
        y1 = oy + (max(rs) + 1) * cell
        L.append(f'<rect x="{x0-4}" y="{y0-4}" width="{x1-x0+8}" height="{y1-y0+8}" rx="10" fill="none" stroke="{col[0]}" stroke-width="3.5"/>')
    L.append(f'<text x="{ox+gw/2}" y="{oy+gh+30}" font-family="sans-serif" font-size="13" fill="#475569" text-anchor="middle">{esc(note)}</text>')
    L.append(f'<text x="{ox+gw/2}" y="{oy+gh+50}" font-family="sans-serif" font-size="12.5" fill="#94a3b8" text-anchor="middle">{esc(note2)}</text>')


y_grid = 190
xp = [120, 480, 840]
blue = [("#2563eb", "#eff6ff"), ("#1d4ed8", "#dbeafe")]
green = [("#16a34a", "#f0fdf4"), ("#15803d", "#dcfce7")]
purp = [("#7c3aed", "#f5f3ff"), ("#6d28d9", "#ede9fe")]

# TP：行向
draw(xp[0], y_grid, "基座 TP（行向）", "view(-1, tp=2)",
     [[0, 1], [2, 3]], blue, "每行一组 → [0,1] · [2,3]", "")
# DCP：同行向，复用 TP 的卡
draw(xp[1], y_grid, "DCP（复用 TP 的卡）", "reshape(-1, dcp=2)",
     [[0, 1], [2, 3]], green, "此处 dcp==tp → 整组当 DCP", "dcp ≤ tp，把 TP 组再细分")
# PCP：列向，transpose 跳取
draw(xp[2], y_grid, "PCP（列向，跨 tp 跳取）", "transpose(3,4).reshape(-1, pcp=2)",
     [[0, 2], [1, 3]], purp, "每列一组 → [0,2] · [1,3]", "把 pcp 换到末尾再切")

# 底部：enable_cp 运行期分流
by = y_grid + gh + 78
L.append(f'<rect x="120" y="{by}" width="{W-240}" height="56" rx="12" fill="#fdf4ff" stroke="#a21caf" stroke-width="1.8"/>')
L.append(f'<text x="{W/2}" y="{by+23}" font-family="sans-serif" font-size="14.5" font-weight="bold" fill="#86198f" text-anchor="middle">CP 不增 world：只在已有的卡上换维度切。组的排布在此讲清</text>')
L.append(f'<text x="{W/2}" y="{by+44}" font-family="monospace" font-size="12.5" fill="#a21caf" text-anchor="middle">attention 篇只在运行期用 enable_cp() 决定是否经这些组做 CP all_gather</text>')

L.append('</svg>')
open("cp_layout.svg", "w", encoding="utf-8").write('\n'.join(L))
print("wrote cp_layout.svg")
