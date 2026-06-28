#!/usr/bin/env python3
"""排布代数：同一张 all_ranks 网格，沿不同维度切出三种正交的组（样例 A）。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1160, 620
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(f'<text x="{W/2}" y="42" font-family="sans-serif" font-size="25" font-weight="bold" fill="#0f172a" text-anchor="middle">一张网格，三种切法：样例 A（world=8, dp=2, tp=4）</text>')
L.append(f'<text x="{W/2}" y="72" font-family="monospace" font-size="14.5" fill="#64748b" text-anchor="middle">all_ranks = arange(8).reshape(ExternalDP=1, dp=2, pp=1, pcp=1, tp=4)  →  投影成 dp×tp = 2×4</text>')

# 网格布局：dp 是行（2 行），tp 是列（4 列）
DP, TP = 2, 4
grid = [[r * TP + c for c in range(TP)] for r in range(DP)]

cell = 58
gw = TP * cell
gh = DP * cell


def draw_grid(ox, oy, title, subtitle, groups, gcolors, axis_note):
    """groups: list of list of (r,c)；同组同色边框。"""
    L.append(f'<text x="{ox+gw/2}" y="{oy-40}" font-family="sans-serif" font-size="17" font-weight="bold" fill="#0f172a" text-anchor="middle">{esc(title)}</text>')
    L.append(f'<text x="{ox+gw/2}" y="{oy-18}" font-family="sans-serif" font-size="12.5" fill="#64748b" text-anchor="middle">{esc(subtitle)}</text>')
    # 先画每个组的底色块（外扩一点形成包围感）
    cell_group = {}
    for gi, g in enumerate(groups):
        for (r, c) in g:
            cell_group[(r, c)] = gi
    # 画单元格
    for r in range(DP):
        for c in range(TP):
            x = ox + c * cell
            y = oy + r * cell
            gi = cell_group[(r, c)]
            col = gcolors[gi % len(gcolors)]
            L.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="6" fill="{col[1]}" stroke="{col[0]}" stroke-width="2"/>')
            L.append(f'<text x="{x+cell/2}" y="{y+cell/2+7}" font-family="monospace" font-size="20" font-weight="bold" fill="{col[0]}" text-anchor="middle">{grid[r][c]}</text>')
    # 组的描边（粗框圈住同组连续单元）—— 用每组的包围盒
    for gi, g in enumerate(groups):
        col = gcolors[gi % len(gcolors)]
        rs = [p[0] for p in g]
        cs = [p[1] for p in g]
        x0 = ox + min(cs) * cell
        y0 = oy + min(rs) * cell
        x1 = ox + (max(cs) + 1) * cell
        y1 = oy + (max(rs) + 1) * cell
        L.append(f'<rect x="{x0-3}" y="{y0-3}" width="{x1-x0+6}" height="{y1-y0+6}" rx="9" fill="none" stroke="{col[0]}" stroke-width="3.5"/>')
    L.append(f'<text x="{ox+gw/2}" y="{oy+gh+30}" font-family="sans-serif" font-size="13" fill="#475569" text-anchor="middle">{esc(axis_note)}</text>')


# 三个面板横排
y_grid = 200
xs_panels = [90, 470, 850]

# 面板1：全局 TP（行向）
blue = [("#2563eb", "#eff6ff")]
draw_grid(xs_panels[0], y_grid, "基座全局 TP（行向）",
          "all_ranks.view(-1, tp=4)",
          [[(0, c) for c in range(4)], [(1, c) for c in range(4)]],
          blue * 1 + [("#2563eb", "#eff6ff"), ("#1d4ed8", "#dbeafe")],
          "每行一组 → [0,1,2,3] · [4,5,6,7]")

# 面板2：MC2 整块
draw_grid(xs_panels[1], y_grid, "MC2（整块 EP 域）",
          "transpose(1,2).reshape(-1, dp·pcp·tp=8)",
          [[(r, c) for r in range(2) for c in range(4)]],
          [("#d97706", "#fff7ed")],
          "整块一组 → [0..7]，横跨两个 DP")

# 面板3：细粒度 mlp_tp 列向
greens = [("#16a34a", "#f0fdf4"), ("#0891b2", "#ecfeff"), ("#7c3aed", "#f5f3ff"), ("#dc2626", "#fef2f2")]
draw_grid(xs_panels[2], y_grid, "细粒度 mlp_tp（列向，沿 DP）",
          "rank_grid(pp,dp,tp)，沿 DP 取每列",
          [[(0, c), (1, c)] for c in range(4)],
          greens,
          "每列一组 → [0,4]·[1,5]·[2,6]·[3,7]")

# 底部结论条
by = y_grid + gh + 70
L.append(f'<rect x="90" y="{by}" width="{W-180}" height="62" rx="12" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>')
L.append(f'<text x="{W/2}" y="{by+26}" font-family="sans-serif" font-size="15" font-weight="bold" fill="#0f172a" text-anchor="middle">三步法：把目标维 transpose 到末尾 → reshape(-1, 该维 size) → unbind 拆成组列表</text>')
L.append(f'<text x="{W/2}" y="{by+48}" font-family="sans-serif" font-size="13.5" fill="#475569" text-anchor="middle">全局 TP 行向、模块 TP 列向（沿 DP 借 rank）→ 两者正交；MC2 整块收成一个专家并行域</text>')

L.append('</svg>')
open("reshape_algebra.svg", "w", encoding="utf-8").write('\n'.join(L))
print("wrote reshape_algebra.svg")
