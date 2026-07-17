#!/usr/bin/env python3
"""fig-m6-transpose: before-after 模板。同一 Blocked 布局的两种转置读法——
左: tensor 视角(元素 -> 线程), 右: warp 视角(线程 -> 元素)。互为逆映射,
不重算, 只换 -use-hw-view 遍历方向。数字全部来自
explainer/traces/layout_decode.txt(两段真实输出)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PANEL_W, PAD, TOP = 430, 40, 96
GAP_MID = 60

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {2*PANEL_W + 2*PAD + GAP_MID} 560">']
w = 2 * PANEL_W + 2 * PAD + GAP_MID
h = 560
L[0] = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">'
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
          'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker></defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

L.append(f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="16" '
         f'font-weight="bold" fill="#0f172a">'
         f'{esc("同一份映射, 两种转置读法(-use-hw-view 切换)")}</text>')

left_x = PAD
right_x = PAD + PANEL_W + GAP_MID

# ---- 左面板: tensor 视角(问: 元素(4,0)在谁手里?) ----
L.append(f'<text x="{left_x}" y="58" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#1e3a8a">'
         f'{esc("tensor 视角:元素 (4,0) 在谁手里?")}</text>')
L.append(f'<rect x="{left_x}" y="70" width="{PANEL_W}" height="26" rx="6" '
         f'fill="#eef2ff" stroke="#6366f1"/>')
L.append(f'<text x="{left_x+10}" y="88" font-family="sans-serif" font-size="12" '
         f'fill="#1e3a8a">{esc("按 tensor 下标行主序遍历 64 格,印 elementMapping[i]")}</text>')

grid_top = 112
rows_tensor = [
    "T0:0 … T7:0",
    "T8:0 … T15:0",
    "T16:0 … T23:0",
    "T24:0 … T31:0",
    "T32:0 … T39:0",
    "T40:0 … T47:0",
    "T48:0 … T55:0",
    "T56:0 … T63:0",
]
ROW_H = 30
for i, txt in enumerate(rows_tensor):
    y = grid_top + i * ROW_H
    hl = (i == 4)  # 第 5 行 = row(4,*) -> 元素 (4,0) 所在行
    L.append(f'<rect x="{left_x}" y="{y}" width="{PANEL_W}" height="{ROW_H-4}" rx="5" '
              f'fill="{"#fef3c7" if hl else "#f8fafc"}" '
              f'stroke="{"#d97706" if hl else "#cbd5e1"}" stroke-width="{2 if hl else 1}"/>')
    L.append(f'<text x="{left_x+10}" y="{y+ROW_H/2+1}" font-family="sans-serif" '
              f'font-size="12" fill="#334155">{esc(f"行{i}: ")}</text>')
    L.append(f'<text x="{left_x+58}" y="{y+ROW_H/2+1}" font-family="sans-serif" '
              f'font-size="12" font-weight="{"bold" if hl else "normal"}" '
              f'fill="{"#78350f" if hl else "#334155"}">{esc(txt)}</text>')
lbl_y = grid_top + 4 * ROW_H + ROW_H / 2 + 1
L.append(f'<text x="{left_x+PANEL_W-8}" y="{lbl_y}" text-anchor="end" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" '
          f'fill="#78350f">{esc("← 元素(4,0)首格 = T32:0")}</text>')

# ---- 右面板: warp 视角(问: Warp1 的 32 条 lane 各拿哪个?) ----
L.append(f'<text x="{right_x}" y="58" font-family="sans-serif" font-size="13" '
         f'font-weight="bold" fill="#92400e">'
         f'{esc("warp 视角:Warp1 的 32 条 lane 各拿哪个?")}</text>')
L.append(f'<rect x="{right_x}" y="70" width="{PANEL_W}" height="26" rx="6" '
         f'fill="#fff7ed" stroke="#ea580c"/>')
L.append(f'<text x="{right_x+10}" y="88" font-family="sans-serif" font-size="12" '
         f'fill="#7c2d12">{esc("外层按 Block/Warp 分节,内层印 threadMapping[...]")}</text>')

L.append(f'<rect x="{right_x}" y="{grid_top}" width="{PANEL_W}" height="{ROW_H-4}" rx="5" '
          f'fill="#dbeafe" stroke="#3b82f6"/>')
L.append(f'<text x="{right_x+10}" y="{grid_top+ROW_H/2+1}" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#1e3a8a">'
          f'{esc("Warp0: (0,0),(0,1),…,(3,7)  — 32 个坐标, 覆盖 tensor 上半 rows0-3")}</text>')

hl_y = grid_top + ROW_H
L.append(f'<rect x="{right_x}" y="{hl_y}" width="{PANEL_W}" height="{ROW_H-4}" rx="5" '
          f'fill="#fef3c7" stroke="#d97706" stroke-width="2"/>')
L.append(f'<text x="{right_x+10}" y="{hl_y+ROW_H/2+1}" font-family="sans-serif" '
          f'font-size="12" font-weight="bold" fill="#78350f">'
          f'{esc("Warp1: (4,0),(4,1),…,(7,7)  — 32 个坐标, 覆盖 tensor 下半 rows4-7")}</text>')

L.append(f'<text x="{right_x}" y="{hl_y+ROW_H+34}" font-family="sans-serif" font-size="12" '
          f'fill="#7c2d12">'
          f'{esc("Warp1 首坐标 (4,0) — 与左图元素(4,0)→T32 互为逆映射")}</text>')

# ---- 中间互逆箭头 ----
mid_y = grid_top + 4 * ROW_H + ROW_H / 2
L.append(f'<line x1="{left_x+PANEL_W+6}" y1="{mid_y-10}" x2="{right_x-6}" y2="{mid_y-10}" '
          'stroke="#d97706" stroke-width="2.2" marker-end="url(#a)"/>')
L.append(f'<line x1="{right_x-6}" y1="{mid_y+10}" x2="{left_x+PANEL_W+6}" y2="{mid_y+10}" '
          'stroke="#d97706" stroke-width="2.2" marker-end="url(#a)"/>')
L.append(f'<text x="{(left_x+PANEL_W+right_x)/2}" y="{mid_y-16}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#92400e">'
          f'{esc("互逆")}</text>')

# ---- 底部结论 ----
left_bottom = grid_top + len(rows_tensor) * ROW_H
note_y = left_bottom + 46
L.append(f'<text x="{PAD}" y="{note_y}" font-family="sans-serif" font-size="13" '
          f'fill="#0f172a">'
          f'{esc("同一次四重循环一次性建好 elementMapping 与 threadMapping,两视角只换遍历方向, 不重算 LinearLayout::apply。")}</text>')

L.append('</svg>')
out = Path(__file__).parent / "fig-m6-transpose.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
