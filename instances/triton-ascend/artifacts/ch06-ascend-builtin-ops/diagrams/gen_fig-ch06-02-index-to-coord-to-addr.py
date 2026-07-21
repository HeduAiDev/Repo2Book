#!/usr/bin/env python3
"""fig-ch06-02-index-to-coord-to-addr — tensor-flow 模板。
一次 gather_out_to_ub 的三级映射:index 值只决定 dim 轴那一维坐标,
其余维由格子自身位置给出;坐标点乘 stride 再乘元素字节数才是地址。
越界的格子在第二级就被摘掉,输出里留下的是 other,不是脏数据。

四条竖向流水线(对应 index tile 的 4 个格子),自顶向下走
「index 值 → 是否 < boundary → 源坐标 → 字节偏移 → 结果」五级。
数据取自 traces/mem_semantics.json(dossier m2 worked_example,已在 explainer 核过)。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def text_w(s, size):
    return sum(size * (1.0 if ord(ch) > 0x2E7F else 0.58) for ch in s)


def fit(s, maxw, base, floor=8.5):
    size = base
    while size > floor and text_w(s, size) > maxw:
        size -= 0.5
    return size


W, H = 1360, 900
PAD = 50
BLUE, RED, GREEN, GRAY = "#1d4ed8", "#b91c1c", "#15803d", "#94a3b8"

TITLE = "一次 gather_out_to_ub 的三级映射:index 值 → 源坐标 → 字节地址"
SUB = "index tile [[0,3],[5,1]](shape 2x2)· src 是 4x3 的 fp32 表 · src_stride=[3,1] · dim=0 · index_boundary=4"

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
     '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     f'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{RED}"/></marker></defs>',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{W/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{W/2}" y="56" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUB)}</text>']

# ── 顶部:index tile 2x2 小网格(标出行列位置) ─────────────────────────
GRID_X, GRID_Y, CELL = 560, 84, 62
INDEX_TILE = [[0, 3], [5, 1]]
L.append(f'<text x="{GRID_X-8}" y="{GRID_Y-10}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="#334155">{esc("输入:index tile(格子自身位置 = 非 dim 轴坐标)")}</text>')
for r in range(2):
    for c in range(2):
        v = INDEX_TILE[r][c]
        x, y = GRID_X + c * CELL, GRID_Y + r * CELL
        oob = v >= 4
        fill = "#fee2e2" if oob else "#eff6ff"
        stroke = RED if oob else BLUE
        L.append(f'<rect x="{x}" y="{y}" width="{CELL-4}" height="{CELL-4}" rx="6" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
        L.append(f'<text x="{x+(CELL-4)/2}" y="{y+26}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="16" font-weight="bold" '
                 f'fill="{stroke}">{esc(str(v))}</text>')
        L.append(f'<text x="{x+(CELL-4)/2}" y="{y+44}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="9" fill="#64748b">'
                 f'{esc(f"({r},{c})")}</text>')

# ── 四条竖向流水线 ─────────────────────────────────────────────────────
LANES = [
    dict(pos="(0,0)", idx=0, oob=False, coord="(0,0)", off=0, val="0.0"),
    dict(pos="(0,1)", idx=3, oob=False, coord="(3,1)", off=40, val="10.0"),
    dict(pos="(1,0)", idx=5, oob=True, coord=None, off=None, val="-1.0(other)"),
    dict(pos="(1,1)", idx=1, oob=False, coord="(1,1)", off=16, val="4.0"),
]
LANE_W, LANE_GAP, TOP = 280, 46, 240
n = len(LANES)
total_w = n * LANE_W + (n - 1) * LANE_GAP
LX0 = (W - total_w) / 2
ROW_H = 92
STAGE_LABELS = ["① index 值(dim 轴坐标)", "② < index_boundary(4) ?",
                "③ 源坐标 (row,col)", "④ 字节偏移", "⑤ 结果格子值"]
for s, lab in enumerate(STAGE_LABELS):
    L.append(f'<text x="{PAD}" y="{TOP+s*ROW_H+18}" font-family="sans-serif" font-size="11.5" '
             f'font-weight="bold" fill="#475569">{esc(lab)}</text>')

for i, ln in enumerate(LANES):
    lx = LX0 + i * (LANE_W + LANE_GAP)
    cx = lx + LANE_W / 2
    L.append(f'<text x="{cx}" y="{TOP-24}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="#0f172a">'
             f'{esc("格子 " + ln["pos"])}</text>')
    color = RED if ln["oob"] else GREEN
    stages_text = [
        (f'index = {ln["idx"]}', BLUE),
        ("否 → 摘出地址计算" if ln["oob"] else "是 → 继续算坐标", color),
        (ln["coord"] if ln["coord"] else "不算地址", color),
        (f'{ln["off"]}' if ln["off"] is not None else "—", color),
        (ln["val"], color),
    ]
    for s, (txt, tcol) in enumerate(stages_text):
        y = TOP + s * ROW_H
        box_h = ROW_H - 14
        fill = "#fee2e2" if (ln["oob"] and s >= 1) else "#f8fafc"
        L.append(f'<rect x="{lx}" y="{y}" width="{LANE_W}" height="{box_h}" rx="8" '
                 f'fill="{fill}" stroke="{tcol}" stroke-width="1.8"/>')
        fs = fit(txt, LANE_W - 24, 13)
        L.append(f'<text x="{cx}" y="{y+box_h/2+5}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="{fs}" font-weight="bold" '
                 f'fill="{tcol}">{esc(txt)}</text>')
        if s < len(stages_text) - 1:
            marker = "r" if ln["oob"] else "a"
            L.append(f'<line x1="{cx}" y1="{y+box_h}" x2="{cx}" y2="{y+ROW_H-4}" '
                     f'stroke="{tcol}" stroke-width="1.6" marker-end="url(#{marker})"/>')
    # 连接顶部 index tile 格子 -> 该 lane 首格
    r, c = int(ln["pos"][1]), int(ln["pos"][3])
    gx = GRID_X + c * CELL + (CELL - 4) / 2
    gy = GRID_Y + r * CELL + (CELL - 4)
    L.append(f'<path d="M {gx} {gy} L {gx} {TOP-40} L {cx} {TOP-40} L {cx} {TOP-4}" '
             f'fill="none" stroke="{GRAY}" stroke-width="1.2" stroke-dasharray="3,3" '
             f'marker-end="url(#a)"/>')

# ── 底部结果 tile 汇总 ─────────────────────────────────────────────────
RES_Y = TOP + len(STAGE_LABELS) * ROW_H + 20
L.append(f'<text x="{PAD}" y="{RES_Y}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="#334155">'
         f'{esc("结果 tile(与 index tile 同形,非此即彼:合法读 或 other 填充):")}</text>')
RES = [[0.0, 10.0], [-1.0, 4.0]]
RCELL = 70
RGX, RGY = PAD + 20, RES_Y + 16
for r in range(2):
    for c in range(2):
        v = RES[r][c]
        oob = (r == 1 and c == 0)
        x, y = RGX + c * RCELL, RGY + r * RCELL
        fill, stroke = ("#fee2e2", RED) if oob else ("#ecfdf5", GREEN)
        L.append(f'<rect x="{x}" y="{y}" width="{RCELL-6}" height="{RCELL-6}" rx="6" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
        L.append(f'<text x="{x+(RCELL-6)/2}" y="{y+(RCELL-6)/2+5}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="14" font-weight="bold" '
                 f'fill="{stroke}">{esc(str(v))}</text>')

foot = ("地址公式恒为 base + (Σ src_coord[d]·src_stride[d]) × 4 字节,如 (3,1) → (3×3+1)×4 = 40;"
        "index=5 越界(≥4)在第二级就被摘出地址计算,换成 other=-1.0,不产生任何脏读。")
L.append(f'<text x="{RGX+2*RCELL+30}" y="{RGY+50}" font-family="sans-serif" font-size="12" '
         f'fill="#64748b">{esc(foot[:62])}</text>')
L.append(f'<text x="{RGX+2*RCELL+30}" y="{RGY+72}" font-family="sans-serif" font-size="12" '
         f'fill="#64748b">{esc(foot[62:])}</text>')

H_ACTUAL = RGY + 2 * RCELL + 20
L.append('</svg>')
svg = '\n'.join(L)
svg = svg.replace(f'viewBox="0 0 {W} {H}"', f'viewBox="0 0 {W} {H_ACTUAL}"')
svg = svg.replace(f'<rect width="{W}" height="{H}" fill="white"/>',
                  f'<rect width="{W}" height="{H_ACTUAL}" fill="white"/>')
out = Path(__file__).with_name('fig-ch06-02-index-to-coord-to-addr.svg')
out.write_text(svg, encoding='utf-8')
print(f'wrote {out} ({W}x{H_ACTUAL})')
