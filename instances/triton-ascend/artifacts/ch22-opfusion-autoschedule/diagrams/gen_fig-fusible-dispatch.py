#!/usr/bin/env python3
"""state-table 变体：isFusible 按 fusionKind_ 分派到 per-kind 兼容表。
上半：9 分支 switch 的芯片带（FusibleHelper.cpp:L557-L582），高亮 3 个本图要对比的分支。
下半：ShallowCV / MixCV / SingleCube 三种 kind 的兼容表对比（同一套流程、不同关卡开合）。
数字：9(switch分支数)/14(ShallowCV patternA×patternB 各14类)/13(=14-1，含kMatmul外13类vector)/
10(MixCV 非matmul patternA×patternB 各10类，互兼容10×10子块)/0(SingleCube 恒 false，即 0)，
均据源码 FusibleHelper.cpp。全坐标计算，零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "isFusible：按 FusionKind 分派到 per-kind 兼容表"
SUBTITLE = "同一个 switch（9 分支，FusibleHelper.cpp:L557-L582）——换一支，兼容表和结果就完全不同"

# 上：9 分支芯片带，按源码 case 顺序
CASES = ["PureElemwise", "AnyPB", "LastAxisPBR", "AnyPBR",
         "ShallowCV", "ShallowVV", "MixCV", "MixC2", "SingleCube"]
HILITE = {"ShallowCV": "#3b82f6", "MixCV": "#d97706", "SingleCube": "#64748b"}

PAD, TOP = 46, 96
CHIP_W, CHIP_H, CHIP_GAP = 118, 40, 8
n = len(CASES)
chips_w = n * CHIP_W + (n - 1) * CHIP_GAP

# 下：三栏对比表
COLS = ["ShallowCV（case 5/9）", "MixCV（case 7/9）", "SingleCube（case 9/9）"]
COL_KEY = ["ShallowCV", "MixCV", "SingleCube"]
ROW_LABELS = ["兼容表规模", "matmul 后可接", "举例：一条边判定", "isFusible 该边结果"]
CELLS = {
    "ShallowCV": [
        "14×14 类\n（patternA/patternB 各 14，含 kMatmul）",
        "全部 13 类 vector pattern\n（elementwise/reduce/broadcast…）",
        "%7 log → %9 matmul",
        "true",
    ],
    "MixCV": [
        "10×10 非 matmul 子块\n（matmul 另循独立不对称规则）",
        "仅 elementwise / zeroRank-elemwise\n（reduce/broadcast 被拒，不对称）",
        "matmul → reduce\n（被拒，返回 false）",
        "false",
    ],
    "SingleCube": [
        "无兼容表\n分支直接返回",
        "（不适用，SingleCube\n本身即单算子核）",
        "任意 patternA、patternB",
        "恒 false（0）",
    ],
}
RESULT_COLOR = {"true": ("#ecfdf5", "#047857"), "false": ("#fee2e2", "#b91c1c"),
                "恒 false（0）": ("#fee2e2", "#b91c1c")}

LABEL_W, COL_W, ROW_H, HEADER_H = 118, 300, 60, 40
table_top = TOP + CHIP_H + 78
w = PAD * 2 + max(chips_w, LABEL_W + COL_W * len(COLS))
h = table_top + HEADER_H + ROW_H * len(ROW_LABELS) + 70

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-24}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD-6}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 芯片带
chips_x0 = PAD + (max(chips_w, LABEL_W + COL_W * len(COLS)) - chips_w) / 2
L.append(f'<text x="{PAD}" y="{TOP-10}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">isFusible(patternA, patternB) 的 9 支 switch（分派到各自兼容表）</text>')
for i, name in enumerate(CASES):
    x = chips_x0 + i * (CHIP_W + CHIP_GAP)
    hi = HILITE.get(name)
    if hi:
        fill = {"#3b82f6": "#dbeafe", "#d97706": "#fef3c7", "#64748b": "#f1f5f9"}[hi]
        L.append(f'<rect x="{x}" y="{TOP}" width="{CHIP_W}" height="{CHIP_H}" rx="6" '
                  f'fill="{fill}" stroke="{hi}" stroke-width="2.5"/>')
        tf = hi
    else:
        L.append(f'<rect x="{x}" y="{TOP}" width="{CHIP_W}" height="{CHIP_H}" rx="6" '
                  f'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
        tf = "#94a3b8"
    fw_attr = 'font-weight="bold" ' if hi else ''
    L.append(f'<text x="{x+CHIP_W/2}" y="{TOP+CHIP_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="{tf}" '
              f'{fw_attr}>{esc(name)}</text>')

# 箭头：芯片带 -> 表
arrow_y1 = TOP + CHIP_H
arrow_y2 = table_top - 10
cx_all = chips_x0 + chips_w / 2
L.append(f'<line x1="{cx_all}" y1="{arrow_y1+4}" x2="{cx_all}" y2="{arrow_y2}" '
          'stroke="#64748b" stroke-width="1.6" marker-end="url(#a)"/>')
L.append(f'<text x="{cx_all}" y="{(arrow_y1+arrow_y2)/2+18}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="#64748b">'
          f'展开 3 支代表性兼容表（边界差异最大）</text>')

# 表头
table_x0 = PAD + (max(chips_w, LABEL_W + COL_W * len(COLS)) - (LABEL_W + COL_W * len(COLS))) / 2
col_x = [table_x0 + LABEL_W + i * COL_W for i in range(len(COLS))]
for j, name in enumerate(COLS):
    x = col_x[j]
    hi = HILITE[COL_KEY[j]]
    fill = {"#3b82f6": "#dbeafe", "#d97706": "#fef3c7", "#64748b": "#f1f5f9"}[hi]
    L.append(f'<rect x="{x}" y="{table_top}" width="{COL_W-10}" height="{HEADER_H}" rx="4" '
              f'fill="{fill}" stroke="{hi}" stroke-width="2"/>')
    L.append(f'<text x="{x+(COL_W-10)/2}" y="{table_top+HEADER_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="{hi}" '
              f'font-weight="bold">{esc(name)}</text>')

row_y = [table_top + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]
for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(f'<text x="{table_x0+LABEL_W-14}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    for j, key in enumerate(COL_KEY):
        cx = col_x[j]
        text = CELLS[key][i]
        is_result_row = (i == len(ROW_LABELS) - 1)
        if is_result_row and text in RESULT_COLOR:
            fill, stroke = RESULT_COLOR[text]
            L.append(f'<rect x="{cx}" y="{ry+5}" width="{COL_W-10}" height="{ROW_H-10}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
            tf = stroke
        else:
            L.append(f'<rect x="{cx}" y="{ry+5}" width="{COL_W-10}" height="{ROW_H-10}" rx="4" '
                      f'fill="#f8fafc" stroke="#e2e8f0" stroke-width="1"/>')
            tf = "#334155"
        lines = text.split("\n")
        n_lines = len(lines)
        y0 = ry + ROW_H / 2 - (n_lines - 1) * 8 + 4
        fw_attr2 = 'font-weight="bold" ' if is_result_row else ''
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-10)/2}" y="{y0+k*15}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="10.5" fill="{tf}" '
                      f'{fw_attr2}>{esc(line)}</text>')

foot_y = h - 22
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">ShallowCV 最宽松（cube 与全部 13 类 vector 互融）、MixCV 分两制（10×10 非 matmul 子块 + matmul 独立不对称规则）、'
          f'SingleCube 恒不融合——同一套判据、不同 kind 判若两表。</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-fusible-dispatch.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
