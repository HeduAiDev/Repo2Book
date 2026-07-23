#!/usr/bin/env python3
"""fig-m3-split: state-table 模板。8 个 idx 为列，contMask/discMask/combined/OOB 为行。
展示 decomposeAndMask 把混合掩码拆成 contMask（安全护栏，收窄到 [0,6)）与
discMask（逐元素选择）——combined 只会落在 contMask 为真的安全范围内。
全坐标由循环计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "decomposeAndMask：contMask 收窄安全范围，discMask 逐元素选择"
SUBTITLE = "示例：BLOCK=8，contMask = (idx<6)，discMask = 运行期值谓词（真值位置 {1,3,4,6}）"

IDX = list(range(8))
CONT = [True, True, True, True, True, True, False, False]           # idx<6
DISC = [False, True, False, True, True, False, True, False]         # {1,3,4,6}
COMBINED = [c and d for c, d in zip(CONT, DISC)]                    # {1,3,4}
OOB_IF_NO_CONT = [False]*6 + [True, True]                           # {6,7}

ROW_LABELS = ["contMask (idx<6)", "discMask", "combined = cont ∧ disc", "无 contMask 全载是否 OOB"]

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 220, 78, 50, 34, 108, 30
w = PAD * 2 + LABEL_W + COL_W * len(IDX)
h = TOP + HEADER_H + ROW_H * len(ROW_LABELS) + PAD + 60
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(IDX))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROW_LABELS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+22}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# column headers = idx
for j, idx in enumerate(IDX):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
             'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="12" fill="white" '
             f'font-weight="bold">{esc(f"idx={idx}")}</text>')

BOOL_COLOR = {True: ("#ecfdf5", "#047857", "T"), False: ("#f1f5f9", "#64748b", "F")}

def draw_row(i, values_render, extra_highlight=None):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
             f'font-family="sans-serif" font-size="13" font-weight="bold" '
             f'fill="#374151">{esc(ROW_LABELS[i])}</text>')
    for j in range(len(IDX)):
        cx = col_x[j]
        text, fill, stroke, bold = values_render[j]
        L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if bold else 1}"/>')
        fw = 'font-weight="bold" ' if bold else ''
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+5}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="12.5" {fw}fill="{stroke}">{esc(text)}</text>')

# row0: contMask
vals = []
for c in CONT:
    fill, stroke, t = BOOL_COLOR[c]
    vals.append((t, fill, stroke, False))
draw_row(0, vals)

# row1: discMask
vals = []
for d in DISC:
    fill, stroke, t = BOOL_COLOR[d]
    vals.append((t, fill, stroke, False))
draw_row(1, vals)

# row2: combined (highlight selected = True)
vals = []
for cb in COMBINED:
    if cb:
        vals.append(("T ✓选中", "#dbeafe", "#1d4ed8", True))
    else:
        vals.append(("F", "#f1f5f9", "#64748b", False))
draw_row(2, vals)

# row3: OOB if no contMask
vals = []
for oob in OOB_IF_NO_CONT:
    if oob:
        vals.append(("OOB", "#fee2e2", "#b91c1c", True))
    else:
        vals.append(("安全", "#ecfdf5", "#047857", False))
draw_row(3, vals)

foot_y = h - 30
L.append(f'<rect x="{PAD}" y="{foot_y-30}" width="{w-2*PAD}" height="52" rx="8" '
         'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1"/>')
L.append(f'<text x="{PAD+16}" y="{foot_y-8}" font-family="sans-serif" font-size="12.5" '
         f'fill="#334155">{esc("不拆则全载会触碰 idx∈{6,7} 共 2 个 OOB 位置；拆后 load 用 contMask 收窄到 [0,6)，")}</text>')
L.append(f'<text x="{PAD+16}" y="{foot_y+10}" font-family="sans-serif" font-size="12.5" '
         f'fill="#334155">{esc("combined 只会在这个安全范围内选中 {1,3,4} 共 3 个，零越界。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-m3-split.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
