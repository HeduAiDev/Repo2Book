#!/usr/bin/env python3
"""state-table 模板:op 路由到 (aiv?,cube?) 分拣表。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK = "#0f172a"
GRAY = "#64748b"
YES = "#15803d"
YES_BG = "#dcfce7"
NO_BG = "#f1f5f9"
NO = "#94a3b8"

TITLE = "op 路由到 (aiv?, cube?):分拣规则一览"
SUBTITLE = "copy 只去 aiv、fixpipe 只去 cube;骨架 op(scf.for/scope/yield)两侧都要;sync 按自身 tcore_type;其余按 ch16 的 valueTypes(DAGScope.cpp:L164-261)"

ROWS = [
    ("dot", "valueTypes=CUBE_ONLY", False, True),
    ("addf", "valueTypes=VECTOR_ONLY", True, False),
    ("copy", "专规:CopyOp", True, False),
    ("fixpipe", "专规:FixpipeOp", False, True),
    ("scf.for / scope.scope / scf.yield", "专规:结构 op", True, True),
    ("sync_block_set(tcore_type=CUBE)", "专规:按属性", False, True),
    ("sync_block_set(tcore_type=VECTOR)", "专规:按属性", True, False),
]

LABEL_W, AIV_W, CUBE_W, RULE_W = 340, 140, 140, 300
ROW_H, HEADER_H, TOP, PAD = 50, 46, 130, 40
W = PAD * 2 + LABEL_W + AIV_W + CUBE_W + RULE_W
H = TOP + HEADER_H + ROW_H * len(ROWS) + 170

x_label = PAD
x_aiv = PAD + LABEL_W
x_cube = x_aiv + AIV_W
x_rule = x_cube + CUBE_W

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="{INK}">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+24}" font-family="sans-serif" font-size="12" fill="{GRAY}">{esc(SUBTITLE)}</text>']

# header row
HEADERS = [(x_label, LABEL_W, "op"), (x_aiv, AIV_W, "needsMoveAiv"),
           (x_cube, CUBE_W, "needsMoveCube"), (x_rule, RULE_W, "规则来源")]
for x, w, name in HEADERS:
    L.append(f'<rect x="{x}" y="{TOP}" width="{w}" height="{HEADER_H}" fill="#334155"/>')
    L.append(f'<text x="{x+w/2}" y="{TOP+HEADER_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" fill="white">{esc(name)}</text>')

for i, (op, rule, aiv, cube) in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H
    bg = "#ffffff" if i % 2 == 0 else "#f8fafc"
    L.append(f'<rect x="{PAD}" y="{ry}" width="{W-2*PAD}" height="{ROW_H}" fill="{bg}" stroke="#e2e8f0"/>')
    L.append(f'<text x="{x_label+14}" y="{ry+ROW_H/2+5}" font-family="sans-serif" font-size="12.5" '
              f'font-weight="bold" fill="{INK}">{esc(op)}</text>')
    for x, w, val in [(x_aiv, AIV_W, aiv), (x_cube, CUBE_W, cube)]:
        cxm, cym = x + w / 2, ry + ROW_H / 2
        if val:
            L.append(f'<circle cx="{cxm}" cy="{cym}" r="14" fill="{YES_BG}" stroke="{YES}" stroke-width="1.5"/>')
            L.append(f'<text x="{cxm}" y="{cym+5}" text-anchor="middle" font-family="sans-serif" '
                      f'font-size="13" font-weight="bold" fill="{YES}">是</text>')
        else:
            L.append(f'<circle cx="{cxm}" cy="{cym}" r="14" fill="{NO_BG}" stroke="{NO}" stroke-width="1.2"/>')
            L.append(f'<text x="{cxm}" y="{cym+5}" text-anchor="middle" font-family="sans-serif" '
                      f'font-size="13" fill="{NO}">否</text>')
    L.append(f'<text x="{x_rule+16}" y="{ry+ROW_H/2+5}" font-family="sans-serif" font-size="11.5" '
              f'fill="{GRAY}">{esc(rule)}</text>')

for x in [x_aiv, x_cube, x_rule]:
    L.append(f'<line x1="{x}" y1="{TOP}" x2="{x}" y2="{TOP+HEADER_H+ROW_H*len(ROWS)}" stroke="#cbd5e1"/>')

leg_y = TOP + HEADER_H + ROW_H * len(ROWS) + 40
L.append(f'<circle cx="{PAD+14}" cy="{leg_y}" r="10" fill="{YES_BG}" stroke="{YES}" stroke-width="1.3"/>')
L.append(f'<text x="{PAD+14}" y="{leg_y+4}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" font-weight="bold" fill="{YES}">是</text>')
L.append(f'<text x="{PAD+34}" y="{leg_y+4}" font-family="sans-serif" font-size="11.5" '
          f'fill="{INK}">进该 scope</text>')
L.append(f'<circle cx="{PAD+180}" cy="{leg_y}" r="10" fill="{NO_BG}" stroke="{NO}" stroke-width="1.1"/>')
L.append(f'<text x="{PAD+180}" y="{leg_y+4}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="{NO}">否</text>')
L.append(f'<text x="{PAD+200}" y="{leg_y+4}" font-family="sans-serif" font-size="11.5" '
          f'fill="{INK}">不进该 scope</text>')

CAP = "分拣表一列 aiv、一列 cube：勾两个的是骨架 op(循环/分支)，各勾一个的是计算/搬运/同步 op。这张表就是 SplitScope 双遍的输入。"
cap_y = leg_y + 42
L.append(f'<text x="{PAD}" y="{cap_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="{INK}">{esc(CAP)}</text>')

L.append('</svg>')
out = Path(__file__).parent / "fig-m12-routing-table.svg"
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
