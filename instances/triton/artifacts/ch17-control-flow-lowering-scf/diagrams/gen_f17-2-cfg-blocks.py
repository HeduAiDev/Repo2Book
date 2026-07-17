#!/usr/bin/env python3
"""f17-2-cfg-blocks: 带 return 的 if 下降成非结构化 CFG(cf.cond_br + 手写 φ 汇合)。
实按 traces/ch17_traces.json -> ir.K3_if_return_cfg 的真实基本块画:
bb0(entry,cond_br)->bb1(then,tt.return 离场)/bb2(else,cf.br bb4);
bb3(no predecessors 死块,cf.br bb4);bb4(2 preds 汇合)。
右侧红线边注框说明 endif_block 通常还会挂块参数(本例分支只写内存,无变量要合并)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

def box(cx, cy, w, h, fill, stroke, lines, fs=12.5, bold_first=True, dashed=False, sw=1.6):
    x, y = cx - w / 2, cy - h / 2
    dash = ' stroke-dasharray="6,4"' if dashed else ''
    out = [f'<rect x="{x:.1f}" y="{y:.1f}" width="{w}" height="{h}" rx="9" '
           f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{dash}/>']
    n = len(lines)
    y0 = cy - (n - 1) * 8
    for k, line in enumerate(lines):
        fw = 'font-weight="bold" ' if (bold_first and k == 0) else ''
        out.append(f'<text x="{cx:.1f}" y="{y0 + k*16:.1f}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="{fs}" fill="#0f172a" {fw}>{esc(line)}</text>')
    return out

def arrow(x1, y1, x2, y2, color="#334155", label=None, lx=None, ly=None, sw=1.6, dashed=False):
    dash = ' stroke-dasharray="6,4"' if dashed else ''
    out = [f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
           f'stroke="{color}" stroke-width="{sw}" marker-end="url(#a)"{dash}/>']
    if label:
        tx = lx if lx is not None else (x1 + x2) / 2
        ty = ly if ly is not None else (y1 + y2) / 2 - 6
        out.append(f'<text x="{tx:.1f}" y="{ty:.1f}" text-anchor="middle" '
                    f'font-family="sans-serif" font-size="11" fill="{color}">{esc(label)}</text>')
    return out

BOX_W, BOX_H = 260, 76
PAD, TOP, VGAP, HGAP = 40, 76, 62, 90
NOTE_W = 300

BB0_CX = PAD + BOX_W + HGAP / 2 + 20
BB0_Y = TOP
BB1_X = PAD
BB2_X = BB0_CX * 2 - PAD - BOX_W  # 对称位置由 BB0_CX 与 BB1_X 推出
ROW2_Y = BB0_Y + BOX_H + VGAP
BB3_Y = ROW2_Y
BB3_X = BB2_X + BOX_W + HGAP + 40  # bb3 单独放右侧,和 bb2 同一行,隔开表示"无前驱"
ROW3_Y = ROW2_Y + BOX_H + VGAP + 20
BB4_CX = (BB1_X + BB2_X) / 2 + BOX_W / 2

DIAGRAM_W = BB3_X + BOX_W + PAD
W = DIAGRAM_W + NOTE_W + PAD
H = ROW3_Y + BOX_H + PAD + 46

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="16" font-weight="bold" '
          f'fill="#0f172a">{esc("带 return 的 if 下降成非结构化 CFG(实按 K3_if_return_cfg 真实基本块)")}</text>')
L.append(f'<text x="{PAD}" y="50" font-family="sans-serif" font-size="12" '
          f'fill="#475569">{esc("cf.cond_br 分岔 -> then 直接离场 / else+死块汇入 endif;没有单一出口")}</text>')

# bb0 entry
L += box(BB0_CX, BB0_Y, BOX_W, BOX_H, "#e2e8f0", "#475569",
          ["^bb0 (entry)", "%0 = arith.cmpi sgt", "cf.cond_br %0, ^bb1, ^bb2"], fs=11.5)

# bb1 then (死端,tt.return)
bb1_cy = ROW2_Y
L += box(BB1_X + BOX_W/2, bb1_cy, BOX_W, BOX_H, "#fee2e2", "#b91c1c",
          ["^bb1  // pred: ^bb0", "store 1; tt.return", "(直接离场,无后继)"], fs=11.5)

# bb2 else
bb2_cy = ROW2_Y
L += box(BB2_X + BOX_W/2, bb2_cy, BOX_W, BOX_H, "#dbeafe", "#1d4ed8",
          ["^bb2  // pred: ^bb0", "cf.br ^bb4"], fs=11.5)

# bb3 死块 (no predecessors)
bb3_cy = ROW2_Y
L += box(BB3_X + BOX_W/2, bb3_cy, BOX_W, BOX_H, "#f1f5f9", "#94a3b8",
          ["^bb3:  // no predecessors", "cf.br ^bb4", "(死块,不可达)"], fs=11.5, dashed=True)

# bb4 汇合
bb4_cy = ROW3_Y
L += box(BB4_CX, bb4_cy, BOX_W, BOX_H, "#fef3c7", "#d97706",
          ["^bb4:  // 2 preds: ^bb2, ^bb3", "store 2; tt.return", "(endif 汇合块)"], fs=11.5)

# 箭头
L += arrow(BB0_CX - BOX_W*0.22, BB0_Y + BOX_H, BB1_X + BOX_W/2, bb1_cy - BOX_H/2, "#1d4ed8", "cond=真")
L += arrow(BB0_CX + BOX_W*0.22, BB0_Y + BOX_H, BB2_X + BOX_W/2, bb2_cy - BOX_H/2, "#334155", "cond=假")
L += arrow(BB2_X + BOX_W/2, bb2_cy + BOX_H/2, BB4_CX - BOX_W*0.2, bb4_cy - BOX_H/2, "#334155", "cf.br")
L += arrow(BB3_X + BOX_W/2, bb3_cy + BOX_H/2, BB4_CX + BOX_W*0.2, bb4_cy - BOX_H/2, "#94a3b8", "cf.br", dashed=True)

# bb1 死端标注(无箭头进 bb4)
L.append(f'<text x="{BB1_X + BOX_W/2}" y="{bb1_cy + BOX_H/2 + 22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" fill="#b91c1c">{esc("无边到 ^bb4(then 已 return 离场)")}</text>')

# 右侧红线边注框
note_x = DIAGRAM_W + 10
note_y = ROW2_Y - 10
note_h = ROW3_Y + BOX_H - note_y
L.append(f'<rect x="{note_x}" y="{note_y}" width="{NOTE_W-20}" height="{note_h}" rx="10" '
          'fill="#fef2f2" stroke="#dc2626" stroke-width="1.6" stroke-dasharray="5,3"/>')
note_lines = [
    "红线:",
    "endif_block 通常还会",
    "add_argument()——分支各自",
    "写了同名变量时,汇合块用",
    "块参数接住(手写 φ,",
    "L634, L646-L653)。",
    "本例分支只写内存、没有",
    "变量要合并,故 ^bb4 本身",
    "没有块参数,不影响 CFG。",
]
for i, t in enumerate(note_lines):
    fw = 'bold' if i == 0 else 'normal'
    L.append(f'<text x="{note_x+16}" y="{note_y+24+i*20}" font-family="sans-serif" '
              f'font-size="11.5" font-weight="{fw}" fill="#7f1d1d">{esc(t)}</text>')

foot_y = H - 22
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("Triton v3.2.0 headless 实测:ir.K3_if_return_cfg;op_counts.K3_if_return_cfg.cf.br=2")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("f17-2-cfg-blocks.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} size={W}x{H}")
