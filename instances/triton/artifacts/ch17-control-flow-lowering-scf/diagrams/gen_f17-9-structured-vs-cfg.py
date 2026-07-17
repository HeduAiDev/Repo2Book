#!/usr/bin/env python3
"""f17-9-structured-vs-cfg: 控制流写法决定下降路径与可优化性——左=结构化 scf.for
(挂 num_stages,能被流水线 pass 吃),右=带 return 的 cf.cond_br CFG(非结构化,吃不了)。
底部加"循环内 return"编译期直接 raise 的第三种结局。before-after 双面板 + 底部结局条。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

PANEL_W, PAD, TOP = 420, 44, 106
BOX_H = 56
w = PAD * 2 + PANEL_W * 2 + 90
h = TOP + 3 * (BOX_H + 22) + 190

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']
L.append(f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="16" font-weight="bold" '
          f'fill="#0f172a">{esc("控制流写法决定下降路径与可优化性")}</text>')
L.append(f'<text x="{PAD}" y="50" font-family="sans-serif" font-size="12" '
          f'fill="#475569">{esc("分派判据:contains_return + scf_stack(code_generator.py:L688-L697)")}</text>')

lx = PAD
rx = PAD + PANEL_W + 90
cx_l = lx + PANEL_W / 2
cx_r = rx + PANEL_W / 2

L.append(f'<text x="{cx_l}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#15803d">{esc("结构化路径:scf.for")}</text>')
L.append(f'<text x="{cx_r}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" font-weight="bold" fill="#b91c1c">{esc("非结构化路径:带 return 的 cf CFG")}</text>')

steps_l = [
    ("scf.for { ... }", False),
    ("{ tt.num_stages = 3,", False),
    ("  tt.loop_unroll_factor = 2 }", True),
]
steps_r = [
    ("cf.cond_br = 1", False),
    ("^bbN: // no predecessors", True),
    ("(死块=1,无 num_stages 可挂)", False),
]
y0 = TOP
for i in range(3):
    yl = y0 + i * (BOX_H + 22)
    ltext, lhot = steps_l[i]
    fill_l = "#dcfce7" if lhot else "#f0fdf4"
    stroke_l = "#15803d"
    L.append(f'<rect x="{lx}" y="{yl}" width="{PANEL_W}" height="{BOX_H}" rx="9" '
              f'fill="{fill_l}" stroke="{stroke_l}" stroke-width="{2 if lhot else 1.3}"/>')
    L.append(f'<text x="{cx_l}" y="{yl+BOX_H/2+5}" text-anchor="middle" font-family="monospace" '
              f'font-size="13" fill="#0f172a">{esc(ltext)}</text>')

    rtext, rhot = steps_r[i]
    fill_r = "#fee2e2" if rhot else "#fef2f2"
    stroke_r = "#b91c1c"
    L.append(f'<rect x="{rx}" y="{yl}" width="{PANEL_W}" height="{BOX_H}" rx="9" '
              f'fill="{fill_r}" stroke="{stroke_r}" stroke-width="{2 if rhot else 1.3}"/>')
    L.append(f'<text x="{cx_r}" y="{yl+BOX_H/2+5}" text-anchor="middle" font-family="monospace" '
              f'font-size="12.5" fill="#0f172a">{esc(rtext)}</text>')
    if i < 2:
        L.append(f'<line x1="{cx_l}" y1="{yl+BOX_H}" x2="{cx_l}" y2="{yl+BOX_H+18}" '
                  'stroke="#15803d" stroke-width="1.4" marker-end="url(#a)"/>')
        L.append(f'<line x1="{cx_r}" y1="{yl+BOX_H}" x2="{cx_r}" y2="{yl+BOX_H+18}" '
                  'stroke="#b91c1c" stroke-width="1.4" marker-end="url(#a)"/>')

y_note = y0 + 3 * (BOX_H + 22) - 10
L.append(f'<text x="{cx_l}" y="{y_note}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#15803d">{esc("流水线 pass 能直接吃(第 29/30 章)")}</text>')
L.append(f'<text x="{cx_r}" y="{y_note}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#b91c1c">{esc("结构化 pass 吃不了这条 CFG")}</text>')

# 第三种结局:循环内 return
third_y = y_note + 40
third_w = PANEL_W * 2 + 90
L.append(f'<rect x="{lx}" y="{third_y}" width="{third_w}" height="60" rx="10" '
          'fill="#fef3c7" stroke="#b45309" stroke-width="1.8" stroke-dasharray="6,4"/>')
L.append(f'<text x="{lx+third_w/2}" y="{third_y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="#92400e">'
          f'{esc("第三种结局:循环内 return -> 编译期直接 raise(根本无法下降)")}</text>')
L.append(f'<text x="{lx+third_w/2}" y="{third_y+44}" text-anchor="middle" font-family="monospace" '
          f'font-size="11.5" fill="#92400e">{esc("K7_return_in_for_error: Cannot have return statements inside while/for")}</text>')

foot_y = third_y + 60 + 34
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("把 return 写进循环/分支,等于亲手把这段代码挪出结构化优化的射程")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="10.5" '
          f'fill="#64748b">{esc("Triton v3.2.0 headless 实测:ir.K1_for_range_attrs / op_counts.K3_if_return_cfg / ir.K7_return_in_for_error")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("f17-9-structured-vs-cfg.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} size={w}x{h}")
