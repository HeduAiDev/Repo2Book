#!/usr/bin/env python3
"""flow 模板：compile_hint 按值类型五路分派成属性，再由 annotation.mark 贴标——不改原 op。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

W, PAD = 1280, 40
BOX_W = 880
CX = PAD + BOX_W / 2
COL1_W, COL2_W = 190, 330
COL3_X = PAD + COL1_W + 30 + COL2_W + 30
COL3_W = BOX_W - COL1_W - 30 - COL2_W - 30

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} 900">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
     'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#047857"/></marker></defs>',
     f'<rect width="{W}" height="900" fill="white"/>',
     f'<text x="{PAD}" y="32" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">{esc("compile_hint：贴条，不是改写")}</text>',
     f'<text x="{PAD}" y="52" font-family="sans-serif" font-size="12.5" fill="#64748b">'
     f'{esc("值按 Python 类型五路分派成 MLIR 属性，再由 annotation.mark 挂到目标 handle 上——原 op 一个字节不动；均在 AST → ttir（ast_to_ttir）阶段 emit")}</text>']

LEGEND = [("#dbeafe", "#1d4ed8", "正常落属性"),
          ("#fff7ed", "#c2410c", "反直觉的坑"),
          ("#ede9fe", "#6d28d9", "仅内部入口可达"),
          ("#fee2e2", "#b91c1c", "终止：抛错")]
LEG_X = PAD + BOX_W + 24
for i, (fl, st, txt) in enumerate(LEGEND):
    ly = 74 + i * 21
    L.append(f'<rect x="{LEG_X}" y="{ly}" width="16" height="12" rx="3" '
             f'fill="{fl}" stroke="{st}" stroke-width="1.3"/>')
    L.append(f'<text x="{LEG_X+22}" y="{ly+11}" font-family="sans-serif" font-size="11" '
             f'fill="{st}">{esc(txt)}</text>')

def box(y, h, lines, fill, stroke, x=None, bw=BOX_W):
    bx = PAD if x is None else x
    L.append(f'<rect x="{bx}" y="{y}" width="{bw}" height="{h}" rx="9" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    n = len(lines)
    y0 = y + h / 2 - (n - 1) * 9 + 5
    for k, (line, small) in enumerate(lines):
        fs = 11 if small else 13
        fw = '' if small else 'font-weight="bold" '
        L.append(f'<text x="{bx+bw/2}" y="{y0+k*16}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs}" {fw}'
                  f'fill="{stroke}">{esc(line)}</text>')
    return bx, y, bw, h

def varrow(x, y1, y2):
    L.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="#334155" '
              'stroke-width="1.6" marker-end="url(#a)"/>')

y = 72
box(y, 30, [("compile_hint(handle, key, hint_val)", False)], "#e2e8f0", "#334155")
y2 = y + 30
varrow(CX, y2, y2 + 22)
L.append(f'<text x="{CX+16}" y="{y2+16}" font-family="sans-serif" font-size="10.5" '
         f'fill="#64748b">{esc("aux_ops.py:L136-L151")}</text>')

y = y2 + 22
box(y, 30, [("compile_hint_impl(...)  ——  按 type(hint_val) 五路分派  ·  aux_ops.py:L114-L133", False)],
    "#eef2ff", "#4338ca")
y2 = y + 30
varrow(CX, y2, y2 + 20)

# 5-way dispatch table
y = y2 + 20
# 前五路 = 成功分派（结果汇入 create_annotation_mark）；末行 = 终止分支，无出边
rows = [
    ("bool", "hint_val=False", "#bool<false>\n（bool 判在最前，不会掉进假值分支）", "#dbeafe", "#1d4ed8", False),
    ("假值 (falsy)", "hint_val=None / 0", "unit attr\n（整数 0 走 not hint_val 分支，不是 i32！）", "#fff7ed", "#c2410c", False),
    ("int", "hint_val=4", "4 : i32", "#dbeafe", "#1d4ed8", False),
    ("constexpr", "hint_val 是 core.constexpr 实例\n仅内部入口可达：直呼 compile_hint_impl",
     "get_str_attr(hint_val.value)\n⇒ 独立一路：落成字符串属性", "#ede9fe", "#6d28d9", False),
    ("list", "hint_val=[1, 2]", "[1, 2] : i64", "#dbeafe", "#1d4ed8", False),
    ("其余类型（如 float）", "hint_val 为 float 等其他类型",
     "ValueError: Unsupported hint value type\n✕ 终止分支：不产属性，不接 annotation.mark",
     "#fee2e2", "#b91c1c", True),
]
row_h = 54
row_gap = 10
col1_x = PAD
col2_x = PAD + COL1_W + 30
BUS_X = PAD + BOX_W + 40          # 成功属性的汇流总线，走在表格右侧
bus_centers = []
term_bottom = None
for i, (typ, ex, out, fill, stroke, terminal) in enumerate(rows):
    ry = y + i * (row_h + row_gap) + (14 if terminal else 0)
    cy = ry + row_h / 2
    dash = ' stroke-dasharray="5,4"' if terminal else ''
    L.append(f'<rect x="{col1_x}" y="{ry}" width="{COL1_W}" height="{row_h}" rx="7" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>')
    L.append(f'<text x="{col1_x+COL1_W/2}" y="{cy+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="{stroke}">{esc(typ)}</text>')
    L.append(f'<line x1="{col1_x+COL1_W}" y1="{cy}" x2="{col2_x-4}" y2="{cy}" '
              f'stroke="#94a3b8" stroke-width="1.4"{dash} marker-end="url(#a)"/>')
    L.append(f'<rect x="{col2_x}" y="{ry}" width="{COL2_W}" height="{row_h}" rx="7" '
              f'fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.2"{dash}/>')
    ex_lines = ex.split("\n")
    ey0 = cy - (len(ex_lines) - 1) * 8 + 4
    for k, el in enumerate(ex_lines):
        L.append(f'<text x="{col2_x+COL2_W/2}" y="{ey0+k*15}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11.5" fill="#334155">{esc(el)}</text>')
    L.append(f'<line x1="{col2_x+COL2_W}" y1="{cy}" x2="{COL3_X-4}" y2="{cy}" '
              f'stroke="#94a3b8" stroke-width="1.4"{dash} marker-end="url(#a)"/>')
    L.append(f'<rect x="{COL3_X}" y="{ry}" width="{COL3_W}" height="{row_h}" rx="7" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>')
    out_lines = out.split("\n")
    oy0 = cy - (len(out_lines) - 1) * 8 + 4
    for k, ol in enumerate(out_lines):
        L.append(f'<text x="{COL3_X+COL3_W/2}" y="{oy0+k*15}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" fill="{stroke}">{esc(ol)}</text>')
    if terminal:
        term_bottom = ry + row_h
    else:
        bus_centers.append(cy)
        L.append(f'<line x1="{COL3_X+COL3_W}" y1="{cy}" x2="{BUS_X}" y2="{cy}" '
                  'stroke="#047857" stroke-width="1.4"/>')
        L.append(f'<circle cx="{BUS_X}" cy="{cy}" r="3" fill="#047857"/>')

# 汇流：五路成功属性 → create_annotation_mark（ValueError 行不接出边）
y_conv = term_bottom + 34
y_mark = y_conv + 38
L.append(f'<polyline points="{BUS_X},{bus_centers[0]} {BUS_X},{y_conv} {CX},{y_conv} {CX},{y_mark-4}" '
          'fill="none" stroke="#047857" stroke-width="2" marker-end="url(#g)"/>')
L.append(f'<text x="{BUS_X-8}" y="{y_conv-10}" text-anchor="end" font-family="sans-serif" '
          f'font-size="11" fill="#047857">{esc("五路成功分派出的属性在此汇入")}</text>')

y = y_mark
box(y, 48, [("create_annotation_mark(handle, key, attr)", False),
            ("→ annotation::MarkOp + setAttr", True)], "#ecfdf5", "#047857")
y2 = y + 48
L.append(f'<text x="{CX}" y="{y2+18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#4d7c0f">{esc("ascend_ir.cc:L597-L603——落地的是旁挂标记 op，原 op 不改")}</text>')

# side notes: 两个入口的不对称（SIMT 门控 / constexpr 可达性）
SIDE_H = 98
side_y = y2 + 40
L.append(f'<rect x="{PAD}" y="{side_y}" width="{BOX_W}" height="{SIDE_H}" rx="9" '
          'fill="#f8fafc" stroke="#94a3b8" stroke-width="1.3" stroke-dasharray="5,4"/>')
L.append(f'<text x="{PAD+18}" y="{side_y+22}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#475569">'
          f'{esc("两个入口并不等价：compile_hint（公开）严于 compile_hint_impl（内部）")}</text>')
for k, (txt, col) in enumerate([
        ("SIMT 模式下 compile_hint 早退，0 个 mark；但 multibuffer 直呼 compile_hint_impl，仍发 1 个 hivm.multi_buffer=2:i32", "#475569"),
        ("——生效中的门控只写在 compile_hint 里（aux_ops.py:L137-L139）；compile_hint_impl 里那份是注释掉的 FIXME（L115-L118），不拦任何东西", "#c2410c"),
        ("——constexpr 一路同理：公开入口 L146-L150 先把它解成裸 str，impl 没有 str 分支 ⇒ ValueError；只有直呼 impl 才走得到 get_str_attr", "#6d28d9")]):
    L.append(f'<text x="{PAD+18}" y="{side_y+42+k*19}" font-family="sans-serif" font-size="11" '
              f'fill="{col}">{esc(txt)}</text>')

foot_y = side_y + SIDE_H + 30
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">'
          f'{esc("五路分派 + 旁挂标记：compile_hint 只负责“告诉编译器一件事”，从不改动被提示的那个 op 本身。")}</text>')
h_final = foot_y + 20
L[0] = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h_final}">'
L[2] = f'<rect width="{W}" height="{h_final}" fill="white"/>'
L.append('</svg>')
out = Path(__file__).with_name("fig-ch08-m14-compile-hint-dispatch.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
