#!/usr/bin/env python3
"""flow 模板(定制):backend.add_stages 把空 stages 字典按插入序填成 5 级有序字典——
全书降级链骨架。左:调用前(空字典)。右:调用后(5 行,末行契约不同→高亮)。
改造点:ROWS(ir_name, 函数名, 返回类型)。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

ROWS = [
    ("ttir", "make_ttir(module, metadata)", "str", False),
    ("ttgir", "make_ttgir(module, metadata)", "str", False),
    ("llir", "make_llir(module, metadata)", "str", False),
    ("ptx", "make_ptx(module, metadata)", "str", False),
    ("cubin", "make_cubin(module, metadata)", "bytes", True),
]

LEFT_W, LEFT_H = 220, 110
ROW_W, ROW_H, ROW_GAP = 560, 46, 10
KEY_W = 90
PAD, TOP = 50, 110
n = len(ROWS)
RIGHT_X = PAD + LEFT_W + 300
w = RIGHT_X + ROW_W + PAD
h = TOP + n * (ROW_H + ROW_GAP) - ROW_GAP + PAD + 130

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker>'
     '</defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="38" font-family="sans-serif" font-size="17" font-weight="bold" '
     f'fill="#0f172a">{esc("backend.add_stages 把空 stages 字典按插入序填成 5 级有序字典——全书降级链的骨架")}</text>',
     f'<text x="{PAD}" y="62" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc("compile() 只递一个空字典;不同后端的 add_stages 填不同内容(本图=CUDABackend)")}</text>']

# 左:调用前
left_y = TOP + (n * (ROW_H + ROW_GAP) - ROW_GAP) / 2 - LEFT_H / 2
L.append(f'<rect x="{PAD}" y="{left_y}" width="{LEFT_W}" height="{LEFT_H}" rx="10" '
          'fill="#f1f5f9" stroke="#94a3b8" stroke-width="1.6" stroke-dasharray="5,4"/>')
L.append(f'<text x="{PAD+LEFT_W/2}" y="{left_y+34}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#475569">{esc("调用前")}</text>')
L.append(f'<text x="{PAD+LEFT_W/2}" y="{left_y+62}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14" fill="#334155">{esc("stages = {}")}</text>')
L.append(f'<text x="{PAD+LEFT_W/2}" y="{left_y+86}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#94a3b8">{esc("compile() 造的空有序字典")}</text>')

# 中:箭头 + 调用标签
arr_y = TOP + (n * (ROW_H + ROW_GAP) - ROW_GAP) / 2
L.append(f'<line x1="{PAD+LEFT_W+8}" y1="{arr_y}" x2="{RIGHT_X-8}" y2="{arr_y}" '
          'stroke="#334155" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{(PAD+LEFT_W+RIGHT_X)/2}" y="{arr_y-14}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" '
          f'fill="#334155">{esc("backend.add_stages(stages)")}</text>')
L.append(f'<text x="{(PAD+LEFT_W+RIGHT_X)/2}" y="{arr_y+22}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10.5" '
          f'fill="#64748b">{esc("按序 stages[ir_name]=pass_fn")}</text>')

# 右:调用后 —— 5 行
L.append(f'<text x="{RIGHT_X}" y="{TOP-14}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#475569">{esc("调用后:stages(有序,共 5 项)")}</text>')
for i, (key, fn, ret, hot) in enumerate(ROWS):
    y = TOP + i * (ROW_H + ROW_GAP)
    fill, stroke = ("#fef3c7", "#b45309") if hot else ("#dbeafe", "#1d4ed8")
    L.append(f'<rect x="{RIGHT_X}" y="{y}" width="{ROW_W}" height="{ROW_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if hot else 1.4}"/>')
    L.append(f'<rect x="{RIGHT_X}" y="{y}" width="{KEY_W}" height="{ROW_H}" rx="8" '
              f'fill="{stroke}"/>')
    L.append(f'<rect x="{RIGHT_X+KEY_W/2}" y="{y}" width="{KEY_W/2}" height="{ROW_H}" fill="{stroke}"/>')
    L.append(f'<text x="{RIGHT_X+KEY_W/2}" y="{y+ROW_H/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="white">{esc(key)}</text>')
    L.append(f'<text x="{RIGHT_X+KEY_W+18}" y="{y+ROW_H/2+5}" font-family="sans-serif" '
              f'font-size="12.5" fill="{"#92400e" if hot else "#0f172a"}">{esc(fn)}</text>')
    ret_x = RIGHT_X + ROW_W - 16
    L.append(f'<text x="{ret_x}" y="{y+ROW_H/2+5}" text-anchor="end" font-family="sans-serif" '
              f'font-size="12.5" font-weight="bold" '
              f'fill="{stroke}">{esc("-> " + ret)}</text>')
    if i > 0:
        py = TOP + (i - 1) * (ROW_H + ROW_GAP) + ROW_H
        L.append(f'<line x1="{RIGHT_X+KEY_W/2}" y1="{py}" x2="{RIGHT_X+KEY_W/2}" y2="{y-2}" '
                  'stroke="#94a3b8" stroke-width="1.4" marker-end="url(#a)"/>')

last_y = TOP + (n - 1) * (ROW_H + ROW_GAP)
L.append(f'<text x="{RIGHT_X+ROW_W}" y="{last_y+ROW_H+20}" text-anchor="end" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" '
          f'fill="#b45309">{esc("末级(唯 1 项)返 bytes,其余 4 项返 str——串联契约")}</text>')

foot_y = TOP + n * (ROW_H + ROW_GAP) - ROW_GAP + 66
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12.5" '
          f'fill="#1e293b" font-weight="bold">'
          f'{esc("结论:add_stages 是 BaseBackend 的抽象钩子——本图只给 5 级骨架全貌,")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+22}" font-family="sans-serif" font-size="12.5" '
          f'fill="#1e293b" font-weight="bold">'
          f'{esc("每个 make_xxx 内部具体跑哪些 pass 是后续章节的内容,不同后端可填不同 stages。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch14-add-stages-skeleton.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size {w}x{h}")
