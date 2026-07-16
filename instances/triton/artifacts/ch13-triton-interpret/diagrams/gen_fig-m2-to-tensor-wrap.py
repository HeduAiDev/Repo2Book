#!/usr/bin/env python3
"""fig-m2-to-tensor-wrap: before-after 模板（代码级）。
ASTTransformer.visit_Assign 把核体每条赋值的右值包进 to_tensor(...)——
连裸标量 c=2 也被提升为 tl.tensor。全坐标计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

ROWS = [  # (idx, before, after, note, highlight)
    ("[0]", "pid = tl.program_id(0)",
     "pid = to_tensor(tl.program_id(0), ib, False)",
     "program_id 返回值被包一层", False),
    ("[1]", "offs = pid*BLOCK + tl.arange(0,BLOCK)",
     "offs = to_tensor(pid*BLOCK+tl.arange(0,BLOCK), ib, False)",
     "整个偏移表达式被包一层", False),
    ("[2]", "c = 2",
     "c = to_tensor(2, ib, False)",
     "裸标量提升为 int32 张量——最能说明『为何不能原样跑 Python』", True),
    ("[3]", "y = tl.load(x_ptr+offs) * c",
     "y = to_tensor(tl.load(x_ptr+offs)*c, ib, False)",
     "读入并乘常量后的结果同样被包一层", False),
    ("store", "tl.store(y_ptr+offs, y)",
     "tl.store(y_ptr+offs, y)", "未触碰：ASTTransformer 只改写 Assign", False),
]

IDX_W, BEFORE_W, AFTER_W, NOTE_W = 56, 300, 380, 300
ROW_H = 46
HEADER_H = 34
TOP = 96
PAD = 30
col_x = {
    "idx": PAD,
    "before": PAD + IDX_W,
    "after": PAD + IDX_W + BEFORE_W,
    "note": PAD + IDX_W + BEFORE_W + AFTER_W,
}
W = PAD * 2 + IDX_W + BEFORE_W + AFTER_W + NOTE_W
H = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 30

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">'
     f'{esc("ASTTransformer.visit_Assign：核体每条赋值都被套上 to_tensor 张量外套")}</text>',
     f'<text x="{PAD}" y="52" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc("visit_Assign@interpreter.py:L1111  →  to_tensor@semantic.py:L111-126")}</text>']

headers = [("idx", "#"), ("before", "改写前 (BEFORE)"), ("after", "改写后 (AFTER)"), ("note", "右值发生了什么")]
for key, title in headers:
    cw = {"idx": IDX_W, "before": BEFORE_W, "after": AFTER_W, "note": NOTE_W}[key]
    L.append(f'<rect x="{col_x[key]}" y="{TOP}" width="{cw-4}" height="{HEADER_H}" '
              'fill="#3b82f6" stroke="#1e3a5f" stroke-width="1"/>')
    L.append(f'<text x="{col_x[key]+(cw-4)/2}" y="{TOP+HEADER_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" fill="white">'
              f'{esc(title)}</text>')

for i, (idx, before, after, note, hot) in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H
    fill = "#fef3c7" if hot else ("#f8fafc" if i % 2 == 0 else "white")
    stroke = "#d97706" if hot else "#e2e8f0"
    sw = 2 if hot else 1
    L.append(f'<rect x="{PAD}" y="{ry}" width="{IDX_W+BEFORE_W+AFTER_W+NOTE_W-4}" '
              f'height="{ROW_H-4}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    L.append(f'<text x="{col_x["idx"]+IDX_W/2-2}" y="{ry+ROW_H/2+1}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="{"#b45309" if hot else "#475569"}">{esc(idx)}</text>')
    L.append(f'<text x="{col_x["before"]+8}" y="{ry+ROW_H/2+1}" '
              f'font-family="monospace" font-size="11" fill="#334155">{esc(before)}</text>')
    L.append(f'<text x="{col_x["after"]+8}" y="{ry+ROW_H/2+1}" '
              f'font-family="monospace" font-size="10.5" '
              f'fill="{"#92400e" if hot else "#0f766e"}">{esc(after)}</text>')
    WRAP = 22
    if len(note) > WRAP:
        L.append(f'<text x="{col_x["note"]+8}" y="{ry+ROW_H/2-7}" '
                  f'font-family="sans-serif" font-size="10.5" fill="#334155">{esc(note[:WRAP])}</text>')
        L.append(f'<text x="{col_x["note"]+8}" y="{ry+ROW_H/2+9}" '
                  f'font-family="sans-serif" font-size="10.5" fill="#334155">{esc(note[WRAP:])}</text>')
    else:
        L.append(f'<text x="{col_x["note"]+8}" y="{ry+ROW_H/2+1}" '
                  f'font-family="sans-serif" font-size="10.5" fill="#334155">{esc(note)}</text>')

foot_y = H - PAD + 8
foot = "4 条赋值 → 恰 4 次 to_tensor 包裹；1 条 store 语句 0 改写（语句总数不变，只做减法不做加法）"
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc(foot)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-m2-to-tensor-wrap.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
