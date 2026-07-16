#!/usr/bin/env python3
"""fig-m7-numpy-vs-ir: before-after 模板（同名接口，换掉内核）。
真 Builder 的 create_* 建 IR 节点交后端编译；InterpreterBuilder 同名同签名，
但直接用 numpy 当场算出数值——不建 IR。全坐标计算，零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

ROWS = [
    ("program_id 实现", "return builder.create_get_program_id(axis)  # 建 IR 节点",
     "return grid_idx[axis]  # 直接取值，不建 IR", "interpreter.py:L358-361"),
    ("二元算子公共实现", "builder.create_add/create_mul(...)  # 建 tt.* IR 节点",
     "binary_op: TensorHandle(op(lhs.data, rhs.data), ...)", "interpreter.py:L416-417"),
    ("create_add", "IR 节点 tt.add，交后端编译成 GPU 指令",
     "binary_op(..., np.add)", "interpreter.py:L431"),
    ("create_mul", "IR 节点 tt.mul，交后端编译成 GPU 指令",
     "binary_op(..., np.multiply)", "interpreter.py:L424"),
    ("访存实现", "IR 节点 tt.load/tt.store，交后端编译",
     "create_masked_load/store -> 按地址读写 host 内存", "interpreter.py:L376-385"),
]

LABEL_W, BEFORE_W, AFTER_W = 170, 380, 380
ROW_H = 58
HEADER_H = 34
TOP = 118
PAD = 30
col_x = {"label": PAD, "before": PAD + LABEL_W, "after": PAD + LABEL_W + BEFORE_W}
W = PAD * 2 + LABEL_W + BEFORE_W + AFTER_W
H = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 30

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
     f'<rect width="{W}" height="{H}" fill="white"/>',
     f'<text x="{PAD}" y="30" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">'
     f'{esc("InterpreterBuilder：同名同签名的 create_*，换掉了内核")}</text>',
     f'<text x="{PAD}" y="52" font-family="sans-serif" font-size="12" fill="#64748b">'
     f'{esc("左：真 Builder——建 IR 节点交后端编译。右：InterpreterBuilder——numpy 当场算出数值，不建 IR")}</text>']

headers = [("label", "接口"), ("before", "真 Builder（建 IR）"), ("after", "InterpreterBuilder（numpy 直算）")]
for key, title in headers:
    cw = {"label": LABEL_W, "before": BEFORE_W, "after": AFTER_W}[key]
    fill = "#64748b" if key == "label" else ("#94a3b8" if key == "before" else "#3b82f6")
    L.append(f'<rect x="{col_x[key]}" y="{TOP}" width="{cw-4}" height="{HEADER_H}" '
              f'fill="{fill}" stroke="#1e3a5f" stroke-width="1"/>')
    L.append(f'<text x="{col_x[key]+(cw-4)/2}" y="{TOP+HEADER_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" fill="white">'
              f'{esc(title)}</text>')

for i, (label, before, after, anchor) in enumerate(ROWS):
    ry = TOP + HEADER_H + i * ROW_H
    bg = "#f8fafc" if i % 2 == 0 else "white"
    L.append(f'<rect x="{PAD}" y="{ry}" width="{LABEL_W+BEFORE_W+AFTER_W-4}" '
              f'height="{ROW_H-4}" fill="{bg}" stroke="#e2e8f0" stroke-width="1"/>')
    L.append(f'<text x="{col_x["label"]+8}" y="{ry+ROW_H/2+1}" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="#334155">{esc(label)}</text>')
    L.append(f'<text x="{col_x["before"]+8}" y="{ry+20}" '
              f'font-family="monospace" font-size="10.5" fill="#475569">{esc(before[:44])}</text>')
    if len(before) > 44:
        L.append(f'<text x="{col_x["before"]+8}" y="{ry+36}" '
                  f'font-family="monospace" font-size="10.5" fill="#475569">{esc(before[44:])}</text>')
    L.append(f'<text x="{col_x["after"]+8}" y="{ry+20}" '
              f'font-family="monospace" font-size="10.5" font-weight="bold" '
              f'fill="#0f766e">{esc(after[:44])}</text>')
    if len(after) > 44:
        L.append(f'<text x="{col_x["after"]+8}" y="{ry+36}" '
                  f'font-family="monospace" font-size="10.5" font-weight="bold" '
                  f'fill="#0f766e">{esc(after[44:])}</text>')
    L.append(f'<text x="{col_x["after"]+8}" y="{ry+ROW_H-10}" '
              f'font-family="sans-serif" font-size="9.5" fill="#94a3b8">{esc(anchor)}</text>')

foot_y = H - PAD + 8
foot = "同名接口 create_add/create_mul/create_get_program_id/create_masked_load 全套复用，芯子从『建 IR』换成『直接算数值』"
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#64748b">{esc(foot)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig-m7-numpy-vs-ir.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
