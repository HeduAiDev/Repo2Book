#!/usr/bin/env python3
"""state-table 模板(仿 example-softmax-trace.py,行标签 + 双世界对照列):
CodeGenerator 穿梭的两个世界:constexpr(编译期) vs tensor(运行期)。
7 行维度对照,每格手工按内容折成 <=2 行以适应宽表。
改造点:ROWS(维度名, constexpr 文案, tensor 文案)。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "CodeGenerator 的两个世界:constexpr(编译期) <-> tensor(运行期)"
SUBTITLE = "本章主线心智模型;此二分在本章现身 4 处(visit_Assign/visit_FunctionDef/call_JitFunction/visit_BinOp)"

LABEL_COL = "维度"
COLS = ["constexpr 世界(编译期)", "tensor 世界(运行期)"]
ROWS = [
    ("是什么", ["Python 值", "(int/str/dtype/tl.constexpr 包装)"],
              ["language.tensor", "(handle=MLIR Value + type)"]),
    ("何时已知", ["编译时已知,可折叠", "(BLOCK=1024 -> arith.constant)"],
                ["运行期才有值", "IR 里只是一个 SSA 名"]),
    ("判据函数", ["_is_constexpr"], ["_is_triton_tensor", "_is_triton_value"]),
    ("在 visit_Assign", ["_unwrap 成裸值赋进 Python 变量", "不建 op"],
                        ["to_tensor 物化成 SSA 句柄"]),
    ("在 visit_FunctionDef", ["取 Python 值,idx 不加", "(不占 IR 位)"],
                             ["fn.args(idx) 取句柄", "idx++"]),
    ("在 call_JitFunction", ["抽进 constants,进 mangle 名", "(不进 IR 参数)"],
                            ["arg.handle 成为 tt.call", "的 SSA 操作数"]),
    ("在 visit_BinOp", ["两边都 constexpr", "-> 纯 Python 运算"],
                       ["有 tensor -> __add__(_builder=)", "建 op"]),
]

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 175, 400, 76, 40, 100, 30
w = PAD * 2 + LABEL_W + COL_W * len(COLS)
h = TOP + HEADER_H + ROW_H * len(ROWS) + PAD + 26
col_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(ROWS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+20}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

WORLD_COLOR = [("#eff6ff", "#1d4ed8"), ("#fdf4ff", "#a21caf")]  # constexpr=蓝, tensor=紫

for j, name in enumerate(COLS):
    x = col_x[j]
    fill, stroke = WORLD_COLOR[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-10}" height="{HEADER_H-6}" rx="3" '
              f'fill="{stroke}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-10)/2}" y="{TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')

for i, (label, c0, c1) in enumerate(ROWS):
    ry = row_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#374151">{esc(label)}</text>')
    for j, lines in enumerate((c0, c1)):
        cx = col_x[j]
        fill, stroke = WORLD_COLOR[j]
        L.append(f'<rect x="{cx}" y="{ry+5}" width="{COL_W-10}" height="{ROW_H-10}" rx="4" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
        n = len(lines)
        y0 = ry + ROW_H / 2 - (n - 1) * 9 + 4
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-10)/2}" y="{y0+k*17}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12" fill="{stroke}">{esc(line)}</text>')
    if i < len(ROWS) - 1:
        L.append(f'<line x1="{PAD}" y1="{ry+ROW_H}" x2="{PAD+LABEL_W+COL_W*len(COLS)-10}" '
                  f'y2="{ry+ROW_H}" stroke="#e2e8f0" stroke-width="1"/>')

foot_y = h - PAD + 6
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("每遇一个节点先问:这是哪个世界的东西——左列折叠进代码不建 op,右列建真 SSA op")}</text>')
L.append('</svg>')
out = Path(__file__).with_name("ch16-fig-constexpr-tensor-worlds.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}, size {w}x{h}")
