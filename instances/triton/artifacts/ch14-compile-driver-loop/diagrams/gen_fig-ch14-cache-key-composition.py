#!/usr/bin/env python3
"""state-table 模板:磁盘缓存键 = 5 段乘性拼接,列=5 个组成段,行=触发条件/覆盖范围/典型动作。
triton_key 列高亮(覆盖面最广,19 项)。改造点:COLS/ROW_LABELS/CELLS/HIGHLIGHT_COL。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "磁盘缓存键 = 5 段乘性拼接——任一段变，整键变，该维度覆盖的 kernel 集体 miss"
SUBTITLE = "key = triton_key()-src.hash()-backend.hash()-options.hash()-env_vars(compiler.py:L231)"

COLS = ["triton_key()", "src.hash()", "backend.hash()", "options.hash()", "env_vars"]
ROW_LABELS = ["改哪里触发", "覆盖范围", "典型改动"]

CELLS = {
    "改哪里触发": [
        "编译器任一源码\n(compiler/backends\n/language)",
        "kernel 源码,或一份 IR 文件本身",
        "换 GPU 型号 / ptxas 版本",
        "调 num_warps/num_stages 等选项",
        "改相关环境变量取值",
    ],
    "覆盖范围": [
        "编译器身份:19 项静态输入\n(frontend+compiler\n+backends+language+so)",
        "单个 kernel / 单份 IR 自身",
        "该 target 的工具链 + 算力位",
        "该次编译的选项组合",
        "命中的具体变量子集",
    ],
    "典型改动": [
        "hack 一个 pass 的实现",
        "改 @jit 源码,或手改一份 .ttgir",
        "sm80 -> sm90,或升级 ptxas",
        "num_warps: 4 -> 8",
        "export TRITON_XXX=1",
    ],
}

HIGHLIGHT_COL = "triton_key()"
COLOR_HOT = ("#fef3c7", "#b45309")
COLOR_COLD = ("#e2e8f0", "#475569")

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 130, 232, 84, 40, 112, 40
n_cols = len(COLS)
n_rows = len(ROW_LABELS)
W = PAD * 2 + LABEL_W + COL_W * n_cols
H = TOP + HEADER_H + ROW_H * n_rows + PAD + 110

col_x = [PAD + LABEL_W + i * COL_W for i in range(n_cols)]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(n_rows)]

L = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
    '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
    f'<rect width="{W}" height="{H}" fill="white"/>',
    f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="17" '
    f'font-weight="bold" fill="#1e293b">{esc(TITLE)}</text>',
    f'<text x="{PAD}" y="58" font-family="sans-serif" font-size="12.5" '
    f'fill="#64748b">{esc(SUBTITLE)}</text>',
]

# 列头
for j, name in enumerate(COLS):
    x = col_x[j]
    hot = (name == HIGHLIGHT_COL)
    fill, stroke = COLOR_HOT if hot else COLOR_COLD
    L.append(
        f'<rect x="{x}" y="{TOP}" width="{COL_W-10}" height="{HEADER_H-6}" rx="5" '
        f'fill="{stroke}" stroke="{stroke}" stroke-width="1.5"/>'
    )
    L.append(
        f'<text x="{x+(COL_W-10)/2}" y="{TOP+(HEADER_H-6)/2+5}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="13.5" fill="white" '
        f'font-weight="bold">{esc(name)}</text>'
    )

# 行标签 + 单元格
for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(
        f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+5}" text-anchor="end" '
        f'font-family="sans-serif" font-size="13" font-weight="bold" '
        f'fill="#374151">{esc(row)}</text>'
    )
    for j, name in enumerate(COLS):
        cx = col_x[j]
        text = CELLS[row][j]
        lines = text.split("\n")
        hot = (name == HIGHLIGHT_COL)
        fill, stroke = COLOR_HOT if hot else COLOR_COLD
        L.append(
            f'<rect x="{cx}" y="{ry+5}" width="{COL_W-10}" height="{ROW_H-10}" rx="5" '
            f'fill="{fill if hot else "#f8fafc"}" stroke="{stroke}" '
            f'stroke-width="{2 if hot else 1}"/>'
        )
        text_fill = stroke if hot else "#374151"
        weight_attr = 'font-weight="bold" ' if hot else ''
        n_lines = len(lines)
        y0 = ry + ROW_H / 2 - (n_lines - 1) * 8 + 4
        for k, line in enumerate(lines):
            L.append(
                f'<text x="{cx+(COL_W-10)/2}" y="{y0+k*15}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="11.5" fill="{text_fill}" '
                f'{weight_attr}>{esc(line)}</text>'
            )

# 行分隔线
for i in range(1, n_rows):
    ry = row_y[i]
    L.append(
        f'<line x1="{PAD}" y1="{ry}" x2="{W-PAD}" y2="{ry}" '
        f'stroke="#e5e7eb" stroke-width="1"/>'
    )

legend_y = TOP + HEADER_H + ROW_H * n_rows + 34
L.append(f'<rect x="{PAD}" y="{legend_y-14}" width="20" height="16" rx="3" '
          f'fill="{COLOR_HOT[0]}" stroke="{COLOR_HOT[1]}" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+28}" y="{legend_y-2}" font-family="sans-serif" font-size="12" '
          f'fill="#334155">{esc("触发面最广——19 项任一变，全部 kernel 磁盘缓存集体 miss")}</text>')

concl_y = legend_y + 30
L.append(
    f'<text x="{PAD}" y="{concl_y}" font-family="sans-serif" font-size="12.5" '
    f'fill="#1e293b" font-weight="bold">{esc("结论:5 段各管一个维度——triton_key 覆盖整个编译器(改一行源码→全部 kernel miss)，其余四段各自独立、互不覆盖。")}</text>'
)
L.append(
    f'<text x="{PAD}" y="{concl_y+22}" font-family="sans-serif" font-size="11" '
    f'fill="#64748b">{esc("与按实参特化的内存 launch 缓存键(另一章)正交:那把键管同一 kernel 换实参，这把键管编译器/kernel/工具链/选项/环境本身变没变。")}</text>'
)

L.append("</svg>")
out = Path(__file__).with_name("fig-ch14-cache-key-composition.svg")
out.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {out}, size {W}x{H}")
