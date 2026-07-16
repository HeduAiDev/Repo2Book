#!/usr/bin/env python3
"""state-table 模板:同一份 add_kernel 源码 4 次发射的缓存键演化。
列=发射序号，行=n_elements/对齐位/BLOCK_SIZE(constexpr)/缓存键/动作。
L4 与 L1 同键(绿=命中)，L2/L3 各改一个维度(红=重编)。全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "同一份 add_kernel，4 次发射 → 3 个缓存键"
SUBTITLE = "add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr)；x/y/output 皆 float32 张量、16 对齐(D)"

COLS = ["L1", "L2", "L3", "L4"]
COL_SUB = [
    "n=1024, BLOCK=1024",
    "n=1024, BLOCK=512",
    "n=1000, BLOCK=1024",
    "n=1024, BLOCK=1024（重复）",
]
ROW_LABELS = ["n_elements", "n 对齐位", "BLOCK_SIZE (constexpr)", "缓存键", "缓存动作"]

CELLS = {
    "n_elements": ["1024", "1024", "1000", "1024"],
    "n 对齐位": ["D (1024%16=0)", "D (1024%16=0)", "N (1000%16=8)", "D (1024%16=0)"],
    "BLOCK_SIZE (constexpr)": ["1024", "512", "1024", "1024"],
    "缓存键": [
        "*fp32*fp32*fp32i32DDDD\n((1024,), {})",
        "*fp32*fp32*fp32i32DDDD\n((512,), {})",
        "*fp32*fp32*fp32i32DDDN\n((1024,), {})",
        "*fp32*fp32*fp32i32DDDD\n((1024,), {})",
    ],
    "缓存动作": ["未命中 → 编第 1 份", "constexpr 变 → 编第 2 份", "对齐位 D→N → 编第 3 份", "同 L1 键 → 命中不编"],
}

# 每列相对 L1 的语义：miss(首编) / recompile(因某维变化重编) / hit(命中复用)
COL_STATUS = ["miss", "recompile", "recompile", "hit"]
COLOR = {
    "miss": ("#fef9c3", "#a16207"),
    "recompile": ("#fee2e2", "#b91c1c"),
    "hit": ("#dcfce7", "#15803d"),
}

# 每列中，相对 L1 发生变化的行（用于高亮该格文字加粗+变色，除去 miss/hit 行本身的整列色）
CHANGED_ROWS = {
    "L2": ["BLOCK_SIZE (constexpr)", "缓存键"],
    "L3": ["n 对齐位", "缓存键"],
}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 190, 250, 66, 46, 108, 40
n_cols = len(COLS)
n_rows = len(ROW_LABELS)
W = PAD * 2 + LABEL_W + COL_W * n_cols
H = TOP + HEADER_H + ROW_H * n_rows + PAD + 96

col_x = [PAD + LABEL_W + i * COL_W for i in range(n_cols)]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(n_rows)]

L = [
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">',
    '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
    'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
    f'<rect width="{W}" height="{H}" fill="white"/>',
    f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="18" '
    f'font-weight="bold" fill="#1e293b">{esc(TITLE)}</text>',
    f'<text x="{PAD}" y="58" font-family="sans-serif" font-size="12.5" '
    f'fill="#64748b">{esc(SUBTITLE)}</text>',
]

# 列头（每列按 status 上色边框，标 Ln + desc）
for j, name in enumerate(COLS):
    x = col_x[j]
    status = COL_STATUS[j]
    fill, stroke = COLOR[status]
    L.append(
        f'<rect x="{x}" y="{TOP}" width="{COL_W-10}" height="{HEADER_H-6}" rx="5" '
        f'fill="{stroke}" stroke="{stroke}" stroke-width="1.5"/>'
    )
    L.append(
        f'<text x="{x+(COL_W-10)/2}" y="{TOP+20}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="14" fill="white" '
        f'font-weight="bold">{esc(name)}</text>'
    )
    L.append(
        f'<text x="{x+(COL_W-10)/2}" y="{TOP+37}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="10.5" fill="white">{esc(COL_SUB[j])}</text>'
    )

# 行标签 + 单元格
for i, row in enumerate(ROW_LABELS):
    ry = row_y[i]
    L.append(
        f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+5}" text-anchor="end" '
        f'font-family="sans-serif" font-size="13" font-weight="bold" '
        f'fill="#374151">{esc(row)}</text>'
    )
    is_key_row = row == "缓存键"
    is_action_row = row == "缓存动作"
    for j, name in enumerate(COLS):
        cx = col_x[j]
        text = CELLS[row][j]
        lines = text.split("\n")
        status = COL_STATUS[j]
        changed = row in CHANGED_ROWS.get(name, [])
        # 底色规则：action 行 / key 行按该列 status 上色；其余行仅 changed 时上色
        if is_action_row or is_key_row:
            fill, stroke = COLOR[status]
            box_fill, box_stroke = fill, stroke
            do_box = True
        elif changed:
            fill, stroke = COLOR[status]
            box_fill, box_stroke = fill, stroke
            do_box = True
        else:
            do_box = False
        if do_box:
            L.append(
                f'<rect x="{cx}" y="{ry+5}" width="{COL_W-10}" height="{ROW_H-10}" rx="5" '
                f'fill="{box_fill}" stroke="{box_stroke}" stroke-width="2"/>'
            )
            text_fill = box_stroke
            weight_attr = 'font-weight="bold" '
        else:
            text_fill = "#374151"
            weight_attr = ""
        n = len(lines)
        y0 = ry + ROW_H / 2 - (n - 1) * 8 + 4
        fsz = 12 if not is_key_row else 11
        for k, line in enumerate(lines):
            L.append(
                f'<text x="{cx+(COL_W-10)/2}" y="{y0+k*15}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="{fsz}" fill="{text_fill}" '
                f'{weight_attr}>{esc(line)}</text>'
            )

# 行分隔线
for i in range(1, n_rows):
    ry = row_y[i]
    L.append(
        f'<line x1="{PAD}" y1="{ry}" x2="{W-PAD}" y2="{ry}" '
        f'stroke="#e5e7eb" stroke-width="1"/>'
    )

# 图例
legend_y = TOP + HEADER_H + ROW_H * n_rows + 34
legend_items = [("miss", "首次未命中/编译"), ("recompile", "维度变化 → 重编"), ("hit", "逐维全同 → 命中复用")]
lx = PAD
for key, desc in legend_items:
    fill, stroke = COLOR[key]
    L.append(f'<rect x="{lx}" y="{legend_y-14}" width="20" height="16" rx="3" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(
        f'<text x="{lx+28}" y="{legend_y-2}" font-family="sans-serif" font-size="12" '
        f'fill="#334155">{esc(desc)}</text>'
    )
    lx += 28 + 12 * len(desc) + 40

# 结论条
concl_y = legend_y + 30
L.append(
    f'<text x="{PAD}" y="{concl_y}" font-family="sans-serif" font-size="12.5" '
    f'fill="#1e293b" font-weight="bold">结论：凡进缓存键的维度（dtype / 指针对齐 / constexpr 值）任一变化即触发重编译；constexpr 若喂连续变化的值，缓存条目无上界 → 首次迭代编译风暴。</text>'
)

L.append("</svg>")
out = Path(__file__).with_name("fig-launch-cache-key.svg")
out.write_text("\n".join(L), encoding="utf-8")
print(f"wrote {out}")
