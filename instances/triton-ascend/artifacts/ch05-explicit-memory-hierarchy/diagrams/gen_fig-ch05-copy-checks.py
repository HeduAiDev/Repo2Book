#!/usr/bin/env python3
"""fig-ch05-copy-checks — state-table 模板（al.copy 地址空间放行矩阵）。
行=src(UB/L1)、列=dst(UB/L1/L0C)；绿=放行→create_copy_buffer，红=语言层
TypeError（未生成 IR）。芯片门禁(is_910_95)是矩阵之外的独立先决条件，单独
画一条横幅。未在 traces/m3_copy.json 里出现的组合(L1→UB、L1→L0C)标为
「本例未覆盖」，不臆造判定结果。全部坐标由常量/循环计算，文本全 esc()。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


TITLE = "al.copy 地址空间放行矩阵"
SUBTITLE = "行=src、列=dst；绿=放行→create_copy_buffer，红=语言层 TypeError(未生成 IR)"

DST_COLS = ["UB", "L1", "L0C"]
SRC_ROWS = ["UB", "L1"]

# cell: (status, lines)  status in {"pass","reject","na"}
CELLS = {
    ("UB", "UB"): ("pass", ["放行", "create_copy_buffer"]),
    ("UB", "L1"): ("pass", ["放行", "create_copy_buffer"]),
    ("UB", "L0C"): ("reject", ["拒：dst's AddressSpace", "must be UB or L1"]),
    ("L1", "UB"): ("na", ["本例未覆盖"]),
    ("L1", "L1"): ("reject", ["拒：src's AddressSpace", "must be UB"]),
    ("L1", "L0C"): ("na", ["本例未覆盖"]),
}
COLOR = {
    "pass": ("#dcfce7", "#15803d", "#14532d"),
    "reject": ("#fee2e2", "#b91c1c", "#7f1d1d"),
    "na": ("#f1f5f9", "#94a3b8", "#64748b"),
}

LABEL_W, COL_W, ROW_H, HEADER_H, TOP, PAD = 100, 250, 100, 42, 106, 40
w = PAD * 2 + LABEL_W + COL_W * len(DST_COLS)
grid_h = HEADER_H + ROW_H * len(SRC_ROWS)
BANNER_H = 70
h = TOP + grid_h + 50 + BANNER_H + 130

col_x = [PAD + LABEL_W + i * COL_W for i in range(len(DST_COLS))]
row_y = [TOP + HEADER_H + i * ROW_H for i in range(len(SRC_ROWS))]

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="18" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="56" text-anchor="middle" font-family="sans-serif" font-size="12.5" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# 列头（dst）
L.append(f'<text x="{PAD+LABEL_W/2}" y="{TOP-12}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11.5" fill="#94a3b8">{esc("src ＼ dst")}</text>')
for j, name in enumerate(DST_COLS):
    x = col_x[j]
    L.append(f'<rect x="{x}" y="{TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="4" '
              'fill="#334155"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TOP+(HEADER_H-6)/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" fill="white" '
              f'font-weight="bold">{esc("dst=" + name)}</text>')

# 行标签 + 单元格
for i, src in enumerate(SRC_ROWS):
    ry = row_y[i]
    L.append(f'<rect x="{PAD}" y="{ry}" width="{LABEL_W-10}" height="{ROW_H-8}" rx="4" '
              'fill="#334155"/>')
    L.append(f'<text x="{PAD+(LABEL_W-10)/2}" y="{ry+(ROW_H-8)/2+5}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="14" fill="white" font-weight="bold">'
              f'{esc("src=" + src)}</text>')
    for j, dst in enumerate(DST_COLS):
        cx = col_x[j]
        status, lines = CELLS[(src, dst)]
        fill, stroke, text_fill = COLOR[status]
        L.append(f'<rect x="{cx}" y="{ry}" width="{COL_W-8}" height="{ROW_H-8}" rx="6" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
        n = len(lines)
        y0 = ry + (ROW_H - 8) / 2 - (n - 1) * 9 + 5
        for k, line in enumerate(lines):
            L.append(f'<text x="{cx+(COL_W-8)/2}" y="{y0+k*18}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                      f'fill="{text_fill}">{esc(line)}</text>')

# 芯片门禁横幅（矩阵之外的独立先决条件，先于矩阵判定）
banner_y = TOP + grid_h + 50
L.append(f'<rect x="{PAD}" y="{banner_y}" width="{w-2*PAD}" height="{BANNER_H}" rx="8" '
          f'fill="#fef2f2" stroke="#b91c1c" stroke-width="1.8" stroke-dasharray="7,5"/>')
L.append(f'<text x="{PAD+20}" y="{banner_y+26}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#7f1d1d">'
          f'{esc("芯片门禁（矩阵之外、优先于上表判定）：is_910_95=False")}</text>')
L.append(f'<text x="{PAD+20}" y="{banner_y+48}" font-family="sans-serif" font-size="12" '
          f'fill="#b91c1c">'
          f'{esc("→ 拒：RuntimeError: only supported on Ascend910_95（即便 UB→UB 本应放行也直接短路）")}</text>')

# 图例
leg_y = banner_y + BANNER_H + 34
LEGEND = [("#15803d", "#dcfce7", "放行 → 建 create_copy_buffer"),
          ("#b91c1c", "#fee2e2", "拒 → TypeError / RuntimeError（不生成 IR）"),
          ("#94a3b8", "#f1f5f9", "本例未覆盖(非臆造)")]
for i, (stroke, fill, label) in enumerate(LEGEND):
    lx = PAD + i * 320
    L.append(f'<rect x="{lx}" y="{leg_y}" width="18" height="18" rx="3" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{lx+26}" y="{leg_y+14}" font-family="sans-serif" font-size="11.5" '
              f'fill="#334155">{esc(label)}</text>')

# 脚注
foot_y = leg_y + 46
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("旧接口 copy_from_ub_to_l1 是同矩阵里 dst 只留 L1 一列的更严子集(dst=UB 时反被拒)。")}</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">{esc("数据来自 host 实测；校验顺序见 third_party/ascend/language/cann/extension/semantic.py:L113-127。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name('fig-ch05-copy-checks.svg')
out.write_text('\n'.join(L), encoding='utf-8')
print(f'wrote {out} ({w}x{h})')
