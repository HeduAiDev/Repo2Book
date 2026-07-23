#!/usr/bin/env python3
"""tiling 变体：hfusion.gather 语义——index 决定去 src 哪一列取值，写入 output。
三个矩阵左到右：index(决策) -> src(数据，命中列高亮) -> output(结果，同色边框)。
每行 i 一种颜色，箭头 index[i][j] -> src[i][k=命中列]，直观呈现
『每行按自己的 index 挑列，行与列位可独立切分，但 gather 轴必须整行扫描』。
数值全部来自 explainer worked_example（src[i][k]=10*(i+1)+k 手算复核，
gather 语义锚 HFusionStructuredOps.td:L202-L215），无 traces 路径印上图。
全坐标计算，零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "hfusion.gather：index 挑列、src 取值、写入 output"
SUBTITLE = "output[i][j] = src[i][index[i][j]]（HFusionStructuredOps.td:L202-L215，axis=1）"

SRC = [[10, 11, 12, 13], [20, 21, 22, 23], [30, 31, 32, 33]]
INDEX = [[3, 0], [1, 2], [0, 3]]
OUTPUT = [[13, 10], [21, 22], [30, 33]]
ROW_COLORS = ["#2563eb", "#7c3aed", "#059669"]  # i=0,1,2

M, K = 3, 4   # src 行数、gather 轴（k=0..3）
J = 2         # index/output 列数（j 轴）

CELL_W, CELL_H = 52, 40
PAD, TOP = 70, 150
HEADER_H = 26
GAP_IDX_SRC = 90
GAP_SRC_OUT = 90
ROWLABEL_W = 40

idx_x = PAD + ROWLABEL_W
src_x = idx_x + J * CELL_W + GAP_IDX_SRC
out_x = src_x + K * CELL_W + GAP_SRC_OUT

grid_top = TOP + HEADER_H
row_y = [grid_top + i * CELL_H for i in range(M)]

w = out_x + J * CELL_W + PAD
CALLOUT_H = 96
h = row_y[-1] + CELL_H + 40 + CALLOUT_H + PAD

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-32}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD-12}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']


def matrix_block(x0, ncols, header_fill, header_stroke, header_note, cell_values, cell_style):
    """cell_style(i, col) -> (fill, stroke, text_fill, bold)"""
    L.append(f'<rect x="{x0}" y="{TOP}" width="{ncols*CELL_W}" height="{HEADER_H}" rx="4" '
              f'fill="{header_fill}" stroke="{header_stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{x0+ncols*CELL_W/2}" y="{TOP+HEADER_H-8}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
              f'fill="{header_stroke}">{esc(header_note)}</text>')
    for i, row in enumerate(cell_values):
        ry = row_y[i]
        for j, val in enumerate(row):
            cx = x0 + j * CELL_W
            fill, stroke, tfill, bold = cell_style(i, j)
            L.append(f'<rect x="{cx}" y="{ry}" width="{CELL_W-4}" height="{CELL_H-4}" rx="4" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if bold else 1.2}"/>')
            L.append(f'<text x="{cx+(CELL_W-4)/2}" y="{ry+(CELL_H-4)/2+5}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="13" '
                      f'font-weight="{"bold" if bold else "normal"}" '
                      f'fill="{tfill}">{esc(str(val))}</text>')


# 行标签 i=0,1,2（放最左，index 矩阵左侧）
for i in range(M):
    ry = row_y[i]
    L.append(f'<text x="{PAD+ROWLABEL_W-10}" y="{ry+(CELL_H-4)/2+5}" text-anchor="end" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="{ROW_COLORS[i]}">i={i}</text>')

# 1) index 矩阵（决策，绿色系表头：j 轴可 tile）
matrix_block(idx_x, J, "#f0fdf4", "#16a34a", "index（j 轴，可 tile）", INDEX,
             lambda i, j: ("#f0fdf4", ROW_COLORS[i], ROW_COLORS[i], True))

# 2) src 矩阵（数据，橙色表头：gather 轴 k 不可 tile；被命中的列按行上色，其余保持中性）
SRC_ROW_HIT = [{INDEX[i][j] for j in range(J)} for i in range(M)]  # 每行 i 命中的列集合

def src_cell_style(i, k):
    if k in SRC_ROW_HIT[i]:
        return ("#fef9ee", ROW_COLORS[i], ROW_COLORS[i], True)
    return ("#fff7ed", "#fdba74", "#9a3412", False)

matrix_block(src_x, K, "#fff7ed", "#c2410c", f"src（gather 轴 k=0..{K-1}，不可 tile）",
             SRC, src_cell_style)

# 3) output 矩阵（结果，边框同行色）
matrix_block(out_x, J, "#eff6ff", "#1d4ed8", "output（= src[i][index[i][j]]）",
             OUTPUT, lambda i, j: ("#eff6ff", ROW_COLORS[i], ROW_COLORS[i], True))

# 箭头：index[i][j] -> src[i][命中列 k]（同行同色），共 M*J = 6 条。
# y 偏移刻意避开单元格文字垂直中心（文字位于 ry+23 附近）：j=0 走单元格上沿、
# j=1 走单元格下沿，两条水平线互不相撞、也不压中间数字。
ARROW_Y_OFFSET = [8, 30]
for i in range(M):
    color = ROW_COLORS[i]
    ry = row_y[i]
    for j in range(J):
        k = INDEX[i][j]
        y = ry + ARROW_Y_OFFSET[j]
        x1 = idx_x + j * CELL_W + (CELL_W - 4)
        x2 = src_x + k * CELL_W + 2
        L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" '
                  f'stroke="{color}" stroke-width="1.6" marker-end="url(#a)" opacity="0.85"/>')

co_y = row_y[-1] + CELL_H + 40
L.append(f'<rect x="{PAD}" y="{co_y}" width="{w-2*PAD}" height="{CALLOUT_H}" rx="6" '
          'fill="#f8fafc" stroke="#475569" stroke-width="1.5"/>')
L.append(f'<text x="{PAD+14}" y="{co_y+22}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="#334155">为什么 i、j 可切、gather 轴 k 不可切：'
          f'</text>')
NOTE1 = "沿 k=0..3 遍历中恰有一个 k 满足 index[i][j]==k（相等唯一）→ 每个 output[i][j] 被写恰一次；"
NOTE2 = "要确定『去哪列』必须看过整行候选列 k，切开一段 k 只见部分候选、命中判定失真——gather 轴不可 tile。"
NOTE3 = "i、j 维互不相干（各行/各输出列独立），可分头处理；本例 M·J·K = 3·2·4 = 24 次比较，输出 3×2 = 6 个元素。"
for idx, note in enumerate([NOTE1, NOTE2, NOTE3]):
    L.append(f'<text x="{PAD+14}" y="{co_y+42+idx*18}" font-family="sans-serif" font-size="11.5" '
              f'fill="#334155">{esc(note)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch21-m4-gather.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
