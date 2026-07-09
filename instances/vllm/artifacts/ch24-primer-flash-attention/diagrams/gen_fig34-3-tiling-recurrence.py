#!/usr/bin/env python3
"""tiling 模板(定制):4x4 打分矩阵切 2x2 块,query 行 0 沿两个 KV 列块推进;
下方状态表跟踪 (m_i,l_i,O_i) 每步演化,末值对照标准 softmax。
全坐标计算,零魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "FlashAttention 分块递推 — 4×4 打分矩阵切 2×2 块,追踪 query 行 0"
SUBTITLE = "seq_len=4, d=2, block_r=block_c=2;query 行 0 依次经过 KV 块 1(列 0-1)、KV 块 2(列 2-3)"

PAD, TOP = 40, 96
N = 4          # 矩阵边长
CELL = 52      # 每格像素
GRID_X = PAD
GRID_Y = TOP

BLOCK1_FILL, BLOCK2_FILL = "#dbeafe", "#fef3c7"   # KV 块1(冷色)/块2(暖色)
ROW0_STROKE = "#b91c1c"                             # query 行0 高亮描边
OTHER_FILL = "#f1f5f9"

grid_w = N * CELL
grid_h = N * CELL

# 右侧:running 量说明框
side_x = GRID_X + grid_w + 70
side_y = GRID_Y + 6
RUNNING = ["running m_i(见过的最高分)", "running l_i(归一累计)", "running O_i(加权输出)"]
SIDE_BOX_W = 250

# 下方:状态演化表
LABEL_W, COL_W, ROW_H, HEADER_H = 150, 220, 46, 32
COLS2 = ["KV 块 1 后", "KV 块 2 后", "标准 softmax(参照)"]
ROWS2 = ["局部 m~ / l~", "m_i 新", "l_i 新", "O_i 新"]
CELLS2 = {
    "局部 m~ / l~": ["m~=0.7071, l~=1.4931", "m~=0.7071, l~=2.0", "—"],
    "m_i 新":       ["0.7071", "0.7071", "0.7071"],
    "l_i 新":       ["1.4931", "3.4931", "(定义于分块内部)"],
    "O_i 新":       ["[0.6698, 0.3302]", "[0.8588, 0.7137]", "[0.8588, 0.7137]"],
}
HL_COL = 2  # 最终对照列高亮
table_w = LABEL_W + COL_W * len(COLS2)
tcol_x = [PAD + LABEL_W + i * COL_W for i in range(len(COLS2))]

TABLE_TOP = GRID_Y + grid_h + 96
trow_y = [TABLE_TOP + HEADER_H + i * ROW_H for i in range(len(ROWS2))]

foot_y = TABLE_TOP + HEADER_H + ROW_H * len(ROWS2) + 26

w = max(PAD + table_w + PAD, side_x + SIDE_BOX_W + PAD)
h = foot_y + 24

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-14}" font-family="sans-serif" font-size="16.5" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+10}" font-family="sans-serif" font-size="12" '
     f'fill="#475569">{esc(SUBTITLE)}</text>']

# 4x4 网格:非 row0 的格子淡灰,row0 按所属 KV 块上色并用红框高亮
for r in range(N):
    for c in range(N):
        x = GRID_X + c * CELL
        y = GRID_Y + r * CELL
        if r == 0:
            fill = BLOCK1_FILL if c < 2 else BLOCK2_FILL
        else:
            fill = OTHER_FILL
        L.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                  f'fill="{fill}" stroke="#94a3b8" stroke-width="1"/>')
        if r == 0:
            L.append(f'<text x="{x+CELL/2}" y="{y+CELL/2+4}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="11" fill="#334155">'
                      f'{esc(f"S(0,{c})")}</text>')
# query 行0 整体红框高亮
L.append(f'<rect x="{GRID_X}" y="{GRID_Y}" width="{grid_w}" height="{CELL}" '
          f'fill="none" stroke="{ROW0_STROKE}" stroke-width="3" rx="2"/>')
# KV 块分界(粗虚线,列1/2 之间)
L.append(f'<line x1="{GRID_X+2*CELL}" y1="{GRID_Y-6}" x2="{GRID_X+2*CELL}" '
          f'y2="{GRID_Y+grid_h+6}" stroke="#1d4ed8" stroke-width="2.5" stroke-dasharray="6,4"/>')
# 行/列轴标签
for c in range(N):
    L.append(f'<text x="{GRID_X+c*CELL+CELL/2}" y="{GRID_Y-14}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" fill="#64748b">K{c}</text>')
for r in range(N):
    L.append(f'<text x="{GRID_X-12}" y="{GRID_Y+r*CELL+CELL/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="11" fill="#64748b">Q{r}</text>')
L.append(f'<text x="{GRID_X+CELL}" y="{GRID_Y+grid_h+28}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" '
          f'fill="#1e40af">{esc("KV 块 1(列 0-1)")}</text>')
L.append(f'<text x="{GRID_X+3*CELL}" y="{GRID_Y+grid_h+28}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" font-weight="bold" '
          f'fill="#b45309">{esc("KV 块 2(列 2-3)")}</text>')
L.append(f'<text x="{GRID_X}" y="{GRID_Y+grid_h+52}" font-family="sans-serif" '
          f'font-size="11.5" fill="#b91c1c">'
          f'{esc("红框=被追踪的 query 行 0;局部块最大仅 2×2,4×4 完整表从未物化")}</text>')

# 右侧:说明手里始终攥着的三个 running 量
L.append(f'<text x="{side_x}" y="{side_y-14}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#0f172a">{esc("手里攥着的三个量")}</text>')
for i, txt in enumerate(RUNNING):
    y = side_y + i * 40
    L.append(f'<rect x="{side_x}" y="{y}" width="{SIDE_BOX_W}" height="30" rx="6" '
              'fill="#ecfdf5" stroke="#047857" stroke-width="1.5"/>')
    L.append(f'<text x="{side_x+12}" y="{y+20}" font-family="sans-serif" font-size="12" '
              f'fill="#047857">{esc(txt)}</text>')

# 下方:状态演化表(KV 块1 / KV 块2 / 标准 softmax 参照)
L.append(f'<text x="{PAD}" y="{TABLE_TOP-14}" font-family="sans-serif" font-size="14" '
          f'font-weight="bold" fill="#1e40af">{esc("query 行 0 的 (m,l,O) 递推 —— 处理完第 2 个 KV 块即与标准 softmax 差在浮点舍入内")}</text>')
for j, name in enumerate(COLS2):
    x = tcol_x[j]
    fill = "#059669" if j == HL_COL else "#3b82f6"
    L.append(f'<rect x="{x}" y="{TABLE_TOP}" width="{COL_W-8}" height="{HEADER_H-6}" rx="3" '
              f'fill="{fill}" stroke="#1e3a5f" stroke-width="1.5"/>')
    L.append(f'<text x="{x+(COL_W-8)/2}" y="{TABLE_TOP+(HEADER_H-6)/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="white" '
              f'font-weight="bold">{esc(name)}</text>')
for i, row in enumerate(ROWS2):
    ry = trow_y[i]
    L.append(f'<text x="{PAD+LABEL_W-16}" y="{ry+ROW_H/2+4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="#374151">{esc(row)}</text>')
    for j in range(len(COLS2)):
        cx = tcol_x[j]
        is_final_row = (row == "O_i 新")
        hl = is_final_row and j >= 1
        if hl:
            L.append(f'<rect x="{cx}" y="{ry+4}" width="{COL_W-8}" height="{ROW_H-8}" rx="4" '
                      'fill="#ecfdf5" stroke="#047857" stroke-width="2"/>')
        text_fill = "#047857" if hl else "#374151"
        weight = 'font-weight="bold" ' if hl else ''
        L.append(f'<text x="{cx+(COL_W-8)/2}" y="{ry+ROW_H/2+4}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11.5" fill="{text_fill}" '
                  f'{weight}>{esc(CELLS2[row][j])}</text>')

FOOT = "KV 块2 后 O_i=[0.8588,0.7137] 与标准 softmax(QKᵀ)V 差约 1e-16(float64 舍入内恒等) —— 全程最大局部块仅 2×2,4×4 完整打分表从未落地 HBM。"
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="12" '
          f'fill="#64748b">{esc(FOOT)}</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig34-3-tiling-recurrence.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
