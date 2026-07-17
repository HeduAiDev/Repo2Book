#!/usr/bin/env python3
"""fig-compose-invert-rref: 三段式(state-table 变体)。
① getMatrix:4 个 base -> 3x4 比特矩阵(m5)。② compose:只对 base 求值(m6,精简展示)。
③ invertAndCompose:[B|A] 拼接做 RREF,左半->单位阵,右半->复合 bases(m7)。
数据来自 explainer.json m5/m6/m7,LinearLayout.cpp:65-113,813-841,887-923。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "复合 = 比特矩阵乘,求逆 = GF(2) RREF——两处调用扛住布局代数全部重活"
SUBTITLE = "lib/Tools/LinearLayout.cpp:65-113(getMatrix), 813-841(compose), 887-923(invertAndCompose)"

PAD = 36
W = 1040

# ---------- 段① getMatrix ----------
BASE_LABELS = ["base1\nL(0,1)", "base2\nL(0,2)", "base3\nL(1,0)", "base4\nL(2,0)"]
BASE_VALS = ["(0b01,0b1)", "(0b10,0b0)", "(0b10,0b0)", "(0b11,0b0)"]
MATRIX_ROWS = [
    ("row0", "1001", "0b1001"),
    ("row1", "0111", "0b0111"),
    ("row2", "1000", "0b1000"),
]

CELL1, ROWLABEL_W1 = 56, 60
GRID1_W = ROWLABEL_W1 + 4 * CELL1
grid1_x0 = (W - GRID1_W) / 2
S1_TITLE_Y = PAD + 70
BASE_ROW_Y = S1_TITLE_Y + 16
BASE_W, BASE_H, BASE_GAP = 108, 54, 14
GRID1_TOP = BASE_ROW_Y + BASE_H + 46

# ---------- 段② compose(精简) ----------
S2_TITLE_Y = GRID1_TOP + 3 * CELL1 + 56
COMPOSE_LINES = [
    "L_bases=[2,1], O_bases=[1,3]",
    "composed_base0=O(L(1))=O(2)=3   composed_base1=O(L(2))=O(1)=1  →  composed=[3,1]",
    "非幂次验证 x=3:L(3)=3, O(3)=2, apply([3,1],3)=3⊕1=2  一致 ✓",
]

# ---------- 段③ invertAndCompose RREF ----------
S3_TITLE_Y = S2_TITLE_Y + 16 + len(COMPOSE_LINES) * 18 + 34
BEFORE = [["0", "1", "1", "0"], ["1", "1", "1", "1"]]
AFTER = [["1", "0", "0", "1"], ["0", "1", "1", "0"]]
CELL2 = 46
GRID2_W = 4 * CELL2
GRID2_GAP = 130
grid2_total = GRID2_W * 2 + GRID2_GAP
grid2_x0 = (W - grid2_total) / 2
GRID2_TOP = S3_TITLE_Y + 44

H = GRID2_TOP + 2 * CELL2 + 130

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
          'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="16" '
         f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD + 20}" font-family="sans-serif" font-size="11" '
         f'fill="#64748b">{esc(SUBTITLE)}</text>')

# ---- 段① ----
L.append(f'<text x="{PAD}" y="{S1_TITLE_Y}" font-family="sans-serif" font-size="14" '
         f'font-weight="bold" fill="#1e40af">① getMatrix:4 个 base → 3×4 比特矩阵</text>')
base_total_w = 4 * BASE_W + 3 * BASE_GAP
base_x0 = (W - base_total_w) / 2
for i in range(4):
    x = base_x0 + i * (BASE_W + BASE_GAP)
    L.append(f'<rect x="{x}" y="{BASE_ROW_Y}" width="{BASE_W}" height="{BASE_H}" rx="6" '
              f'fill="#bfdbfe" stroke="#1d4ed8" stroke-width="1.5"/>')
    name, sub = BASE_LABELS[i].split("\n")
    L.append(f'<text x="{x + BASE_W/2}" y="{BASE_ROW_Y + 18}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="11" font-weight="bold" '
              f'fill="#1e3a8a">{esc(name)} {esc(sub)}</text>')
    L.append(f'<text x="{x + BASE_W/2}" y="{BASE_ROW_Y + 38}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="#1e40af">{esc(BASE_VALS[i])}</text>')
    cx = grid1_x0 + ROWLABEL_W1 + i * CELL1 + CELL1 / 2
    L.append(f'<line x1="{x + BASE_W/2}" y1="{BASE_ROW_Y + BASE_H}" x2="{cx}" '
              f'y2="{GRID1_TOP}" stroke="#94a3b8" stroke-width="1.2" marker-end="url(#a)"/>')

L.append(f'<text x="{grid1_x0 + ROWLABEL_W1 + 4*CELL1/2}" y="{GRID1_TOP - 10}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="#64748b">codomain 位宽 (2,1) → 3 行</text>')
for ri, (rlabel, bits, hexv) in enumerate(MATRIX_ROWS):
    ry = GRID1_TOP + ri * CELL1
    L.append(f'<text x="{grid1_x0 + ROWLABEL_W1 - 10}" y="{ry + CELL1/2 + 4}" text-anchor="end" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="#334155">{esc(rlabel)}</text>')
    for ci, bit in enumerate(bits):
        cx = grid1_x0 + ROWLABEL_W1 + ci * CELL1
        fill = "#93c5fd" if bit == "1" else "#f1f5f9"
        stroke = "#1d4ed8" if bit == "1" else "#cbd5e1"
        L.append(f'<rect x="{cx+2}" y="{ry+2}" width="{CELL1-4}" height="{CELL1-4}" rx="6" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        L.append(f'<text x="{cx+CELL1/2}" y="{ry+CELL1/2+5}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="14" font-weight="bold" '
                  f'fill="#1e3a8a">{esc(bit)}</text>')
    L.append(f'<text x="{grid1_x0 + ROWLABEL_W1 + 4*CELL1 + 16}" y="{ry + CELL1/2 + 4}" '
              f'font-family="sans-serif" font-size="12" fill="#334155">= {esc(hexv)}</text>')
L.append(f'<text x="{grid1_x0}" y="{GRID1_TOP + 3*CELL1 + 24}" font-family="sans-serif" '
         f'font-size="11" fill="#64748b">assert numCols=4≤64 且 numRows=3≤64'
         f'(每行打包进单个 uint64)</text>')

# ---- 段② ----
L.append(f'<text x="{PAD}" y="{S2_TITLE_Y}" font-family="sans-serif" font-size="14" '
         f'font-weight="bold" fill="#1e40af">② compose:O∘L 只需对 L 的每个 base 求 O 值(等价比特矩阵相乘)</text>')
for i, line in enumerate(COMPOSE_LINES):
    L.append(f'<text x="{PAD}" y="{S2_TITLE_Y + 20 + i * 18}" font-family="sans-serif" '
              f'font-size="12" fill="#334155">{esc(line)}</text>')

# ---- 段③ ----
L.append(f'<text x="{PAD}" y="{S3_TITLE_Y}" font-family="sans-serif" font-size="14" '
         f'font-weight="bold" fill="#1e40af">③ invertAndCompose:拼接 [B(outer)|A(this)] 做 GF(2) RREF</text>')


def draw_grid(x0, y0, rows, left_hi=None, right_hi=None):
    for ri, row in enumerate(rows):
        for ci, bit in enumerate(row):
            cx = x0 + ci * CELL2
            cy = y0 + ri * CELL2
            fill, stroke = "#f1f5f9", "#cbd5e1"
            if left_hi and ci < 2:
                fill, stroke = left_hi
            elif right_hi and ci >= 2:
                fill, stroke = right_hi
            L.append(f'<rect x="{cx+2}" y="{cy+2}" width="{CELL2-4}" height="{CELL2-4}" rx="6" '
                      f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
            L.append(f'<text x="{cx+CELL2/2}" y="{cy+CELL2/2+5}" text-anchor="middle" '
                      f'font-family="sans-serif" font-size="13" font-weight="bold" '
                      f'fill="#1e3a8a">{esc(bit)}</text>')
    # 分隔线(col1,2 | col3,4)
    dx = x0 + 2 * CELL2
    L.append(f'<line x1="{dx}" y1="{y0-4}" x2="{dx}" y2="{y0+2*CELL2+4}" '
              f'stroke="#0f172a" stroke-width="2" stroke-dasharray="4,3"/>')


draw_grid(grid2_x0, GRID2_TOP, BEFORE)
L.append(f'<text x="{grid2_x0 + GRID2_W/2}" y="{GRID2_TOP - 10}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="#64748b">拼接前:B(outer)=[2,3] | A(this)=[3,2]</text>')

arrow_y = GRID2_TOP + CELL2
ax1 = grid2_x0 + GRID2_W + 14
ax2 = grid2_x0 + GRID2_W + GRID2_GAP - 14
L.append(f'<line x1="{ax1}" y1="{arrow_y}" x2="{ax2}" y2="{arrow_y}" '
         f'stroke="#16a34a" stroke-width="2.2" marker-end="url(#a)"/>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{arrow_y - 12}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" '
         f'fill="#16a34a">RREF</text>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{arrow_y + 20}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9" '
         f'fill="#64748b">f2reduce::</text>')
L.append(f'<text x="{(ax1+ax2)/2}" y="{arrow_y + 32}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9" '
         f'fill="#64748b">inplace_rref_strided</text>')

after_x0 = grid2_x0 + GRID2_W + GRID2_GAP
draw_grid(after_x0, GRID2_TOP, AFTER,
          left_hi=("#bbf7d0", "#047857"), right_hi=("#bfdbfe", "#1d4ed8"))
L.append(f'<text x="{after_x0 + GRID2_W/2}" y="{GRID2_TOP - 10}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" fill="#64748b">RREF 后</text>')
L.append(f'<text x="{after_x0}" y="{GRID2_TOP + 2*CELL2 + 20}" font-family="sans-serif" '
         f'font-size="11" font-weight="bold" fill="#047857">左半=单位阵(=对 B 求逆)</text>')
L.append(f'<text x="{after_x0}" y="{GRID2_TOP + 2*CELL2 + 36}" font-family="sans-serif" '
         f'font-size="11" font-weight="bold" fill="#1d4ed8">右半 → C_bases=[2,1]</text>')

L.append(f'<text x="{PAD}" y="{GRID2_TOP + 2*CELL2 + 60}" font-family="sans-serif" '
         f'font-size="11" fill="#334155">调用点:f2reduce::inplace_rref_strided —— '
         f'LinearLayout.cpp:912(求逆复合)、:151(求秩)</text>')
L.append(f'<text x="{PAD}" y="{GRID2_TOP + 2*CELL2 + 78}" font-family="sans-serif" '
         f'font-size="11" fill="#334155">语义核对:A(x)=B(C(x)) 对 x=0,1,2,3 全部成立</text>')

caption = ("并排拼 B|A 两张比特矩阵做一次 RREF:左半被消成单位阵(=把 B 求逆),"
           "右半自动浮现出合成配方 C=[2,1];布局的复合/求逆/求秩全归到这一个 GF(2) RREF 调用。")
L.append(f'<text x="{PAD}" y="{H - 14}" font-family="sans-serif" font-size="12" '
         f'fill="#0f172a">{esc(caption)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-compose-invert-rref.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={W}x{H}")
