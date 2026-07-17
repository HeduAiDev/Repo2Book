#!/usr/bin/env python3
"""fig-bases-xor-linear: flow 模板,两段。
① GF(2) 矩阵-向量乘 = 按输入 1-bit 挑 base 列再全部异或(m4 的紧凑写法例子)。
② 换一组 base = 换一种布局:恒等 / 转置 / broadcast 三行对比(m2)。
数据来自 explainer.json m2-bases / m4-gf2-algebra,LinearLayout.h:106-148,218-259。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

TITLE = "bases + xor 线性律:布局-向量乘 = 选列异或;换一组 base = 换一种布局"
SUBTITLE = "GF(2):加法=xor、乘法=按位 and  ——  include/triton/Tools/LinearLayout.h:218-259"

# ---- 段① 数据:输入 6=0b0110,四列 base=[1,2,14,12] ----
COLS = [("col1", 1, 0), ("col2", 2, 1), ("col3", 14, 1), ("col4", 12, 0)]  # (label, base值, 该列 bit)
XOR_RESULT = 12
EQ_TEXT = "(1×0) ⊕ (2×1) ⊕ (14×1) ⊕ (12×0) = 2 ⊕ 14 = 12"

# ---- 段② 数据:三种 base 组合 = 三种布局 ----
ROWS2 = [
    ("1D 恒等(8 元素)", "bases = [L(1), L(2), L(4)] = [1, 2, 4]", "#3b82f6"),
    ("1D 全零(整维压扁)", "bases = [0, 0, 0]", "#94a3b8"),
    ("2D→2D 转置(对调每个 base 分量)",
     "L(0,1)=(1,0)  L(0,2)=(2,0)  L(1,0)=(0,1)  L(2,0)=(0,2)", "#7c3aed"),
    ("2D→1D broadcast(L(x,y)=x,y 维 base 置 0)",
     "L(0,1)=0  L(0,2)=0  L(1,0)=1  L(2,0)=2", "#059669"),
]

PAD = 36
W = 1000
COL_W, COL_H, COL_GAP = 160, 64, 24
SEC1_TOP = PAD + 70
XOR_Y = SEC1_TOP + COL_H + 70
SEC2_TITLE_Y = XOR_Y + 100
ROW_H, ROW_GAP = 46, 14
SEC2_TOP = SEC2_TITLE_Y + 30
H = SEC2_TOP + len(ROWS2) * (ROW_H + ROW_GAP) + 70

cols_total_w = len(COLS) * COL_W + (len(COLS) - 1) * COL_GAP
cols_x0 = (W - cols_total_w) / 2

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
          'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{PAD}" font-family="sans-serif" font-size="17" '
         f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>')
L.append(f'<text x="{PAD}" y="{PAD + 20}" font-family="sans-serif" font-size="12" '
         f'fill="#64748b">{esc(SUBTITLE)}</text>')

# ---- 段① ----
L.append(f'<text x="{PAD}" y="{SEC1_TOP - 26}" font-family="sans-serif" font-size="14" '
         f'font-weight="bold" fill="#1e40af">① 输入 x=6=0b0110 只挑第 2、3 列(bit=1 的列)再异或</text>')
col_x = []
for i, (label, base, bit) in enumerate(COLS):
    x = cols_x0 + i * (COL_W + COL_GAP)
    col_x.append(x + COL_W / 2)
    fill, stroke = ("#bfdbfe", "#1d4ed8") if bit == 1 else ("#f1f5f9", "#94a3b8")
    L.append(f'<rect x="{x}" y="{SEC1_TOP}" width="{COL_W}" height="{COL_H}" rx="8" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    L.append(f'<text x="{x + COL_W/2}" y="{SEC1_TOP + 20}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="#0f172a">{esc(label)} (base={base})</text>')
    tcolor = "#1e3a8a" if bit == 1 else "#94a3b8"
    L.append(f'<text x="{x + COL_W/2}" y="{SEC1_TOP + 42}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" fill="{tcolor}">'
              f'{esc("bit=" + str(bit) + ("  ← 挑中" if bit == 1 else "  (跳过)"))}</text>')
    if bit == 1:
        L.append(f'<line x1="{x + COL_W/2}" y1="{SEC1_TOP + COL_H}" '
                  f'x2="{(col_x[0] if False else W/2)}" y2="{XOR_Y}" '
                  f'stroke="#1d4ed8" stroke-width="1.5" marker-end="url(#a)" opacity="0.75"/>')

xor_w, xor_h = 200, 54
xor_x = W / 2 - xor_w / 2
L.append(f'<rect x="{xor_x}" y="{XOR_Y}" width="{xor_w}" height="{xor_h}" rx="10" '
         f'fill="#dcfce7" stroke="#16a34a" stroke-width="2"/>')
L.append(f'<text x="{W/2}" y="{XOR_Y + 22}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#14532d">全部异或(xor)</text>')
L.append(f'<text x="{W/2}" y="{XOR_Y + 41}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="#14532d">2 ⊕ 14 = {XOR_RESULT}</text>')
L.append(f'<text x="{PAD}" y="{XOR_Y + xor_h + 22}" font-family="sans-serif" font-size="12" '
         f'fill="#334155">{esc(EQ_TEXT)}</text>')

# ---- 段② ----
L.append(f'<text x="{PAD}" y="{SEC2_TITLE_Y}" font-family="sans-serif" font-size="14" '
         f'font-weight="bold" fill="#1e40af">② 换一组 base = 换一种布局(bases 即 L 在 2 幂次点的取值)</text>')
label_w = 320
for i, (name, formula, color) in enumerate(ROWS2):
    y = SEC2_TOP + i * (ROW_H + ROW_GAP)
    L.append(f'<rect x="{PAD}" y="{y}" width="{label_w}" height="{ROW_H}" rx="6" '
              f'fill="{color}" opacity="0.12" stroke="{color}" stroke-width="1.5"/>')
    L.append(f'<text x="{PAD + 12}" y="{y + ROW_H/2 + 4}" font-family="sans-serif" '
              f'font-size="12" font-weight="bold" fill="{color}">{esc(name)}</text>')
    L.append(f'<text x="{PAD + label_w + 20}" y="{y + ROW_H/2 + 4}" font-family="sans-serif" '
              f'font-size="12" fill="#0f172a">{esc(formula)}</text>')

L.append(f'<text x="{PAD}" y="{H - 30}" font-family="sans-serif" font-size="12" '
         f'fill="#334155">{esc("在 GF(2) 上(加=xor、乘=and),布局-向量乘 = 按输入 1-bit 挑 base 列再全部异或;")}</text>')
L.append(f'<text x="{PAD}" y="{H - 12}" font-family="sans-serif" font-size="12" '
         f'fill="#334155">{esc("bases 就是 L 在 2 幂次点的取值——换一组 base 就换一种布局(恒等/转置/broadcast 只是不同 base)。")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-bases-xor-linear.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={W}x{H}")
