#!/usr/bin/env python3
"""fig-ch14-m6-partial-scalarize — layout 模板:16x8 张量的部分标量化。
dim0(size16,unstructured)每行一次 scf.for 迭代(红色左侧条);dim1(size8,structured,
f32 满 32 字节对齐)每行整段 1x8 向量搬(蓝色格子)。格子数与 unstructure_mix.mlir
%18 tensor<16x8x!ptr<f32>> 的真实形状严格一致,不做省略。数据取自
explainer m6.figure_specs.numbers。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

ROWS, COLS = 16, 8
CELL_W, CELL_H = 34, 22
GAP = 2
LABEL_W = 74
PAD, TOP = 44, 150
RIGHT_W = 430

grid_w = COLS * (CELL_W + GAP) - GAP
w = PAD + LABEL_W + grid_w + 40 + RIGHT_W + PAD
h = TOP + ROWS * (CELL_H + GAP) + 140

grid_x0 = PAD + LABEL_W

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="34" text-anchor="middle" font-family="sans-serif" '
     f'font-size="17.5" fill="#0f172a">'
     f'{esc("混合态 [unstructured(16), structured(8)] 的部分标量化")}</text>',
     f'<text x="{w/2}" y="56" text-anchor="middle" font-family="sans-serif" '
     f'font-size="12.5" fill="#475569">'
     f'{esc("dim0 离散 → 1 层 scf.for 逐行循环;dim1 连续且 32 字节对齐 → 每行 1×8 向量整段搬")}</text>',
     f'<text x="{w/2}" y="76" text-anchor="middle" font-family="sans-serif" '
     f'font-size="12" fill="#64748b">'
     f'{esc("ptr %18: tensor<16x8x!ptr<f32>>(unstructure_mix.mlir @indirect_mix_kernel)")}</text>']

# 列标题
L.append(f'<text x="{grid_x0 + grid_w/2}" y="{TOP-58}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="13" fill="#1e40af">'
         f'{esc("dim1(size 8, structured → 向量切片)")}</text>')
for c in range(COLS):
    cx = grid_x0 + c * (CELL_W + GAP) + CELL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-38}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="10" fill="#94a3b8">{c}</text>')

# 行标题
L.append(f'<text x="{PAD}" y="{TOP-14}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="#991b1b">{esc("dim0(16,unstructured → scf.for)")}</text>')

ROW_FILL_A = "#dbeafe"
ROW_FILL_B = "#bfdbfe"
HL_ROW = 3  # 高亮示例迭代行,标注 IR

for r in range(ROWS):
    y = TOP + r * (CELL_H + GAP)
    fill = ROW_FILL_A if r % 2 == 0 else ROW_FILL_B
    stroke = "#1d4ed8" if r == HL_ROW else "#60a5fa"
    sw = 2.4 if r == HL_ROW else 1
    # 左侧红色条:代表该行是一次 scf.for 迭代(unstructured 维)
    L.append(f'<rect x="{PAD}" y="{y}" width="{LABEL_W-10}" height="{CELL_H}" rx="4" '
              f'fill="{"#fecaca" if r == HL_ROW else "#fee2e2"}" '
              f'stroke="{"#dc2626" if r == HL_ROW else "#f87171"}" stroke-width="1"/>')
    L.append(f'<text x="{PAD+(LABEL_W-10)/2}" y="{y+CELL_H/2+4}" text-anchor="middle" '
              f'font-family="monospace" font-size="10" fill="#7f1d1d">{esc(f"iv={r}")}</text>')
    for c in range(COLS):
        x = grid_x0 + c * (CELL_W + GAP)
        L.append(f'<rect x="{x}" y="{y}" width="{CELL_W}" height="{CELL_H}" '
                  f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')

# 高亮行的大括号 + IR 引出线
hl_y = TOP + HL_ROW * (CELL_H + GAP)
hl_cy = hl_y + CELL_H / 2
callout_x = grid_x0 + grid_w + 40
L.append(f'<line x1="{grid_x0+grid_w}" y1="{hl_cy}" x2="{callout_x}" y2="{hl_cy}" '
          'stroke="#1d4ed8" stroke-width="1.5" stroke-dasharray="4,3"/>')
box_y = hl_cy - 58
L.append(f'<rect x="{callout_x}" y="{box_y}" width="{RIGHT_W-30}" height="130" rx="10" '
          'fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>')
L.append(f'<text x="{callout_x+16}" y="{box_y+22}" font-family="sans-serif" font-size="11.5" '
          f'font-weight="bold" fill="#1e3a8a">{esc(f"第 iv={HL_ROW} 轮循环体(CHECK 钉死的 IR)")}</text>')
IR_LINES = [
    "tensor.extract_slice",
    "  %25[%iv,0][1,8][1,1] {DiscreteMemAccess}",
    "%33 = tt.load ... : tensor<1x8x!ptr<f32>>",
    "insert_slice %33 into %29[%iv,0][1,8][1,1]",
]
for i, line in enumerate(IR_LINES):
    L.append(f'<text x="{callout_x+16}" y="{box_y+44+i*20}" font-family="monospace" '
              f'font-size="10.5" fill="#1e40af">{esc(line)}</text>')

# 图例
legend_y = TOP + ROWS * (CELL_H + GAP) + 34
LEGEND = [("#fee2e2", "#f87171", "dim0 行标:一次 scf.for 迭代(iv)"),
          ("#dbeafe", "#60a5fa", "dim1 格子:1×8 连续向量切片(不循环)")]
lx = PAD
for fill, stroke, label in LEGEND:
    L.append(f'<rect x="{lx}" y="{legend_y}" width="16" height="16" rx="3" '
              f'fill="{fill}" stroke="{stroke}"/>')
    L.append(f'<text x="{lx+24}" y="{legend_y+13}" font-family="sans-serif" font-size="11.5" '
              f'fill="#334155">{esc(label)}</text>')
    lx += 330

foot_y = legend_y + 44
L.append(f'<text x="{w/2}" y="{foot_y}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12.5" fill="#334155">'
         f'{esc("总计 16 次循环 × 每次 8 个连续 f32(32 字节,32%32=0 对齐)——而非把整块打成 16×8=128 个散点访存")}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-ch14-m6-partial-scalarize.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
