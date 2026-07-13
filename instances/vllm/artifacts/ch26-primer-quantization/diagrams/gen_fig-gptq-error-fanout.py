#!/usr/bin/env python3
"""fig-gptq-error-fanout: GPTQ 机制的空间形态——一份误差沿逆 Hessian 列方向一次性扇出。
量化第 0 列产生缩放误差 e_0 = -0.0267 后，这份误差沿 (H⁻¹)_{:,0} 方向摊给右侧所有未量化列：
[0.95,-0.4,0.55] 被推到 [0.9172,-0.3525,0.4069]。误差没被消灭，只是搬到输出最不敏感的方向上。
zoom-in 结构图（非逐列流程表）：一行 4 格，第 0 格量化后三条箭头扇出到右侧三格。
数字全部来自 explainer/traces/m3.json + 补偿公式 arXiv:2210.17323 §3 Eq.2。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(str(s))

# ---- numbers from m3.json ----
E0 = "-0.0267"      # per_column_rows[0].err_scaled
COL0 = {"idx": 0, "w": "0.1", "q": "0.1929", "hdiag": "3.4803", "quantized": True}
COLS = [
    {"idx": 1, "before": "0.95",  "after": "0.9172",  "hdiag": "0.8521"},
    {"idx": 2, "before": "-0.4",  "after": "-0.3525", "hdiag": "0.9526"},
    {"idx": 3, "before": "0.55",  "after": "0.4069",  "hdiag": "0.9361"},
]

PAD = 40
CELL_W, CELL_H = 168, 76
GAP = 74
CELL_Y = 250
START_X = 70
def cell_x(i): return START_X + i * (CELL_W + GAP)
def cell_cx(i): return cell_x(i) + CELL_W / 2

w = cell_x(3) + CELL_W + PAD
TITLE_Y = 46
SUB_Y = 72
FAN_LABEL_Y = 120
VAL_Y = CELL_Y + CELL_H + 26         # before->after under each right cell
HDIAG_Y = VAL_Y + 22                 # hessian diag under each cell
FORM_Y = HDIAG_Y + 40
FORM_H = 52
CAP_Y = FORM_Y + FORM_H + 34
h = CAP_Y + 22

L = []
def add(s): L.append(s)

add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">')
add('<defs>'
    '<marker id="r" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
    '</defs>')
add(f'<rect width="{w}" height="{h}" fill="white"/>')

# title / subtitle
add(f'<text x="{PAD}" y="{TITLE_Y}" font-family="sans-serif" font-size="19" '
    f'font-weight="bold" fill="#0f172a">量化第 0 列的一份误差，沿逆 Hessian 第 0 列方向一次性扇出</text>')
add(f'<text x="{PAD}" y="{SUB_Y}" font-family="sans-serif" font-size="13" '
    f'fill="#64748b">误差没被消灭，只是被搬到输出最不敏感的方向上——右侧未量化列整体挪一小步。</text>')

# fan-out banner label (above cells)
add(f'<text x="{PAD}" y="{FAN_LABEL_Y}" font-family="sans-serif" '
    f'font-size="13.5" font-weight="bold" fill="#dc2626">'
    f'一份误差 e₀ = {esc(E0)} 从 col 0 沿 (H⁻¹)_&#123;:,0&#125; 方向扇向右侧三列 ↘</text>')

# ---- fan-out arrows: from col0 top-center up to a control, down to each target top-center ----
sx, sy = cell_cx(0), CELL_Y            # source: col0 top edge center
for c in COLS:
    tx, ty = cell_cx(c["idx"]), CELL_Y  # target: top edge center
    ctrl_x = (sx + tx) / 2
    ctrl_y = CELL_Y - 118               # arc apex above cells
    add(f'<path d="M {sx:.1f} {sy} Q {ctrl_x:.1f} {ctrl_y} {tx:.1f} {ty}" '
        f'fill="none" stroke="#dc2626" stroke-width="1.8" marker-end="url(#r)"/>')

# ---- cells ----
# col0 (quantized, snapped to gridline)
x0 = cell_x(0)
add(f'<rect x="{x0}" y="{CELL_Y}" width="{CELL_W}" height="{CELL_H}" rx="8" '
    f'fill="#fee2e2" stroke="#dc2626" stroke-width="2.5"/>')
add(f'<text x="{cell_cx(0)}" y="{CELL_Y+24}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="13.5" font-weight="bold" fill="#b91c1c">col 0（刚量化）</text>')
add(f'<text x="{cell_cx(0)}" y="{CELL_Y+46}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="13" fill="#0f172a">{esc(COL0["w"])} → 格线 {esc(COL0["q"])}</text>')
add(f'<text x="{cell_cx(0)}" y="{CELL_Y+66}" text-anchor="middle" font-family="sans-serif" '
    f'font-size="12" fill="#b91c1c">缩放误差 e₀ = {esc(E0)}</text>')

# right cells (unquantized, pushed)
for c in COLS:
    cx = cell_x(c["idx"])
    add(f'<rect x="{cx}" y="{CELL_Y}" width="{CELL_W}" height="{CELL_H}" rx="8" '
        f'fill="#dbeafe" stroke="#1d4ed8" stroke-width="1.5"/>')
    add(f'<text x="{cell_cx(c["idx"])}" y="{CELL_Y+28}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="13.5" font-weight="bold" fill="#1e40af">'
        f'col {c["idx"]}（未量化）</text>')
    add(f'<text x="{cell_cx(c["idx"])}" y="{CELL_Y+54}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="14" fill="#0f172a">'
        f'{esc(c["before"])} → {esc(c["after"])}</text>')
    # hessian diag under each cell
    add(f'<text x="{cell_cx(c["idx"])}" y="{HDIAG_Y}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="11.5" fill="#475569">'
        f'[Ĥ⁻¹]_{c["idx"]}{c["idx"]} = {esc(c["hdiag"])}</text>')
# col0 hessian diag
add(f'<text x="{cell_cx(0)}" y="{HDIAG_Y}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="11.5" fill="#475569">'
    f'[Ĥ⁻¹]_00 = {esc(COL0["hdiag"])}</text>')
# label the before->after row meaning
add(f'<text x="{PAD}" y="{VAL_Y}" font-family="sans-serif" font-size="12" '
    f'fill="#475569">补偿前 → 补偿后：</text>')

# ---- formula box (严谨) ----
add(f'<rect x="{PAD}" y="{FORM_Y}" width="{w-2*PAD}" height="{FORM_H}" rx="8" '
    f'fill="#f8fafc" stroke="#94a3b8" stroke-width="1" stroke-dasharray="5 3"/>')
add(f'<text x="{PAD+16}" y="{FORM_Y+22}" font-family="sans-serif" font-size="13" '
    f'fill="#334155">扇出方向与幅度由补偿公式给定：'
    f'δ_F = -(w_q - quant(w_q)) / [H_F⁻¹]_qq · (H_F⁻¹)_&#123;:,q&#125;</text>')
add(f'<text x="{PAD+16}" y="{FORM_Y+42}" font-family="sans-serif" font-size="11.5" '
    f'fill="#64748b">（arXiv:2210.17323 §3 Eq.2；q=0 时即沿逆 Hessian 第 0 列 (H⁻¹)_&#123;:,0&#125;）</text>')

# ---- caption ----
add(f'<text x="{PAD}" y="{CAP_Y}" font-family="sans-serif" font-size="13.5" '
    f'font-weight="bold" fill="#0f172a">'
    f'一列被推歪，误差沿 (H⁻¹)_&#123;:,0&#125; 摊向右侧三列——量化被搬到最不痛的方向，而非被消灭。</text>')

add('</svg>')
out = Path(__file__).with_name("fig-gptq-error-fanout.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
