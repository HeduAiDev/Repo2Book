#!/usr/bin/env python3
"""fig-quant-epiphany: 顿悟头图（落差揭示，一图只打一拳）。
那一下 = 「256 档只是名义；一个 100x 离群通道把 tensor absmax 撑到 163.4783 之后，
普通通道真正用上的只剩 1.78 档」。
视觉主轴 = 同宽两把尺子的落差：你以为(满格 256 绿) vs 其实(普通通道只剩左端 1.78 格红细缝)。
量化落差 256→1.78 直接量在图上：细缝按 1.78/256 真比例画（≈0.7%），再用放大镜读出 1.78 档。
数字全部来自 explainer/traces/m2.json。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(str(s))

# ---- numbers from m2.json ----
FULL_LEVELS = 256          # params.full_levels
TENSOR_ABSMAX = 163.4783   # params.tensor_absmax_m
NORMAL_ABSMAX = 1.136      # rows[0][1]
NORMAL_LEVELS = 1.78       # rows[0][2] == min_nonoutlier_levels
OUTLIER_LEVELS = 256.0     # outlier_levels

PAD = 40
RULER_X = 220
RULER_W = 780
RULER_R = RULER_X + RULER_W
w = RULER_R + PAD
BAR_H = 44

sliver_frac = NORMAL_LEVELS / FULL_LEVELS          # 1.78/256
sliver_w = RULER_W * sliver_frac                    # true-scale sliver width (~5.4px)

L = []
def add(s): L.append(s)

# geometry cursors
TITLE_Y = 46
SUB_Y = 72
BAR_A_Y = 120                     # 你以为
TRANS_Y = BAR_A_Y + BAR_H + 40    # 但其实
BAR_B_Y = TRANS_Y + 34            # 其实
CALL_Y = BAR_B_Y + BAR_H + 92     # 放大镜 callout
CALL_H = 84
CALL_W = 380
PUNCH_Y = CALL_Y + CALL_H + 52
CAP_Y = PUNCH_Y + 34
h = CAP_Y + 24

add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">')
add('<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
    '<marker id="g" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '</defs>')
add(f'<rect width="{w}" height="{h}" fill="white"/>')

# title / subtitle
add(f'<text x="{PAD}" y="{TITLE_Y}" font-family="sans-serif" font-size="20" '
    f'font-weight="bold" fill="#0f172a">你以为 8-bit 给每个数 256 个刻度……</text>')
add(f'<text x="{PAD}" y="{SUB_Y}" font-family="sans-serif" font-size="13.5" '
    f'fill="#64748b">一个 100× 离群通道把 tensor absmax 撑到 {esc(TENSOR_ABSMAX)} 之后，'
    f'刻度尺被它一人占满。</text>')

# ---------- Ruler A : 你以为（满格 256 绿）----------
add(f'<text x="{PAD}" y="{BAR_A_Y+BAR_H/2+5}" font-family="sans-serif" font-size="14" '
    f'font-weight="bold" fill="#047857">你以为</text>')
add(f'<rect x="{RULER_X}" y="{BAR_A_Y}" width="{RULER_W}" height="{BAR_H}" rx="4" '
    f'fill="#a7f3d0" stroke="#047857" stroke-width="1.5"/>')
# ruler ticks (32 evenly, purely for ruler feel)
NTICK = 32
for i in range(NTICK + 1):
    tx = RULER_X + RULER_W * i / NTICK
    th = 12 if i % 8 == 0 else 7
    add(f'<line x1="{tx:.1f}" y1="{BAR_A_Y}" x2="{tx:.1f}" y2="{BAR_A_Y+th}" '
        f'stroke="#047857" stroke-width="1"/>')
add(f'<text x="{RULER_X+RULER_W/2}" y="{BAR_A_Y+BAR_H/2+5}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="14" font-weight="bold" fill="#065f46">'
    f'普通通道 → {esc(FULL_LEVELS)} 档随便用</text>')

# transition
add(f'<text x="{PAD}" y="{TRANS_Y+6}" font-family="sans-serif" font-size="16" '
    f'font-weight="bold" fill="#dc2626">但其实 ↓</text>')

# ---------- Ruler B : 其实（同宽，普通通道只剩左端 1.78 格红细缝）----------
add(f'<text x="{PAD}" y="{BAR_B_Y+BAR_H/2+5}" font-family="sans-serif" font-size="14" '
    f'font-weight="bold" fill="#b91c1c">其实</text>')
# grey wasteland = 被离群通道没收的 254 档
add(f'<rect x="{RULER_X}" y="{BAR_B_Y}" width="{RULER_W}" height="{BAR_H}" rx="4" '
    f'fill="#e2e8f0" stroke="#94a3b8" stroke-width="1.5"/>')
for i in range(NTICK + 1):
    tx = RULER_X + RULER_W * i / NTICK
    th = 12 if i % 8 == 0 else 7
    add(f'<line x1="{tx:.1f}" y1="{BAR_B_Y}" x2="{tx:.1f}" y2="{BAR_B_Y+th}" '
        f'stroke="#94a3b8" stroke-width="1"/>')
# tiny true-scale red sliver at far left = 普通通道真正用到的 1.78 档
add(f'<rect x="{RULER_X}" y="{BAR_B_Y}" width="{sliver_w:.2f}" height="{BAR_H}" '
    f'fill="#dc2626" stroke="#dc2626"/>')
# wasteland label
add(f'<text x="{RULER_X+RULER_W*0.52}" y="{BAR_B_Y+BAR_H/2+5}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="13.5" fill="#64748b">'
    f'← 254 档空转：普通值全挤进最左那道红缝 →</text>')
# outlier pin at far right
add(f'<line x1="{RULER_R}" y1="{BAR_B_Y-12}" x2="{RULER_R}" y2="{BAR_B_Y+BAR_H+8}" '
    f'stroke="#b91c1c" stroke-width="2.5"/>')
add(f'<text x="{RULER_R}" y="{BAR_B_Y-18}" text-anchor="end" font-family="sans-serif" '
    f'font-size="12.5" fill="#b91c1c">离群通道 100× → 值到 {esc(TENSOR_ABSMAX)}，独占 {esc(int(OUTLIER_LEVELS))} 档</text>')
# end scale labels
add(f'<text x="{RULER_X}" y="{BAR_B_Y+BAR_H+18}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="11.5" fill="#475569">0</text>')
add(f'<text x="{RULER_R}" y="{BAR_B_Y+BAR_H+18}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="11.5" fill="#475569">{esc(TENSOR_ABSMAX)}</text>')

# ---------- 放大镜 callout : 把红缝放大读出 1.78 档 ----------
CALL_X = RULER_X
# zoom lines from sliver top-corners down to callout top-corners
add(f'<line x1="{RULER_X:.1f}" y1="{BAR_B_Y+BAR_H}" x2="{CALL_X}" y2="{CALL_Y}" '
    f'stroke="#dc2626" stroke-width="1" stroke-dasharray="4 3"/>')
add(f'<line x1="{RULER_X+sliver_w:.2f}" y1="{BAR_B_Y+BAR_H}" x2="{CALL_X+CALL_W}" y2="{CALL_Y}" '
    f'stroke="#dc2626" stroke-width="1" stroke-dasharray="4 3"/>')
add(f'<text x="{RULER_X+sliver_w+10:.1f}" y="{BAR_B_Y+BAR_H-4}" font-family="sans-serif" '
    f'font-size="11" fill="#dc2626">放大这道红缝 ↓</text>')
# callout box (blown-up sliver)
add(f'<rect x="{CALL_X}" y="{CALL_Y}" width="{CALL_W}" height="{CALL_H}" rx="6" '
    f'fill="#fee2e2" stroke="#dc2626" stroke-width="2"/>')
# blown-up mini ruler 0..1.136 with ~2 ticks
MR_X = CALL_X + 24
MR_W = CALL_W - 48
MR_Y = CALL_Y + 30
add(f'<line x1="{MR_X}" y1="{MR_Y}" x2="{MR_X+MR_W}" y2="{MR_Y}" '
    f'stroke="#b91c1c" stroke-width="2"/>')
for k in range(3):  # 0, 1, ~1.78 ticks
    tx = MR_X + MR_W * (k / NORMAL_LEVELS) if k <= NORMAL_LEVELS else MR_X + MR_W
    tx = min(tx, MR_X + MR_W)
    add(f'<line x1="{tx:.1f}" y1="{MR_Y-7}" x2="{tx:.1f}" y2="{MR_Y+7}" '
        f'stroke="#b91c1c" stroke-width="2"/>')
    add(f'<text x="{tx:.1f}" y="{MR_Y-11}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="10.5" fill="#b91c1c">{k}</text>')
add(f'<text x="{CALL_X+CALL_W/2}" y="{CALL_Y+CALL_H-14}" text-anchor="middle" '
    f'font-family="sans-serif" font-size="13" font-weight="bold" fill="#b91c1c">'
    f'普通通道 absmax {esc(NORMAL_ABSMAX)} → 只够 {esc(NORMAL_LEVELS)} 档（不到 2 格）</text>')

# ---------- punch line ----------
add(f'<text x="{PAD}" y="{PUNCH_Y}" font-family="sans-serif" font-size="19" '
    f'font-weight="bold" fill="#0f172a">落差：'
    f'<tspan fill="#047857">{esc(FULL_LEVELS)} 档</tspan> → '
    f'<tspan fill="#dc2626">{esc(NORMAL_LEVELS)} 档</tspan>'
    f'，254 档被一个离群通道没收。</text>')
add(f'<text x="{PAD}" y="{CAP_Y}" font-family="sans-serif" font-size="13" '
    f'fill="#475569">本章三篇论文（SmoothQuant / GPTQ / AWQ）全在抢救这被没收的 254 档。</text>')

add('</svg>')
out = Path(__file__).with_name("fig-quant-epiphany.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w}x{h})")
