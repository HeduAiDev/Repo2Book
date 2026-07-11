#!/usr/bin/env python3
"""tensor-flow 模板:draft 层交叉注意力的 Q/K/V 拆分——Q 只由 draft token(H_d)产生,
K/V 由 [H_t(context); H_d(query)] 沿序列轴拼接而成,注意力在拼接后的长序列上非因果进行。
数字来自 explainer/traces/cross_attention.json(mechanism cross-attention-qkv-split)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "Q 只出自 draft token,K/V 由 [target 上下文; draft token] 拼接——非因果打分"
SUBTITLE = "num_ctx=3(target 上下文)、block_size=4(bonus+3 mask)——K/V 拼接后序列长 = 3+4 = 7"

PAD, TOP = 46, 110
w = 980

# H_t (context) box, H_d box -> both feed K/V; H_d alone feeds Q
HT_X, HD_X = PAD, PAD + 260
SRC_Y = TOP
SRC_W, SRC_H = 200, 56

H = 620
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {H}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker></defs>',
     f'<rect width="{w}" height="{H}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="15.5" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

# H_t box
L.append(f'<rect x="{HT_X}" y="{SRC_Y}" width="{SRC_W}" height="{SRC_H}" rx="8" '
          'fill="#bfdbfe" stroke="#2563eb" stroke-width="1.6"/>')
L.append(f'<text x="{HT_X+SRC_W/2}" y="{SRC_Y+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#1e3a8a">H_t(target 上下文)</text>')
L.append(f'<text x="{HT_X+SRC_W/2}" y="{SRC_Y+40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#1e40af">num_ctx = 3</text>')

# H_d box
L.append(f'<rect x="{HD_X}" y="{SRC_Y}" width="{SRC_W}" height="{SRC_H}" rx="8" '
          'fill="#ede9fe" stroke="#7c3aed" stroke-width="1.6"/>')
L.append(f'<text x="{HD_X+SRC_W/2}" y="{SRC_Y+22}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#4c1d95">H_d(draft token)</text>')
L.append(f'<text x="{HD_X+SRC_W/2}" y="{SRC_Y+40}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="10.5" fill="#5b21b6">block_size = 4(bonus+3 mask)</text>')

# K/V box (below, receives from both)
KV_Y = SRC_Y + SRC_H + 76
KV_X = HT_X + 40
KV_W = 360
L.append(f'<rect x="{KV_X}" y="{KV_Y}" width="{KV_W}" height="60" rx="8" '
          'fill="#fef3c7" stroke="#d97706" stroke-width="1.8"/>')
L.append(f'<text x="{KV_X+KV_W/2}" y="{KV_Y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#92400e">K/V = concat([H_t; H_d])</text>')
L.append(f'<text x="{KV_X+KV_W/2}" y="{KV_Y+44}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#92400e">序列长 = 3 + 4 = 7</text>')

# arrows H_t -> KV, H_d -> KV
L.append(f'<line x1="{HT_X+SRC_W/2}" y1="{SRC_Y+SRC_H}" x2="{KV_X+KV_W*0.28}" y2="{KV_Y}" '
          'stroke="#2563eb" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<line x1="{HD_X+SRC_W/2}" y1="{SRC_Y+SRC_H}" x2="{KV_X+KV_W*0.72}" y2="{KV_Y}" '
          'stroke="#7c3aed" stroke-width="1.8" marker-end="url(#b)"/>')

# Q box (only from H_d)
Q_X = HD_X + SRC_W + 90
Q_Y = KV_Y
L.append(f'<rect x="{Q_X}" y="{Q_Y}" width="200" height="60" rx="8" '
          'fill="#ede9fe" stroke="#7c3aed" stroke-width="1.8"/>')
L.append(f'<text x="{Q_X+100}" y="{Q_Y+24}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#4c1d95">Q(仅 H_d 投影)</text>')
L.append(f'<text x="{Q_X+100}" y="{Q_Y+44}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#5b21b6">query 位数 = 4</text>')
L.append(f'<line x1="{HD_X+SRC_W*0.85}" y1="{SRC_Y+SRC_H}" x2="{Q_X+40}" y2="{Q_Y}" '
          'stroke="#7c3aed" stroke-width="1.8" marker-end="url(#b)" stroke-dasharray="5,3"/>')

# attention score box (below, fed by both K/V and Q)
ATT_Y = KV_Y + 60 + 76
ATT_X = KV_X - 20
ATT_W = 380
L.append(f'<rect x="{ATT_X}" y="{ATT_Y}" width="{ATT_W}" height="66" rx="8" '
          'fill="#e2e8f0" stroke="#334155" stroke-width="1.8"/>')
L.append(f'<text x="{ATT_X+ATT_W/2}" y="{ATT_Y+26}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="13" font-weight="bold" fill="#0f172a">非因果 softmax(Q x K^T)</text>')
L.append(f'<text x="{ATT_X+ATT_W/2}" y="{ATT_Y+46}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="#334155">打分矩阵 4 x 7(query 4 行 x K/V 7 列)</text>')

L.append(f'<line x1="{KV_X+KV_W/2}" y1="{KV_Y+60}" x2="{ATT_X+ATT_W*0.3}" y2="{ATT_Y}" '
          'stroke="#d97706" stroke-width="1.8" marker-end="url(#a)"/>')
L.append(f'<line x1="{Q_X+100}" y1="{Q_Y+60}" x2="{ATT_X+ATT_W*0.75}" y2="{ATT_Y}" '
          'stroke="#7c3aed" stroke-width="1.8" marker-end="url(#b)"/>')

foot_y = ATT_Y + 66 + 40
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">扰动最后一个 mask 位(idx 3),bonus 位(idx 0)输出变化 ‖Δ‖=0.2788——非因果:后位能影响前位。</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">移除注入的 context K/V(3 列),输出变化 ‖Δ‖=1.0215——注入的 K/V 确实参与了注意力打分。</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-cross-attention-qkv.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
