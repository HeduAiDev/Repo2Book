#!/usr/bin/env python3
"""tensor-flow 模板:MTP(Multi-Token Prediction)模块因果链。深度 k 把上一深度隐状态 h^{k-1}
与 t_{i+k} 的 embedding 组合送入 TRM_k,输出 h^k 再喂给深度 k+1——串行因果链;每深一层,
有效预测窗口收缩一个位置(T-k)。数字来自 explainer/traces/mtp_causal_chain.json(T=6)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "MTP 模块因果链:深度间隐状态串行传递,有效窗口逐层收缩"
SUBTITLE = "深度 k:TRM_k(RMSNorm(h^{k-1}), RMSNorm(Emb(t_{i+k}))) -> h^k —— T=6(输入序列长度)"

DEPTHS = [1, 2, 3]
VALID_LEN = {1: 5, 2: 4, 3: 3}

BOX_W, BOX_H, GAP_X, PAD, TOP = 150, 62, 74, 46, 90
EMB_W, EMB_H = 128, 44
h0_W = 96

w = PAD * 2 + h0_W + GAP_X + len(DEPTHS) * (BOX_W + GAP_X) - GAP_X

chain_y0 = TOP + 60
strip_y0 = chain_y0 + BOX_H + 42 + EMB_H + 46
foot_y0 = strip_y0 + 40 + 32
h = foot_y0 + 20

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#334155"/></marker>'
     '<marker id="b" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{PAD}" y="{PAD-12}" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#1e40af">{esc(TITLE)}</text>',
     f'<text x="{PAD}" y="{PAD+8}" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

chain_y = TOP + 60
by = chain_y

# h0 box (initial hidden state from main model)
x0 = PAD
L.append(f'<rect x="{x0}" y="{by}" width="{h0_W}" height="{BOX_H}" rx="8" '
          'fill="#e2e8f0" stroke="#64748b" stroke-width="1.5"/>')
L.append(f'<text x="{x0+h0_W/2}" y="{by+BOX_H/2-2}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="12" fill="#334155">h^0</text>')
L.append(f'<text x="{x0+h0_W/2}" y="{by+BOX_H/2+14}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="9.5" fill="#64748b">(主模型输出)</text>')

prev_right = x0 + h0_W
depth_x = []
for i, k in enumerate(DEPTHS):
    x = prev_right + GAP_X
    depth_x.append(x)
    # arrow h^{k-1} -> depth box
    L.append(f'<line x1="{prev_right}" y1="{by+BOX_H/2}" x2="{x}" y2="{by+BOX_H/2}" '
              'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
    L.append(f'<text x="{(prev_right+x)/2}" y="{by+BOX_H/2-8}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="#334155">h^{k-1}</text>')
    # depth box
    L.append(f'<rect x="{x}" y="{by}" width="{BOX_W}" height="{BOX_H}" rx="8" '
              'fill="#ede9fe" stroke="#7c3aed" stroke-width="1.5"/>')
    L.append(f'<text x="{x+BOX_W/2}" y="{by+BOX_H/2-2}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="#5b21b6">深度 {k}: TRM_{k}</text>')
    L.append(f'<text x="{x+BOX_W/2}" y="{by+BOX_H/2+16}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="#6d28d9">valid_len={VALID_LEN[k]}</text>')
    # embedding input from below
    ey = by + BOX_H + 42
    ex = x + BOX_W / 2 - EMB_W / 2
    L.append(f'<rect x="{ex}" y="{ey}" width="{EMB_W}" height="{EMB_H}" rx="6" '
              'fill="#dbeafe" stroke="#2563eb" stroke-width="1.5"/>')
    L.append(f'<text x="{ex+EMB_W/2}" y="{ey+EMB_H/2+4}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="#1d4ed8">Emb(t_i+{k})</text>')
    L.append(f'<line x1="{ex+EMB_W/2}" y1="{ey}" x2="{x+BOX_W/2}" y2="{by+BOX_H}" '
              'stroke="#2563eb" stroke-width="1.5" marker-end="url(#b)"/>')
    prev_right = x + BOX_W

# final h^D output arrow
final_x = prev_right + 40
L.append(f'<line x1="{prev_right}" y1="{by+BOX_H/2}" x2="{final_x}" y2="{by+BOX_H/2}" '
          'stroke="#334155" stroke-width="1.5" marker-end="url(#a)"/>')
L.append(f'<text x="{(prev_right+final_x)/2}" y="{by+BOX_H/2-8}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="10" fill="#334155">h^3</text>')

# valid-window shrink strip
strip_y = by + BOX_H + 42 + EMB_H + 46
L.append(f'<text x="{PAD}" y="{strip_y-14}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="#0f172a">有效预测窗口每深一层收缩一个位置(T-k)</text>')
win_labels = [("T=6", "输入序列", "#e2e8f0", "#334155")]
for k in DEPTHS:
    win_labels.append((f"T-{k}={VALID_LEN[k]}", f"深度{k}有效窗口",
                        "#ede9fe" if False else "#fef3c7", "#b45309"))
wx = PAD
WIN_W = 120
for i, (val, lab, fill, stroke) in enumerate(win_labels):
    L.append(f'<rect x="{wx}" y="{strip_y}" width="{WIN_W}" height="40" rx="6" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{wx+WIN_W/2}" y="{strip_y+17}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="{stroke}">{esc(val)}</text>')
    L.append(f'<text x="{wx+WIN_W/2}" y="{strip_y+32}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="9.5" fill="{stroke}">{esc(lab)}</text>')
    if i < len(win_labels) - 1:
        L.append(f'<line x1="{wx+WIN_W+4}" y1="{strip_y+20}" x2="{wx+WIN_W+26}" '
                  f'y2="{strip_y+20}" stroke="#64748b" stroke-width="1.5" marker-end="url(#a)"/>')
    wx += WIN_W + 30

foot_y = strip_y + 40 + 32
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11" '
          f'fill="#64748b">紫=深度 Transformer block(串行因果链,h^k 只能等 h^k-1 算完才能算);'
          f'蓝=每深度独有的 token embedding 输入;embedding 层与输出头跨深度共享</text>')
L.append('</svg>')
out = Path(__file__).with_name("fig33-mtp-causal-chain.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
