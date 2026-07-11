#!/usr/bin/env python3
"""state-table 模板:块内位置权重 w_k = exp(-(k-1)/gamma) 指数衰减,早位置犯错的加权损失更高。
数字来自 explainer/traces/position_loss.json(mechanism training-anchor-mask-weighted-loss,gamma=4)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "块内位置权重按 exp(-(k-1)/gamma) 指数衰减——早位置犯错的加权损失明显更高"
SUBTITLE = "gamma=4:w_1=1.0(基准)到 w_4=0.4724(末位仅约首位 47%)"

WEIGHTS = [("k=1", 1.0), ("k=2", 0.7788), ("k=3", 0.6065), ("k=4", 0.4724)]

PAD, TOP = 46, 118
BAR_W, BAR_GAP = 100, 44
SCALE = 220  # px per unit weight
LOSS_W = 300
w = PAD * 2 + LOSS_W * 2 + 60  # 驱动画布宽度的是下方两个损失对比框,须容纳它们
BASELINE_Y = TOP + SCALE + 10

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} 620">',
     '<rect width="' + str(w) + '" height="620" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="15.5" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>',
     f'<line x1="{PAD}" y1="{BASELINE_Y}" x2="{w-PAD}" y2="{BASELINE_Y}" '
     'stroke="#94a3b8" stroke-width="1.5"/>']

bars_total_w = len(WEIGHTS) * BAR_W + (len(WEIGHTS) - 1) * BAR_GAP
start_x = (w - bars_total_w) / 2
for i, (lab, val) in enumerate(WEIGHTS):
    bx = start_x + i * (BAR_W + BAR_GAP)
    bar_h = val * SCALE
    by = BASELINE_Y - bar_h
    shade = 220 - i * 34
    fill = f"rgb({shade},{min(255,shade+30)},255)"
    L.append(f'<rect x="{bx}" y="{by}" width="{BAR_W}" height="{bar_h}" rx="6" '
              f'fill="{fill}" stroke="#1d4ed8" stroke-width="1.6"/>')
    L.append(f'<text x="{bx+BAR_W/2}" y="{by-10}" text-anchor="middle" font-family="sans-serif" '
              f'font-size="14" font-weight="bold" fill="#1e3a8a">{val}</text>')
    L.append(f'<text x="{bx+BAR_W/2}" y="{BASELINE_Y+22}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" fill="#334155">{esc(lab)}</text>')

# loss comparison boxes below
LOSS_Y = BASELINE_Y + 66
LOSS_H = 84
lx1 = PAD
lx2 = PAD + LOSS_W + 60
L.append(f'<text x="{PAD}" y="{LOSS_Y-14}" font-family="sans-serif" font-size="13" '
          f'font-weight="bold" fill="#0f172a">同幅误差:错在早位置 vs 错在晚位置的加权损失</text>')
L.append(f'<rect x="{lx1}" y="{LOSS_Y}" width="{LOSS_W}" height="{LOSS_H}" rx="8" '
          'fill="#fee2e2" stroke="#dc2626" stroke-width="1.8"/>')
L.append(f'<text x="{lx1+LOSS_W/2}" y="{LOSS_Y+30}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" fill="#991b1b">位置 1 犯错的加权损失</text>')
L.append(f'<text x="{lx1+LOSS_W/2}" y="{LOSS_Y+58}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="20" font-weight="bold" fill="#b91c1c">3.4995</text>')
L.append(f'<rect x="{lx2}" y="{LOSS_Y}" width="{LOSS_W}" height="{LOSS_H}" rx="8" '
          'fill="#dcfce7" stroke="#16a34a" stroke-width="1.8"/>')
L.append(f'<text x="{lx2+LOSS_W/2}" y="{LOSS_Y+30}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" fill="#166534">位置 4 犯错的加权损失</text>')
L.append(f'<text x="{lx2+LOSS_W/2}" y="{LOSS_Y+58}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="20" font-weight="bold" fill="#15803d">1.6531</text>')

foot_y = LOSS_Y + LOSS_H + 40
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">早位置犯错代价约是晚位置的 2.1169 倍(3.4995 / 1.6531)——训练目标本身就编码「早位置更值钱」。</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11.5" '
          f'fill="#334155">块扩散不是多轮迭代去噪,而是单次前向内的掩码填空 + 位置加权——与「并行=每位置同等重要」的直觉恰好相反。</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-position-weight-decay.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
