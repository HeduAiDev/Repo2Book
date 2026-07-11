#!/usr/bin/env python3
"""before-after 模板:自回归起草 vs 块扩散起草——耗时随块大小 gamma 的变化对比。
左panel: 自回归,T_draft = gamma*t_step 随 gamma 线性上涨(前向次数= gamma)。
右panel: 块扩散,T_draft 恒 = t_parallel(前向次数恒为 1),与 gamma 无关。
数字来自 explainer/traces/block_diffusion.json(见 explainer.json mechanism block-diffusion-parallel-drafting)。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

GAMMAS = [4, 8, 16]
AR_CALLS = {4: 4, 8: 8, 16: 16}
AR_T = {4: 0.8, 8: 1.6, 16: 3.2}
DIFF_CALLS = 1
DIFF_T = 0.5

TITLE = "自回归起草耗时随块大小线性涨,块扩散起草恒定不变"
SUBTITLE = "T_draft(自回归) = gamma * t_step(t_step=0.2)  vs  T_draft(块扩散) = t_parallel(恒 0.5),与 gamma 无关"

PANEL_W = 320
BAR_W = 56
BAR_GAP = 40
PAD = 46
TOP = 118
SCALE = 70  # px per unit T_draft
MAX_T = 3.2
BASELINE_Y = TOP + MAX_T * SCALE + 10

w = PAD * 2 + PANEL_W * 2 + 70
h = BASELINE_Y + 140

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{w/2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="16" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{w/2}" y="52" text-anchor="middle" font-family="sans-serif" font-size="12" '
     f'fill="#64748b">{esc(SUBTITLE)}</text>']

def panel(px, title, calls_map, t_map, color, stroke, is_const):
    cx = px + PANEL_W / 2
    L.append(f'<text x="{cx}" y="{TOP-30}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="14" font-weight="bold" fill="#0f172a">{esc(title)}</text>')
    # baseline
    L.append(f'<line x1="{px}" y1="{BASELINE_Y}" x2="{px+PANEL_W}" y2="{BASELINE_Y}" '
             'stroke="#94a3b8" stroke-width="1.5"/>')
    total_bars_w = len(GAMMAS) * BAR_W + (len(GAMMAS) - 1) * BAR_GAP
    start_x = cx - total_bars_w / 2
    for i, g in enumerate(GAMMAS):
        bx = start_x + i * (BAR_W + BAR_GAP)
        t_val = t_map[g]
        bar_h = t_val * SCALE
        by = BASELINE_Y - bar_h
        L.append(f'<rect x="{bx}" y="{by}" width="{BAR_W}" height="{bar_h}" rx="4" '
                  f'fill="{color}" stroke="{stroke}" stroke-width="1.5"/>')
        L.append(f'<text x="{bx+BAR_W/2}" y="{by-24}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11.5" fill="#334155">'
                  f'前向×{calls_map[g]}</text>')
        L.append(f'<text x="{bx+BAR_W/2}" y="{by-8}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                  f'fill="{stroke}">{t_val}</text>')
        L.append(f'<text x="{bx+BAR_W/2}" y="{BASELINE_Y+20}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" fill="#334155">'
                  f'{esc("gamma=" + str(g))}</text>')
    return

panel(PAD, "自回归起草(逐 token 自回归)", AR_CALLS, AR_T, "#fecaca", "#dc2626", False)
panel(PAD + PANEL_W + 70, "块扩散起草(一次前向出整块)", {g: DIFF_CALLS for g in GAMMAS},
      {g: DIFF_T for g in GAMMAS}, "#bbf7d0", "#16a34a", True)

# connecting arrow with delta annotation
mid_y = TOP - 4
L.append(f'<line x1="{PAD+PANEL_W+8}" y1="{mid_y}" x2="{PAD+PANEL_W+62}" y2="{mid_y}" '
         'stroke="#64748b" stroke-width="2" marker-end="url(#a)"/>')
L.append(f'<text x="{PAD+PANEL_W+35}" y="{mid_y-8}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" fill="#64748b">{esc("换起草方式")}</text>')

foot_y = BASELINE_Y + 60
L.append(f'<text x="{PAD}" y="{foot_y}" font-family="sans-serif" font-size="11.5" '
         f'fill="#334155">红=自回归:前向次数恒 = gamma(4/8/16),T_draft 随 gamma 线性涨(0.8/1.6/3.2)</text>')
L.append(f'<text x="{PAD}" y="{foot_y+20}" font-family="sans-serif" font-size="11.5" '
         f'fill="#334155">绿=块扩散:前向次数恒 = 1(与 gamma 无关),T_draft 恒 = 0.5——gamma=16 时自回归是块扩散的 6.4 倍</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-block-diffusion-vs-ar.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
