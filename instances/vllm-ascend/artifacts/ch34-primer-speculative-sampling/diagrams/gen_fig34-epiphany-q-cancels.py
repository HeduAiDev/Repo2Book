#!/usr/bin/env python3
"""顿悟头图(落差对比):只锚一个洞见——草稿好坏只买速度,永远动不了输出分布。
上半「天真做法:草稿抽了直接当输出 ⇒ 分布 = q」,token C 被系统性高估 3 倍(q=0.3 对真实 p=0.1),警示色。
下半「其实:接受质量 min(p,q) + 残差质量 (p-min) 逐 token 精确拼回 p」,C 被拉回 0.1,q 在加法里整个相消。
数字全部来自 explainer traces(accept_reject / dist_preserving),不即兴。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TOKENS = ["A", "B", "C", "D"]
Q      = {"A": 0.4, "B": 0.2, "C": 0.3, "D": 0.1}   # 草稿分布 q
P      = {"A": 0.5, "B": 0.3, "C": 0.1, "D": 0.1}   # 目标分布 p
ACCEPT = {"A": 0.4, "B": 0.2, "C": 0.1, "D": 0.1}   # min(p,q)
RESID  = {"A": 0.1, "B": 0.1, "C": 0.0, "D": 0.0}   # (1-beta)p' = p - min(p,q), beta=0.8
HERO   = "C"                                          # 落差主角:q 高估最狠处

# --- 几何 ---
BAR_W, GAP = 96, 78
BARS_X0 = 170
SCALE = 190 / 0.5            # 满格 0.5 -> 190px
TITLE_H = 46
PANEL_PLOT_H = 190
MID_H = 92                   # 中间「q 相消」带
w = BARS_X0 + len(TOKENS) * (BAR_W + GAP) - GAP + 300
top_title_y   = 42
base_top      = top_title_y + TITLE_H + PANEL_PLOT_H
mid_top       = base_top + 30
bot_title_y   = mid_top + MID_H + 6
base_bot      = bot_title_y + TITLE_H + PANEL_PLOT_H
foot_y        = base_bot + 66
legend_y      = base_bot + 96
h = legend_y + 60

RED_F, RED_S   = "#fca5a5", "#dc2626"
GRAY_F, GRAY_S = "#e2e8f0", "#94a3b8"
BLUE_F, BLUE_S = "#bfdbfe", "#2563eb"
GRN_F, GRN_S   = "#bbf7d0", "#15803d"
DASH           = "#94a3b8"

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 8" refX="8" refY="4" markerWidth="9" '
     'markerHeight="7" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#dc2626"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

def bar_x(i): return BARS_X0 + i * (BAR_W + GAP)

# ============ 上半:天真做法 ============
L.append(f'<text x="{BARS_X0-110}" y="{top_title_y}" font-family="sans-serif" font-size="17" '
         f'font-weight="bold" fill="#b91c1c">天真做法:草稿抽了直接当输出 &#8658; 分布就是 q（错）</text>')
for i, tok in enumerate(TOKENS):
    bx, q, p = bar_x(i), Q[tok], P[tok]
    qh = q * SCALE
    hero = (tok == HERO)
    fill, stroke = (RED_F, RED_S) if hero else (GRAY_F, GRAY_S)
    L.append(f'<rect x="{bx}" y="{base_top-qh}" width="{BAR_W}" height="{qh}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if hero else 1.5}"/>')
    L.append(f'<text x="{bx+BAR_W/2}" y="{base_top-qh+22}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="14" font-weight="bold" '
             f'fill="{"#7f1d1d" if hero else "#475569"}">{q:.1f}</text>')
    # 真实目标 p 参考短线(露出 C 的超额)
    py = base_top - p * SCALE
    L.append(f'<line x1="{bx-8}" y1="{py}" x2="{bx+BAR_W+8}" y2="{py}" '
             f'stroke="{DASH}" stroke-dasharray="5,3" stroke-width="1.5"/>')
    # token 标签
    L.append(f'<text x="{bx+BAR_W/2}" y="{base_top+24}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="15" font-weight="bold" fill="#0f172a">{esc(tok)}</text>')
# 主角 C 的高估注释
cx = bar_x(TOKENS.index(HERO))
c_qh = Q[HERO] * SCALE
ann_x = cx + BAR_W + 22
L.append(f'<text x="{ann_x}" y="{base_top-c_qh+4}" font-family="sans-serif" font-size="13.5" '
         f'font-weight="bold" fill="#b91c1c">C：q=0.3</text>')
L.append(f'<text x="{ann_x}" y="{base_top-c_qh+24}" font-family="sans-serif" font-size="13.5" '
         f'fill="#b91c1c">= 真实 p=0.1 的 3 倍 &#10007;</text>')
L.append(f'<text x="{ann_x}" y="{base_top-c_qh+44}" font-family="sans-serif" font-size="11.5" '
         f'fill="#94a3b8">虚线 = 真实目标 p</text>')
# 基线
L.append(f'<line x1="{BARS_X0-16}" y1="{base_top}" x2="{bar_x(len(TOKENS)-1)+BAR_W+16}" '
         f'y2="{base_top}" stroke="#94a3b8" stroke-width="1"/>')

# ============ 中间:q 相消带 ============
band_cx = BARS_X0 + (len(TOKENS)*(BAR_W+GAP)-GAP)/2
L.append(f'<line x1="{cx+BAR_W/2}" y1="{mid_top}" x2="{cx+BAR_W/2}" y2="{mid_top+MID_H-18}" '
         f'stroke="#dc2626" stroke-width="2.5" marker-end="url(#a)"/>')
L.append(f'<rect x="{band_cx-250}" y="{mid_top+18}" width="500" height="46" rx="10" '
         f'fill="#fef2f2" stroke="#dc2626" stroke-width="1.5"/>')
L.append(f'<text x="{band_cx}" y="{mid_top+40}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14.5" font-weight="bold" fill="#b91c1c">拒绝 + 残差重采样：把 C 从 0.3 拉回 0.1</text>')
L.append(f'<text x="{band_cx}" y="{mid_top+58}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" fill="#7f1d1d">min(p,q) 的 +q 与残差 (p−min) 的 −q 逐 token 精确相消</text>')

# ============ 下半:其实(拼回 p) ============
L.append(f'<text x="{BARS_X0-110}" y="{bot_title_y}" font-family="sans-serif" font-size="17" '
         f'font-weight="bold" fill="#166534">其实:接受质量 + 残差质量 = p，与草稿分布 q 无关（对）</text>')
for i, tok in enumerate(TOKENS):
    bx, acc, res, p = bar_x(i), ACCEPT[tok], RESID[tok], P[tok]
    acc_h, res_h = acc * SCALE, res * SCALE
    hero = (tok == HERO)
    L.append(f'<rect x="{bx}" y="{base_bot-acc_h}" width="{BAR_W}" height="{max(acc_h,1)}" '
             f'fill="{BLUE_F}" stroke="{BLUE_S}" stroke-width="1.5"/>')
    L.append(f'<text x="{bx+BAR_W/2}" y="{base_bot-acc_h/2+5}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="12.5" fill="#1d4ed8">{acc:.1f}</text>')
    if res_h > 0:
        L.append(f'<rect x="{bx}" y="{base_bot-acc_h-res_h}" width="{BAR_W}" height="{res_h}" '
                 f'fill="{GRN_F}" stroke="{GRN_S}" stroke-width="1.5"/>')
        L.append(f'<text x="{bx+BAR_W/2}" y="{base_bot-acc_h-res_h/2+5}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="12.5" fill="#15803d">{res:.1f}</text>')
    # 合计 = p
    tot_fill = "#b91c1c" if hero else "#0f172a"
    L.append(f'<text x="{bx+BAR_W/2}" y="{base_bot-acc_h-res_h-10}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="14" font-weight="bold" fill="{tot_fill}">={p:.1f}</text>')
    L.append(f'<text x="{bx+BAR_W/2}" y="{base_bot+24}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="15" font-weight="bold" fill="#0f172a">{esc(tok)}</text>')
# 主角 C 拉回注释:放在右侧空白(D 右外),避让 D 柱
c_acc_h = ACCEPT[HERO] * SCALE
note_x = bar_x(len(TOKENS)-1) + BAR_W + 30
L.append(f'<text x="{note_x}" y="{base_bot-90}" font-family="sans-serif" '
         f'font-size="13.5" font-weight="bold" fill="#166534">C：从 0.3</text>')
L.append(f'<text x="{note_x}" y="{base_bot-70}" font-family="sans-serif" '
         f'font-size="13.5" font-weight="bold" fill="#166534">拉回 0.1</text>')
L.append(f'<text x="{note_x}" y="{base_bot-50}" font-family="sans-serif" '
         f'font-size="13.5" fill="#166534">= 真实 p &#10003;</text>')
L.append(f'<line x1="{BARS_X0-16}" y1="{base_bot}" x2="{bar_x(len(TOKENS)-1)+BAR_W+16}" '
         f'y2="{base_bot}" stroke="#94a3b8" stroke-width="1"/>')

# ============ 收束 + 图例 ============
L.append(f'<text x="{BARS_X0-110}" y="{foot_y}" font-family="sans-serif" font-size="15" '
         f'font-weight="bold" fill="#0f172a">一句话：草稿好坏只买速度（接受率 &#946;=0.8），永远动不了输出分布一个字。</text>')
legend = [(RED_F, RED_S, "被 q 拉偏（天真做法高估）"),
          (BLUE_F, BLUE_S, "接受质量 min(p,q)"),
          (GRN_F, GRN_S, "残差质量 (1−&#946;)p′ = p−min")]
lx = BARS_X0 - 110
for fill, stroke, label in legend:
    L.append(f'<rect x="{lx}" y="{legend_y}" width="17" height="17" rx="3" '
             f'fill="{fill}" stroke="{stroke}"/>')
    L.append(f'<text x="{lx+25}" y="{legend_y+14}" font-family="sans-serif" font-size="12.5" '
             f'fill="#334155">{label}</text>')
    lx += 260
L.append(f'<line x1="{lx}" y1="{legend_y+9}" x2="{lx+25}" y2="{legend_y+9}" '
         f'stroke="{DASH}" stroke-dasharray="5,3" stroke-width="1.5"/>')
L.append(f'<text x="{lx+31}" y="{legend_y+14}" font-family="sans-serif" font-size="12.5" '
         f'fill="#334155">真实目标 p 参考线</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig34-epiphany-q-cancels.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  ({w}x{h})")
