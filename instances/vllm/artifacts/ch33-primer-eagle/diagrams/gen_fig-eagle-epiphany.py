#!/usr/bin/env python3
"""顿悟头图 fig-eagle-epiphany（§2.5 五步法）。
那一下：草稿的原料——特征 f——是目标前向白送的；朴素路线要从零重算一个完整小模型，
EAGLE 只捡起白送的半成品接着画一步（两层可训练）反而更准。
视觉主轴=落差对比：左『从零重算』(整个 7B 模型/宽重/0.6) vs 右『接着画』(流水线上白送的岔口/0.24B/0.8)。
所有坐标由常量/循环算出；文本全 esc()。数字来自条目 numbers（arXiv:2401.15077 §1/§3.1/§4）。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(str(s))

# ---- 视觉语言 ----
NAVY = "#0f172a"; MUTE = "#475569"
RED_F, RED_S = "#fee2e2", "#dc2626"      # 朴素/贵
GRN_F, GRN_S = "#dcfce7", "#16a34a"      # EAGLE/省
AMB_F, AMB_S = "#fef08a", "#d97706"      # 白送 高亮
NEU_F, NEU_S = "#e2e8f0", "#64748b"      # 中性 chip
PIP_F, PIP_S = "#dbeafe", "#3b82f6"      # 目标前向流水线

W, H = 1160, 796
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>'
         '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
         'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
         '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
         'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
         '<marker id="ae" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
         'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker>'
         '<marker id="aw" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" '
         'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#d97706"/></marker>'
         '</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

def text(x, y, s, size=14, anchor="middle", fill=NAVY, bold=False, italic=False):
    b = ' font-weight="bold"' if bold else ''
    it = ' font-style="italic"' if italic else ''
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="sans-serif" '
             f'font-size="{size}"{b}{it} fill="{fill}">{esc(s)}</text>')

def box(cx, y, w, h, s, fill, stroke, size=14, bold=False, sw=2, tfill=NAVY, rx=8):
    L.append(f'<rect x="{cx-w/2:.1f}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
    text(cx, y + h/2 + size*0.35, s, size=size, bold=bold, fill=tfill)

def varrow(cx, y0, y1, mk="url(#ag)", color="#64748b", sw=1.8):
    L.append(f'<line x1="{cx}" y1="{y0}" x2="{cx}" y2="{y1}" stroke="{color}" '
             f'stroke-width="{sw}" marker-end="{mk}"/>')

# ===== A. 标题带 =====
text(W/2, 36, "顿悟：草稿的原料是白送的", size=25, bold=True)
text(W/2, 65, "目标模型每次验证前向，已经把每个位置的『下一步念头』——特征 f——算好白送出来了",
     size=14.5, fill=MUTE)

# ===== 面板几何 =====
PT = 96
LCX, RCX = 300, 858            # 左右面板中心
PANEL_W = 500
LX0, RX0 = LCX-PANEL_W/2, RCX-PANEL_W/2
PANEL_TOP, PANEL_BOT = 84, 512
# 面板底板
for cx, fill, stroke in [(LCX, "#fef7f7", RED_S), (RCX, "#f5fdf8", GRN_S)]:
    L.append(f'<rect x="{cx-PANEL_W/2:.1f}" y="{PANEL_TOP}" width="{PANEL_W}" '
             f'height="{PANEL_BOT-PANEL_TOP}" rx="14" fill="{fill}" stroke="{stroke}" '
             f'stroke-width="1.5" stroke-dasharray="2 4"/>')

# 中缝：以为→其实 反转揭示
text(W/2, 150, "以为", size=13, fill=RED_S, bold=True)
text(W/2, 300, "其实", size=15, fill=GRN_S, bold=True)
L.append(f'<line x1="{LX0+PANEL_W+8:.1f}" y1="255" x2="{RX0-8:.1f}" y2="255" '
         f'stroke="{GRN_S}" stroke-width="3" marker-end="url(#ae)"/>')

# ===== B. 左面板：从零重算 =====
text(LCX, PANEL_TOP+26, "❶ 从零重算", size=17, bold=True, fill=RED_S)
# token 起点
box(LCX, 128, 130, 34, "prompt token", NEU_F, NEU_S, size=13)
varrow(LCX, 162, 190, mk="url(#ar)", color=RED_S)
# 一整个模型：宽重方块，内部堆多层 → 视觉体量
MBW, MBX, MBY, MBH = 300, LCX-150, 196, 150
L.append(f'<rect x="{MBX}" y="{MBY}" width="{MBW}" height="{MBH}" rx="10" '
         f'fill="{RED_F}" stroke="{RED_S}" stroke-width="2.5"/>')
NLAYER = 6
lay_h = 12; gap = (MBH - 40 - NLAYER*lay_h) / (NLAYER-1)
for i in range(NLAYER):
    ly = MBY + 14 + i*(lay_h+gap)
    L.append(f'<rect x="{MBX+22}" y="{ly:.1f}" width="{MBW-44}" height="{lay_h}" rx="3" '
             f'fill="#fca5a5" stroke="{RED_S}" stroke-width="0.8"/>')
text(LCX, MBY+MBH-14, "完整小 LLM 草稿模型", size=13.5, bold=True, fill=RED_S)
varrow(LCX, MBY+MBH, MBY+MBH+26, mk="url(#ar)", color=RED_S)
box(LCX, MBY+MBH+28, 130, 34, "草稿 token", NEU_F, NEU_S, size=13)
# 落差注记
text(LCX, 430, "整个模型从 token 从零重算", size=13.5, fill=NAVY)
box(LCX, 448, 340, 44, "给 70B 当草稿要一个完整 7B 模型", RED_F, RED_S,
    size=13.5, bold=True, tfill=RED_S)

# ===== C. 右面板：接着画一步（流水线上白送的岔口）=====
text(RCX, PANEL_TOP+26, "❷ 接着画一步", size=17, bold=True, fill=GRN_S)
# 目标前向流水线（横向）：token → 目标多层前向 → 特征 f（白送）
py = 130; ph = 40
tok_w, mid_w, f_w = 96, 170, 150
tx = RX0 + 28
box(tx+tok_w/2, py, tok_w, ph, "token", PIP_F, PIP_S, size=12.5)
mx = tx+tok_w+38
box(mx+mid_w/2, py, mid_w, ph, "目标模型 · 多层前向", PIP_F, PIP_S, size=12.5)
fx = mx+mid_w+40
FCX = fx+f_w/2
L.append(f'<rect x="{fx}" y="{py-3}" width="{f_w}" height="{ph+6}" rx="9" '
         f'fill="{AMB_F}" stroke="{AMB_S}" stroke-width="2.5"/>')
text(FCX, py+15, "特征 f", size=13.5, bold=True, fill=NAVY)
text(FCX, py+31, "✦ 白送", size=12, bold=True, fill=AMB_S)
# 流水线内部箭头
L.append(f'<line x1="{tx+tok_w}" y1="{py+ph/2}" x2="{mx-4}" y2="{py+ph/2}" '
         f'stroke="{PIP_S}" stroke-width="1.6" marker-end="url(#ag)"/>')
L.append(f'<line x1="{mx+mid_w}" y1="{py+ph/2}" x2="{fx-4}" y2="{py+ph/2}" '
         f'stroke="{PIP_S}" stroke-width="1.6" marker-end="url(#ag)"/>')
# 从 f 岔出一条细支 → 草稿头（就一小节）
HCX, HY, HW, HH = RCX, 238, 400, 76
L.append(f'<line x1="{FCX}" y1="{py+ph+3}" x2="{FCX}" y2="{HY-24}" '
         f'stroke="{AMB_S}" stroke-width="2.2" stroke-dasharray="5 3"/>')
L.append(f'<line x1="{FCX}" y1="{HY-24}" x2="{HCX+40}" y2="{HY-2}" '
         f'stroke="{AMB_S}" stroke-width="2.2" stroke-dasharray="5 3" marker-end="url(#aw)"/>')
text(FCX-14, HY-32, "捡起半成品，接着画", size=12, anchor="end", fill=AMB_S, italic=True)
# 草稿头 node（compact，注结构=数字③）
L.append(f'<rect x="{HCX-HW/2:.1f}" y="{HY}" width="{HW}" height="{HH}" rx="10" '
         f'fill="{GRN_F}" stroke="{GRN_S}" stroke-width="2.5"/>')
text(HCX, HY+24, "草稿头 · 仅 2 层可训练", size=14.5, bold=True, fill=GRN_S)
text(HCX, HY+48, "[token emb ; f] 宽 2h → FC 降回 h → 单层 decoder", size=12, fill=NAVY)
text(HCX, HY+64, "→ 共享 LM Head（冻结）→ token", size=12, fill=NAVY)
varrow(HCX, HY+HH, HY+HH+24, mk="url(#ae)", color=GRN_S)
box(HCX, HY+HH+26, 130, 34, "草稿 token", NEU_F, NEU_S, size=13)
# 落差注记
text(RCX, 430, "白送的 f 上只多两层，接着画一步", size=13.5, fill=NAVY)
box(RCX, 448, 400, 44, "草稿头 0.24B（7B 目标）～ 0.99B（70B 目标）", GRN_F, GRN_S,
    size=13, bold=True, tfill=GRN_S)

# ===== D. 底部：量化落差做成视觉尺度 =====
BY = 540
text(60, BY, "量化落差", size=15, bold=True, anchor="start")
BARX = 320   # 条形起点：左侧留出组标题 + 系列名两行空间
def bar_row(y, glabel, pairs, unit, vmax, maxlen=430, note=""):
    # 组标题：独占一行，左对齐，不与系列名/条形相撞
    text(60, y, glabel, size=13.5, anchor="start", bold=True, fill=MUTE)
    for i, (name, val, fill, stroke) in enumerate(pairs):
        yy = y + 14 + i*30
        blen = max(10, val/vmax*maxlen)
        L.append(f'<rect x="{BARX}" y="{yy}" width="{blen:.1f}" height="20" rx="4" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
        text(BARX-10, yy+15, name, size=12, anchor="end", fill=NAVY)
        text(BARX+blen+8, yy+15, f"{val}{unit}", size=12.5, anchor="start", bold=True, fill=stroke)
    if note:
        text(BARX+maxlen+14, y+14+30+15, note, size=12.5, anchor="start", bold=True, fill=NAVY)

bar_row(BY+30,
        "体量（给 70B 当草稿的可训练规模）",
        [("朴素 · 完整模型", 7.0, RED_F, RED_S),
         ("EAGLE · 草稿头", 0.99, GRN_F, GRN_S)],
        "B", 7.0, note="≈7× 更小")
bar_row(BY+130,
        "草稿准确率",
        [("token 层（Medusa）", 0.6, RED_F, RED_S),
         ("特征层（EAGLE）", 0.8, GRN_F, GRN_S)],
        "", 1.0, maxlen=360, note="两层可训练反而更准")

# 收口一句
text(W/2, H-18, "草稿的原料零成本，EAGLE 只在特征空间续一步——省，且更准。",
     size=14.5, bold=True, fill=NAVY)

L.append('</svg>')
out = Path(__file__).with_name("fig-eagle-epiphany.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
