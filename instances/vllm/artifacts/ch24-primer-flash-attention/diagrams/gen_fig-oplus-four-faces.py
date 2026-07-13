#!/usr/bin/env python3
"""⊕ 算子四副面孔统一架构图:中央一个 ⊕ 框写通式(换公共基准 M=max → 相加),
四个卫星面孔框各标一种状态对 + 一行微型数值见证,视觉重点=『状态对在换,代数没换』。
数字来自 explainer traces(online_softmax_merge_operator / flashattention_tiling / lse_merge)。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

# 四副面孔:位置(0=TL,1=TR,2=BL,3=BR), 标题色, 头, 状态对, 数值见证多行
FACES = [
    {"pos": 0, "color": ("#eff6ff", "#2563eb"),
     "head": "面孔① 单遍递推 · §二", "pair": "状态对 (m, d)",
     "lines": ["A = [1,3] → (3, 1.1353)",
               "B = [2,5] → (5, 1.0498)",
               "A ⊕ B = (5, 1.2034)"]},
    {"pos": 1, "color": ("#f0fdf4", "#16a34a"),
     "head": "面孔② FlashAttention 分块 · §三", "pair": "状态对 (m, ℓ, O)",
     "lines": ["块1: (0.7071, 1.4931, [0.6698, 0.3302])",
               "块2: (0.7071, 3.4931, [0.8588, 0.7137])",
               "同一 ⊕ 逐块推进 O"]},
    {"pos": 2, "color": ("#fefce8", "#ca8a04"),
     "head": "面孔③ LSE 合并（对数域）· §六", "pair": "状态对 (lse, O)",
     "lines": ["(1.4003, [0.5,0.5]) ⊕ (0.3536, [2.0,0.0])",
               "→ O = [0.8898, 0.3701]"]},
    {"pos": 3, "color": ("#fdf4ff", "#c026d3"),
     "head": "面孔④ split-KV 多块归并 · §六末", "pair": "状态对 (O, lse)",
     "lines": ["长 KV 切 n 块 → 各出 (O, lse)",
               "→ tree-reduce 归并",
               "（纯结构，无数值）"]},
]

PAD = 40
CARD_W = 372
CARD_H = 150
CENTER_GAP = 316          # 两列卡片间距(容中央 hub)
COL_GAP_TOP = 96
ROW_GAP = 210             # 上下两行卡片间距

col_x = [PAD, PAD + CARD_W + CENTER_GAP]
row_y = [COL_GAP_TOP, COL_GAP_TOP + CARD_H + ROW_GAP]

w = PAD + CARD_W + CENTER_GAP + CARD_W + PAD
h = row_y[1] + CARD_H + PAD + 8

# 中央 hub
HUB_W, HUB_H = 240, 168
hub_x = (col_x[0] + CARD_W + col_x[1]) / 2 - HUB_W / 2
hub_y = (row_y[0] + row_y[1] + CARD_H) / 2 - HUB_H / 2

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#6366f1"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

# ---- 标题 ----
L.append(f'<text x="{PAD}" y="42" font-family="sans-serif" font-size="21" '
         f'font-weight="bold" fill="#0f172a">{esc("一个 ⊕，四副面孔：状态对在换，代数没换")}</text>')
L.append(f'<text x="{PAD}" y="66" font-family="sans-serif" font-size="13" '
         f'fill="#64748b">{esc("同一套代数（先换到公共基准 M=max，再相加）贯穿全章四种状态对")}</text>')

def card_rect(pos):
    cx = col_x[pos % 2]
    cy = row_y[pos // 2]
    return cx, cy

# ---- 连接线:hub 四角 → 各卡片内侧角(方向 hub→面孔,箭头在面孔端)----
hub_corners = {
    0: (hub_x, hub_y),                  # TL
    1: (hub_x + HUB_W, hub_y),          # TR
    2: (hub_x, hub_y + HUB_H),          # BL
    3: (hub_x + HUB_W, hub_y + HUB_H),  # BR
}
for f in FACES:
    p = f["pos"]
    cx, cy = card_rect(p)
    hxc, hyc = hub_corners[p]
    if p % 2 == 0:   # 左列:卡片内侧角=右边
        tx = cx + CARD_W
    else:            # 右列:卡片内侧角=左边
        tx = cx
    if p // 2 == 0:  # 上行:卡片内侧角=下边
        ty = cy + CARD_H
    else:            # 下行:卡片内侧角=上边
        ty = cy
    L.append(f'<line x1="{hxc:.1f}" y1="{hyc:.1f}" x2="{tx:.1f}" y2="{ty:.1f}" '
             f'stroke="#94a3b8" stroke-width="1.6" marker-end="url(#a)"/>')

# ---- 四副面孔卡片 ----
for f in FACES:
    cx, cy = card_rect(f["pos"])
    fill, stroke = f["color"]
    L.append(f'<rect x="{cx}" y="{cy}" width="{CARD_W}" height="{CARD_H}" rx="10" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>')
    # 头(彩条)
    L.append(f'<rect x="{cx}" y="{cy}" width="{CARD_W}" height="28" rx="10" '
             f'fill="{stroke}"/>')
    L.append(f'<rect x="{cx}" y="{cy+14}" width="{CARD_W}" height="14" '
             f'fill="{stroke}"/>')
    L.append(f'<text x="{cx+14}" y="{cy+19}" font-family="sans-serif" font-size="13" '
             f'font-weight="bold" fill="white">{esc(f["head"])}</text>')
    # 状态对
    L.append(f'<text x="{cx+14}" y="{cy+50}" font-family="sans-serif" font-size="14" '
             f'font-weight="bold" fill="{stroke}">{esc(f["pair"])}</text>')
    # 数值见证行
    for li, ln in enumerate(f["lines"]):
        L.append(f'<text x="{cx+14}" y="{cy+74+li*22}" font-family="monospace" '
                 f'font-size="12.5" fill="#334155">{esc(ln)}</text>')

# ---- 中央 hub ----
L.append(f'<rect x="{hub_x:.1f}" y="{hub_y:.1f}" width="{HUB_W}" height="{HUB_H}" rx="14" '
         f'fill="#eef2ff" stroke="#6366f1" stroke-width="3"/>')
L.append(f'<text x="{hub_x+HUB_W/2:.1f}" y="{hub_y+54:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="46" font-weight="bold" '
         f'fill="#4338ca">{esc("⊕")}</text>')
for li, ln in enumerate(["同一套代数", "① 换到公共基准 M=max", "② 再相加"]):
    fw = "bold" if li == 0 else "normal"
    fs = 14 if li == 0 else 12.5
    L.append(f'<text x="{hub_x+HUB_W/2:.1f}" y="{hub_y+82+li*24:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="{fs}" font-weight="{fw}" '
             f'fill="#3730a3">{esc(ln)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-oplus-four-faces.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
