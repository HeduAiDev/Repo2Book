#!/usr/bin/env python3
"""顿悟图头图(落差揭示):算得更多却更快——因为搬得少 9.2×。
视觉主轴=反转对比:三条度量各画朴素 vs Flash 双条,条长按真实值等比缩放;
唯一 Flash 更长的一行(计算量)标红=反直觉锚点,HBM 9.2× 落差做成最宽的视觉尺度。
所有数字来自 arXiv:2205.14135 Fig.2 左表(GPT-2 medium, N=1024, d=64, 16 heads, batch 64)。
"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

# 度量:名称, 朴素值, Flash值, 单位, 反转类型, 右列注记(两行), 注记色
# 顺序=因果读法:①计算量↑(悖论)→②HBM↓9.2×(主宰)→③时间↓5.7×(结果)
METRICS = [
    {"tag": "① 计算量", "unit": "GFLOPs", "std": 66.6, "flash": 75.2,
     "note": ["▲ Flash 反而多", "+13% 计算"], "note_color": "#dc2626",
     "flip": "flash_more"},
    {"tag": "② HBM 读写", "unit": "GB", "std": 40.3, "flash": 4.4,
     "note": ["9.2×", "HBM 落差"], "note_color": "#1e3a8a",
     "flip": "std_more", "big": True},
    {"tag": "③ 运行时间", "unit": "ms", "std": 41.7, "flash": 7.3,
     "note": ["▼ Flash", "快 5.7×"], "note_color": "#047857",
     "flip": "std_more"},
]

C_STD = ("#94a3b8", "#475569")     # 朴素:slate
C_FLASH = ("#10b981", "#047857")   # Flash:emerald

PAD = 40
BARX = PAD + 96          # 条起点(左留 tag)
MAXBAR = 430
RATIOW = 150
BAR_H = 34
BAR_GAP = 9
HEADER_H = 26
GROUP_GAP = 34
TOP = 118

w = BARX + MAXBAR + RATIOW + PAD
group_block = HEADER_H + 2 * BAR_H + BAR_GAP
groups_bottom = TOP + len(METRICS) * group_block + (len(METRICS) - 1) * GROUP_GAP
strip_y = groups_bottom + 24
h = strip_y + 58

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="6" '
     'markerHeight="4" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>']

# ---- 标题:一拳洞见 ----
L.append(f'<text x="{PAD}" y="46" font-family="sans-serif" font-size="24" '
         f'font-weight="bold" fill="#0f172a">{esc("算得更多，却快 5.7×")}</text>')
L.append(f'<text x="{PAD}" y="74" font-family="sans-serif" font-size="14" '
         f'fill="#64748b">{esc("FlashAttention：时间的主宰是搬运，不是计算")}</text>')

# ---- 图例(右上)----
lx = w - PAD - 210
ly = 40
for i, (label, col) in enumerate([("朴素注意力", C_STD), ("FlashAttention", C_FLASH)]):
    yy = ly + i * 24
    L.append(f'<rect x="{lx}" y="{yy}" width="20" height="14" rx="3" '
             f'fill="{col[0]}" stroke="{col[1]}" stroke-width="1"/>')
    L.append(f'<text x="{lx+28}" y="{yy+12}" font-family="sans-serif" '
             f'font-size="12.5" fill="#334155">{esc(label)}</text>')

# ---- 三条度量组 ----
for gi, m in enumerate(METRICS):
    gy = TOP + gi * (group_block + GROUP_GAP)
    # 组标题
    L.append(f'<text x="{PAD}" y="{gy+18}" font-family="sans-serif" font-size="15" '
             f'font-weight="bold" fill="#0f172a">{esc(m["tag"]+"（"+m["unit"]+"）")}</text>')
    scale = MAXBAR / max(m["std"], m["flash"])
    bars_top = gy + HEADER_H
    for bi, (which, val, col) in enumerate(
            [("朴素", m["std"], C_STD), ("Flash", m["flash"], C_FLASH)]):
        by = bars_top + bi * (BAR_H + BAR_GAP)
        bw = val * scale
        # 行 tag(右对齐,条左侧)
        L.append(f'<text x="{BARX-10}" y="{by+BAR_H/2+4}" text-anchor="end" '
                 f'font-family="sans-serif" font-size="12" fill="#64748b">{esc(which)}</text>')
        # 条
        L.append(f'<rect x="{BARX}" y="{by}" width="{bw:.1f}" height="{BAR_H}" rx="4" '
                 f'fill="{col[0]}" stroke="{col[1]}" stroke-width="1.5"/>')
        # 数值:长条内(白字右对齐),短条外(深字)
        vtxt = f'{val:g} {m["unit"]}'
        if bw > 84:
            L.append(f'<text x="{BARX+bw-8}" y="{by+BAR_H/2+5}" text-anchor="end" '
                     f'font-family="sans-serif" font-size="14" font-weight="bold" '
                     f'fill="white">{esc(vtxt)}</text>')
        else:
            L.append(f'<text x="{BARX+bw+8}" y="{by+BAR_H/2+5}" '
                     f'font-family="sans-serif" font-size="14" font-weight="bold" '
                     f'fill="{col[1]}">{esc(vtxt)}</text>')
    # 右列注记(反转/落差)
    rcx = BARX + MAXBAR + RATIOW / 2 + 8
    rcy = bars_top + BAR_H + BAR_GAP / 2
    big = m.get("big")
    n0_size = 26 if big else 15
    L.append(f'<text x="{rcx}" y="{rcy}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="{n0_size}" font-weight="bold" fill="{m["note_color"]}">{esc(m["note"][0])}</text>')
    L.append(f'<text x="{rcx}" y="{rcy+ (24 if big else 18)}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="{13 if big else 12.5}" font-weight="bold" '
             f'fill="{m["note_color"]}">{esc(m["note"][1])}</text>')

# ---- 反转强调:计算量组(Flash 更长)画一根红色向右小箭头贴 Flash 条尾 ----
# 计算量组 Flash 条:gi=0, bi=1
g0y = TOP
flash_by = g0y + HEADER_H + (BAR_H + BAR_GAP)
flash_bw = METRICS[0]["flash"] * (MAXBAR / max(METRICS[0]["std"], METRICS[0]["flash"]))
std_bw = METRICS[0]["std"] * (MAXBAR / max(METRICS[0]["std"], METRICS[0]["flash"]))
# 从朴素条尾指向 Flash 条尾,凸显“更长”
ay = flash_by - BAR_GAP / 2
L.append(f'<line x1="{BARX+std_bw}" y1="{ay}" x2="{BARX+flash_bw-2}" y2="{ay}" '
         f'stroke="#dc2626" stroke-width="2" marker-end="url(#a)" '
         f'stroke-dasharray="4,3"/>')

# ---- 底部洞见带 ----
L.append(f'<rect x="{PAD}" y="{strip_y}" width="{w-2*PAD}" height="38" rx="8" '
         f'fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>')
STRIP = "HBM 读写砍 9.2× → 端到端快 5.7×；N×N 打分表从未落地 HBM（计算反而多 13% 也无妨）"
L.append(f'<text x="{w/2}" y="{strip_y+24}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13.5" font-weight="bold" fill="#1e3a8a">{esc(STRIP)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig-fa-epiphany.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
