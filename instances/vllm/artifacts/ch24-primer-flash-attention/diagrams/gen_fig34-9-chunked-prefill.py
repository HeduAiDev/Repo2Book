#!/usr/bin/env python3
"""tiling 模板(定制):50×50 因果注意力矩阵,下三角(含对角线)可见、上三角掩码;
query 轴按 16/16/18 三段染色,chunk 边界是水平横切线——每段可见窗恰覆盖从第 0 列到
该段对角线的全部历史列。与 cascade 沿 KV 轴竖切、需 ⊕ 合并成对照。
全坐标计算,零手写魔数。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

TITLE = "chunked prefill 在因果矩阵上的切法 — 50×50 下三角,query 轴切 16/16/18"
SUBTITLE = "切割线水平(沿 query 轴),不切断任何一行的历史列;逐块输出直接拼接 == 一次性整段(实测偏差精确 0)"

N = 50
CHUNKS = [16, 16, 18]                 # query 轴三段大小
# 段起止(绝对行),及每段累积可见 KV 列数(= 段末行 + 1)
seg_bounds = []
s = 0
for sz in CHUNKS:
    seg_bounds.append((s, s + sz - 1))   # [start, end] 闭区间
    s += sz
CUM_COLS = [e + 1 for (_, e) in seg_bounds]   # 16, 32, 50

CELL = 9
PAD_L, PAD_T = 118, 92                 # 左给 query 轴标签,上给标题+key 轴标签
grid_w = N * CELL
grid_h = N * CELL
GX, GY = PAD_L, PAD_T

# 三段配色:可见格填色(浅)+ 段标签色(深)
BAND_FILL = ["#dbeafe", "#d1fae5", "#ffedd5"]     # 蓝 / 绿 / 橙(浅)
BAND_DARK = ["#2563eb", "#059669", "#ea580c"]     # 蓝 / 绿 / 橙(深)
MASK_FILL = "#f8fafc"
DIAG_STROKE = "#0f172a"
BOUND_STROKE = "#b91c1c"               # chunk 边界(横切)

def band_of(r):
    for k, (a, b) in enumerate(seg_bounds):
        if a <= r <= b:
            return k
    return len(seg_bounds) - 1

# 右侧段说明框
side_x = GX + grid_w + 46
SIDE_W = 250

def txt_w(s, fs=11):
    """粗估文本宽度(CJK 全宽 ~fs,ASCII ~0.55fs),用于图例排版避免越界。"""
    tot = 0.0
    for ch in s:
        tot += fs if ord(ch) > 0x2E7F else fs * 0.55
    return tot

# 图例项(色块 4 项 + 红虚线 1 项)先行布局,据此定画布宽,杜绝末项越界
LEG_ITEMS = [("块1 query 行 0–15", BAND_FILL[0], BAND_DARK[0]),
             ("块2 行 16–31", BAND_FILL[1], BAND_DARK[1]),
             ("块3 行 32–49", BAND_FILL[2], BAND_DARK[2]),
             ("未来 key(掩码)", MASK_FILL, "#94a3b8")]
LEG_DASH = "chunk 边界(query 轴横切)"
leg_x0 = 24
_lx = leg_x0
for _t, _f, _s in LEG_ITEMS:
    _lx += 14 + 6 + txt_w(_t) + 24
_lx += 26 + 6 + txt_w(LEG_DASH)   # 红虚线段 + 标签
leg_end = _lx

w = max(side_x + SIDE_W + 24, leg_end + 24)
h = GY + grid_h + 132

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">',
     '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
     'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>',
     f'<rect width="{w}" height="{h}" fill="white"/>',
     f'<text x="{24}" y="{34}" font-family="sans-serif" font-size="16.5" '
     f'font-weight="bold" fill="#0f172a">{esc(TITLE)}</text>',
     f'<text x="{24}" y="{56}" font-family="sans-serif" font-size="12" '
     f'fill="#475569">{esc(SUBTITLE)}</text>']

# 上三角掩码区(c > r):一整块浅色多边形
# 顶点:(GX, GY) 右上三角 -> 从对角线上方到右上角
L.append(f'<polygon points="{GX},{GY} {GX+grid_w},{GY} {GX+grid_w},{GY+grid_h}" '
         f'fill="{MASK_FILL}" stroke="none"/>')

# 下三角(含对角线)可见格:逐格填色,按行所属段上色
for r in range(N):
    k = band_of(r)
    fill = BAND_FILL[k]
    y = GY + r * CELL
    for c in range(r + 1):
        x = GX + c * CELL
        L.append(f'<rect x="{x}" y="{y}" width="{CELL}" height="{CELL}" '
                 f'fill="{fill}" stroke="#cbd5e1" stroke-width="0.4"/>')

# 对角线(可见窗右缘 = 每行的最新可见 key)
L.append(f'<line x1="{GX}" y1="{GY}" x2="{GX+grid_w}" y2="{GY+grid_h}" '
         f'stroke="{DIAG_STROKE}" stroke-width="1.4"/>')

# 外框
L.append(f'<rect x="{GX}" y="{GY}" width="{grid_w}" height="{grid_h}" '
         f'fill="none" stroke="#94a3b8" stroke-width="1.2"/>')

# chunk 边界:水平横切线(在段之间,即 row 16 与 row 32 的上边缘)
for (a, b) in seg_bounds[1:]:
    yb = GY + a * CELL
    L.append(f'<line x1="{GX-8}" y1="{yb}" x2="{GX+grid_w+8}" y2="{yb}" '
             f'stroke="{BOUND_STROKE}" stroke-width="2.6" stroke-dasharray="7,4"/>')
# 边界短标(紧贴每条横切线左端上方,落在网格外左侧留白,不与右侧框重叠)
for (a, b) in seg_bounds[1:]:
    yb = GY + a * CELL
    L.append(f'<text x="{GX-14}" y="{yb-4}" text-anchor="end" font-family="sans-serif" '
             f'font-size="9.5" fill="{BOUND_STROKE}">{esc("切")}</text>')

# 轴标签
L.append(f'<text x="{GX+grid_w/2}" y="{GY-24}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12.5" font-weight="bold" fill="#334155">{esc("key 位置(历史 KV 列)0 → 49")}</text>')
# y 轴标题(竖排通过旋转;pivot 右移留边,防旋转 bbox 越出画布左缘)
_yc = GY + grid_h / 2
L.append(f'<text x="{56}" y="{_yc}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="12" font-weight="bold" fill="#334155" '
         f'transform="rotate(-90 56 {_yc})">{esc("query 位置 0 → 49")}</text>')
# 刻度(0/16/32/49)两轴
for t in [0, 16, 32, 49]:
    # x 轴刻度
    xt = GX + t * CELL + CELL / 2
    L.append(f'<text x="{xt}" y="{GY-8}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="10" fill="#64748b">{t}</text>')
    # y 轴刻度
    yt = GY + t * CELL + CELL / 2 + 3
    L.append(f'<text x="{GX-12}" y="{yt}" text-anchor="end" font-family="sans-serif" '
             f'font-size="10" fill="#64748b">{t}</text>')

# 掩码区标注(置于上三角高处,避开 row-16 横切线)
L.append(f'<text x="{GX+grid_w*0.58}" y="{GY+grid_h*0.16}" font-family="sans-serif" '
         f'font-size="11" fill="#94a3b8">{esc("未来 key:掩码")}</text>')
L.append(f'<text x="{GX+grid_w*0.58}" y="{GY+grid_h*0.16+15}" font-family="sans-serif" '
         f'font-size="11" fill="#94a3b8">{esc("(softmax 权重 = 0)")}</text>')

# 右侧:三段说明(query 行区间 + 累积可见 KV 列数)
L.append(f'<text x="{side_x}" y="{GY-4}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="#0f172a">{esc("三段(一段 = 一次 varlen 调用)")}</text>')
for k, (a, b) in enumerate(seg_bounds):
    by = GY + 8 + k * 62
    L.append(f'<rect x="{side_x}" y="{by}" width="{SIDE_W}" height="52" rx="6" '
             f'fill="{BAND_FILL[k]}" stroke="{BAND_DARK[k]}" stroke-width="1.8"/>')
    L.append(f'<text x="{side_x+12}" y="{by+20}" font-family="sans-serif" font-size="12" '
             f'font-weight="bold" fill="{BAND_DARK[k]}">{esc(f"块 {k+1}:query 行 {a}–{b}")}</text>')
    L.append(f'<text x="{side_x+12}" y="{by+38}" font-family="sans-serif" font-size="11.5" '
             f'fill="#334155">{esc(f"累积可见 KV 列 0–{b}(共 {CUM_COLS[k]} 列)")}</text>')

# 右侧下方:与 cascade 的对照
cy = GY + 8 + 3 * 62 + 12
L.append(f'<rect x="{side_x}" y="{cy}" width="{SIDE_W}" height="86" rx="6" '
         f'fill="#fef2f2" stroke="#b91c1c" stroke-width="1.5"/>')
L.append(f'<text x="{side_x+12}" y="{cy+20}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="#b91c1c">{esc("对照 cascade")}')
L.append('</text>')
cascade_lines = [
    "cascade 沿 KV 轴竖切前缀/后缀,",
    "各出 (O, lse),必须 ⊕ 合并;",
    "此处沿 query 轴横切,行本独立,",
    "拼接即可 —— 连合并都不需要。",
]
for i, t in enumerate(cascade_lines):
    L.append(f'<text x="{side_x+12}" y="{cy+38+i*15}" font-family="sans-serif" '
             f'font-size="10.8" fill="#7f1d1d">{esc(t)}</text>')

# 底部结论
foot_y = GY + grid_h + 40
L.append(f'<text x="{24}" y="{foot_y}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="#b91c1c">'
         f'{esc("三块逐块输出拼接 vs 一次性整段因果注意力:max|差| = 0.0(逐字节相同,非近似)")}</text>')
FOOT2 = ("因果第 i 行的输出只依赖位置 ≤ i 的 KV,是绝对位置的纯函数——切点落在 query 轴,"
         "每行的历史列(第 0 列到对角线)完整落在它所属段的可见窗内,没有一行被切断。")
L.append(f'<text x="{24}" y="{foot_y+22}" font-family="sans-serif" font-size="11.5" '
         f'fill="#64748b">{esc(FOOT2)}</text>')

# 图例(布局与上方宽度预算同源)
lg_y = foot_y + 44
lx = leg_x0
for txt, fill, stroke in LEG_ITEMS:
    L.append(f'<rect x="{lx}" y="{lg_y-10}" width="14" height="14" rx="2" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    L.append(f'<text x="{lx+20}" y="{lg_y+2}" font-family="sans-serif" font-size="11" '
             f'fill="#475569">{esc(txt)}</text>')
    lx += 14 + 6 + txt_w(txt) + 24
# 红色虚线图例:chunk 边界
L.append(f'<line x1="{lx}" y1="{lg_y-3}" x2="{lx+26}" y2="{lg_y-3}" '
         f'stroke="{BOUND_STROKE}" stroke-width="2.6" stroke-dasharray="7,4"/>')
L.append(f'<text x="{lx+32}" y="{lg_y+2}" font-family="sans-serif" font-size="11" '
         f'fill="#475569">{esc(LEG_DASH)}</text>')

L.append('</svg>')
out = Path(__file__).with_name("fig34-9-chunked-prefill.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
