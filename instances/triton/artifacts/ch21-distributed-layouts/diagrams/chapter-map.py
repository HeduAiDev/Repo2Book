#!/usr/bin/env python3
"""ch21《Distributed 布局：Blocked、Slice、MMA 与 DotOperand 编码》—— 本章地图。

本章为**自然标题章**（chapter.md 无 `## N.M` 编号，只有自然标题），故站牌
**禁用 §N.M 徽标**，改用标题词本身（如 "Blocked 三元组"，取自
"## Blocked 三元组：把张量切成每线程一块连续元素" 的冒号前半句）。

四条泳道 = 本章四类 distributed 布局的心智分组：
  0 公共骨架  —— 所有 distributed 布局共享的 4 级计算层级 + 后端接缝（同一基类）
  1 Blocked 布局族 —— 本章 stakes 核心：三元组 → order 合并 → 自动推导 → getElemsPerThread
  2 降维投影 —— SliceEncoding，寄生在 Blocked 之上（squeeze）
  3 Tensor Core 编码 —— NvidiaMma → DotOperand，硬件定死的一对布局

入口（绿）= 承接上一章「布局是函数 L」；出口（橙）= 通向 shared 编码 / LinearLayout /
mma 深化三条后续脉络。

■ 站牌自适应宽度：自然标题站牌比 §N.M 短徽标长得多，若沿用固定 46px 胶囊会在
  紧凑列距下压穿相邻节点。改为：胶囊宽度按 cjk_text_width() 实测文本宽度算，
  且贴节点**顶边居中跨界**（而非右上角）、宽度上限夹到 NODE_W 内——保证胶囊
  水平范围不超出自身节点的列宽，不会喙到隔壁列。字号不够放就用 fit_font_size()
  逐步缩小（下限 8px），而不是任由文字溢出胶囊。

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def fit_font_size(text, max_width, base_size, min_size=8.0, step=0.5):
    """从 base_size 起逐步缩小字号，直到 cjk_text_width(text, size) <= max_width
    或触底 min_size——自然标题站牌/长符号名比 §N.M 短徽标长得多，靠计算不靠手改。"""
    size = base_size
    while size > min_size and cjk_text_width(text, size) > max_width:
        size -= step
    return round(size, 1)


# ---------------- DATA(本章数据) ----------------
LANES = ["公共骨架", "Blocked 布局族", "降维投影", "Tensor Core 编码"]  # 泳道,上→下

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌(标题词本身,禁用 §))
NODES = [
    ("hierarchy", 0, 0, 0, "DistributedEncodingTrait",
     "CTA→Warp→Thread→Value 四级层级", "四级计算层级"),
    ("blocked", 1, 1, 0, "BlockedEncodingAttr",
     "三元组+order:每线程一块连续元素", "Blocked 三元组"),
    ("order", 1, 2, 0, "order",
     "getContigPerThread=sizePerThread→合并", "order 与合并访存"),
    ("builder", 1, 3, 0, "AttrBuilder",
     "从 shape+numWarps 反解各级 tile", "自动推导 builder"),
    ("elems", 1, 4, 0, "getElemsPerThread",
     "ceil(shapePerCTA/t)×sizePerThread", "getElemsPerThread"),
    ("slice", 2, 2, 0, "SliceEncodingAttr",
     "squeeze(dim):expand_dims 的逆", "SliceEncoding"),
    ("mma", 3, 1, 0, "NvidiaMmaEncodingAttr",
     "versionMajor 分派三代 Tensor Core", "NvidiaMmaEncoding"),
    ("dotop", 3, 2, 0, "DotOperandEncodingAttr",
     "kWidth=32/bitwidth 摆盘操作数", "DotOperandEncoding"),
    ("seam", 0, 5, 0, "DistributedEncoding",
     "AMD/Nvidia 矩阵乘布局同继承一行", "后端接缝"),
]
EDGES = [  # (src_id, dst_id) —— 调用/派生边,统一主线蓝
    ("hierarchy", "blocked"), ("hierarchy", "mma"),
    ("blocked", "order"), ("blocked", "slice"),
    ("order", "builder"), ("builder", "elems"),
    ("mma", "dotop"),
    ("elems", "seam"), ("slice", "seam"), ("dotop", "seam"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("Blocked 全链路", [
        (0, "四级计算层级"), (1, "Blocked 三元组"), (2, "order 与合并访存"),
        (3, "自动推导 builder"), (4, "getElemsPerThread"), (5, "后端接缝"),
    ], True),
    ("降维投影", [
        (0, "四级计算层级"), (1, "Blocked 三元组"), (2, "SliceEncoding"), (5, "后端接缝"),
    ], False),
    ("Tensor Core 编码", [
        (0, "四级计算层级"), (1, "NvidiaMmaEncoding"), (2, "DotOperandEncoding"), (5, "后端接缝"),
    ], False),
]
LEGEND = [
    ("#22c55e", "入口:承接上一章「布局是函数 L」"),
    ("#3b82f6", "章内主线:公共骨架派生四类具体 encoding"),
    ("#f97316", "出口:通向 shared 编码/LinearLayout/mma 深化"),
]
TITLE = "Distributed 布局家族剖面(源码走线 + 标题站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 190, 60
COL_GAP, ROW_GAP = 28, 20
EDGE_MARGIN, STUB_W, STUB_H = 12, 60, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 24  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 30, 24, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_H = 20
BADGE_HPAD = 8  # 胶囊左右内边距
NODE_INNER_PAD = 10  # 节点内文本左右各留白,估算可用宽度用

n_cols = max(n[2] for n in NODES) + 1
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_lane = [0] * len(LANES)
for _id, lane, col, row, *_ in NODES:
    rows_per_lane[lane] = max(rows_per_lane[lane], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_lane]
band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for bh in band_h:
    band_top.append(_cum)
    _cum += bh
lanes_bottom = _cum

NODE_XY = {}
for nid, lane, col, row, *_ in NODES:
    x = COLX[col]
    y = band_top[lane] + LANE_LABEL_H + BAND_PAD + row * (NODE_H + ROW_GAP)
    NODE_XY[nid] = (x, y)
NODE_BY_ID = {n[0]: n for n in NODES}

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text, max_w):
    """§ 徽标胶囊的自然标题变体:宽度按文本实测算、夹到 max_w 以内(字号不够放就
    fit_font_size 缩小),居中挂在 (cx,cy)。禁止定长 46px——自然标题词比 §N.M
    短徽标长得多，定长会在紧凑列距下喙到隔壁节点/路线。"""
    size = fit_font_size(text, max_w - 2 * BADGE_HPAD, 11.0, min_size=8.0)
    bw = min(max_w, cjk_text_width(text, size) + 2 * BADGE_HPAD)
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="{size}" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11.5) + 34

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩(给入口/出口箭头一个可附着的框,兼表达"调用方在画布外")
ex, ey = NODE_XY["hierarchy"]; ey += NODE_H / 2
xx, xy = NODE_XY["seam"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝,先画边再画节点盖住端点毛刺)
# 多条边汇入同一节点时,终点 y 各偏移,否则重合的终点看不出"汇合"。
_dst_total = {}
for _, dst in EDGES:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    y_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 顶边居中站牌——自然标题版:站牌宽度夹到
# NODE_W 内、居中跨界，不用 §N.M 版本的右上角小胶囊，避免长文本喙到隔壁列)
inner_w = NODE_W - 2 * NODE_INNER_PAD
for nid, lane, col, row, symbol, phrase, station in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_size = fit_font_size(symbol, inner_w, 13.0, min_size=9.5)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.44:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    sub_size = fit_font_size(phrase, inner_w, 10.5, min_size=8.0)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.76:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{sub_size}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W / 2, y, station, NODE_W - 8)

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(站牌=图上节点顶边站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first = COLX[stops[0][0]] + NODE_W / 2
    x_last = COLX[stops[-1][0]] + NODE_W / 2
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for col, station in stops:
        L += badge(COLX[col] + NODE_W / 2, ry, station, NODE_W - 8)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w:.0f}x{h:.0f}, aspect {w / h:.2f}:1)")
