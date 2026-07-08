#!/usr/bin/env python3
"""第 31 章(【原理篇·论文精读】量化数学：从 scale/zero-point 到 GPTQ、AWQ、SmoothQuant)
——本章地图:论文推导剖面图。

本章是 primer(原理篇)章,正文用自然标题(一、二、三…),无 `## N.M` 编号——按契约禁用
§N.M 徽标,站牌改用标题词本身;符号真实性核对改对 book/papers/ch31-primer-quantization/
*.md 论文包(paper.md=GPTQ arXiv:2210.17323、paper-awq.md=AWQ arXiv:2306.00978、
paper-smoothquant.md=SmoothQuant arXiv:2211.10438)+ 正文(lint_chapter_map.py 对
kind=primer 章的口径)。

本章的真实形状是"先立公共地基、再三篇论文各自分道扬镳、最后汇总对比、落回代码框架"——
一次 fork(地基→三条论文路线)+ 一次 merge(三条路线→数值推演→落地)。三条泳道纵向
堆叠(问题与基础 / 三条论文路线 / 汇总与落地),同一套全局列坐标(不折行、不需要跨段
桥接),因为整章能在 ≤5 列内画完(画布预算:宽 ≤1500 且宽高比 ≤2.6:1)。

节点做了两处合并(不改变"一节一站牌"的阅读路线,只是图上的方框数收窄,合并只发生在
"同一篇论文/同一主题的相邻两节"这种最安全的地方——具体哪两节合了、合并后底部阅读
路线仍逐节可查,见下方 READING_ORDER 与 ROUTES 的 9 站设计):
  - 二(均匀量化基础)+ 三(粒度与 INT8 GEMM 硬件约束)→ 一个"foundation"方框
    (两节话题连贯:先讲 scale/zero-point,再讲为什么这些 scale 只能摆在外维——
    是同一条论证链的两步,不是两件事)。
  - 四(GPTQ 二阶补偿)+ 五(GPTQ 三大工程优化)→ 一个"gptq"方框
    (同一篇论文的"数学推导"与"工程加速",工程加速不改变数学结果,合并成一站
    不损失论点)。
  合并后,GPTQ/AWQ/SmoothQuant 三个方框就都是"一篇论文一站",与 AWQ(六)、
  SmoothQuant(七)对齐,视觉上更贴合"三篇论文分道扬镳"的核心论点。
  底部阅读路线的站牌不受此合并影响——READING_ORDER 仍列出全部 9 个自然标题,
  各自有独立的时间轴站位(不复用节点列号,见 ch24 同款技法),保证读者能精确
  跳到"四"或"五"而不是只能跳到笼统的"GPTQ"。

用法: python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):逐字符判定——
    全角(ord>0x2E80)按 1.0×size,半角(ASCII/拉丁等)按 0.58×size,求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(本章数据) ----------------
LANES = ["问题与基础", "三条论文路线（各自分道扬镳）", "汇总与落地"]  # 泳道,上→下

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号/公式名, 一行短语, 站牌文字)
# 自然标题章:站牌用标题词本身,禁用 §N.M。
NODES = [
    ("entry", 0, 0, 0,
     "outlier 塌缩",
     "SmoothQuant §3：outlier 级数塌缩",
     "动机"),
    ("foundation", 0, 1, 0,
     "scale / zero-point",
     "INT8 GEMM 只能外维反量化",
     "基础与硬件约束"),
    ("gptq", 1, 2, 0,
     "GPTQ：二阶补偿",
     "OBQ §3 Eq.2 补偿；GPTQ §4 懒惰批",
     "GPTQ"),
    ("awq", 1, 2, 1,
     "AWQ：激活感知缩放",
     "AWQ Eq.4-5：网格搜索最优缩放 s",  # 避免写成"§3.2"——natural-heading 章的
     # chapter-map 全文禁 §N.M 形式(即便是论文小节号,也会被 lint_chapter_map 的
     # 徽标正则误当成章节徽标拦下),故这里只保留无歧义的 Eq 编号,不带小节号。
     "AWQ"),
    ("smoothquant", 1, 2, 2,
     "SmoothQuant：迁移变换",
     "SmoothQuant §4：迁移，α=0.5 甜点",
     "SmoothQuant"),
    ("showdown", 2, 3, 0,
     "四法同台对比",
     "四法各自相对基线的降幅对比",
     "数值推演"),
    ("exit", 2, 4, 0,
     "落地：QuantType",
     "input_scale / weight_scale 形状对齐",
     "落地"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝；一次 fork + 一次 merge
    ("entry", "foundation"),
    ("foundation", "gptq"), ("foundation", "awq"), ("foundation", "smoothquant"),
    ("gptq", "showdown"), ("awq", "showdown"), ("smoothquant", "showdown"),
    ("showdown", "exit"),
]

# 底部阅读路线的 9 个站牌(与正文一~九节一一对应,独立于图上节点的合并——
# 不复用节点列号,见 ch24 同款技法:折/合并后同一节点会被多个站牌共用同一
# x 位置,若仍借节点列号,不同站牌会叠在一起分不清)。
READING_ORDER = [
    "动机", "均匀量化基础", "粒度与硬件约束",
    "GPTQ 二阶补偿", "GPTQ 三大工程优化",
    "AWQ 激活感知缩放", "SmoothQuant 迁移难度",
    "数值推演", "落地",
]
# (路线名, [站牌文字,...] 按阅读顺序取 READING_ORDER 的子序列, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("全程精读（三篇论文都读）", READING_ORDER, True),
    ("只读 W4A16（GPTQ+AWQ）",
     ["动机", "均匀量化基础", "粒度与硬件约束",
      "GPTQ 二阶补偿", "GPTQ 三大工程优化", "AWQ 激活感知缩放", "数值推演", "落地"], False),
    ("只读 W8A8（SmoothQuant）",
     ["动机", "均匀量化基础", "粒度与硬件约束", "SmoothQuant 迁移难度", "数值推演", "落地"], False),
]
LEGEND = [
    ("#22c55e", "入口：从第 30 章收尾处接入"),
    ("#3b82f6", "章内主线：基础→三篇论文分支→汇总→落地"),
    ("#f97316", "出口：数学交给第 32 章的框架消费"),
]
TITLE = "第 31 章 · 量化数学：GPTQ / AWQ / SmoothQuant 三篇论文推导剖面图"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
BADGE_FONT_SIZE = 11
BADGE_PAD_X = 14
BADGE_H = 20


def badge_width(text):
    return max(46.0, cjk_text_width(text, BADGE_FONT_SIZE) + BADGE_PAD_X * 2)


NODE_H = 68
COL_GAP, ROW_GAP = 26, 18
EDGE_MARGIN, STUB_W, STUB_H = 14, 64, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 16  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44

# 节点宽度:同一批节点统一宽度(保列对齐),按本章最长的符号名/短语算(零手写魔数)
_SYMBOL_FONT, _PHRASE_FONT = 13, 10.5
_NODE_TEXT_PAD = 20
NODE_W = max(
    190,
    max(cjk_text_width(sym, _SYMBOL_FONT) for *_, sym, _, _ in NODES) + _NODE_TEXT_PAD,
    max(cjk_text_width(ph, _PHRASE_FONT) for *_, ph, _ in NODES) + _NODE_TEXT_PAD,
)

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


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy)——宽度按文字自适应(见 badge_width)。"""
    bw = badge_width(text)
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="{BADGE_FONT_SIZE}" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.1f} {h:.1f}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN))
) + '</defs>')
L.append(f'<rect width="{w:.1f}" height="{h:.1f}" fill="white"/>')

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
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w:.1f}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w:.1f}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w:.1f}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩(给入口/出口箭头一个可附着的框,兼表达"调用方在画布外")
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["exit"]; xy += NODE_H / 2
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

# 调用边(主线蓝)。多条边汇入同一节点(showdown 收 3 条)时,终点 y 各偏移(间距 16px),
# 让"汇合"在视觉上看得出来,而不是叠成一条线断头。
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

# 节点(圆角框 + 真实符号/公式名 + 一行短语 + 右上角站牌)
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W:.1f}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{_SYMBOL_FONT}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{_PHRASE_FONT}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    bw = badge_width(sec)
    L += badge(x + NODE_W - bw / 2 + 8, y, sec)

# 底部阅读路线:9 个站牌按 READING_ORDER 均匀分布在整个画布宽度上(独立于图上节点的
# 列号——foundation/gptq 各合并了两节,若仍借节点列号,不同站牌会叠在同一 x 位置)。
# 时间轴左端起点让给最长路线名的实际宽度,不留固定魔数空档。
_route_label_w = max(cjk_text_width(name, 12) for name, *_ in ROUTES)
_route_left = 16 + _route_label_w + 40  # 额外留白:sans-serif 实际渲染宽度常比 cjk_text_width 估算略宽
_n_stops = len(READING_ORDER)
_route_x = {name: _route_left + i * (w - PAD_R - _route_left) / (_n_stops - 1)
            for i, name in enumerate(READING_ORDER)}

L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first, x_last = _route_x[stops[0]], _route_x[stops[-1]]
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for sec in stops:
        L += badge(_route_x[sec], ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w:.0f}x{h:.0f}, NODE_W={NODE_W:.0f})")
