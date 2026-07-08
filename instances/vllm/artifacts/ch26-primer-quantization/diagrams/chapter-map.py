#!/usr/bin/env python3
"""第 26 章(【原理篇·论文精读】量化数学)——本章地图:源码/论文剖面图。

本章是 primer(原理篇)章,正文用自然标题(一、二、三…),无 `## N.M` 编号——按契约
禁用 §N.M 徽标,站牌改用标题词本身;符号真实性核对改对 book/papers/ch26-primer-
quantization/*.md 论文包 + 正文(lint_chapter_map.py 对 kind=primer 章的口径)。

形状:一个真实的"共享底座→三分支→再汇合→落地"剖面,不是单纯按阅读顺序摆成一条
直线——§二/§三共用同一段 quant_utils 参考实现(group_size 就是粒度旋钮),GPTQ/AWQ/
SmoothQuant 三法(§四/五/六)各自的数学都是在这同一份均匀量化网格上做误差控制,
§七的同台称重把三法的参考实现重跑一遍摆在一起比,§八先讲三法共享的 quant_config→
apply 统一插座,FP8 的 e8m0 装载(仍在§八)是这个插座具体落到 FP8 时的一个特例分支。
三层泳道(底座/三法/落地)堆叠,天然产生斜线连边(上层→下层),不需要额外的桥接带
——这就是模板本身(entry/dispatch→fast_attn/full_attn→fast_kernel/full_kernel→exit)
已经支持的形状,只是把两分支扩到三分支。

阅读路线底部时间轴改用独立均匀分布(不复用图上节点列号)——GPTQ/AWQ/SmoothQuant
三个节点共享同一列(col=1,只是行不同),若路线条也用列号,三个站牌会叠在同一 x 位置。

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
LANES = ["均匀量化底座", "三种误差控制法(论文数学 + vLLM 参考实现)", "vLLM 统一落地面"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌文字)
NODES = [
    ("base",        0, 0, 0, "w_s / group_size",     "scale/zp 底座 + 粒度旋钮",   "均匀底座与粒度"),
    ("gptq",        1, 1, 0, "ops.gptq_gemm",         "Hessian 二阶补偿",           "GPTQ"),
    ("awq",         1, 1, 1, "ops.awq_gemm",          "激活感知缩放",               "AWQ"),
    ("smoothquant", 1, 1, 2, "fp8_max",               "难度迁移到权重",             "SmoothQuant"),
    ("comparison",  2, 2, 0, "w_ref",                 "同制式内比 RTN",             "三法同台称重"),
    ("surface",     2, 3, 0, "quant_method.apply",    "统一插座,按层分发",          "统一调用面"),
    ("fp8e8m0",     2, 4, 0, "float8_e8m0fnu",        "scale 取整 2 的幂",          "FP8 e8m0 装载"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝
    ("base", "gptq"), ("base", "awq"), ("base", "smoothquant"),         # 三法共享同一份均匀量化网格
    ("gptq", "comparison"), ("awq", "comparison"), ("smoothquant", "comparison"),  # 同台称重重跑三法参考实现
    ("comparison", "surface"),   # 从"离线算什么"过渡到"运行期怎么消费"(统一插座)
    ("surface", "fp8e8m0"),      # FP8 是同一 apply 插座落到块量化时的具体装载分支
]
# 阅读顺序上的 7 个站牌(与正文一~八节一一对应,granularity 并入 base 一站)
READING_ORDER = ["均匀底座与粒度", "GPTQ", "AWQ", "SmoothQuant", "三法同台称重", "统一调用面", "FP8 e8m0 装载"]
# (路线名, [站牌文字,...] 按阅读顺序取 READING_ORDER 的子序列, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("全程精读", READING_ORDER, True),
    ("只读 AWQ 落地", ["均匀底座与粒度", "AWQ", "统一调用面"], False),
    ("只看落地面", ["均匀底座与粒度", "三法同台称重", "统一调用面", "FP8 e8m0 装载"], False),
]
LEGEND = [
    ("#22c55e", "入口:从上一章的量化动机进入"),
    ("#3b82f6", "章内主线:共享底座→三法→落地面"),
    ("#f97316", "出口:回到 vLLM 统一调用面之外"),
]
TITLE = "第 26 章 · 量化数学:均匀底座 → GPTQ/AWQ/SmoothQuant → vLLM 落地面剖面图"

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


NODE_H = 58
COL_GAP, ROW_GAP = 34, 18
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44

_SYMBOL_FONT, _PHRASE_FONT = 12.5, 10.5
_NODE_TEXT_PAD = 20
NODE_W = max(
    190,
    max(cjk_text_width(sym, _SYMBOL_FONT) for *_, sym, _, _ in NODES) + _NODE_TEXT_PAD,
    max(cjk_text_width(ph, _PHRASE_FONT) for *_, ph, _ in NODES) + _NODE_TEXT_PAD,
)
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 24  # 左右各留:接口桩 + 一段箭头

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

# 阅读路线的站牌 x 坐标独立均匀分布,不复用 NODE 列号——GPTQ/AWQ/SmoothQuant 三个
# 节点共享同一列(col=1,只是泳道内行不同),若路线条也用列号,三个站牌会叠在同一 x。
# 起点还要给左侧路线名文字留够宽度(按最长路线名估算),否则长路线名会被第一个站牌压住。
_route_label_w = max(cjk_text_width(name, 12) for name, _, _ in ROUTES)
_route_x_left = max(PAD_L, 16 + _route_label_w + 24)
_route_x_right = w - PAD_R - NODE_W
ROUTE_X = {
    name: _route_x_left + i * (_route_x_right - _route_x_left) / (len(READING_ORDER) - 1) + NODE_W / 2
    for i, name in enumerate(READING_ORDER)
}


def badge(cx, cy, text):
    """§/站牌徽标胶囊,居中挂在 (cx,cy)——宽度按文字自适应。"""
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
ex, ey = NODE_XY["base"]; ey += NODE_H / 2
xx, xy = NODE_XY["fp8e8m0"]; xy += NODE_H / 2
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

# 调用边(主线蓝)。多条边汇入同一节点时,终点 y 各偏移,否则重合的终点看不出"汇合"。
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

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌)
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

# 底部阅读路线:站牌 x 坐标用 ROUTE_X(独立均匀分布),不复用图上节点列号。
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first = ROUTE_X[stops[0]]
    x_last = ROUTE_X[stops[-1]]
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for sec in stops:
        L += badge(ROUTE_X[sec], ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w:.0f}x{h:.0f}, ratio {w / h:.2f}:1)")
