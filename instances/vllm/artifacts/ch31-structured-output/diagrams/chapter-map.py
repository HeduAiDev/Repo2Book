#!/usr/bin/env python3
"""第 31 章「本章地图」——约束解码 I 的源码剖面图。

走线太长(前端校验 → 引擎异步编译 → 后端两层契约 → 调度器晋级 → 交棒),
所以折成上下两段泳道:上段末尾一个「接下段」桩,下段开头一个「承上段」桩,
同一条主线。视觉语言(§徽标胶囊 / 入口绿 / 出口橙 / 主线蓝 / 路线条实线蓝-虚线灰)
沿用 references/example-chapter-map.py,只改 DATA 与折行/多行符号的排版逻辑。

六项自查(渲染 → Read PNG 亲眼看后如实记录):
  claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
  arrows_attached=True     cjk_rendered=True         reading_order_clear=True
[FIX-ROUND-2](本轮改动后重渲 + 重新 Read PNG 复核,替换第一轮记录):
  - 第一轮 no_overlap 为 False:`validate_xgrammar_grammar` 与
    `StructuredOutputGrammar` 两个长符号名按 (NODE_W-16) 估算不越框,但粗体实际
    渲染比 cjk_text_width() 的半角系数宽,PNG 上两处文字压在节点框左右边线上。
    本轮把可用宽度收到 (NODE_W-TEXT_INSET)、字号下限降到 10.2 重渲,复核确认
    两处文字完整落在框内,现在的 True 是重新看过 PNG 的结果。
  - 其余五项两轮均为 True(数字 18.3 KB / 词表 1/32 / 五分支 / 四家实现 /
    六方法逐个对正文;箭头两端均贴边;中文无豆腐块;上段→接下段→承上段→下段
    的折行顺序在图上显式)。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算:全角(ord>0x2E80)按 1.0×size,半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = [
    "上段 · 前端:请求校验期(进引擎之前)",
    "上段 · 引擎:输入处理线程",
    "下段 · 编译线程 + 后端两层契约",
    "下段 · 调度器:晋级与交棒",
]

# (节点id, 泳道下标, 列, 泳道内行号, 符号名(str 或多行 list), 一行短语, §编号)
NODES = [
    ("params",   0, 0, 0, "StructuredOutputsParams",   "六种写法,互斥只许一种",   "§31.1"),
    ("key",      0, 1, 0, "get_structured_output_key", "归一成枚举+规格串面单",   "§31.2"),
    ("validate", 0, 2, 0, "validate_xgrammar_grammar", "auto 阶梯试编,定后端",    "§31.3"),
    ("rewrite",  0, 3, 0, "choice_as_grammar",         "原地改写:choice 转 EBNF", "§31.2"),
    ("init",     1, 4, 0, "grammar_init",              "惰性建后端,编译进线程池", "§31.4"),

    ("create",   2, 0, 0, "_create_grammar",           "工作线程里取面单开编",    "§31.4"),
    ("compile",  2, 1, 0, "compile_grammar",           "xgrammar 五分支分派",     "§31.6"),
    ("backend",  2, 1, 1, "StructuredOutputBackend",   "四家实现,能力各不同",     "§31.7"),
    ("grammar",  2, 2, 0, "StructuredOutputGrammar",   "请求级六方法契约",        "§31.5"),
    ("bitmask",  2, 2, 1, "fill_bitmask",              "一行 18.3 KB,词表 1/32",  "§31.8"),
    ("promote",  3, 3, 0, ["_try_promote_blocked", "_waiting_request"],
                                                       "读 grammar 就绪才放行",   "§31.4"),
    ("handoff",  3, 4, 0, "get_grammar_bitmask",       "交棒下一章:批量装配",     "§31.9"),
]
EDGES = [  # 主线调用边(左→右)
    ("params", "key"), ("key", "validate"), ("validate", "rewrite"), ("rewrite", "init"),
    ("create", "compile"), ("create", "backend"),
    ("compile", "grammar"), ("backend", "grammar"),
    ("grammar", "promote"), ("promote", "handoff"),
]
# 同列的「属于」关系(非调用边):虚线蓝 + 一行注记
EDGES_V = [("grammar", "bitmask", "六方法之一")]

ENTRY_NODE, ENTRY_LABEL = "params", "调用方"
EXIT_NODE, EXIT_LABEL = "handoff", "交棒下一章"
FOLD_OUT_NODE, FOLD_OUT_LABEL = "init", "接下段"     # 上段末:主线折行出去
FOLD_IN_NODE, FOLD_IN_LABEL = "create", "承上段"     # 下段首:主线折行回来

# (路线名, [(列, §编号), ...], 是否高亮)
ROUTES = [
    ("主线:校验→编译→放行", [(0, "§31.1"), (1, "§31.2"), (2, "§31.3"), (3, "§31.4"), (4, "§31.9")], True),
    ("细读:契约与 xgrammar 实现", [(0, "§31.4"), (1, "§31.6"), (2, "§31.5"), (4, "§31.9")], False),
    ("对照:四后端与掩码开销", [(1, "§31.7"), (2, "§31.8")], False),
]
LEGEND = [
    ("solid", "#22c55e", "入口:带约束的请求进来"),
    ("solid", "#3b82f6", "章内主线调用边"),
    ("solid", "#f97316", "出口:交棒下一章"),
    ("dash", "#3b82f6", "折行:上段末接下段首"),
]
TITLE = "第 31 章 · 约束解码 I 源码走线(上下两段折行 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]  # 交替背景,仅装饰
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 196, 62
COL_GAP, ROW_GAP = 36, 22
EDGE_MARGIN, STUB_W, STUB_H = 16, 78, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 26
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 28, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 24, 46
BADGE_W, BADGE_H = 48, 20
SYM_SIZE, SUB_SIZE = 12.5, 10.5
SYM_MIN = 10.2
# 粗体渲染比 cjk_text_width() 的估算宽,长符号名按框宽再留一档余量(见 [FIX-ROUND-2])
TEXT_INSET = 30

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
    NODE_XY[nid] = (COLX[col], band_top[lane] + LANE_LABEL_H + BAND_PAD + row * (NODE_H + ROW_GAP))

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def fit_size(text, size, limit):
    """长符号名自动缩字号以不越框(下限 SYM_MIN,再长就该在 DATA 里拆多行)。"""
    tw = cjk_text_width(text, size)
    return size if tw <= limit else max(SYM_MIN, size * limit / tw)


def badge(cx, cy, text):
    bx, by = cx - BADGE_W / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BADGE_W}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


def stub(x, y_center, label, color, fill, text_fill):
    return [
        f'<rect x="{x:.1f}" y="{y_center - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
        f'rx="{STUB_H / 2}" fill="{fill}" stroke="{color}" stroke-width="1.3"/>',
        f'<text x="{x + STUB_W / 2:.1f}" y="{y_center + 4:.1f}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
        f'fill="{text_fill}">{esc(label)}</text>',
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

# 图例(4 种语义:入口绿/主线蓝/出口橙/折行蓝虚线)
_lx, _ly = PAD_L, TOP_PAD + TITLE_H + 15
for kind, color, label in LEGEND:
    if kind == "solid":
        L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    else:
        L.append(f'<line x1="{_lx}" y1="{_ly - 4}" x2="{_lx + 16}" y2="{_ly - 4}" stroke="{color}" '
                 f'stroke-width="2.4" stroke-dasharray="5,3"/>')
    L.append(f'<text x="{_lx + 22}" y="{_ly}" font-family="sans-serif" font-size="11.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 22 + cjk_text_width(label, 11.5) + 30

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                 f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口 / 出口 / 折行接口桩
ex, ey = NODE_XY[ENTRY_NODE]
ey += NODE_H / 2
L += stub(EDGE_MARGIN, ey, ENTRY_LABEL, C_ENTRY, "#dcfce7", "#166534")
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')

xx, xy = NODE_XY[EXIT_NODE]
xy += NODE_H / 2
sx = w - EDGE_MARGIN - STUB_W
L += stub(sx, xy, EXIT_LABEL, C_EXIT, "#ffedd5", "#9a3412")
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

fox, foy = NODE_XY[FOLD_OUT_NODE]
foy += NODE_H / 2
L += stub(sx, foy, FOLD_OUT_LABEL, C_MAIN, "#dbeafe", "#1e40af")
L.append(f'<line x1="{fox + NODE_W:.1f}" y1="{foy:.1f}" x2="{sx:.1f}" y2="{foy:.1f}" '
         f'stroke="{C_MAIN}" stroke-width="2" stroke-dasharray="6,4" marker-end="url(#mMain)"/>')

fix_, fiy = NODE_XY[FOLD_IN_NODE]
fiy += NODE_H / 2
L += stub(EDGE_MARGIN, fiy, FOLD_IN_LABEL, C_MAIN, "#dbeafe", "#1e40af")
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{fiy:.1f}" x2="{fix_:.1f}" y2="{fiy:.1f}" '
         f'stroke="{C_MAIN}" stroke-width="2" stroke-dasharray="6,4" marker-end="url(#mMain)"/>')

# 主线调用边:多条汇入同一节点时终点 y 错开,否则看不出「汇合」
_dst_total = {}
for _, dst in EDGES:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]
    x2, y2 = NODE_XY[dst]
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    y_off = (i - (n - 1) / 2) * 16 if n > 1 else 0
    L.append(f'<line x1="{x1 + NODE_W:.1f}" y1="{y1 + NODE_H / 2:.1f}" x2="{x2:.1f}" '
             f'y2="{y2 + NODE_H / 2 + y_off:.1f}" stroke="{C_MAIN}" stroke-width="2" '
             f'marker-end="url(#mMain)"/>')

# 同列「属于」关系:虚线蓝竖边 + 左侧注记
for src, dst, note in EDGES_V:
    x1, y1 = NODE_XY[src]
    _x2, y2 = NODE_XY[dst]
    cx = x1 + NODE_W / 2
    L.append(f'<line x1="{cx:.1f}" y1="{y1 + NODE_H:.1f}" x2="{cx:.1f}" y2="{y2:.1f}" '
             f'stroke="{C_MAIN}" stroke-width="1.6" stroke-dasharray="4,3" marker-end="url(#mMain)"/>')
    L.append(f'<text x="{cx - 8:.1f}" y="{(y1 + NODE_H + y2) / 2 + 3.5:.1f}" text-anchor="end" '
             f'font-family="sans-serif" font-size="9.5" fill="{C_NODE_SUB}">{esc(note)}</text>')

# 节点:圆角框 + 真实符号名(可多行) + 一行短语 + 右上角 § 徽标
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    lines = symbol if isinstance(symbol, list) else [symbol]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
             f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    if len(lines) == 1:
        sym_y = [y + NODE_H * 0.44]
        sub_y = y + NODE_H * 0.74
    else:
        sym_y = [y + NODE_H * (0.32 + 0.20 * k) for k in range(len(lines))]
        sub_y = y + NODE_H * 0.85
    for text_line, ty in zip(lines, sym_y):
        fs = fit_size(text_line, SYM_SIZE, NODE_W - TEXT_INSET)
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ty:.1f}" text-anchor="middle" '
                 f'font-family="sans-serif" font-size="{fs:.1f}" font-weight="bold" '
                 f'fill="{C_NODE_TITLE}">{esc(text_line)}</text>')
    fs = fit_size(phrase, SUB_SIZE, NODE_W - TEXT_INSET)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{sub_y:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="{fs:.1f}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 8, y, sec)

# 底部阅读路线:复用列坐标,§ 徽标与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 16:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
             f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first = COLX[stops[0][0]] + NODE_W / 2
    x_last = COLX[stops[-1][0]] + NODE_W / 2
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
             f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for col, sec in stops:
        L += badge(COLX[col] + NODE_W / 2, ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  ({w} x {h}, ratio {w / h:.2f}:1, {len(NODES)} nodes)")
