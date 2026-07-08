#!/usr/bin/env python3
"""第 8 章「本章地图」——init_ascend_model_parallel 源码剖面图。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py 模板改写：
不可变机制(esc/cjk_text_width/入口绿-出口橙-主线蓝/路线实线蓝-虚线灰/图例规则)
原样保留，只改下面的 DATA + 一处必要的几何泛化(见下)。

本章 narrative/chapter.md 全部是**自然标题**(无 `## N.M` 编号)，按契约：禁用
§N.M 徽标，站牌改用标题词本身(取自各节标题的一个可核实子串，如"时序证据"取自
"## 时序证据：基座先建，昇腾后叠")。

[必要泛化] 模板原版 badge() 用固定 BADGE_W=46，因为编号章的站牌永远是等长的
"§N.M"。本章站牌是长短不一的中文短语(短至 "MC2" 3 字符，长至 "取用、哨兵与
销毁" 8 字符)，固定宽度要么裁字要么留白过多——改成按 cjk_text_width() 逐条
计算宽度(min 36px)，右边界仍钉在节点右边框 +8px 处(与原版数学等价:
cx = x+NODE_W-width/2+8 时 right_edge = cx+width/2 = x+NODE_W+8 与 width 无关，
所以哪怕站牌变宽也不会撞到右侧下一列节点，只会向节点内部多占一点左边距)。
底部路线徽标同理居中动态宽度。除了这处宽度计算，配色/形状/图例规则不变。

节点预算 11 ≤ 12：
  主线(6, 蓝色实边连成一条链): entry → grid → mc2 → finegrained → flashcomm2 → exit
  卫星(5, 不挂边, 仅作"随时可查"的旁证/细节站): import_reuse / algebra /
  numchunks / cp_layout / factory_reuse。

设计要点：
- 主线是"排布代数"这条真实控制流:worker 触发 → all_ranks 5D 网格 → 三个
  昇腾专属组依次建(MC2/细粒度TP/flashcomm2，源码里就是这个先后顺序)→ 取用/
  销毁出口。CP 组(PCP/DCP)由基座建、不在昇腾主线的调用链上，因此画成卫星
  (cp_layout)，对齐在 grid 同一列，呼应"同一张网格"这条论点，而不是编一条
  不存在的调用边把它接进主线。
- import_reuse / cp_layout / factory_reuse 三个"复用证据 + CP 归口"卫星统一
  收进独立的第 4 条泳道(而不是分别贴在 entry/grid/exit 正下方)。
  [FIX-ROUND-1](渲染→Read PNG 发现后本轮修正，未改变任何论点/符号)：
  初版把 import_reuse 放在 entry 正下方(同列不同行)，Read PNG 发现
  entry→grid 的主线箭头斜穿而下时从 import_reuse 节点框的右上角"擦边而过"
  (数学上仍在框外，但间距仅 8px，肉眼看像穿框，不是一张干净的图)。改法:
  把 import_reuse 连同 cp_layout / factory_reuse 一起下沉到独立的第 4 泳道
  (与主线所在的 0-2 泳道之间完全没有斜线穿越)，重渲后 Read PNG 确认三处
  斜线(entry→grid、grid→mc2、flashcomm2→exit)都不再靠近任何卫星节点框。
- numchunks 是"细粒度 TP"节点的算法细节卫星，与 ch20 的 state/split 卫星同构
  (同列不同行)。

用法:python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算——全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["触发 / 出入口层", "排布代数根", "昇腾专属组(MC2 / 细粒度 TP / flashcomm2)", "复用证据 + CP 归口(基座)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名(可含 "\n" 机械换行,不改变拼写), 一行短语, 站牌(标题词,自然标题章禁用 §N.M))
NODES = [
    ("entry", 0, 0, 0,
     "_init_worker_distributed\n_environment",
     "worker 时序:基座先建、\n昇腾后叠",
     "时序证据"),
    ("grid", 1, 1, 0,
     "all_ranks",
     "5D 网格\n(ExternalDP·dp·pp·pcp·tp)",
     "5D 网格"),
    ("algebra", 1, 1, 1,
     "transpose\nreshape / unbind",
     "三步法:切出任意\n维度的组",
     "排布代数"),
    ("mc2", 2, 2, 0,
     "_MC2",
     "transpose(1,2)+reshape\n收成专家域,兼总哨兵",
     "MC2"),
    ("finegrained", 2, 3, 0,
     "_create_or_get_group",
     "沿 DP 借 rank,\n与全局 TP 正交",
     "细粒度 TP"),
    ("numchunks", 2, 3, 1,
     "num_chunks",
     "dp // group_size,\n两轮切块示例",
     "两轮切块"),
    ("flashcomm2", 2, 4, 0,
     "_FLASHCOMM2_OTP",
     "strided 跨步重排,\n通信-计算重叠",
     "flashcomm2"),
    ("exit", 0, 5, 0,
     "get_mc2_group()",
     "取用带 assert 断言\n销毁 destroy() 置 None",
     "取用、哨兵与销毁"),
    ("import_reuse", 3, 0, 0,
     "GroupCoordinator",
     "同 init_model_parallel_group\n都从基座 import 进来",
     "复用而非替换"),
    ("cp_layout", 3, 1, 0,
     "_PCP\n_DCP",
     "同一张网格,\npcp/dcp 换维度切",
     "CP 归口"),
    ("factory_reuse", 3, 5, 0,
     "init_model_parallel\n_group",
     "昇腾每建一组都调它,\n吐出 GroupCoordinator",
     "复用的接缝"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝;其余 5 个是卫星节点,不挂边
    ("entry", "grid"),
    ("grid", "mc2"),
    ("mc2", "finegrained"),
    ("finegrained", "flashcomm2"),
    ("flashcomm2", "exit"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
# 路线名刻意压短(≤6 字)——文字区只有到第一个站牌左边界前的窄条,长句会压到站牌上。
ROUTES = [
    ("通读主线",
     [(0, "时序证据"), (1, "5D 网格"), (2, "MC2"), (3, "细粒度 TP"), (4, "flashcomm2"), (5, "取用、哨兵与销毁")], True),
    ("比三种切法",
     [(2, "MC2"), (3, "细粒度 TP"), (4, "flashcomm2")], False),
    ("看两处旁证",
     [(0, "复用而非替换"), (1, "CP 归口"), (5, "复用的接缝")], False),
]
LEGEND = [("#22c55e", "入口:从上层调用进入"), ("#3b82f6", "章内主线调用边"), ("#f97316", "出口:返回上层")]
TITLE = "第 8 章 · init_ascend_model_parallel 源码剖面(5D 网格排布代数 + 三种切法 + CP 归口)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
# 本章符号名/部分短语偏长——用机械换行(见 NODES 里的 "\n")；NODE_W 只需装下
# "半个符号名"，NODE_H 加高两行以容纳最多 2 行符号 + 最多 2 行短语。
NODE_W, NODE_H = 185, 90
COL_GAP, ROW_GAP = 30, 22
EDGE_MARGIN, STUB_W, STUB_H = 12, 60, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_H, BADGE_PAD_X, BADGE_MIN_W = 20, 14, 36  # 站牌高度固定,宽度按文字动态算(见下)

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


def _badge_width(text):
    """站牌宽度按文字动态算(自然标题站牌长短不一，不能像 §N.M 那样固定宽度)。"""
    return max(BADGE_MIN_W, cjk_text_width(text, 11) + BADGE_PAD_X)


def badge_topright(x, y, node_w, text):
    """§ 徽标胶囊贴节点右上角。right_edge = x+node_w+8 恒定(与 width 无关，
    因为 cx = x+node_w-width/2+8 时 cx+width/2 恰好把 width/2 消掉)——
    哪怕站牌变宽也不会撞到右侧下一列节点，只会向节点内部多占一点左边距。"""
    width = _badge_width(text)
    cx = x + node_w - width / 2 + 8
    return _badge_rect_text(cx, y, width, text)


def badge_centered(cx, cy, text):
    """路线徽标:居中挂在 (cx,cy)，宽度同样动态算。"""
    width = _badge_width(text)
    return _badge_rect_text(cx, cy, width, text)


def _badge_rect_text(cx, cy, width, text):
    bx, by = cx - width / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{width:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
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

# 调用边(主线蓝,先画边再画节点盖住端点毛刺)
# 多条边汇入同一节点时,终点 y 各偏移(间距 16px),否则重合的终点看不出"汇合"。
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

# 节点(圆角框 + 真实符号名(1~2 行) + 一行短语(1~2 行,始终锚在节点下半区) + 右上角站牌)
SYMBOL_1LINE_Y, SYMBOL_2LINE_Y1, SYMBOL_2LINE_Y2 = 34, 24, 40
PHRASE_1LINE_Y, PHRASE_2LINE_Y1, PHRASE_2LINE_Y2 = 71, 66, 80
for nid, lane, col, row, symbol, phrase, tag in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_lines = symbol.split("\n")
    sym_ys = [y + SYMBOL_1LINE_Y] if len(sym_lines) == 1 else [y + SYMBOL_2LINE_Y1, y + SYMBOL_2LINE_Y2]
    for line, ly in zip(sym_lines, sym_ys):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ly:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(line)}</text>')
    phrase_lines = phrase.split("\n")
    phrase_ys = [y + PHRASE_1LINE_Y] if len(phrase_lines) == 1 else [y + PHRASE_2LINE_Y1, y + PHRASE_2LINE_Y2]
    for line, ly in zip(phrase_lines, phrase_ys):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ly:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(line)}</text>')
    L += badge_topright(x, y, NODE_W, tag)

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first = COLX[stops[0][0]] + NODE_W / 2
    x_last = COLX[stops[-1][0]] + NODE_W / 2
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for col, tag in stops:
        L += badge_centered(COLX[col] + NODE_W / 2, ry, tag)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}: {w:.0f}x{h:.0f}")

# [SELF-CHECK](渲染→Read PNG 亲眼看后如实记录，见 figure-manifest.json 对应条目)
