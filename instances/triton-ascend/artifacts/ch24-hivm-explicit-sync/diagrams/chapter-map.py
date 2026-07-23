#!/usr/bin/env python3
"""第 24 章「本章地图」——HIVM 显式同步源码剖面图。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py（及同实例 ch23
chapter-map.py）体例改写：不可变机制(esc/cjk_text_width/badge/配色/图例规则)原样保留，
只改 DATA + 蛇形走线的方向感知端点。

节点预算 12（§24.1…§24.12），恰是本章 12 节，逐一挂 §24.N 站牌。为压住画布宽度、
避免一行 12 框横向无限延展，走线折成「两层泳道 · 蛇形下行」：
- 泳道 0「① 核内流水同步（§24.1→24.8）」占两行：row0 左→右（§24.1→§24.4），
  在 §24.4 列垂直下探到 row1，row1 右→左（§24.5→§24.8）——一条连续的蛇。
- 泳道 1「② 跨核同步 + 小结」一行：§24.9→§24.10→§24.11→§24.12（出口）。
- §24.8 列垂直下探进泳道 1 的 §24.9，两层在此接力。

边端点方向感知：dst 在右→从 src 右缘出；dst 在左→从 src 左缘出；同列→垂直下探。
相邻节点直线不穿越第三个框。全部符号名取自 chapter.md/dossier.json 原样子串（linter
只核带 `_`/`(`/内部 `.` 的 token；驼峰类名如 GraphSyncSolver 不触发核对但仍逐字取真）。

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算——全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
# 泳道 1 标签须短(≤ ~9 CJK 字)：§24.8→§24.9 的垂直下探线在 c0 中心 x≈184 处穿过泳道 1
# 标签所在 y 带，标签过长会与该线相撞；范围信息留给节点/路线/标题承载。
LANES = ["① 核内流水同步（§24.1→24.8，蛇形下行）", "② 跨核同步 + 小结"]  # 泳道,上→下
LANE_ROWS = [2, 1]  # 每条泳道内的行数(泳道 0 折两行装 8 框)

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名(可含 "\n" 机械换行), 一行短语, §编号)
NODES = [
    # 泳道 0 · row0：左→右
    ("n241", 0, 0, 0, "AutoInjectSync",              "六级异步引擎无自动\n依赖，同步须进 IR",       "§24.1"),
    ("n242", 0, 1, 0, "set_flag / wait_flag\npipe_barrier", "核内同步三件套 op\n与语义",             "§24.2"),
    ("n243", 0, 2, 0, "InjectSync\n六道工序",         "一趟 pass 的\n总览流水线",                     "§24.3"),
    ("n244", 0, 3, 0, "MemAlias",                     "内存别名分析建\n生产者–消费者边",              "§24.4"),
    # 泳道 0 · row1：右→左(接 §24.4 垂直下探)
    ("n245", 0, 3, 1, "barrier / flag\n决策二分",      "同 pipe 插 barrier\n异 pipe 插 flag",          "§24.5"),
    ("n246", 0, 2, 1, "GraphSyncSolver",              "最小同步集：可达即\n冗余，断路即补插",         "§24.6"),
    ("n247", 0, 1, 1, "SyncEventId\nAllocation",      "event id 分池 +\n生命周期复用（图着色）",      "§24.7"),
    ("n248", 0, 0, 1, "MoveSyncState",                "循环里同步外提：\nset 提前 / wait 沉后",       "§24.8"),
    # 泳道 1：左→右(接 §24.8 垂直下探)
    ("n249",  1, 0, 0, "sync_block_set /\nsync_block_wait", "Cube↔Vector 经\n显存 gm 握手",           "§24.9"),
    ("n2410", 1, 1, 0, "InjectBlockMixSync /\nShallowSync",  "只对 MIX 核注入\n两条融合路径",           "§24.10"),
    ("n2411", 1, 2, 0, "sync_block_lock /\nunlock",         "跨核旗记账 +\n块间锁严格互斥",           "§24.11"),
    ("n2412", 1, 3, 0, "两层同步\n一套分析",                "带同步的 HIVM IR\n续降到 AscendC",        "§24.12"),
]
# 调用边(主线蓝),相邻节点直线不穿第三框；端点方向由列序自动定(见下)。
EDGES = [
    ("n241", "n242"), ("n242", "n243"), ("n243", "n244"),   # row0 左→右
    ("n244", "n245"),                                        # 垂直下探(c3, row0→row1)
    ("n245", "n246"), ("n246", "n247"), ("n247", "n248"),   # row1 右→左
    ("n248", "n249"),                                        # 垂直下探(c0, 泳道0→泳道1)
    ("n249", "n2410"), ("n2410", "n2411"), ("n2411", "n2412"),  # 泳道1 左→右
]
# (路线名, [§编号,...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)——badge 沿线均匀落点
ROUTES = [
    ("全通读（§24.1→24.12）", ["§24.1", "§24.3", "§24.5", "§24.8", "§24.10", "§24.12"], True),
    ("只搭第一层 · 核内流水同步", ["§24.2", "§24.4", "§24.6", "§24.7"], False),
    ("只搭第二层 · 跨核同步 + 小结", ["§24.9", "§24.10", "§24.11", "§24.12"], False),
]
LEGEND = [("#22c55e", "入口:从上层调用进入"), ("#3b82f6", "章内主线调用边"), ("#f97316", "出口:返回上层")]
TITLE = "第 24 章 · HIVM 显式同步源码剖面（两层同步:核内流水 + 跨核握手 · § 讲解站牌）"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 170, 90
COL_GAP, ROW_GAP = 30, 24
EDGE_MARGIN, STUB_W, STUB_H = 12, 55, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_W, BADGE_H = 46, 20

n_cols = max(n[2] for n in NODES) + 1
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in LANE_ROWS]
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


def badge_w(text):
    """站牌胶囊宽度——按文字自适应,不用固定 BADGE_W 截断(§24.10/§24.12 比 §24.N 长一位)。"""
    return max(BADGE_W, cjk_text_width(text, 11) + 14)


def badge(cx, cy, text):
    """§ 徽标胶囊,居中挂在 (cx,cy)。"""
    bw = badge_w(text)
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
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

# 入口/出口接口桩
ex, ey = NODE_XY["n241"]; ey += NODE_H / 2
xx, xy = NODE_XY["n2412"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("上一章 IR")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("降到下一章")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝) —— 方向感知端点：右邻从右缘出、左邻从左缘出、同列垂直下探。
for src, dst in EDGES:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    cs = NODE_BY_ID[src][2]; cd = NODE_BY_ID[dst][2]
    if cd > cs:      # 向右
        p1 = (xs_ + NODE_W, ys_ + NODE_H / 2); p2 = (xd, yd + NODE_H / 2)
    elif cd < cs:    # 向左
        p1 = (xs_, ys_ + NODE_H / 2); p2 = (xd + NODE_W, yd + NODE_H / 2)
    else:            # 同列：垂直下探
        p1 = (xs_ + NODE_W / 2, ys_ + NODE_H); p2 = (xd + NODE_W / 2, yd)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名(1~2 行) + 一行短语(1~2 行) + 右上角 § 徽标)
SYMBOL_1LINE_Y, SYMBOL_2LINE_Y1, SYMBOL_2LINE_Y2 = 34, 24, 40
PHRASE_1LINE_Y, PHRASE_2LINE_Y1, PHRASE_2LINE_Y2 = 71, 66, 80
for nid, lane, col, row, symbol, phrase, sec in NODES:
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
    L += badge(x + NODE_W - badge_w(sec) / 2 + 8, y, sec)

# 底部阅读路线:badge 沿线均匀落点(蛇形布局下列坐标不再对应阅读序,故不锚列)
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
# badge 起点须让过左侧路线名(最长名如"只搭第二层 · 跨核同步 + 小结")，否则首个 badge 压住名字。
ROUTE_LABEL_W = max(cjk_text_width(name, 12) for name, _stops, _hi in ROUTES)
ROUTE_X0 = 16 + ROUTE_LABEL_W + 28
ROUTE_X1 = COLX[n_cols - 1] + NODE_W / 2
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{ROUTE_X0:.1f}" y1="{ry:.1f}" x2="{ROUTE_X1:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    n = len(stops)
    for si, sec in enumerate(stops):
        bx = ROUTE_X0 + (ROUTE_X1 - ROUTE_X0) * (si / (n - 1) if n > 1 else 0.5)
        L += badge(bx, ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}: {w:.0f}x{h:.0f}")
