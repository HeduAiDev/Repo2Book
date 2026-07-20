#!/usr/bin/env python3
"""第 6 章「本章地图」——ParentRequest 扇出与归并剖面。

改写自 .claude/skills/svg-diagram/references/example-chapter-map.py（几何/配色
不可变部分照抄），并参考已验收的 instances/vllm/artifacts/ch39-engine-core/
diagrams/chapter-map.py 的两处做法：
  - split_symbol()：真实符号名在节点宽度下装不下时,在离中点最近的下划线处
    拆两行(不加省略号,两段仍是原符号的连续子串,lint 子串核对仍能命中)。
  - 用较宽的 NODE_W(220) + 每列多行(row)堆叠同一泳道内的多个节点,把全章
    9 个节点压进 5 列以内,不让画布横向爆宽。

本章真实控制流(§6.2-§6.7)只有一条主干(扇出→登记→出引擎→归并),没有
ch39 那种"两条几乎不相交的主线",所以边全部是"main"(蓝实线真实调用),
不需要 transition 型边。

■ 不可变(全书统一视觉语言,未改动):
  1. §徽标胶囊 badge()；2. 入口=绿#22c55e/出口=橙#f97316 接口桩；
  3. 章内主线调用边=蓝#3b82f6；4. 底部路线条(高亮=实线蓝/次要=虚线灰)；
  5. >2 种语义色画图例；6. cjk_text_width() 做宽度估算。

■ 本章数据设计要点:
  - `add_request()` 的岔路口(is_pooling or n==1 ?)不单开一个 dispatch 节点——
    真实代码里判据就在 entry 函数体内,没有独立的分派函数,拆出来反而是
    杜撰一个不存在的符号,所以判据写进 entry 的 phrase,岔路本身靠底部两条
    ROUTES(扇出路径 vs 快路径)体现「哪条路线经过哪些列」。
  - §6.4 一节内 ParentRequest.__init__ / get_child_info / _get_child_sampling_params
    三个真实机制各开一个节点、同列纵向堆叠(col1 row0/1/2),都标 §6.4——
    比强行合并成一个大节点更利于"想跳读 seed 递进就找这一格"的选读指引。
  - EngineCore 单开一条泳道(跨进程),体现"n 个 child 出了客户端进程,在引擎
    侧被当成普通独立请求调度"这一章眼(§6.3)——它不是本章重点算法,只画
    一个节点做"路过的中继站",阅读路线里两条主路线都会经过它。
  - 出口节点标 get_outputs(真正的归并函数),phrase 里带出"id 换回
    external_req_id"这一步(同一 §6.6 小节内的下一段代码),避免为了给这半句
    话单开一个节点而超支点预算。
  - 取消(§6.7)不在主干调用链上(它是请求生命周期里独立触发的另一条支线),
    但读者要能从图上找到"取消从哪儿查起"——abort_requests 挂在 register
    节点正下方(同列 row1),因为它读的正是 register 阶段织的两张登记表
    (parent_requests / external_req_ids),边 register→abort 画的是这层
    "数据依赖"而非严格的同步调用序,配色仍用主线蓝(不新增第 4 种语义色,
    避免为一条支线边引入图例负担)。

用法: python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算:全角(ord>0x2E80)按 1.0×size,半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def split_symbol(text, max_w, size):
    """真实符号名在给定字号下装不下节点宽度时,在离中点最近的下划线处拆两行。
    两段各自仍是原符号的连续子串(不加省略号),lint_chapter_map 的子串核对
    对每段仍能命中——不会被判成杜撰符号。找不到下划线就原样返回单行。"""
    if cjk_text_width(text, size) <= max_w:
        return [text]
    positions = [i for i, c in enumerate(text) if c == '_' and i != 0]
    if not positions:
        return [text]
    mid = len(text) // 2
    split_at = min(positions, key=lambda p: abs(p - mid))
    return [text[:split_at], text[split_at:]]


# ---------------- DATA(本章数据) ----------------
LANES = ["客户端进程(扇出 / 登记 / 归并)", "跨进程 · EngineCore(调度)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, [§编号,...])
NODES = [
    ("entry",      0, 0, 0, "add_request",
     "is_pooling/n==1 判快路,否则扇出", ["§6.2"]),
    ("fanout",     0, 1, 0, "ParentRequest",
     "建父状态机,预置归并容器", ["§6.4"]),
    ("childinfo",  0, 1, 1, "get_child_info",
     "派生唯一 child id,登记完成表", ["§6.4"]),
    ("seedcalc",   0, 1, 2, "_get_child_sampling_params",
     "seed+index 递进 / 无 seed 复用", ["§6.4"]),
    ("register",   0, 2, 0, "OutputProcessor.add_request",
     "建独立 RequestState,织两张登记表", ["§6.5"]),
    ("abort",      0, 2, 1, "abort_requests",
     "级联取消全部未完成 child", ["§6.7"]),
    ("enginecore", 1, 3, 0, "EngineCore",
     "n 个 child 当普通请求平等调度", ["§6.3"]),
    ("exit",       0, 4, 0, "get_outputs",
     "攒齐 n 路,换回 external_req_id", ["§6.6"]),
]
# (src_id, dst_id) —— 调用边,统一主线蓝
EDGES = [
    ("entry", "fanout"),
    ("fanout", "childinfo"),
    ("childinfo", "seedcalc"),
    ("childinfo", "register"),
    ("register", "enginecore"),
    ("register", "abort"),
    ("enginecore", "exit"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("扇出路径(n>1,本章主线)", [(0, "§6.2"), (1, "§6.4"), (2, "§6.5"), (3, "§6.3"), (4, "§6.6")], True),
    ("快路径(n==1,见第4章)",   [(0, "§6.2"), (2, "§6.5"), (3, "§6.3"), (4, "§6.6")], False),
    ("取消(级联 abort)",       [(0, "§6.2"), (2, "§6.7")], False),
]
LEGEND = [
    ("#22c55e", "入口:客户端 generate() 调 add_request"),
    ("#3b82f6", "章内主线调用边"),
    ("#f97316", "出口:归并结果返回上层"),
]
TITLE = "第 6 章 · ParentRequest 扇出与归并剖面(源码走线 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 220, 74
TITLE_SIZE, TITLE_LINE_H, SUB_SIZE = 12, 13, 10
COL_GAP, ROW_GAP = 40, 22
EDGE_MARGIN, STUB_W, STUB_H = 16, 60, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 28  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 14
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_W, BADGE_H = 46, 20

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
    """§ 徽标胶囊,居中挂在 (cx,cy)。"""
    bx, by = cx - BADGE_W / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BADGE_W}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
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
# 图例(3 种语义色画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11.5) + 30

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

# 调用边(主线蓝)。多条边汇入同一节点时,终点 y 各偏移,否则重合的终点在视觉上
# 看不出"汇合"(本章无此情形,每个节点至多 1 条入边,但保留通用逻辑)。
# 同列纵向堆叠的节点(如 §6.4 三个节点、register/abort)之间的边是"从上一格
# 掉到下一格",若仍套用"src 右边缘→dst 左边缘"的公式,会画出一条横贯整个
# 节点宽度的对角线、倒穿几何上位于中间的节点框(如 fanout→childinfo 会斜穿
# childinfo 自己的框顶部一角,箭头方向也会指反)。同列(x1==x2)时改画竖直
# 连接线:上一格底边中点→下一格顶边中点。
_dst_total = {}
for _, dst in EDGES:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    if x1 == x2:
        cx = x1 + NODE_W / 2
        if y2 >= y1:
            p1, p2 = (cx, y1 + NODE_H), (cx, y2)
        else:
            p1, p2 = (cx, y1), (cx, y2 + NODE_H)
    else:
        y_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名[必要时拆两行] + 一行短语 + 右上角 § 徽标)
for nid, lane, col, row, symbol, phrase, secs in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    title_lines = split_symbol(symbol, NODE_W - 26, TITLE_SIZE)
    if len(title_lines) == 1:
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.36:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{TITLE_SIZE}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(title_lines[0])}</text>')
    else:
        base_y = y + NODE_H * 0.30
        for li, line in enumerate(title_lines):
            L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{base_y + li * TITLE_LINE_H:.1f}" '
                      f'text-anchor="middle" font-family="sans-serif" font-size="{TITLE_SIZE}" '
                      f'font-weight="bold" fill="{C_NODE_TITLE}">{esc(line)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.86:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{SUB_SIZE}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    bcx = x + NODE_W - BADGE_W / 2 + 8
    for sec in secs:
        L += badge(bcx, y, sec)
        bcx -= (BADGE_W + 6)

# 底部阅读路线:复用列坐标 COLX,§ 徽标与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
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
print(f"wrote {out} ({w:.0f}x{h:.0f})")
