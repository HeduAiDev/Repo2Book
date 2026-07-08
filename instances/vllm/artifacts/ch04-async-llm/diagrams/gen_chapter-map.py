#!/usr/bin/env python3
"""第 4 章(AsyncLLM：三段式异步解耦)——本章地图：请求往返剖面图。

编号标题章（`## 4.1`..`## 4.11`），站牌用 §4.N 徽标。

两段，折成上下两行、各自成段（画布预算：宽 ≤1500 且宽高比 ≤2.6:1——单行 10 列会
远超预算，故折行）：
  上段(§4.3→§4.4 请求登记与跨进程投递)——generate() 调 add_request() 再到
    _add_request()，在这里扇出两路：一路留在本进程登记进 OutputProcessor
    （叶子，登记完即止），另一路把 EngineCoreRequest 经 add_request_async
    送出本进程，抵达画布外的 EngineCore（灰显，机制留待后续章节）；
  下段(§4.5→§4.7→§4.6→§4.3 结果生产与解多路复用)——output_handler 从
    EngineCore 拉一批结果，process_outputs 按 req_id 解多路复用回各自的
    RequestOutputCollector，消费者 generate() 取出、判停、yield 给客户端。
两段之间留一条空白"桥接带"，只画一条跨段箭头（EngineCore → output_handler，
对应 §4.8 的 IPC 往返消息）——呼应正文"发出去→(进程边界)→收回来"的形状，
不再靠横向无限延展，改靠纵向的"浮出"桥接表达。

用法: python3 gen_chapter-map.py → 同目录 chapter-map.svg
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
LANES = [
    "请求登记与跨进程投递（本进程 Frontend → 独立进程 EngineCore）",
    "结果生产与解多路复用（后台任务 → 本进程 Frontend → 消费者）",
]  # 泳道 -> 折成上下两段, 上->下

# (节点id, 段下标(0=上段/1=下段), 段内列, 段内行号, 真实符号名, 一行短语, §徽标或None)
# 两段各自独立编号列 0..4——折行的关键：不再共享跨段列号，画布宽度只由
# "段内最多 5 列" 决定，而不是 10 个节点依次排开的单行宽度。
NODES = [
    ("gen_entry", 0, 0, 0, "generate()",
     "服务器调用,拿到请求队列 q", "§4.3"),
    ("add_request", 0, 1, 0, "add_request()",
     "转请求→建队列→懒启后台", "§4.4"),
    ("fanout", 0, 2, 0, "_add_request",
     "扇出两路:登记+跨进程投递", "§4.4"),
    ("op_add_request", 0, 3, 1, "OutputProcessor.add_request",
     "登记 req_id→queue 映射表", "§4.7"),
    ("add_req_async", 0, 3, 2, "add_request_async",
     "送出 EngineCoreRequest", "§4.4"),
    ("engine_core_ext", 0, 4, 2, "EngineCore",
     "调度+执行,留待后续章节", None),
    ("output_handler", 1, 0, 0, "output_handler()",
     "拉批→分块→sleep(0)让步", "§4.5"),
    ("process_outputs", 1, 1, 0, "process_outputs()",
     "按 req_id 解多路复用", "§4.7"),
    ("collector", 1, 2, 0, "RequestOutputCollector",
     "单槽+Event,merge合帧", "§4.6"),
    ("gen_exit", 1, 3, 0, "generate()",
     "取出→判停→yield 给客户端", "§4.3"),
]
EDGES = [  # (src_id, dst_id) —— 调用边；同段=段内右中→左中，跨段=桥接带下沿/上沿
    ("gen_entry", "add_request"),
    ("add_request", "fanout"),
    ("fanout", "op_add_request"),      # 扇出：本进程登记这一叶
    ("fanout", "add_req_async"),       # 扇出：跨进程投递这一路
    ("add_req_async", "engine_core_ext"),
    ("engine_core_ext", "output_handler"),   # 跨段：浮出——IPC 往返(§4.8)
    ("output_handler", "process_outputs"),
    ("process_outputs", "collector"),
    ("collector", "gen_exit"),
]
# 阅读顺序上的站牌(与图上节点一一对应，engine_core_ext 是本章外范围不设站牌)，
# 用于底部阅读路线的独立时间轴——不复用图上节点的段内列号(折行后同一列号被
# 两段各用一次，若路线条也用列号，不同段的站牌会在同一 x 位置叠在一起)。
READING_ORDER = ["gen_entry", "add_request", "fanout", "op_add_request", "add_req_async",
                 "output_handler", "process_outputs", "collector", "gen_exit"]
# (路线名, [节点id,...] 按阅读顺序取 READING_ORDER 的子序列, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("完整往返：一个请求从发出到收到（对应 §4.9 时序）", READING_ORDER, True),
    ("只看结果怎么收回来（生产者→分发→队列）",
     ["output_handler", "process_outputs", "collector", "gen_exit"], False),
]
LEGEND = [
    ("#22c55e", "入口：服务器调用发起请求"),
    ("#3b82f6", "章内主线：调用边 / 扇出登记边"),
    ("#f97316", "出口：yield 返回客户端"),
]
TITLE = "第 4 章 · AsyncLLM 请求往返剖面（扇出登记 + 生产者-消费者解扇出）"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
C_BRIDGE_CAPTION = "#475569"
C_EXT_FILL, C_EXT_STROKE, C_EXT_TEXT = "#f1f5f9", "#94a3b8", "#64748b"  # 本章外范围节点(灰显)

# ---------------- 几何常量(全计算,零魔数) ----------------
BADGE_FONT_SIZE = 11
BADGE_PAD_X = 14
BADGE_H = 20


def badge_width(text):
    return max(46.0, cjk_text_width(text, BADGE_FONT_SIZE) + BADGE_PAD_X * 2)


NODE_H = 70
COL_GAP, ROW_GAP = 30, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 46
# 桥接带:两段之间的空白间隔,专放一条跨段箭头 + 简短说明文字。
INTER_LANE_GAP = 190

# 节点宽度:同一批节点统一宽度(保列对齐),按本章最长的符号名/短语算
_SYMBOL_FONT, _PHRASE_FONT = 13, 10.5
_NODE_TEXT_PAD = 20
NODE_W = max(
    190,
    max(cjk_text_width(sym, _SYMBOL_FONT) for *_, sym, _, _ in NODES) + _NODE_TEXT_PAD,
    max(cjk_text_width(ph, _PHRASE_FONT) for *_, ph, _ in NODES) + _NODE_TEXT_PAD,
)
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 16  # 左右各留:接口桩 + 一段箭头

n_cols = max(n[2] for n in NODES) + 1  # 段内最多列数(两段各自独立复用这批列号)
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_band = [0] * len(LANES)
for _id, band, col, row, *_ in NODES:
    rows_per_band[band] = max(rows_per_band[band], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_band]

band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for i, bh in enumerate(band_h):
    if i > 0:
        _cum += INTER_LANE_GAP  # 段与段之间插入桥接带(不给背景色,留白给跨段箭头)
    band_top.append(_cum)
    _cum += bh
lanes_bottom = _cum

NODE_XY = {}
for nid, band, col, row, *_ in NODES:
    x = COLX[col]
    y = band_top[band] + LANE_LABEL_H + BAND_PAD + row * (NODE_H + ROW_GAP)
    NODE_XY[nid] = (x, y)
NODE_BY_ID = {n[0]: n for n in NODES}

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
    """§ 徽标胶囊,居中挂在 (cx,cy)——宽度按文字自适应(见 badge_width),
    颜色/圆角/描边视觉语言与模板一致,不变的是"胶囊+靛蓝描边+深靛蓝粗体文字"。"""
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

# 泳道背景 + 标签(桥接带本身不上色,留白给跨段箭头,视觉上与两段区分开)
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w:.1f}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w:.1f}" y2="{band_top[i]:.1f}" '
              f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
    L.append(f'<line x1="0" y1="{band_top[i] + band_h[i]:.1f}" x2="{w:.1f}" y2="{band_top[i] + band_h[i]:.1f}" '
              f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩(给入口/出口箭头一个可附着的框,兼表达"调用方在画布外")
ex, ey = NODE_XY["gen_entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["gen_exit"]; xy += NODE_H / 2
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

# 调用边:同段(band 相同)= 右中→左中(不论段内是否跨行,直线连接均不经过其它
# 节点框——段内相邻列/相邻行之间是空白,fanout 的两条扇出边同理);跨段(band 不同)
# = 桥接带下沿中点→上沿中点(桥接带本身是留白区,没有节点占用,任意左右跨度都
# 不会撞到其它元素)。
bridge_captions = []  # (x, y, text) —— 桥接带箭头旁的简短说明,渲后统一追加避免被箭头压住
for src, dst in EDGES:
    src_band = NODE_BY_ID[src][1]
    dst_band = NODE_BY_ID[dst][1]
    x1, y1 = NODE_XY[src]
    x2, y2 = NODE_XY[dst]
    if src_band == dst_band:
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        p2 = (x2, y2 + NODE_H / 2)
    elif dst_band > src_band:  # 上段→下段:浮出——src 下沿中点→dst 上沿中点
        p1 = (x1 + NODE_W / 2, y1 + NODE_H)
        p2 = (x2 + NODE_W / 2, y2)
    else:  # 下段→上段(本章未用,保留通用性):src 上沿中点→dst 下沿中点
        p1 = (x1 + NODE_W / 2, y1)
        p2 = (x2 + NODE_W / 2, y2 + NODE_H)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    if src_band != dst_band:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        bridge_captions.append((mx + 24, my, "跨进程 IPC 往返：EngineCoreRequest 去 / EngineCoreOutput 回（§4.8）"))

for cx, cy, cap in bridge_captions:
    L.append(f'<text x="{cx:.1f}" y="{cy:.1f}" font-family="sans-serif" font-size="12.5" '
              f'font-style="italic" fill="{C_BRIDGE_CAPTION}">{esc(cap)}</text>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角 § 徽标;本章外范围节点灰显、无徽标)
for nid, band, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    is_ext = sec is None
    fill = C_EXT_FILL if is_ext else C_NODE_FILL
    stroke = C_EXT_STROKE if is_ext else C_NODE_STROKE
    title_fill = C_EXT_TEXT if is_ext else C_NODE_TITLE
    dash = ' stroke-dasharray="6,4"' if is_ext else ''
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W:.1f}" height="{NODE_H}" rx="12" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"{dash}/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{_SYMBOL_FONT}" font-weight="bold" '
              f'fill="{title_fill}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{_PHRASE_FONT}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    if sec:
        bw = badge_width(sec)
        L += badge(x + NODE_W - bw / 2 + 8, y, sec)

# 底部阅读路线:9 个站牌按 READING_ORDER 均匀分布在整个画布宽度上(独立于图上节点的
# 段内列号——折行后同一列号被两段各用一次,若仍借列号会让不同段的站牌叠在同一 x
# 位置)。速览路线的 4 个站牌取 READING_ORDER 中对应下标的同一 x,保持与全程路线
# 纵向对齐。时间轴左端起点让给路线名文字(按最长路线名的实际宽度算,不留固定
# 魔数空档)。
_route_label_w = max(cjk_text_width(name, 12) for name, *_ in ROUTES)
_route_left = 16 + _route_label_w + 24
_n_stops = len(READING_ORDER)
_route_x = {nid: _route_left + i * (w - PAD_R - _route_left) / (_n_stops - 1)
            for i, nid in enumerate(READING_ORDER)}

L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first, x_last = _route_x[stops[0]], _route_x[stops[-1]]
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for nid in stops:
        sec = NODE_BY_ID[nid][-1]
        if sec:
            L += badge(_route_x[nid], ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w:.0f}x{h:.0f}, aspect={w/h:.2f}:1, NODE_W={NODE_W:.0f})")
