#!/usr/bin/env python3
"""第 14 章(抢占与请求生命周期回流)——本章地图:源码剖面图。

本章是编号标题章(`## 14.N`),§徽标用 §14.N,徽标章号须与目录号 ch14 一致。

两段(上下堆叠,画布预算:宽 ≤1500 且宽高比 ≤2.6:1;若单行铺开 8 个节点会横向超宽,
参照 ch24 chapter-map 的折行修复,改上下两段各自成行):
  上段 schedule()——RUNNING 阶段抢占循环(§14.1-§14.2)+ WAITING 阶段双队列守卫
    (§14.3-§14.4),4 节点,段内从左到右按调用顺序排列;
  下段 update_from_output()——回流闭环与终态判定(§14.5-§14.7),4 节点,段内从左
    到右按调用顺序排列。
两段之间只有一条真实的跨拍桥接边:dualqueue(§14.4,上段最右)→update_output(§14.5,
下段最左)——本拍 schedule() 收尾之后,要等模型 forward 返回,下一次才轮到
update_from_output() 被调用,这是本章两个函数入口之间唯一的真实时间缝隙。
guard(§14.3)判定"本拍抢占过就跳过 WAITING"这个分支,不额外画第二条跨段边表示
"直接跳到 update_from_output"——那条捷径靠底部两条阅读路线的站牌序列表达(路线 1
不经过 §14.4,路线 2 不经过 §14.2),避免图上再添一条容易和 dualqueue→update_output
视觉粘连的斜线。

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
LANES = ["schedule()：RUNNING 抢占循环 → WAITING 双队列守卫", "update_from_output()：回流闭环与终态判定"]

# (节点id, 段下标(0=上段 schedule, 1=下段 update_from_output), 段内列, 段内行号,
#  真实符号名, 一行短语, §徽标)
NODES = [
    ("entry",         0, 0, 0, "allocate_slots",
     "RUNNING 阶段要块失败，返回 None", "§14.1"),
    ("preempt",       0, 1, 0, "_preempt_request",
     "LIFO 抢 running 末尾，free KV·回队头", "§14.2"),
    ("guard",         0, 2, 0, "not preempted_reqs",
     "本拍抢占过就跳过 WAITING", "§14.3"),
    ("dualqueue",     0, 3, 0, "skipped_waiting",
     "阻塞态隔离，双队列防饿死", "§14.4"),
    ("update_output", 1, 0, 0, "update_from_output",
     "追加 token·spec 回退·回流入口", "§14.5"),
    ("checkstop",     1, 1, 0, "check_stop",
     "EOS / stop_token / 长度 / 重复", "§14.6"),
    ("free",          1, 2, 0, "_free_request",
     "真完成：登记·释放 KV·删记录", "§14.5"),
    ("exit",          1, 3, 0, "RequestStatus",
     "is_finished = status > PREEMPTED", "§14.7"),
]
EDGES = [  # (src_id, dst_id) —— 调用边;同段=段内左→右主线蓝,跨段=桥接带竖向/斜向蓝
    ("entry", "preempt"),
    ("preempt", "guard"),
    ("guard", "dualqueue"),
    ("dualqueue", "update_output"),   # 唯一跨段边:本拍调度收尾 → 下一次回流调用
    ("update_output", "checkstop"),
    ("checkstop", "free"),
    ("free", "exit"),
]
# 阅读顺序上的 7 个 § 站牌(§14.5 在上文对应两个节点 update_output/free，此处按
# 章节号去重为一个时间轴刻度——路线要表达的是"跳去哪一节"，不是"图上第几个框")。
READING_ORDER = ["§14.1", "§14.2", "§14.3", "§14.4", "§14.5", "§14.6", "§14.7"]
# (路线名, [§徽标,...] 按阅读顺序取 READING_ORDER 的子序列, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("抢占发生：内存紧张路径", ["§14.1", "§14.2", "§14.3", "§14.5", "§14.6", "§14.7"], True),
    ("无抢占：双队列顺畅路径", ["§14.1", "§14.3", "§14.4", "§14.5", "§14.6", "§14.7"], False),
]
LEGEND = [
    ("#22c55e", "入口：延续 ch13 schedule() 的 RUNNING 阶段"),
    ("#3b82f6", "章内主线调用边（含 1 处跨拍桥接）"),
    ("#f97316", "出口：is_finished 判定后返回调用方"),
]
TITLE = "第 14 章 · 抢占循环与请求生命周期回流剖面（源码走线 + § 讲解站牌）"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
C_BRIDGE_CAPTION = "#475569"

# ---------------- 几何常量(全计算,零魔数) ----------------
BADGE_FONT_SIZE = 11
BADGE_PAD_X = 14
BADGE_H = 20


def badge_width(text):
    return max(46.0, cjk_text_width(text, BADGE_FONT_SIZE) + BADGE_PAD_X * 2)


NODE_H = 62
COL_GAP, ROW_GAP = 30, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 46
# 桥接带:两段之间的空白间隔,专放跨段箭头 + 简短说明文字。
INTER_LANE_GAP = 130

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

# 调用边:同段(band 相同)= 段内左→右,右中→左中;跨段(band 不同)= 桥接带上下沿,
# 上中/下中 attach(避免"从右边出、绕回左边"的倒退斜线,均不经过任何节点框内部,
# 因为桥接带本身是留白区,没有节点占用)。
bridge_captions = []  # (x, y, text) —— 桥接带箭头旁的简短说明,渲后统一追加避免被箭头压住
for src, dst in EDGES:
    src_band = NODE_BY_ID[src][1]
    dst_band = NODE_BY_ID[dst][1]
    x1, y1 = NODE_XY[src]
    x2, y2 = NODE_XY[dst]
    if src_band == dst_band:
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        p2 = (x2, y2 + NODE_H / 2)
    elif dst_band > src_band:  # 上段→下段:浮回落地,src 下沿中点→dst 上沿中点
        p1 = (x1 + NODE_W / 2, y1 + NODE_H)
        p2 = (x2 + NODE_W / 2, y2)
    else:  # 下段→上段(本章未用到,保留通用性)
        p1 = (x1 + NODE_W / 2, y1)
        p2 = (x2 + NODE_W / 2, y2 + NODE_H)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    if src_band != dst_band:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        cap = "本拍 schedule() 收尾 → 模型 forward 返回后调用 update_from_output()"
        bridge_captions.append((mx - cjk_text_width(cap, 12.5) / 2, my, cap))

for cx, cy, cap in bridge_captions:
    L.append(f'<text x="{cx:.1f}" y="{cy:.1f}" font-family="sans-serif" font-size="12.5" '
              f'font-style="italic" fill="{C_BRIDGE_CAPTION}">{esc(cap)}</text>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角 § 徽标)
for nid, band, col, row, symbol, phrase, sec in NODES:
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

# 底部阅读路线:7 个 § 站牌按 READING_ORDER 均匀分布在整个画布宽度上(独立于图上
# 节点的段内列号——两段各自复用列 0..3,若路线条仍借列号,§14.1(上段col0)与
# §14.5(下段col0)会叠在同一 x 位置)。时间轴左端起点让给路线名文字(按最长路线名
# 的实际宽度算,不留固定魔数空档)。
_route_label_w = max(cjk_text_width(name, 12) for name, *_ in ROUTES)
_first_badge_half_w = badge_width(READING_ORDER[0]) / 2
_route_left = 16 + _route_label_w + 24 + _first_badge_half_w
_n_stops = len(READING_ORDER)
_route_x = {name: _route_left + i * (w - PAD_R - _route_left) / (_n_stops - 1)
            for i, name in enumerate(READING_ORDER)}

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
    for sec in stops:
        L += badge(_route_x[sec], ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w:.0f}x{h:.0f}, NODE_W={NODE_W:.0f})")
