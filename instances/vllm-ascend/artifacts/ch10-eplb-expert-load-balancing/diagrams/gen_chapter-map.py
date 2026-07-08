#!/usr/bin/env python3
"""第 10 章(Expert 负载均衡 eplb:子进程规划 + D2D 权重热迁移)——本章地图:源码剖面图。

本章是自然标题章(正文标题为"一个临时的重型特性"/"主线①:节拍状态机 EplbUpdator"等,
无 `## N.M` 编号)——按契约禁用 §N.M 徽标,站牌改用标题词本身;badge 用自适应宽度
(参照 ch24-primer-flash-attention 的 badge_width() 写法,而非固定 46px 胶囊)。

两段式布局(参照同为自然标题、同样"主进程↔子进程"跨进程结构的 ch24 桥接带写法):
  下段(推理主进程,①③,含 entry/exit)——构建 → 点火与收集(all_gather) → …
    → forward_before/forward_end 取规划+发起异步搬运 → 主线③ D2D 三态机(一站聚合,
    详见本章 three_state.png/cadence_table.png) → 逐拍归零(exit)。
  上段(EPLB 子进程,②④)——worker_process 唤醒 → do_update 规划主流程 →
    代表策略 DefaultEplb(④,一站聚合 PolicyFactory 派发 + 算法本体,详见
    rebalance_binpack.png) → compose 差集反推 send/recv(详见 map_diff.png)。
两段之间的桥接带画两条跨段箭头:主进程"点火"下潜子进程规划、子进程"回传"浮回主进程
取规划——对应 planner_q(唤醒信号,无界)/ block_update_q(规划结果,maxsize=1)两条
真实队列。

节点预算:9 个真实符号节点(≤12)。已有 5 张细节图(pipeline_overview / cadence_table /
three_state / map_diff / rebalance_binpack)覆盖各主线内部细节,本图故意粗粒度聚合到
"节"一级,不复刻其内容,只负责"从哪进、经过什么阶段、从哪出、跳读去哪节"的导航。

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
LANES = ["EPLB 子进程 · 规划(主线②)与策略(主线④)", "推理主进程 · 节拍(主线①)与异步搬运(主线③)"]
# band 0 = 上段(子进程), band 1 = 下段(主进程,含 entry/exit)

# (节点id, band, 段内列, 段内行, 真实符号名(可含 "\n" 机械换行), 一行短语(可含 "\n"), 站牌文字)
# 站牌是从对应正文标题里蒸馏出的短关键词(仿 ch24-primer-flash-attention 的
# "tiling"/"cascade" 写法,而非照抄整条自然标题原文)——完整标题太长,adaptive
# badge_width() 撑到那个宽度会在 NODE_W=210/COL_GAP=30 的密度下互相压字,
# 短关键词才能既对得上标题、又不越界。见文件开头 station→heading 对照表。
NODES = [
    ("build",         1, 0, 0, "EplbUpdator",
     "构建期拉起三件套\n(loader+子进程+updator)", "构建期"),
    ("gather_wakeup", 1, 1, 0, "compute_and_set_\nmoe_load()",
     "all_gather 收负载\n+ 唤醒子进程", "点火与收集"),
    ("worker_process", 0, 0, 0, "worker_process",
     "planner_q 唤醒后\n跑子进程主循环", "容器与两条队列"),
    ("do_update",     0, 1, 0, "do_update()",
     "读共享态→调策略\n→算 send/recv→打包", "do_update"),
    ("policy",        0, 2, 0, "DefaultEplb\nrebalance_experts",
     "PolicyFactory 派发到此\n冗余副本+贪心装箱+5%门槛", "DefaultEplb"),
    ("compose",       0, 3, 0, "compose_expert_\nupdate_info_greedy",
     "-1 差集反推\n每层 send/recv", "compose"),
    ("fetch",         1, 2, 0, "forward_before",
     "取回规划\n+ 发起异步 P2P", "forward_before"),
    ("d2d",           1, 3, 0, "D2DExpertWeightLoader",
     "generate→asyn→update\n三态搬权重(聚合)", "三态机"),
    ("cycle_end",     1, 4, 0, "update_iteration()",
     "满周期闭环\ncur_iterations 归零", "逐拍归零"),
]
# 站牌→正文标题对照(供自查/盲审核对,不参与渲染):
#   构建期        → "构建期：三件套与子进程"
#   点火与收集    → "点火与收集：借来的 all_gather"
#   容器与两条队列 → "容器与两条队列"(原文即此,未截短)
#   do_update     → "do_update：子进程里的规划主流程"
#   DefaultEplb   → "代表策略：DefaultEplb 怎么铺平负载"
#   compose       → "compose：用 -1 差集反推 send/recv"
#   forward_before → "forward_before / forward_end：节拍点上做的事"
#   三态机        → "主线③：异步 P2P 三态机 D2DExpertWeightLoader"
#   逐拍归零      → "逐拍走一整轮，看清「哪一拍干什么」"
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝;同 band 同行内左→右,跨 band 走桥接带
    ("build", "gather_wakeup"),
    ("gather_wakeup", "worker_process"),   # 跨段(下→上):planner_q 唤醒,潜入子进程规划
    ("worker_process", "do_update"),
    ("do_update", "policy"),
    ("policy", "compose"),
    ("compose", "fetch"),                  # 跨段(上→下):block_update_q 回传,浮回主进程
    ("fetch", "d2d"),
    ("d2d", "cycle_end"),
]
BRIDGE_CAPTION = {
    ("gather_wakeup", "worker_process"): "planner_q 唤醒，潜入子进程规划",
    ("compose", "fetch"): "block_update_q 回传，浮回主进程",
}
# 阅读顺序(与节点一一对应,按正文行文顺序)
READING_ORDER = [
    "构建期", "点火与收集", "容器与两条队列", "do_update", "DefaultEplb",
    "compose", "forward_before", "三态机", "逐拍归零",
]
# (路线名, [站牌文字,...] 按阅读顺序取 READING_ORDER 的子序列, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("通读全流水线(构建→节拍→子进程规划→策略→异步搬运→归零)", READING_ORDER, True),
    ("跳读：只想弄清④策略怎么铺平负载",
     ["容器与两条队列", "do_update", "DefaultEplb", "compose"], False),
]
LEGEND = [
    ("#22c55e", "入口：model_runner 每 step 调用"),
    ("#3b82f6", "章内主线调用/回传边"),
    ("#f97316", "出口：归零，交还推理主循环"),
]
TITLE = "第 10 章 · EPLB 在线热迁移剖面(节拍状态机 + 子进程规划 + 异步 P2P 搬运)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
C_BRIDGE_CAPTION = "#475569"

# ---------------- 几何常量(全计算,零魔数) ----------------
BADGE_FONT_SIZE = 10.5
BADGE_PAD_X = 12
BADGE_H = 20


def badge_width(text):
    """站牌胶囊宽度按文字自适应(本章站牌是完整自然标题短语,比 §N.M 长得多,
    固定 46px 会溢出——照 ch24-primer-flash-attention 的写法用 cjk_text_width 估算)。"""
    return max(46.0, cjk_text_width(text, BADGE_FONT_SIZE) + BADGE_PAD_X * 2)


NODE_W, NODE_H = 210, 92  # 高度留 2 行符号 + 2 行短语
COL_GAP, ROW_GAP = 30, 20
EDGE_MARGIN, STUB_W, STUB_H = 14, 64, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 28  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 50
# 桥接带:两段之间的空白间隔,专放跨段箭头 + 简短说明文字(留白,不上背景色)
INTER_LANE_GAP = 90

n_cols = max(n[2] for n in NODES) + 1  # 段内最多列数(两段各自独立复用这批列号)
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_band = [0] * len(LANES)
for _id, band, col, row, *_ in NODES:
    rows_per_band[band] = max(rows_per_band[band], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_band]

band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for i, bh in enumerate(band_h):
    if i > 0:
        _cum += INTER_LANE_GAP
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
    """站牌徽标胶囊,居中挂在 (cx,cy)——宽度自适应(见 badge_width)。"""
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

# 泳道背景 + 标签(桥接带本身不上色,留白给跨段箭头)
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w:.1f}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w:.1f}" y2="{band_top[i]:.1f}" '
              f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
    L.append(f'<line x1="0" y1="{band_top[i] + band_h[i]:.1f}" x2="{w:.1f}" y2="{band_top[i] + band_h[i]:.1f}" '
              f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩(entry=build, exit=cycle_end,都在下段"推理主进程"——它是本章对外可见的边界)
ex, ey = NODE_XY["build"]; ey += NODE_H / 2
xx, xy = NODE_XY["cycle_end"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#166534">{esc("model_runner")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("下一轮 step")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边:同段(band 相同)= 段内左→右,右中→左中;跨段(band 不同)= 桥接带上下沿,
# 上中/下中 attach(不经过任何节点框内部,桥接带本身是留白区)。
bridge_captions = []  # (x, y, text) —— 桥接带箭头旁的简短说明,渲后统一追加避免被箭头压住
for src, dst in EDGES:
    src_band = NODE_BY_ID[src][1]
    dst_band = NODE_BY_ID[dst][1]
    x1, y1 = NODE_XY[src]
    x2, y2 = NODE_XY[dst]
    if src_band == dst_band:
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        p2 = (x2, y2 + NODE_H / 2)
    elif dst_band > src_band:  # band 数值变大 = 页面上往下走(上段→下段):浮回主进程,
        p1 = (x1 + NODE_W / 2, y1 + NODE_H)  # src(band0,上段)下沿中点
        p2 = (x2 + NODE_W / 2, y2)           # dst(band1,下段)上沿中点
    else:  # band 数值变小 = 页面上往上走(下段→上段):潜入子进程规划,
        p1 = (x1 + NODE_W / 2, y1)           # src(band1,下段)上沿中点
        p2 = (x2 + NODE_W / 2, y2 + NODE_H)  # dst(band0,上段)下沿中点
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    if src_band != dst_band:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        cap = BRIDGE_CAPTION.get((src, dst), "")
        if cap:
            bridge_captions.append((mx + 16, my, cap))

for cx, cy, cap in bridge_captions:
    L.append(f'<text x="{cx:.1f}" y="{cy:.1f}" font-family="sans-serif" font-size="11.5" '
              f'font-style="italic" fill="{C_BRIDGE_CAPTION}">{esc(cap)}</text>')

# 节点(圆角框 + 真实符号名(1~2 行) + 一行短语(1~2 行) + 右上角站牌)
SYMBOL_1LINE_Y, SYMBOL_2LINE_Y1, SYMBOL_2LINE_Y2 = 34, 26, 42
PHRASE_1LINE_Y, PHRASE_2LINE_Y1, PHRASE_2LINE_Y2 = 74, 68, 84
for nid, band, col, row, symbol, phrase, sec in NODES:
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
    bw = badge_width(sec)
    L += badge(x + NODE_W - bw / 2 + 8, y, sec)

# 底部阅读路线:9 个站牌按 READING_ORDER 均匀分布在整个画布宽度上(独立于图上节点的
# 段内列号——上下两段各自复用列号 0..3,若仍借列号会让不同段的站牌叠在同一 x)。
_route_label_w = max(cjk_text_width(name, 12) for name, *_ in ROUTES)
# _route_x[READING_ORDER[0]] 的圆心就是 _route_left(见下),所以留白除了路线名
# 本身的宽度,还要再让出第一枚站牌的半宽,否则站牌越宽、越往左侵蚀路线名文字
# (本章"构建期"站牌较宽 + 首条路线名较长时,原先的固定 24px 留白不够,曾压字)。
_route_left = 16 + _route_label_w + 24 + badge_width(READING_ORDER[0]) / 2
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
print(f"wrote {out} ({w:.0f}x{h:.0f})")
