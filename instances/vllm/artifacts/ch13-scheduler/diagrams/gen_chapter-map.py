#!/usr/bin/env python3
"""第 13 章「本章地图」——schedule() 一拍剖面 + update_from_output() 反馈环。

改写自 .claude/skills/svg-diagram/references/example-chapter-map.py（沿用 ch36
新增的多徽标节点 / transition 边 / split_symbol 三项扩展）。本章代码主线其实是
两个方法：`schedule()`（发出去）和 `update_from_output()`（收回来）——不是同一次
调用栈里的两段，而是 EngineCore busy loop 里前后两次独立调用（中间隔着一次模型
前向）。处理办法：

  - 仍然只画「一个」入口桩(左边缘,绿)和「一个」出口桩(右边缘,橙)——入口挂在
    `schedule()` 本身（§13.1 契约 + §13.2 token_budget 初始化，两个 § 合挂一个
    节点），出口挂在 `update_from_output()`（§13.6，它的返回值回到 EngineCore）。
  - `_update_after_schedule()`（§13.5 乐观推进）到 `update_from_output()`（§13.6）
    之间不存在直接函数调用——中间隔着"模型前向跑完、EngineCore 下一次调用"，画一条
    与"章内主线调用边"(蓝实线,真实调用)视觉上明确不同的边：灰色虚线 + 独立图例
    "章内换题(非函数调用)"，如实标注这只是叙事顺序上的衔接。
  - `running`/`waiting`(RUNNING/WAITING 两阶段)与 `update_after`/`exit` 各自向下
    钻一层("KV 分配/抢占"、"AsyncScheduler 覆写")展示真实的更深一级符号——这两层
    都是"泳道=调用深度"的原教旨用法(与模板示例的 调度层→执行层→算子层 一致)。

■ 不可变(全书统一视觉语言，抄自模板 + ch36 扩展，未改动):
  1. §徽标胶囊 badge()（含 ch36 起支持的"一个节点挂多个 §"）；
  2. 入口=绿#22c55e/出口=橙#f97316 接口桩；
  3. 章内主线调用边=蓝#3b82f6；
  4. 底部路线条(高亮=实线蓝/次要=虚线灰)；
  5. >2 种语义色画图例；6. cjk_text_width() 做宽度估算；
  7. split_symbol() 在符号名装不下节点宽度时于下划线处拆两行(不加省略号)。

■ 本章新增(仅本章需要):
  - entry 节点(`schedule()`)同时挂 §13.1(不分相契约)与 §13.2(token_budget 初始化)
    两块徽标——这两节共享同一个真实入口，不必拆两个节点。
  - transition 边只有一条：`update_after`(§13.5)→`exit`(§13.6)，标注"下一次
    busy loop,模型前向已跑完"，不是函数调用。

用法: python3 gen_chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def split_symbol(text, max_w, size):
    """真实符号名在给定字号下装不下节点宽度时，在离中点最近的下划线处拆两行。
    两段各自仍是原符号的连续子串(不加省略号)，lint_chapter_map 的子串核对
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
LANES = ["schedule() → update_from_output():一拍的发起与收拢", "KV 分配 / 抢占", "AsyncScheduler 覆写(占位机制)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, [§编号,...])
NODES = [
    ("entry",      0, 0, 0, "schedule()",
     "不分相契约;token_budget=预算池", ["§13.1", "§13.2"]),
    ("running",    0, 1, 0, "num_new_tokens",
     "RUNNING:追赶公式,三重min截断", ["§13.3"]),
    ("allocate",   1, 1, 0, "allocate_slots",
     "失败→抢占队尾 self.running.pop()", ["§13.3"]),
    ("waiting",    0, 2, 0, "preempted_reqs",
     "WAITING:if not preempted_reqs才进", ["§13.4"]),
    ("chunked",    1, 2, 0, "enable_chunked_prefill",
     "关→超预算break;开→截token_budget", ["§13.4"]),
    ("sched_out",  0, 3, 0, "SchedulerOutput",
     "scheduled_new_reqs全量/其余增量", ["§13.5"]),
    ("update_after", 0, 4, 0, "_update_after_schedule",
     "乐观推进 num_computed_tokens", ["§13.5"]),
    ("placeholder", 2, 4, 0, "num_output_placeholders",
     "覆写:+=1+cur_num_spec_tokens", ["§13.7"]),
    ("exit",       0, 5, 0, "update_from_output",
     "追加token,check_stop,free", ["§13.6"]),
    ("update_req", 2, 5, 0, "_update_request_with_output",
     "覆写:占位兑现 -=len(new_token_ids)", ["§13.7"]),
]
# (src_id, dst_id, style) —— style 省略即 "main"(蓝实线,真实调用/同函数内控制流);
# "transition" = 灰虚线,章内换题,非函数调用。
EDGES = [
    ("entry", "running"),
    ("running", "waiting"),
    ("waiting", "sched_out"),
    ("sched_out", "update_after"),
    ("running", "allocate"),
    ("waiting", "chunked"),
    ("update_after", "placeholder"),
    ("update_after", "exit", "transition"),
    ("exit", "update_req"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("主线:一拍全流程(推荐)",        [(0, "§13.1"), (1, "§13.3"), (2, "§13.4"), (3, "§13.5"), (4, "§13.5"), (5, "§13.6")], True),
    ("只看 token 预算怎么花",        [(0, "§13.2"), (1, "§13.3"), (2, "§13.4")], False),
    ("只看 SchedulerOutput 全量/增量", [(3, "§13.5"), (5, "§13.6")], False),
    ("只看异步占位(AsyncScheduler)", [(4, "§13.7"), (5, "§13.7")], False),
]
LEGEND = [
    ("#22c55e", "入口:EngineCore busy loop 调用进入"),
    ("#3b82f6", "章内主线调用边(真实调用/同函数控制流)"),
    ("#f97316", "出口:返回上层"),
    ("#94a3b8", "章内换题(非函数调用,隔一次模型前向)"),
]
TITLE = "第 13 章 · schedule() 一拍剖面 + update_from_output() 反馈环(源码走线 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_TRANSITION = "#94a3b8"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 190, 66
TITLE_SIZE, TITLE_LINE_H, SUB_SIZE = 12, 13, 10
COL_GAP, ROW_GAP = 26, 20
EDGE_MARGIN, STUB_W, STUB_H = 14, 64, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 26  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 22, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 42
BADGE_W, BADGE_H = 44, 19

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
        f'font-size="10.5" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN), ("Trans", C_TRANSITION))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14.5" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例;本章 4 色)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11) + 26

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

# 入口/出口接口桩
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["exit"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(main=主线蓝;transition=灰虚线,章内换题非调用)
main_edges = [e for e in EDGES if (e[2] if len(e) > 2 else "main") == "main"]
transition_edges = [e for e in EDGES if len(e) > 2 and e[2] == "transition"]
_dst_total = {}
for src, dst in main_edges:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst in main_edges:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    same_row = abs(y1 - y2) < 1
    if same_row:
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        p2 = (x2, y2 + NODE_H / 2)
    else:
        # 跨泳道竖直钻深:从上边框底部中点连到下方节点顶部中点
        p1 = (x1 + NODE_W / 2, y1 + NODE_H)
        p2 = (x2 + NODE_W / 2, y2)
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    if n > 1 and same_row:
        y_off = (i - (n - 1) / 2) * 16
        p2 = (p2[0], p2[1] + y_off)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# transition 边:本章的 update_after→exit 同处 lane0 同一行、仅列号相邻,
# 直接从 src 右边框中点连到 dst 左边框中点(与 main 边同一落点公式),
# 只是描边换成灰虚线+专属箭头——不需要绕行,没有第三方节点挡在中间。
for src, dst, _style in transition_edges:
    bx, by = NODE_XY[src]
    rx, ry = NODE_XY[dst]
    p1 = (bx + NODE_W, by + NODE_H / 2)
    p2 = (rx, ry + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_TRANSITION}" stroke-width="2" stroke-dasharray="7,5" '
              f'marker-end="url(#mTrans)"/>')

# 节点(圆角框 + 真实符号名[必要时拆两行] + 一行短语 + 右上角 § 徽标[可多个并排])
for nid, lane, col, row, symbol, phrase, secs in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    title_lines = split_symbol(symbol, NODE_W - 24, TITLE_SIZE)
    if len(title_lines) == 1:
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.38:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{TITLE_SIZE}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(title_lines[0])}</text>')
    else:
        base_y = y + NODE_H * 0.30
        for li, line in enumerate(title_lines):
            L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{base_y + li * TITLE_LINE_H:.1f}" '
                      f'text-anchor="middle" font-family="sans-serif" font-size="{TITLE_SIZE}" '
                      f'font-weight="bold" fill="{C_NODE_TITLE}">{esc(line)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.88:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{SUB_SIZE}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    # 右上角 § 徽标:多个并排贴在上边框,不下探进文字区
    bcx = x + NODE_W - BADGE_W / 2 + 6
    for sec in secs:
        L += badge(bcx, y, sec)
        bcx -= (BADGE_W + 5)

# 底部阅读路线:复用列坐标 COLX,§ 徽标与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="11.5" '
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
