#!/usr/bin/env python3
"""第 33 章「本章地图」——KV Connector 契约与调度集成剖面。

改写自 .claude/skills/svg-diagram/references/example-chapter-map.py（几何/配色
不可变部分照抄），并复用已验收的 instances/vllm/artifacts/ch12-engine-core/
diagrams/chapter-map.py 的两个做法：
  - split_symbol()/wrap_text()：真实符号名/短语在节点宽度下装不下时按软断点
    折两行(不加省略号，折出的每段仍是原词的连续子串，lint 子串核对仍能命中)。
  - NODE_H 按"最坏情形"(2 行符号名 + 3 行短语)反推，逐节点显式垒高度，
    避免固定比例锚点压字/溢出。
  - 同列纵向堆叠 + 同列竖直连接线公式(x1==x2 时画上一格底边中点→下一格
    顶边中点)：把"启动期构造"(role→factory→entry)和"alloc→isolate 掉入
    隔离区"这两处纵向关系画清楚，而不是套用横向对角线公式画出穿框的斜线。

本章有两条源码主线(契约 base.py + 调度集成 scheduler.py)，因此三条泳道
按"层"划分而非按调用顺序：
  泳道0 契约与构造(base.py / factory.py，一次性)：KVConnectorRole 定义
    role-split 的根，create_connector 按 role 各造一份——只发生在
    Scheduler.__init__，不是运行时每步都走的路径，所以单独一条竖向
    "启动期"支线喂给运行时入口(仿 ch12 的 mcb→stepfn→entry 写法)。
  泳道1 调度器 WAITING 循环·决策侧(scheduler.py，§33.4 为主)：entry 选队列
    取队头→dispatch 判阻塞态/试提升→(新请求主干)hits 查远程命中→alloc 分配
    +登记→exit 提升回 WAITING/PREEMPTED。alloc 另有一条纵向掉落边通向
    isolate(WAITING_FOR_REMOTE_KVS)——如实反映"这条支线本步到此为止，
    不延伸到 exit"：被隔离的请求本步不会变成 RUNNING，要等下一步 dispatch
    重新判定，这条回环不画成图上的一条真实回边(避免虚构一条会穿过 hits/
    alloc 节点框的倒退箭头)，只在 dispatch 的短语里用文字说明。
  泳道2 KV 到位与释放·回传侧(scheduler.py，§33.5)：recv/update_waiting 是
    dispatch 提升判定背后的两个真实方法(worker 报完成→打标记；提升成功→
    补登记+全命中回退一个 token)，connector_finished 是请求真正结束时的
    另一支线(能否立即释放 block)。三者都不接调用边——它们是"随时可查的
    背景机制"(仿 ch15 的 hashfn/queue/block 卫星节点写法)，不是同一次
    循环迭代里会连续跑到的下一步，画成有向边反而是编造一条并不存在的
    直接调用关系；用同列/邻列的空间位置 + phrase 文字表达"它们服务于
    dispatch 的提升判定"这层关系，足够让读者按 § 跳读。

■ 不可变(全书统一视觉语言，未改动)：
  1. §徽标胶囊 badge()；2. 入口=绿#22c55e / 出口=橙#f97316 接口桩；
  3. 章内主线调用边=蓝#3b82f6；4. 底部路线条(高亮=实线蓝 / 次要=虚线灰)；
  5. >2 种语义色画图例；6. cjk_text_width() 做宽度估算。

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
    """真实符号名在给定字号下装不下节点宽度时,在离中点最近的下划线或点号处
    拆两行(不加省略号,两段各自仍是原符号的连续子串,lint_chapter_map 的子串
    核对对每段仍能命中——不会被判成杜撰符号)。找不到断点就原样返回单行。"""
    if cjk_text_width(text, size) <= max_w:
        return [text]
    positions = [i for i, c in enumerate(text) if c in "_." and i != 0]
    if not positions:
        return [text]
    mid = len(text) // 2
    split_at = min(positions, key=lambda p: abs(p - mid))
    return [text[:split_at], text[split_at:]]


_SOFT_BREAK = set("，；：、 ,;→（）()")


def wrap_text(text, max_w, size):
    """一行短语按宽度贪心换行:逐字符累加,超宽时回溯当前行最近的软断点
    (中英文标点/箭头/空格/括号)处折行,避免把节点框内长短语硬生生粘到相邻
    节点上(no_overlap)。找不到软断点才硬断。"""
    lines, cur = [], ""
    for ch in text:
        trial = cur + ch
        if cjk_text_width(trial, size) <= max_w or not cur:
            cur = trial
            continue
        brk = -1
        for i in range(len(cur) - 1, -1, -1):
            if cur[i] in _SOFT_BREAK:
                brk = i
                break
        if 0 <= brk < len(cur) - 1:
            lines.append(cur[: brk + 1])
            cur = cur[brk + 1 :] + ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


# ---------------- DATA(本章数据) ----------------
LANES = [
    "契约与构造(base.py / factory.py，一次性)",
    "调度器 WAITING 循环 · 决策侧(scheduler.py，§33.4)",
    "KV 到位与释放 · 回传侧(scheduler.py，§33.5)",
]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, [§编号,...])
#
# 列设计说明：entry/dispatch 同列(col1)不同行——真实控制流是
# `_is_blocked_waiting_status(status) and not _try_promote_blocked_waiting_request(...)`，
# 阻塞态判定在 entry 分支里就地做；只有"确实处于阻塞态"的请求才会真的调
# dispatch(_try_promote_blocked_waiting_request)，非阻塞态(全新请求)直接从
# entry 掉头查命中(entry→hits)，不经过 dispatch——entry→hits 和 entry→dispatch
# 是两条并列出边，不是串联，如实反映"dispatch 只服务于阻塞态请求"这一点，
# 不虚构一条"所有请求都先过 dispatch"的假主干。dispatch 提升成功后同样落到
# hits(dispatch→hits)。exit/isolate 同列(col4)不同行——都是"这一轮的终点"，
# 但只有 exit 那一行真的接出口桩：isolate 的请求本步到此为止，不接返回上层。
NODES = [
    ("role", 0, 0, 0, "KVConnectorRole",
     "SCHEDULER=0 / WORKER=1，role-split 的根", ["§33.2"]),
    ("factory", 0, 0, 1, "create_connector",
     "按 role 懒加载；调度器进程只造 SCHEDULER 那份", ["§33.3"]),
    ("entry", 1, 1, 0, "_select_waiting_queue_for_scheduling",
     "选队列取队头；非阻塞态(新请求)直接查命中", ["§33.4"]),
    ("dispatch", 1, 1, 1, "_try_promote_blocked_waiting_request",
     "仅阻塞态请求才查这里：finished_recving_kv_req_ids 命中就提升→查命中，否则重新隔离",
     ["§33.4", "§33.5"]),
    ("hits", 1, 2, 0, "get_num_new_matched_tokens",
     "查远程命中；返回 None 就先跳过这个请求", ["§33.4"]),
    ("alloc", 1, 3, 0, "allocate_slots",
     "delay_cache_blocks 推迟登记；update_state_after_alloc 通知 connector",
     ["§33.4"]),
    ("exit", 1, 4, 0, "RequestStatus.WAITING",
     "提升回 WAITING/PREEMPTED，下一步进 RUNNING", ["§33.5"]),
    ("isolate", 1, 4, 1, "WAITING_FOR_REMOTE_KVS",
     "置状态，prepend 进 step_skipped_waiting 隔离(本步到此为止)", ["§33.4"]),
    ("recv", 2, 1, 0, "_update_from_kv_xfer_finished",
     "worker 报 finished_recving，标记入 finished_recving_kv_req_ids", ["§33.5"]),
    ("update_waiting", 2, 1, 1, "_update_waiting_for_remote_kv",
     "cache_blocks 补登记；整 prompt 全命中回退一个 token", ["§33.5"]),
    ("connector_finished", 2, 4, 0, "_connector_finished",
     "request_finished 裁决：block 立即放还是等 finished_sending", ["§33.5"]),
]
# (src_id, dst_id) —— 调用边,统一主线蓝
EDGES = [
    ("role", "factory"),
    ("factory", "entry"),
    ("entry", "hits"),       # 非阻塞态(新请求)：直接查命中
    ("entry", "dispatch"),   # 阻塞态：转提升判定
    ("dispatch", "hits"),    # 提升成功：落到同一条查命中主干
    ("hits", "alloc"),
    ("alloc", "exit"),
    ("alloc", "isolate"),    # 首次判定需异步等待：隔离，本步到此为止
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("决策侧全流程(一次调度步，推荐)",
     [(1, "§33.4"), (2, "§33.4"), (3, "§33.4"), (4, "§33.5")], True),
    ("阻塞态判定 / 隔离等待(次要)",
     [(1, "§33.4"), (3, "§33.4"), (4, "§33.4")], False),
]
LEGEND = [
    ("#22c55e", "入口：调度循环每步调用"),
    ("#3b82f6", "章内主线调用边"),
    ("#f97316", "出口：提升回 WAITING/PREEMPTED → RUNNING"),
]
TITLE = "第 33 章 · KV Connector 契约与调度集成剖面(源码走线 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
# NODE_H 按"最坏情形"(2 行符号名 + 3 行短语，本章 alloc/dispatch 等长短语节点
# 会触发)反推，保证逐节点文字互不重叠；行数更少的节点顶部锚点固定、底部
# 多留白，不居中但绝不压字。
NODE_W = 178
TITLE_SIZE, TITLE_LINE_H, SUB_SIZE, PHRASE_LINE_H = 12, 13, 10, 12
TITLE_TOP, PHRASE_GAP, NODE_BOTTOM_PAD = 22, 8, 10
MAX_TITLE_LINES, MAX_PHRASE_LINES = 2, 3
NODE_H = (TITLE_TOP + (MAX_TITLE_LINES - 1) * TITLE_LINE_H + PHRASE_GAP + PHRASE_LINE_H
          + (MAX_PHRASE_LINES - 1) * PHRASE_LINE_H + NODE_BOTTOM_PAD)
COL_GAP, ROW_GAP = 26, 20
EDGE_MARGIN, STUB_W, STUB_H = 14, 58, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 26  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
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

# 调用边(主线蓝)。同列(x1==x2)时是"从上一格掉到下一格"的纵向关系(启动期
# role→factory→entry；alloc→isolate 掉入隔离区)，画竖直连接线(上一格底边
# 中点→下一格顶边中点)，不套用"src 右边缘→dst 左边缘"的横向对角线公式——
# 否则会画出一条穿过中间节点框的斜线。多条边汇入同一节点时终点 y 各偏移，
# 避免重合的终点看不出"汇合"(本章暂无汇合，逻辑保留以防后续改数据)。
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

# 节点(圆角框 + 真实符号名[必要时拆两行] + 一行短语[必要时按软断点折多行] +
# 右上角 § 徽标[可多个,从右向左叠放])。符号名与短语各自独立按行显式垒
# 高度，避免"两行符号名+三行短语"这种最坏情形在固定比例下互相压字。
for nid, lane, col, row, symbol, phrase, secs in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    title_lines = split_symbol(symbol, NODE_W - 22, TITLE_SIZE)
    title_base_y = y + TITLE_TOP
    for li, line in enumerate(title_lines):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{title_base_y + li * TITLE_LINE_H:.1f}" '
                  f'text-anchor="middle" font-family="sans-serif" font-size="{TITLE_SIZE}" '
                  f'font-weight="bold" fill="{C_NODE_TITLE}">{esc(line)}</text>')
    phrase_lines = wrap_text(phrase, NODE_W - 14, SUB_SIZE)
    phrase_base_y = title_base_y + (len(title_lines) - 1) * TITLE_LINE_H + PHRASE_GAP + PHRASE_LINE_H
    for pi, line in enumerate(phrase_lines):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{phrase_base_y + pi * PHRASE_LINE_H:.1f}" '
                  f'text-anchor="middle" font-family="sans-serif" font-size="{SUB_SIZE}" '
                  f'fill="{C_NODE_SUB}">{esc(line)}</text>')
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
