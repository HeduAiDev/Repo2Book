#!/usr/bin/env python3
"""第 39 章「本章地图」——弹性 EP 扩缩状态机 + Responses 有状态多轮剖面。

改写自 .claude/skills/svg-diagram/references/example-chapter-map.py。本章
是全书唯一一章内塞了两条几乎不相交的真实代码主线(弹性扩缩的分布式状态机 /
Responses 多轮的会话存储)，两条线各自有自己的入口与出口——不是同一条
调用链上的两段。处理办法：

  - 仍然只画「一个」入口桩(左边缘,绿)和「一个」出口桩(右边缘,橙)——
    入口挂在全章第一个真实机制 reinitialize_distributed()（§39.5，弹性扩
    缩的触发点），出口挂在全章最后一个真实机制 msg_store/response_store
    落库（§39.12，多轮会话的收尾）。这保持了全书统一的「一进一出」视觉语言，
    不引入模板之外的第二套接口桩样式。
  - 两条主线之间(barrier → create_responses)不存在函数调用关系，画一条
    与"章内主线调用边"（蓝实线，代表真实调用）视觉上明确不同的边：灰色
    虚线 + 单独图例"章内换题(非函数调用)"，如实标注这只是叙事顺序上的
    衔接，不是代码调用。这比省略这条边(两块孤悬、读者看不出章内顺序)
    或用蓝色实线(会被误读成"barrier 调用了 create_responses")更诚实。

■ 不可变(全书统一视觉语言，抄自模板，未改动):
  1. §徽标胶囊 badge()；2. 入口=绿#22c55e/出口=橙#f97316 接口桩；
  3. 章内主线调用边=蓝#3b82f6；4. 底部路线条(高亮=实线蓝/次要=虚线灰)；
  5. >2 种语义色画图例；6. cjk_text_width() 做宽度估算。

■ 本章新增(仅本章需要，未改动上面的不可变部分):
  - [FIX-ROUND-2] 弹性 EP 一侧按 §39.3 表格的真实 2×2 分类拆成 **四个角色
    节点**(existing/new/remaining/removing)，而不是合并成两个。第一轮曾把
    `_progress_existing_engine()` 一个节点同时挂 §39.6 和 §39.8 两块徽标——
    但 §39.8 讲的是 `_progress_new_engine()`/`_eep_scale_up_before_kv_init()`
    这两个完全不同的真实符号，`_progress_existing_engine()` 从未出现在
    §39.8 里。盲审顺着 §39.8 徽标跳过去，落在一个和该节内容无关的符号上，
    判定 FAIL。改法：`progress()` 本就按 (scale_type, worker_type) 派发到
    四个真实函数之一(见 §39.4 代码)，四个函数各自只在自己的小节被讲——
    于是四个函数各开一个节点，每个只挂"确实讲它的那一个"§ 徽标，
    `dispatch → {existing,new,remaining,removing} → barrier` 四路都是真实
    调用边(_staged_barrier 是四个函数各自都会走到的公共点，见正文
    "等 engine_count 齐 + staged_barrier(...)" 那几行注释)。
  - `entry`(`reinitialize_distributed()`)只是"已经在跑的引擎"收到
    reconfig 请求的入口——新引擎是全新进程，靠自己 `__init__` 里的
    `_eep_scale_up_before_kv_init()` 直接构造状态机，从未调用
    `reinitialize_distributed()`。所以底部"弹性扩容·新引擎"路线**不挂
    §39.5**，从 `dispatch`(§39.4)开始，不假称新引擎也走 entry 这一站。
  - 一条"transition"型的边(barrier→resp_entry)：灰虚线+专属箭头，和
    "main"型(蓝实线)区分开，见 EDGES 里第三个字段。
  - 节点标题(真实符号名)如果按 cjk_text_width() 在 12px 字号下仍装不进
    节点宽度，就在离中点最近的下划线处拆两行(split_symbol())——纯几何
    换行，不改变符号本身、不加省略号(省略号会被误判成新符号)。

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
    对每段仍能命中——不会被判成杜撰符号。找不到下划线就原样返回单行(允许
    轻微溢出，好过瞎拆断词)。"""
    if cjk_text_width(text, size) <= max_w:
        return [text]
    positions = [i for i, c in enumerate(text) if c == '_' and i != 0]
    if not positions:
        return [text]
    mid = len(text) // 2
    split_at = min(positions, key=lambda p: abs(p - mid))
    return [text[:split_at], text[split_at:]]


# ---------------- DATA(本章数据) ----------------
LANES = ["弹性 EP 扩缩(分布式拓扑)", "Responses 有状态多轮(会话记忆)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, [§编号,...])
NODES = [
    ("entry",       0, 0, 0, "reinitialize_distributed",
     "外部触发扩缩,转非阻塞", ["§39.5"]),
    ("dispatch",    0, 1, 0, "progress",
     "busy loop 每轮调一次,四路分派", ["§39.4"]),
    ("existing",    0, 2, 0, "_progress_existing_engine",
     "存在引擎9步,两次WAIT跨进程握手", ["§39.6"]),
    ("new",         0, 2, 1, "_progress_new_engine",
     "新引擎4步,KV-init前完成扩入", ["§39.8"]),
    ("remaining",   0, 2, 2, "_progress_remaining_engine",
     "余留引擎4步,一气切到新组", ["§39.9"]),
    ("removing",    0, 2, 3, "_progress_removing_engine",
     "被裁引擎3步,完毕发SHUTDOWN_COMPLETE", ["§39.9"]),
    ("barrier",     0, 3, 0, "_staged_barrier",
     "两阶段barrier,容忍rank时间偏序", ["§39.7"]),
    ("resp_entry",  1, 0, 0, "create_responses",
     "取previous_response_id查历史", ["§39.10"]),
    ("non_harmony", 1, 1, 0, "construct_input_messages",
     "非harmony:织历史+上轮输出+新输入", ["§39.11"]),
    ("harmony",     1, 1, 1, "_construct_input_messages_with_harmony",
     "新会话建system/续轮接msg_store", ["§39.12"]),
    ("exit",        1, 2, 0, "msg_store / response_store",
     "落库:共享list令输出自动留存", ["§39.12"]),
]
# (src_id, dst_id, style) —— style 省略即 "main"(蓝实线,真实调用)；
# "transition" = 灰虚线,章内换题,非函数调用。
EDGES = [
    ("entry", "dispatch"),
    ("dispatch", "existing"), ("dispatch", "new"),
    ("dispatch", "remaining"), ("dispatch", "removing"),
    ("existing", "barrier"), ("new", "barrier"),
    ("remaining", "barrier"), ("removing", "barrier"),
    ("barrier", "resp_entry", "transition"),
    ("resp_entry", "non_harmony"), ("resp_entry", "harmony"),
    ("non_harmony", "exit"), ("harmony", "exit"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("弹性扩容·存在引擎(主线)",        [(0, "§39.5"), (1, "§39.4"), (2, "§39.6"), (3, "§39.7")], True),
    ("弹性扩容·新引擎(KV-init前扩入)", [(1, "§39.4"), (2, "§39.8"), (3, "§39.7")], False),
    ("弹性缩容(余留/被裁)",           [(0, "§39.5"), (1, "§39.4"), (2, "§39.9"), (3, "§39.7")], False),
    ("Responses 非harmony",          [(0, "§39.10"), (1, "§39.11"), (2, "§39.12")], True),
    ("Responses harmony",            [(0, "§39.10"), (1, "§39.12"), (2, "§39.12")], False),
]
LEGEND = [
    ("#22c55e", "入口:从上层/客户端调用进入"),
    ("#3b82f6", "章内主线调用边"),
    ("#f97316", "出口:返回上层"),
    ("#94a3b8", "章内换题(非函数调用,仅衔接两条主线)"),
]
TITLE = "第 39 章 · 弹性 EP 扩缩状态机 + Responses 多轮剖面(源码走线 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_TRANSITION = "#94a3b8"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 235, 74
TITLE_SIZE, TITLE_LINE_H, SUB_SIZE = 12, 13, 10
COL_GAP, ROW_GAP = 46, 22
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
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
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN), ("Trans", C_TRANSITION))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例;本章 4 色)
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

# 入口/出口接口桩(全章只有一进一出:入口挂在 §39.5 触发点,出口挂在 §39.12 落库点)
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

# 调用边(main=主线蓝;transition=灰虚线,章内换题非调用)
# 多条边汇入同一节点时,终点 y 各偏移,避免看不出"汇合"——仅对 main 边计数,
# transition 边走独立的折线路由,不占用这个偏移。
main_edges = [e for e in EDGES if (e[2] if len(e) > 2 else "main") == "main"]
transition_edges = [e for e in EDGES if len(e) > 2 and e[2] == "transition"]
_dst_total = {}
for src, dst in main_edges:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst in main_edges:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    y_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# transition 边:直线会斜穿 _progress_remaining_engine() 的框(barrier 右上、
# resp_entry 左下、remaining_engine 恰好卡在两者之间的几何路径上)。改走三段
# 折线:从 barrier 下边中点直落(经过 lane0 第 2 行里 col3 那格本就空着的地方,
# 不与 remaining_engine [col2] 重叠)→ 在两条泳道之间的空档转水平 → 从
# resp_entry 上边中点落下——全程避开所有节点框。
for src, dst, _style in transition_edges:
    bx, by = NODE_XY[src]
    rx, ry = NODE_XY[dst]
    drop_x = bx + NODE_W / 2
    land_x = rx + NODE_W / 2
    turn_y = ry - 14
    pts = [(drop_x, by + NODE_H), (drop_x, turn_y), (land_x, turn_y), (land_x, ry)]
    path_d = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    L.append(f'<path d="{path_d}" fill="none" stroke="{C_TRANSITION}" stroke-width="2" '
              f'stroke-dasharray="7,5" marker-end="url(#mTrans)"/>')

# 节点(圆角框 + 真实符号名[必要时拆两行] + 一行短语 + 右上角 § 徽标[可多个并排])
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
    # 右上角 § 徽标:多个并排贴在上边框,不下探进文字区(标题从 0.30~0.36 起才开始)
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
