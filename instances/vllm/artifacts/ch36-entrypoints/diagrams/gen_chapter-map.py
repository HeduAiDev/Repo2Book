#!/usr/bin/env python3
"""第 36 章「本章地图」——OpenAI 兼容服务器：请求剖面 + 服务器生命周期。

改写自 .claude/skills/svg-diagram/references/example-chapter-map.py。本章正文
自己在 §36.9 小结里点破：全章其实是两套生命周期的交织——「请求的一生」(横向,
§36.4 §36.5 §36.8)是主线,「服务器怎么起来、怎么倒下」(§36.2 §36.3 §36.6 §36.7)
是地基。两者不是两条互不相干的叙事(不同于第 37 章那种真正无函数调用关系的
两条线)——地基的 build_app/init_app_state 直接装配出请求主线要用的中间件和
handler、serve_http 直接 spawn 出 watchdog_task/shutdown_task,全是真实调用边,
不需要第 37 章那种"非调用/仅叙事衔接"的灰色虚线。

入口/出口选点:入口挂在全章第一个真实机制 setup_server()(§36.2,进程启动第一
步);出口挂在 handle_shutdown()(§36.7,§36.7 结尾"从绑端口到关端口...任何一条
退出路径都不会留下孤儿子进程"是全章实质性内容的收官句,35.8/35.9 之后只是补一块
中间件说明和摘要)。请求主线自己的两个终态节点(stream/full generator)留作 lane
内的终端叶子——不额外画第二套接口桩(全书统一"每侧一个"),节点自身的短语
(「推 SSE」「聚合 JSON」)已经把「HTTP 响应从这里出去」交代清楚,盲审复述时
读节点文字即可对上。

■ 不可变(全书统一视觉语言，抄自模板，未改动):
  1. §徽标胶囊 badge()；2. 入口=绿#22c55e/出口=橙#f97316 接口桩；
  3. 章内主线调用边=蓝#3b82f6；4. 底部路线条(高亮=实线蓝/次要=虚线灰)；
  5. >2 种语义色画图例；6. cjk_text_width() 做宽度估算。

■ 本章新增(仅本章需要，未改动上面的不可变部分):
  - split_symbol()：真实符号名太长时优先在下划线处拆两行；没有下划线
    (如 AuthenticationMiddleware 这种纯驼峰名)退化到离中点最近的大写字母
    边界拆分——两段仍都是原符号的连续子串,不引入省略号,lint 的子串核对
    对每段仍能命中。
  - 一个节点可能同时属于两个小节(如 build_app / init_app_state 这一步,
    FastAPI(lifespan=lifespan) 把 §36.6 的 lifespan 钩子也挂在这里)——
    NODES 的 § 字段是列表,右上角并排贴多个徽标(抄自第 37 章的写法)。

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
    """真实符号名在给定字号下装不下节点宽度时拆两行：优先找离中点最近的下划线；
    没有下划线(纯驼峰名如 AuthenticationMiddleware)退化到离中点最近的、非首字符的
    大写字母前拆分。两种拆法切出的两段都仍是原符号的连续子串，不加省略号——
    lint_chapter_map 的子串核对对每段仍能命中，不会被判成杜撰符号。都找不到就
    原样返回单行(允许轻微溢出，好过瞎拆断词)。"""
    if cjk_text_width(text, size) <= max_w:
        return [text]
    mid = len(text) // 2
    underscore_positions = [i for i, c in enumerate(text) if c == '_' and i != 0]
    if underscore_positions:
        split_at = min(underscore_positions, key=lambda p: abs(p - mid))
        return [text[:split_at], text[split_at:]]
    upper_positions = [i for i, c in enumerate(text) if i != 0 and c.isupper()]
    if upper_positions:
        split_at = min(upper_positions, key=lambda p: abs(p - mid))
        return [text[:split_at], text[split_at:]]
    return [text]


# ---------------- DATA(本章数据) ----------------
LANES = ["服务器装配(地基,§36.2 §36.3 §36.6)", "请求处理主线(§36.4 §36.5 §36.8)", "运行期收尾(§36.7)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, [§编号,...])
NODES = [
    ("entry",       0, 0, 0, "setup_server",
     "绑 socket,避开与 ray 的端口竞争", ["§36.2"]),
    ("build_engine", 0, 1, 0, "build_async_engine_client",
     "起 AsyncLLM(接第4章三段式引擎)", ["§36.2"]),
    ("build_app",   0, 2, 0, "build_app / init_app_state",
     "装配 FastAPI(挂 lifespan)+ 注册 OpenAIServingChat 等 handler", ["§36.3", "§36.6"]),
    ("auth_mw",     1, 3, 0, "AuthenticationMiddleware",
     "纯 ASGI 鉴权门,只管 /v1、跳过 OPTIONS", ["§36.8"]),
    ("create_chat", 1, 4, 0, "create_chat_completion",
     "路由 handler,with_cancellation 竞速取消", ["§36.4"]),
    ("render_req",  1, 5, 0, "render_chat_request",
     "校验模型 + 渲染 messages 成 token", ["§36.4"]),
    ("gen_call",    1, 6, 0, "engine_client.generate",
     "接第4章 AsyncLLM,返回异步生成器", ["§36.4"]),
    ("stream_gen",  1, 7, 0, "chat_completion_stream_generator",
     "DELTA:逐帧推 SSE,首块 role/末块 finish/[DONE]", ["§36.5"]),
    ("full_gen",    1, 7, 1, "chat_completion_full_generator",
     "FINAL_ONLY:攒到末个,一次性聚合 JSON", ["§36.5"]),
    ("serve_http",  2, 3, 0, "serve_http",
     "建 uvicorn.Server,起 watchdog/server/shutdown 三 task", ["§36.7"]),
    ("watchdog",    2, 4, 0, "watchdog_loop",
     "每 5s 探测引擎是否暗死", ["§36.7"]),
    ("exit",        2, 5, 0, "handle_shutdown",
     "先 run_in_executor 关引擎,再置 should_exit", ["§36.7"]),
]
EDGES = [  # (src_id, dst_id) —— 全部是真实调用/spawn 边,统一主线蓝
    ("entry", "build_engine"),
    ("build_engine", "build_app"),
    ("build_app", "auth_mw"),
    ("build_app", "serve_http"),
    ("auth_mw", "create_chat"),
    ("create_chat", "render_req"),
    ("render_req", "gen_call"),
    ("gen_call", "stream_gen"),
    ("gen_call", "full_gen"),
    ("serve_http", "watchdog"),
    ("serve_http", "exit"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("流式请求主线(推荐)", [(0, "§36.2"), (2, "§36.3"), (3, "§36.8"), (4, "§36.4"), (6, "§36.4"), (7, "§36.5")], True),
    ("非流式请求(末端聚合)", [(0, "§36.2"), (2, "§36.3"), (3, "§36.8"), (4, "§36.4"), (6, "§36.4"), (7, "§36.5")], False),
    ("服务器装配→关停(地基)", [(0, "§36.2"), (2, "§36.6"), (3, "§36.7"), (5, "§36.7")], False),
]
LEGEND = [("#22c55e", "入口:进程启动"), ("#3b82f6", "章内主线调用/spawn 边"), ("#f97316", "出口:进程优雅退出")]
TITLE = "第 36 章 · OpenAI 兼容服务器:请求剖面 + 服务器生命周期(源码走线 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 145, 76
TITLE_SIZE, TITLE_LINE_H, SUB_SIZE = 11, 12.5, 9.5
COL_GAP, ROW_GAP = 18, 20
EDGE_MARGIN, STUB_W, STUB_H = 12, 58, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 28  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 13
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
        f'<text x="{cx:.1f}" y="{cy + 3.6:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="10.5" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
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
         f'font-size="14.5" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例;本章 3 色)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11.5) + 32

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

# 入口/出口接口桩(全章只有一进一出:入口挂在 §36.2 进程启动第一步,
# 出口挂在 §36.7 优雅关停的收尾函数)
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

# 调用边(主线蓝)。多条边汇入同一节点时,终点 y 各偏移(间距 16px),
# 否则重合的终点在视觉上看不出"汇合"、像一条线断头。
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
    same_col = abs(x2 - x1) < 1
    if same_col:
        # 跨泳道竖直分支(如 build_app -> serve_http):直角折线,避免与其它
        # 同列节点的方框相交(直线会斜穿中间列的节点)。
        mid_y = (y1 + NODE_H + y2 + NODE_H / 2 + y_offset) / 2
        p1x = x1 + NODE_W * 0.28
        pts = [(p1x, y1 + NODE_H), (p1x, mid_y), (x2 + NODE_W * 0.28, mid_y), (x2 + NODE_W * 0.28, y2 + NODE_H / 2 + y_offset)]
        path_d = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        L.append(f'<path d="{path_d}" fill="none" stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    else:
        p2 = (x2, y2 + NODE_H / 2 + y_offset)
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名[必要时拆两行] + 一行短语 + 右上角 § 徽标[可多个并排])
for nid, lane, col, row, symbol, phrase, secs in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    title_lines = split_symbol(symbol, NODE_W - 22, TITLE_SIZE)
    if len(title_lines) == 1:
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.34:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{TITLE_SIZE}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(title_lines[0])}</text>')
    else:
        base_y = y + NODE_H * 0.27
        for li, line in enumerate(title_lines):
            L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{base_y + li * TITLE_LINE_H:.1f}" '
                      f'text-anchor="middle" font-family="sans-serif" font-size="{TITLE_SIZE}" '
                      f'font-weight="bold" fill="{C_NODE_TITLE}">{esc(line)}</text>')
    # 短语可能较长,按 cjk_text_width 拆到两行(节点较窄,单行常装不下)
    phrase_words = phrase
    max_phrase_w = NODE_W - 14
    if cjk_text_width(phrase_words, SUB_SIZE) <= max_phrase_w:
        phrase_lines = [phrase_words]
    else:
        # 从中点向两侧找最近的空格/标点断点,找不到就整句放第一行(允许溢出)
        mid = len(phrase_words) // 2
        break_positions = [i for i, c in enumerate(phrase_words) if c in " ,:()+"]
        if break_positions:
            bp = min(break_positions, key=lambda p: abs(p - mid))
            phrase_lines = [phrase_words[:bp].rstrip(" ,:()+"), phrase_words[bp:].lstrip(" ,:()+")]
        else:
            phrase_lines = [phrase_words]
    phrase_base_y = y + NODE_H * (0.60 if len(title_lines) == 1 else 0.70)
    for pi, pline in enumerate(phrase_lines):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{phrase_base_y + pi * 11:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{SUB_SIZE}" fill="{C_NODE_SUB}">{esc(pline)}</text>')
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
