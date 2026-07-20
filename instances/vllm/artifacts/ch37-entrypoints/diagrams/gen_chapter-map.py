#!/usr/bin/env python3
"""第 37 章「本章地图」——离线 LLM API 剖面:构造分叉(client 三选一) + 同步驱动脊。

改写自 .claude/skills/svg-diagram/references/example-chapter-map.py(及
ch39-engine-core 的 gen_chapter-map.py 里 split_symbol()/多 § 徽标的写法)。

本章两条真实主线按时间先后发生(先构造、后驱动),不是并列的两个话题,所以
LANES 按「阶段」分两条,而不是按代码文件分层——上泳道「构造期」讲
LLM.__init__→from_engine_args→make_client 怎么把 EngineCore 客户端在
SyncMPClient/InprocClient 之间三选一(§37.2/§37.3,反直觉澄清:默认走
SyncMPClient,不是"看起来更像离线"的 InprocClient);下泳道「请求驱动期」讲
四入口→批量提交→while step() 同步循环(§37.4/§37.5)。两条泳道各自独立复用
列坐标 COLX(泳道内行号只在自己泳道内计数),不强行画一条跨泳道调用边——
两阶段是"先后发生在同一个 LLM 实例上",不是一次函数调用,画一条箭头反而
会被误读成"构造期直接调用了 generate()";上下两条泳道 + 泳道名本身的
"构造期→请求驱动期"顺序,已经把这层时间先后关系交代清楚。

■ 不可变(全书 72 章统一视觉语言,抄自模板,未改动):
  1. §徽标胶囊 badge();2. 入口=绿#22c55e/出口=橙#f97316 接口桩;
  3. 章内主线调用边=蓝#3b82f6;4. 底部路线条(高亮=实线蓝/次要=虚线灰);
  5. >2 种语义色画图例;6. cjk_text_width() 做宽度估算。

■ 本章新增(仅本章需要,未改动上面的不可变部分):
  - split_symbol()(抄自 ch39):真实符号名装不下节点宽度时在下划线处拆两行。
  - 泳道按「阶段」而非「代码层」划分(构造期/请求驱动期),因为本章两条主线
    是时间先后关系而非并列关系;两泳道内部各自独立编列(col/row 只在本泳道
    内有意义,靠 COLX 共享同一套横坐标)。
  - 「构造期」泳道内部有一次真实分叉:make_client() 按 (multiprocess_mode,
    asyncio_mode) 选具体客户端类,本章只关心离线默认 (True,False)→
    SyncMPClient 和 env=0 回退 (False,False)→InprocClient 两支(第三支
    (True,True)→AsyncMPClient 是第 4 章的事,不在本章节点预算内,只在
    mkclient 节点的短语里带一句)。两支都画成真实调用边(蓝实线),"哪支是
    默认"这件事交给底部阅读路线的高亮/虚线去表达,不在上方主图里对节点
    做灰显——因为两支在源码里是同一个 if/elif 链的平级分支,视觉上不应有
    主次之分,主次是"运行时默认命中哪支"的语义,那是阅读路线该管的事。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录):
    claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
    arrows_attached=True     cjk_rendered=True         reading_order_clear=True

用法: python3 gen_chapter-map.py → 同目录 chapter-map.svg
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
LANES = ["构造期 · LLM → LLMEngine → EngineCoreClient", "请求驱动期 · 提交 → 同步循环 → 排序返回"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, [§编号,...])
NODES = [
    ("init",       0, 0, 0, "LLM.__init__",
     "构造参数收拢为 EngineArgs", ["§37.2"]),
    ("fea",        0, 1, 0, "from_engine_args",
     "env 变量强制走 mp=True (默认)", ["§37.2"]),
    ("mkclient",   0, 2, 0, "make_client",
     "按(mp,asyncio)三分支选实现", ["§37.3"]),
    ("syncmp",     0, 3, 0, "SyncMPClient",
     "后台进程+ZMQ+阻塞队列(默认命中)", ["§37.3"]),
    ("inproc",     0, 3, 1, "InprocClient",
     "env=0 回退:真进程内,无 ZMQ", ["§37.3"]),
    ("entry4",     1, 0, 0, "generate()",
     "chat/embed/encode 同构入口", ["§37.4"]),
    ("submit",     1, 1, 0, "_render_and_add_requests",
     "FINAL_ONLY+自增id,失败即回滚", ["§37.4"]),
    ("addreq",     1, 2, 0, "LLMEngine.add_request",
     "双注册:输出侧+送后台EngineCore", ["§37.4"]),
    ("runengine",  1, 3, 0, "_run_engine",
     "while 未完成请求: step()", ["§37.5"]),
    ("enginestep", 1, 4, 0, "LLMEngine.step",
     "get_output阻塞;装配;停止串", ["§37.5"]),
]
# (src_id, dst_id) —— 调用边,统一主线蓝;两条泳道各自独立成链,互不跨接
# (构造期→请求驱动期是时间先后而非函数调用,不画跨泳道边,见文件头说明)。
EDGES = [
    ("init", "fea"), ("fea", "mkclient"),
    ("mkclient", "syncmp"), ("mkclient", "inproc"),
    ("entry4", "submit"), ("submit", "addreq"),
    ("addreq", "runengine"), ("runengine", "enginestep"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("反直觉:离线默认 SyncMPClient",
     [(0, "§37.2"), (1, "§37.2"), (2, "§37.3"), (3, "§37.3")], False),
    ("同步驱动脊:四入口→step()",
     [(0, "§37.4"), (1, "§37.4"), (2, "§37.4"), (3, "§37.5"), (4, "§37.5")], True),
]
LEGEND = [("#22c55e", "入口:调用方构造 LLM 并发起任务"), ("#3b82f6", "章内主线调用边"),
          ("#f97316", "出口:返回上层(调用方)")]
TITLE = "第 37 章 · 离线 LLM API 剖面:构造分叉 + 同步驱动脊(源码走线 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 200, 76
TITLE_SIZE, TITLE_LINE_H, SUB_SIZE = 12.5, 13, 10.5
COL_GAP, ROW_GAP = 30, 26
EDGE_MARGIN, STUB_W, STUB_H = 16, 70, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 30  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 16
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 32, 24, 18
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 48
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
# 图例(>2 种语义色必须画图例;本章 3 色)
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

# 入口/出口接口桩:入口挂在构造期第一站(LLM.__init__),出口挂在请求驱动期
# 最后一站(LLMEngine.step,同步循环里真正等 EngineCore 输出的地方)。
ex, ey = NODE_XY["init"]; ey += NODE_H / 2
xx, xy = NODE_XY["enginestep"]; xy += NODE_H / 2
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

# 调用边(主线蓝)。多条边汇入同一节点时,终点 y 各偏移(如 2 条即 ±8px),
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
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.40:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{TITLE_SIZE}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(title_lines[0])}</text>')
    else:
        base_y = y + NODE_H * 0.32
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
