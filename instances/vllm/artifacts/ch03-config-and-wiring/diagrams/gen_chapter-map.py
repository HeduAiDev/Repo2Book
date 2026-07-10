#!/usr/bin/env python3
"""第 3 章「本章地图」——EngineArgs → VllmConfig 两级映射剖面。

改写自 .claude/skills/svg-diagram/references/example-chapter-map.py（沿用
instances/vllm/artifacts/ch37-engine-core 已验证过的 split_symbol() 长符号
拆行技法，处理本章比模板示例更多、更长的真实符号名）。

本章骨架就是 3.13 节小结自己给出的结构：
  第一级映射(3.2-3.3: EngineArgs 扁平参数 → create_engine_config 打包)
  → 推导中枢(3.4: VllmConfig.__post_init__，内部再分两支 3.5 async_scheduling
    三态决策 / 3.9 optimization_level 声明式预设)
  → 第二级映射(3.6-3.8: 三个工厂查表——Executor.get_class / get_scheduler_cls /
    make_client)
  → 汇合(3.11: EngineCore.__init__)。

一个不那么直觉但正文明确写出的事实（§3.11 原句：「两件事在这里发生，正好
对应三个工厂里的两个」）：EngineCore.__init__ 只汇合了 Executor.get_class 和
get_scheduler_cls 两个工厂的产物；第三个工厂 make_client 在更外层、构造
EngineCore 之前就已调用完毕，决定的是"EngineCore 在哪个进程里被构造"，不是
"喂给 EngineCore.__init__ 的一个参数"。同理 compute_hash 是 VllmConfig 自己
的方法，供 torch.compile 缓存查询，跟本章"选类→实例化"这条主线正交（正文
原话："它跟前面的组装/工厂是正交的一条线"）。所以本图没有把 make_client /
compute_hash 画进"汇入出口"的主线，而是放进第 4 条泳道，标注为"旁线"——
仍然从 __post_init__ 分支出来（都是消费同一个已构建好的 VllmConfig），但
不再往右连到 exit。这比把它们硬塞进主链、或者干脆不画（读者就看不到
§3.8/§3.10 在图上落在哪一步）更诚实。

■ 不可变(全书统一视觉语言，抄自模板，未改动):
  1. §徽标胶囊 badge()；2. 入口=绿#22c55e/出口=橙#f97316 接口桩；
  3. 章内主线调用边=蓝#3b82f6；4. 底部路线条(高亮=实线蓝/次要=虚线灰)；
  5. >2 种语义色画图例；6. cjk_text_width() 做宽度估算。

■ 本章特点:
  - 六列布局，第 4 列(col=3→col=4 扇出)一次性容纳六个"消费同一个已构建
    VllmConfig"的兄弟节点(跨 3 条泳道)：两条推导(async_scheduling /
    optimization_level)+两个工厂(Executor.get_class / get_scheduler_cls)+
    两个旁线(compute_hash / make_client)。只有两个工厂继续向右汇入 exit，
    其余四个是终点（对应它们各自不汇入 EngineCore.__init__ 这一事实）。
  - 长符号名(如 `create_engine_config()`)按 cjk_text_width() 在给定字号下
    装不进节点宽度时用 split_symbol() 在最近的下划线处拆两行——纯几何换行，
    不改变符号本身、不加省略号(省略号会被误判成新符号)。

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
LANES = [
    "入口 · 第一级映射(参数→配置)",
    "推导中枢(VllmConfig.__post_init__ 的两条推导支线)",
    "第二级映射 · 工厂查表→汇合出口",
    "旁线 · 不汇入 EngineCore.__init__ 的两件事",
]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, [§编号,...])
NODES = [
    ("entry",      0, 0, 0, "from_engine_args",
     "统一入口,持有 EngineArgs", ["§3.1"]),
    ("eargs",      0, 1, 0, "EngineArgs",
     "扁平参数袋子,默认值借自子 Config", ["§3.2"]),
    ("create_cfg", 0, 2, 0, "create_engine_config()",
     "打包子配置,聚合成 VllmConfig", ["§3.3"]),
    ("post_init",  1, 3, 0, "__post_init__",
     "跨子配置校验与推导中枢", ["§3.4"]),
    ("async_sch",  1, 4, 0, "async_scheduling",
     "三态:pooling/投机/执行器→退化", ["§3.5"]),
    ("opt_lvl",    1, 4, 1, "optimization_level",
     "O0–O3 声明式预设,仅填 None", ["§3.9"]),
    ("exec_fac",   2, 4, 0, "Executor.get_class",
     "按 backend 查表选执行器类", ["§3.6"]),
    ("sched_fac",  2, 4, 1, "get_scheduler_cls()",
     "按 async_scheduling 选调度器类", ["§3.7"]),
    ("hash_fn",    3, 4, 0, "compute_hash()",
     "10 位指纹,决定编译缓存命中", ["§3.10"]),
    ("client_fac", 3, 4, 1, "make_client",
     "按进程/异步模式选 IPC 客户端", ["§3.8"]),
    ("exit",       0, 5, 0, "EngineCore.__init__",
     "执行器/调度器在此实例化落地", ["§3.11"]),
]
# (src_id, dst_id) —— 调用边,统一主线蓝
EDGES = [
    ("entry", "eargs"),
    ("eargs", "create_cfg"),
    ("create_cfg", "post_init"),
    ("post_init", "async_sch"),
    ("post_init", "opt_lvl"),
    ("post_init", "exec_fac"),
    ("post_init", "sched_fac"),
    ("post_init", "hash_fn"),
    ("post_init", "client_fac"),
    ("exec_fac", "exit"),
    ("sched_fac", "exit"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("主线:两级映射全程",
     [(0, "§3.1"), (1, "§3.2"), (2, "§3.3"), (3, "§3.4"), (4, "§3.7"), (5, "§3.11")], True),
    ("执行器工厂路径",
     [(0, "§3.1"), (2, "§3.3"), (3, "§3.4"), (4, "§3.6"), (5, "§3.11")], False),
    ("旁线:三态决策细节",
     [(0, "§3.1"), (3, "§3.4"), (4, "§3.5")], False),
]
LEGEND = [("#22c55e", "入口:从上层调用进入"), ("#3b82f6", "章内主线调用边"), ("#f97316", "出口:返回上层")]
TITLE = "第 3 章 · EngineArgs→VllmConfig 两级映射剖面(源码走线 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 175, 70
TITLE_SIZE, TITLE_LINE_H, SUB_SIZE = 12, 13, 10
COL_GAP, ROW_GAP = 26, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
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
    """§ 徽标胶囊,居中挂在 (cx,cy) —— 节点用它贴右上角,路线legend用它居中挂线上。"""
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

# 调用边(主线蓝)。多条边汇入同一节点时(仅 exit),终点 y 各偏移(间距 16px)，
# 否则重合的终点看不出"汇合"。
# exec_fac/sched_fac → exit 这两条边跨了两条泳道(lane2 直接跳回 lane0)，
# 途中会穿过同一列(col4)里 lane1 的 async_sch/optimization_level 节点——
# 直线连接会斜穿这两个框(及其 § 徽标)。改走"列间空档"折线:从源节点右边
# 先探进 col4/col5 之间的空档、沿空档垂直上行(空档内无任何节点/徽标)、
# 到达出口所在行再水平接入 exit 左边——全程不经过任何节点框。
GAP_ROUTE_DST = {"exit"}
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
    if dst in GAP_ROUTE_DST and NODE_BY_ID[src][1] != NODE_BY_ID[dst][1]:
        # 源节点与目标节点不在同一条泳道——直线连接会纵向穿过同列(col4)里
        # 中间那条泳道的节点/徽标(如 lane1 的 async_sch/optimization_level)。
        # 改走列间空档的折线路由,空档内取两条互不重叠的 x(避开两侧节点/徽标)。
        gap_x = x1 + NODE_W + 9 + i * 8
        pts = [p1, (gap_x, p1[1]), (gap_x, p2[1]), p2]
        path_d = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        L.append(f'<path d="{path_d}" fill="none" stroke="{C_MAIN}" stroke-width="2" '
                  f'marker-end="url(#mMain)"/>')
    else:
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名[必要时拆两行] + 一行短语 + 右上角 § 徽标)
for nid, lane, col, row, symbol, phrase, secs in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    title_lines = split_symbol(symbol, NODE_W - 22, TITLE_SIZE)
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
