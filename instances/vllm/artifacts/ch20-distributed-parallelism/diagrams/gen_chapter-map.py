#!/usr/bin/env python3
"""第 20 章「本章地图」——GroupCoordinator 通信剖面(源码走线 + § 讲解站牌)。

改写自 .claude/skills/svg-diagram/references/example-chapter-map.py。本章内容天然
分两个时间尺度:①启动期(每个 worker 进程只做一次)把群组真正建起来;②运行期
(每次前向调用都会走)GroupCoordinator 对外暴露的三大通信原语。两段各自的"列数"
不同(启动期 2 列,运行期 5 列),若摊成单行 7 列会让画布宽度远超 1500 预算——
改用 ch24 引入的"折成上下两段"手法:上段=启动期,下段=运行期,段间留一条空白
桥接带,只画一条跨段箭头(init_groups → group_coord,这本来就是一次真实的构造
调用,颜色仍是主线蓝,不是 ch36 那种"非调用"的灰虚线)。

■ 不可变(全书统一视觉语言,抄自模板,未改动):
  1. §徽标胶囊 badge();2. 入口=绿#22c55e/出口=橙#f97316 接口桩;
  3. 章内主线调用边=蓝#3b82f6;4. 底部路线条(高亮=实线蓝/次要=虚线灰);
  5. >2 种语义色画图例(本章只用 3 色,不额外加图例项);6. cjk_text_width() 做宽度估算。

■ 本章新增(仅本章需要,未改动上面的不可变部分):
  - `broadcast_tensor_dict`(§20.4)不经过 `device_communicator`——它直接用
    `torch.distributed.broadcast(group=cpu_group/device_group)`,这是正文明确写出的
    (对照 §20.2/§20.5 都委托 `device_communicator`)。所以 `broadcast_dict` 节点
    直接连到出口,不经过 `backend_fallback`(那个节点专属 all_reduce 的
    `CudaCommunicator` 回退链,§20.2)——如果硬把三条通信原语都汇进同一个"后端"
    节点,会把 §20.4 的内容错挂成 §20.2(ch36 踩过这个坑:合并节点只能标一个
    真实站得住脚的 §)。同理 `pp_send_recv` 节点本身就是 "GroupCoordinator.send
    委托 device_communicator.send"(§20.5 原文原话),不需要再单独过一次
    `backend_fallback`。
  - `broadcast_dict`/`pp_send_recv` 到出口的边跳过了 `custom_op`/`backend_fallback`
    所在的两列(它们只在 row0 存在)。若直接画直线,会在两列的 x 范围内斜穿这两个
    节点的框(和 ch36 那条 transition 边同一个几何陷阱)。改走三段折线:先在
    自己的行(row1/row2,与 row0 的两个节点完全不共享 y)水平走到出口列前,再一段
    垂直,最后水平扎进出口节点——全程只经过两列间的空白间隔,不经过任何节点框。
  - 入口挂在 `MultiprocExecutor._init_executor`(§20.7 上半:把群组拉起来),
    出口挂在 `FutureWrapper.result`(§20.7 下半:collective_rpc 的回收)——两处
    确实都在正文 "## 20.7 谁把群组拉起来:MultiprocExecutor" 一节内(含其
    "### 一次 collective_rpc 的广播与回收" 子节),不是牵强凑数。

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


def wrap_line(text, max_w, size, break_chars):
    """通用换行:一行装不下就在 break_chars 里找一个离中点最近的字符处拆两行
    (拆分点本身若是标点/下划线/点号,并入前一行;若是空格,丢弃这个空格)。
    找不到断点就返回原样(允许轻微溢出,好过瞎拆断词——真实符号名靠 wrap_symbol
    专门处理下划线/点号断点,这里通用版主要给中文短语用)。"""
    if cjk_text_width(text, size) <= max_w:
        return [text]
    positions = [i for i, c in enumerate(text) if c in break_chars]
    if not positions:
        return [text]
    # 用「宽度中点」而不是「字符数中点」去找最近断点——中英混排时更准。
    widths = [cjk_text_width(text[:i + 1], size) for i in range(len(text))]
    half = widths[-1] / 2
    split_at = min(positions, key=lambda p: abs(widths[p] - half))
    left = text[:split_at + 1]
    right = text[split_at + 1:]
    if text[split_at] == " ":
        left = text[:split_at]
    return [left, right]


def wrap_symbol(text, max_w, size):
    """真实符号名装不下就在离中点最近的 `_`/`.` 处拆两行——两段仍是原符号的
    连续子串(不加省略号),lint_chapter_map 的子串核对对每段仍能命中。"""
    return wrap_line(text, max_w, size, set("._"))


# ---------------- DATA(本章数据) ----------------
LANES = [
    "编排层 · 启动期把群组建起来(每 worker 进程一次)",
    "抽象层 · GroupCoordinator 对外的三大通信原语(每次前向调用)",
]

# (节点id, 段下标(0=启动期/1=运行期), 段内列, 段内行号, 真实符号名, 一行短语, §编号)
NODES = [
    ("spawn_worker", 0, 0, 0, "MultiprocExecutor._init_executor",
     "建 rpc_broadcast_mq,为每卡 spawn 一个 WorkerProc", "§20.7"),
    ("init_groups", 0, 1, 0, "initialize_model_parallel",
     "5维 rank 张量做变换,切出各维度进程组", "§20.6"),
    ("group_coord", 1, 0, 0, "GroupCoordinator.__init__",
     "cpu_group 走 gloo 管元数据,device_group 走 NCCL 管张量", "§20.1"),
    ("allreduce_dispatch", 1, 1, 0, "GroupCoordinator.all_reduce",
     "world_size 为 1 就短路,否则按 use_custom_op_call 分流", "§20.2"),
    ("broadcast_dict", 1, 1, 1, "broadcast_tensor_dict",
     "metadata 先走 cpu_group,张量再按 is_cpu 选组广播", "§20.4"),
    ("pp_send_recv", 1, 1, 2, "device_communicator.send",
     "阻塞式点对点,把隐藏状态传给下一 PP 段", "§20.5"),
    ("custom_op", 1, 2, 0, "torch.ops.vllm.all_reduce",
     "只传 group_name 字符串,配 fake 实现推断输出形状", "§20.3"),
    ("backend_fallback", 1, 3, 0, "CudaCommunicator",
     "回退链:CustomAllreduce,pynccl,torch.distributed", "§20.2"),
    ("collect_output", 1, 4, 0, "FutureWrapper.result",
     "仅 output_rank 那个 worker 回写,其余丢弃", "§20.7"),
]
# (src_id, dst_id, 边型) —— 边型省略即 "main"(蓝实线,直线);
# "bridge" = 跨段(仍是真实构造调用,同样蓝实线,只是走桥接带);
# "skip"   = 同段内跳过若干列的边,走三段折线避开被跳过列上的节点框。
EDGES = [
    ("spawn_worker", "init_groups"),
    ("init_groups", "group_coord", "bridge"),
    ("group_coord", "allreduce_dispatch"),
    ("group_coord", "broadcast_dict"),
    ("group_coord", "pp_send_recv"),
    ("allreduce_dispatch", "custom_op"),
    ("custom_op", "backend_fallback"),
    ("backend_fallback", "collect_output"),
    ("broadcast_dict", "collect_output", "skip"),
    ("pp_send_recv", "collect_output", "skip"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
# 列号复用 NODES 里「段内列」——构建期路线用启动期段的列号(0,1),
# 其余三条运行期路线用运行期段的列号(0=group_coord..4=collect_output)。
ROUTES = [
    ("构建期(启动一次)",
     [(0, "§20.7"), (1, "§20.6")], False),
    ("运行期·custom-op(推荐)",
     [(0, "§20.1"), (1, "§20.2"), (2, "§20.3"), (3, "§20.2"), (4, "§20.7")], True),
    ("运行期·双群组广播",
     [(0, "§20.1"), (1, "§20.4"), (4, "§20.7")], False),
    ("运行期·PP P2P",
     [(0, "§20.1"), (1, "§20.5"), (4, "§20.7")], False),
]
LEGEND = [
    ("#22c55e", "入口:从上层(engine/调度循环)调用进入"),
    ("#3b82f6", "章内主线调用边(含跨段的真实构造调用)"),
    ("#f97316", "出口:返回上层"),
]
TITLE = "第 20 章 · GroupCoordinator 通信剖面(源码走线 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
TITLE_SIZE, TITLE_LINE_H, SUB_SIZE, SUB_LINE_H = 12.5, 14, 10.5, 12
COL_GAP, ROW_GAP = 28, 18
EDGE_MARGIN, STUB_W, STUB_H = 14, 64, 24
LANE_LABEL_H, BAND_PAD = 22, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 32, 24, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 42
BADGE_W, BADGE_H = 46, 20
INTER_BAND_GAP = 46  # 两段之间的空白桥接带(只放一条跨段箭头,不需要 ch24 那么宽)

_NODE_TEXT_PAD = 18
# 节点宽度:按本章最长的符号名(必要时拆两行取较宽的那半)+ 最长短语(同样必要时
# 拆两行)综合算出一个统一宽度——比直接塞长符号硬撑宽度更省画布。
_max_symbol_half_w = 0
_max_phrase_half_w = 0
for *_, _sym, _phrase, _sec in NODES:
    for cand_w in (140, 160, 180, 200):
        lines = wrap_symbol(_sym, cand_w, TITLE_SIZE)
        if len(lines) <= 2 and max(cjk_text_width(l, TITLE_SIZE) for l in lines) <= cand_w:
            _max_symbol_half_w = max(_max_symbol_half_w, cand_w)
            break
    else:
        _max_symbol_half_w = max(_max_symbol_half_w, cjk_text_width(_sym, TITLE_SIZE))
    for cand_w in (140, 160, 180, 200):
        lines = wrap_line(_phrase, cand_w, SUB_SIZE, set(",，、 "))
        if len(lines) <= 2 and max(cjk_text_width(l, SUB_SIZE) for l in lines) <= cand_w:
            _max_phrase_half_w = max(_max_phrase_half_w, cand_w)
            break
    else:
        _max_phrase_half_w = max(_max_phrase_half_w, cjk_text_width(_phrase, SUB_SIZE))
NODE_W = max(150.0, _max_symbol_half_w + _NODE_TEXT_PAD, _max_phrase_half_w + _NODE_TEXT_PAD)
NODE_H = 62

PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 24  # 左右各留:接口桩 + 一段箭头

n_cols = max(n[2] for n in NODES) + 1
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_band = [0] * len(LANES)
for _id, band, col, row, *_ in NODES:
    rows_per_band[band] = max(rows_per_band[band], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_band]

band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for i, bh in enumerate(band_h):
    if i > 0:
        _cum += INTER_BAND_GAP
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
    """§ 徽标胶囊,居中挂在 (cx,cy)。"""
    bx, by = cx - BADGE_W / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BADGE_W}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
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
# 图例(3 种语义色仍画图例,和模板一致)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11) + 26

# 泳道(段)背景 + 标签(段间桥接带留白,不上色,视觉上与两段区分开)
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w:.1f}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w:.1f}" y2="{band_top[i]:.1f}" '
              f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
    L.append(f'<line x1="0" y1="{band_top[i] + band_h[i]:.1f}" x2="{w:.1f}" y2="{band_top[i] + band_h[i]:.1f}" '
              f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩:入口挂 spawn_worker(段 0),出口挂 collect_output(段 1)——
# 两者本就在不同的段,stub 各自贴在自己节点的 y 高度上,和 ch36 的做法一致。
ex, ey = NODE_XY["spawn_worker"]; ey += NODE_H / 2
xx, xy = NODE_XY["collect_output"]; xy += NODE_H / 2
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

# 调用边(main/bridge/skip 都是蓝实线):main 走左→右中线(同段);bridge 跨段,
# 走上一段节点下边中点→下一段节点上边中点(和 ch24 的跨段桥接边同一种接法,
# 避免"从右边出、绕一个大斜线到左边"的倒退观感);skip 走三段折线避开被跳过列。
_main_edges = [e for e in EDGES if (e[2] if len(e) > 2 else "main") == "main"]
_bridge_edges = [e for e in EDGES if len(e) > 2 and e[2] == "bridge"]
_skip_edges = [e for e in EDGES if len(e) > 2 and e[2] == "skip"]
_dst_total = {}
for e in _main_edges + _skip_edges:
    _dst_total[e[1]] = _dst_total.get(e[1], 0) + 1
_dst_seen = {}


def _next_offset(dst):
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    return (i - (n - 1) / 2) * 16 if n > 1 else 0


for e in _main_edges:
    src, dst = e[0], e[1]
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    p2 = (x2, y2 + NODE_H / 2 + _next_offset(dst))
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

for src, dst, _style in _bridge_edges:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W / 2, y1 + NODE_H)
    p2 = (x2 + NODE_W / 2, y2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# skip 边:src 在自己那一行水平走到「被跳过列」和目的列之间的空档,再转垂直对齐
# 目的节点入口高度,最后一小段水平扎进目的节点左边——三段都不经过任何节点框
# (水平段固定在 src 自己的行高,不会撞到只在 row0 出现的 custom_op/backend_fallback;
# 垂直段走在两列之间的纯留白 COL_GAP 里)。
for src, dst, _style in _skip_edges:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    p2 = (x2, y2 + NODE_H / 2 + _next_offset(dst))
    turn_x = x2 - COL_GAP / 2
    pts = [p1, (turn_x, p1[1]), (turn_x, p2[1]), p2]
    path_d = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    L.append(f'<path d="{path_d}" fill="none" stroke="{C_MAIN}" stroke-width="2" '
              f'marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名[必要时拆两行] + 一行短语[必要时拆两行] + 右上角 § 徽标)
for nid, band, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W:.1f}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    title_lines = wrap_symbol(symbol, NODE_W - 22, TITLE_SIZE)
    phrase_lines = wrap_line(phrase, NODE_W - 16, SUB_SIZE, set(",，、 "))
    n_title, n_phrase = len(title_lines), len(phrase_lines)
    block_h = n_title * TITLE_LINE_H + n_phrase * SUB_LINE_H
    top_y = y + (NODE_H - block_h) / 2 + TITLE_LINE_H * 0.72
    for li, line in enumerate(title_lines):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{top_y + li * TITLE_LINE_H:.1f}" '
                  f'text-anchor="middle" font-family="sans-serif" font-size="{TITLE_SIZE}" '
                  f'font-weight="bold" fill="{C_NODE_TITLE}">{esc(line)}</text>')
    phrase_y0 = top_y + n_title * TITLE_LINE_H + SUB_LINE_H * 0.55
    for li, line in enumerate(phrase_lines):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{phrase_y0 + li * SUB_LINE_H:.1f}" '
                  f'text-anchor="middle" font-family="sans-serif" font-size="{SUB_SIZE}" '
                  f'fill="{C_NODE_SUB}">{esc(line)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 8, y, sec)

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
print(f"wrote {out} ({w:.0f}x{h:.0f}, NODE_W={NODE_W:.0f}, n_cols={n_cols})")
