#!/usr/bin/env python3
"""第 25 章「本章地图」——KV manager + 三个 Scheduler 子类的源码剖面图。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py 模板改写：
不可变机制(esc/cjk_text_width/badge/配色/图例规则)原样保留，只改 DATA 与
"两处独立入口"这一处结构性调整（模板只支持单入口，本章有两个互不隶属的
启动期注入点，见下）。

本章不是单一 forward() 分流链，而是两条互相独立的特化轴线（呼应正文 §25.1
"先看特化是从哪里注入的"——两处注入点分属两条轴）：
  轴 A（上泳道）：调度器选择——check_and_update_config() 按三个互斥开关，
                  在 SchedulerDynamicBatch / RecomputeScheduler /
                  ProfilingChunkScheduler 里三选一（默认都不选，用原生 Scheduler）。
  轴 B（下泳道）：KV manager 选择——KVCacheCoordinator.__init__ 被 patch 后
                  调用昇腾版 get_manager_for_kv_cache_spec，只对压缩 MLA
                  spec 改选 CompressAttentionManager。
两轴在源码里彼此独立生效（一个管"喂哪些请求"，一个管"给多少 block"），
故本图用两条各自独立的 入口→主线→出口 链、共享同一个出口桩收束
（示意"最终都回到引擎的调度步"），而不是编造一条不存在的因果连线把两轴接起来。

节点预算：9 个代码节点（entry_switch/dyn_batch/recompute/profiling/exit +
entry_patch/get_manager/compress_manager/offload_note）≤ 12。
offload_note(§25.7) 是悬空卫星节点——不挂调用边，代表"没细讲的旁支，
见第 13 章"，用法参照 ch20 chapter-map 里 c8 卫星节点的先例。
本章标题为编号标题(## 25.1 ... ## 25.7)，站牌用 §25.N；exit 节点是三个
调度器子类共同的返回点，不对应单一小节，不挂 § 徽标。

多入口的实现：模板原版只在 NODE_XY["entry"]/["exit"] 各挂一个接口桩；
本章有两个独立入口，改成对 ENTRY_IDS 列表逐个画桩（出口仍是单个共享桩，
逻辑不变）。

symbol 支持内嵌 "\\n" 做机械换行（不改变符号拼写，只是排版切分，同 ch20
先例）；换行点选在下划线/驼峰/点号等自然边界。

用法：python3 gen_chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算——全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = [
    "调度器特化 · 三选一，默认原生 Scheduler（§25.1, §25.4–25.6）",
    "KV manager 特化 · 复用查表只重映射一个 spec（§25.2–25.3, §25.7）",
]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名(可含 "\n" 机械换行), 一行短语(可含 "\n"), §编号或"")
NODES = [
    ("entry_switch", 0, 0, 1, "check_and_\nupdate_config",
     "三个开关选调度器类，\n默认一个都不拨", "§25.1"),
    ("dyn_batch", 0, 1, 0, "SchedulerDynamicBatch",
     "refine_budget 查表调预算，\n再 decode 优先重排", "§25.4"),
    ("recompute", 0, 1, 1, "RecomputeScheduler",
     "block 不够时 kv_consumer\n丢请求，回吐重算", "§25.5"),
    ("profiling", 0, 1, 2, "ProfilingChunk\nScheduler",
     "二次模型反解 chunk size，\n收窄本步 token 数", "§25.6"),
    ("exit", 0, 2, 1, "SchedulerOutput",
     "三个子类殊途同归，\n拼好交还引擎调度步", ""),
    ("entry_patch", 1, 0, 0, "KVCacheCoordinator\n.__init__",
     "被 patch 成调用\n昇腾版 manager 工厂", "§25.2"),
    ("get_manager", 1, 1, 0, "get_manager_for_\nkv_cache_spec",
     "查原生 spec_manager_map，\n只对压缩 MLA 改选", "§25.2"),
    ("compress_manager", 1, 2, 0, "CompressAttention\nManager",
     "入口 //= compress_ratio，\n原封不动调 super()", "§25.3"),
    ("offload_note", 1, 3, 0, "KV offload\nmanager",
     "显存不够的另一条支线，\n自成第 13 章（本章不展开）", "§25.7"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝;offload_note 是卫星节点,不挂边
    ("entry_switch", "dyn_batch"), ("entry_switch", "recompute"), ("entry_switch", "profiling"),
    ("dyn_batch", "exit"), ("recompute", "exit"), ("profiling", "exit"),
    ("entry_patch", "get_manager"), ("get_manager", "compress_manager"),
    # 注意:compress_manager 不连去 exit——KV manager 轴是启动期构造好的独立设施,
    # 由运行时的 scheduler 在 allocate_slots 里调用,不是"schedule()返回SchedulerOutput"
    # 这条因果链的一环,连过去是正文没有明确支持的因果关系,故不画。
]
ENTRY_IDS = ["entry_switch", "entry_patch"]  # 两处独立注入点,各挂一个入口桩
EXIT_IDS = ["exit"]  # 共享一个出口桩
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("KV manager 特化链(确定性)", [(0, "§25.2"), (2, "§25.3")], True),
    ("调度器 · 动态预算(SLO)", [(0, "§25.1"), (1, "§25.4")], False),
    ("调度器 · PD 重算", [(0, "§25.1"), (1, "§25.5")], False),
    ("调度器 · profiling chunk", [(0, "§25.1"), (1, "§25.6")], False),
]
LEGEND = [("#22c55e", "入口:从引擎启动/构造调用进入"), ("#3b82f6", "章内主线调用边"), ("#f97316", "出口:交还引擎调度步")]
TITLE = "第 25 章 · KV manager + 调度器的克制特化(两条独立注入轴 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
C_SATELLITE_BORDER = "#cbd5e1"  # 悬空卫星节点(offload_note)用虚线边框,与主线节点区分

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 190, 92
COL_GAP, ROW_GAP = 30, 22
EDGE_MARGIN, STUB_W, STUB_H = 12, 62, 26
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
SATELLITE_IDS = {"offload_note"}  # 悬空节点:虚线边框+浅灰,不挂调用边

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

# 入口接口桩(本章两处独立注入点,各挂一个;出口只有一个共享桩)
for eid in ENTRY_IDS:
    ex, ey = NODE_XY[eid]
    ey += NODE_H / 2
    L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
              f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
    L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
    L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
              f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
for xid in EXIT_IDS:
    xx, xy = NODE_XY[xid]
    xy += NODE_H / 2
    sx = w - EDGE_MARGIN - STUB_W
    L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
              f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
    L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
    L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
              f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝,画在节点下面这条先画后画都行,这里先画边再画节点盖住端点毛刺)
# 多条边汇入同一节点时,终点 y 各偏移(间距 16px),否则重合的终点看不出"汇合"。
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

# 节点(圆角框 + 真实符号名(1~2 行) + 一行短语(1~2 行) + 右上角 § 徽标(sec 为空则不画))
SYMBOL_1LINE_Y, SYMBOL_2LINE_Y1, SYMBOL_2LINE_Y2 = 34, 24, 40
PHRASE_1LINE_Y, PHRASE_2LINE_Y1, PHRASE_2LINE_Y2 = 72, 66, 82
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    is_sat = nid in SATELLITE_IDS
    dash = ' stroke-dasharray="5,4"' if is_sat else ''
    stroke_c = C_SATELLITE_BORDER if is_sat else C_NODE_STROKE
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{stroke_c}" stroke-width="1.5"{dash}/>')
    title_c = C_NODE_SUB if is_sat else C_NODE_TITLE
    sym_lines = symbol.split("\n")
    sym_ys = [y + SYMBOL_1LINE_Y] if len(sym_lines) == 1 else [y + SYMBOL_2LINE_Y1, y + SYMBOL_2LINE_Y2]
    for line, ly in zip(sym_lines, sym_ys):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ly:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                  f'fill="{title_c}">{esc(line)}</text>')
    phrase_lines = phrase.split("\n")
    phrase_ys = [y + PHRASE_1LINE_Y] if len(phrase_lines) == 1 else [y + PHRASE_2LINE_Y1, y + PHRASE_2LINE_Y2]
    for line, ly in zip(phrase_lines, phrase_ys):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ly:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(line)}</text>')
    if sec:
        L += badge(x + NODE_W - BADGE_W / 2 + 8, y, sec)

# 底部阅读路线:复用列坐标 COLX,§ 徽标与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=确定性链路 / 虚线灰=按需三选一)")}</text>')
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
print(f"wrote {out}: {w:.0f}x{h:.0f}")
