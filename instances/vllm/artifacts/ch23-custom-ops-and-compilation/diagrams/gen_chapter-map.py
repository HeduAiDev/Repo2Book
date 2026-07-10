#!/usr/bin/env python3
"""第 23 章「本章地图」——CustomOp 两级 dispatch 剖面。

改写自 .claude/skills/svg-diagram/references/example-chapter-map.py。本章的
两级 dispatch 不是同一条调用链上的两段：第 1 级(单算子,构造期)和第 2 级
(整图,首次前向)各有自己独立的触发点，二者之间没有直接函数调用关系——
真正的联系是「第 1 级选 forward_native，正是为了把算子暴露给第 2 级的
Inductor 去融合」，这是一条概念上的因果线，不是调用边。处理办法参照
ch37 的先例：

  - 两条主线各自的入口/出口都是真实的「返回调用方」，且都不止一个——
    第 1 级的 forward_cuda/forward_native、第 2 级的 PiecewiseBackend/
    unified_attention_with_output 分道之后都各自把控制权交还模型的
    forward，没有再汇合成一个函数。与其虚构一个不存在的"总入口/总出口"
    节点，这里让入口桩扇出两条箭头(对应两个真实构造期/首次前向触发点)、
    出口桩扇入四条箭头(对应四条真实的"返回调用方"路径)——桩的颜色/形状
    仍是模板规定的绿入口、橙出口，只是从"一对一"泛化成"一对多"。
  - 第 1 级 forward_native 与第 2 级 PiecewiseBackend 之间画一条与"章内
    主线调用边"(蓝实线,真实调用)视觉上明确不同的边：灰色虚线 + 独立图例
    "章内咬合(非函数调用)"，如实标注这是 §23.9 讲的两级呼应关系，不是
    代码调用。

■ 不可变(全书统一视觉语言，抄自模板，未改动):
  1. §徽标胶囊 badge()；2. 入口=绿#22c55e/出口=橙#f97316 接口桩；
  3. 章内主线调用边=蓝#3b82f6；4. 底部路线条(高亮=实线蓝/次要=虚线灰)；
  5. >2 种语义色画图例；6. cjk_text_width() 做宽度估算。

■ 本章新增(仅本章需要，未改动上面的不可变部分):
  - 入口桩扇出 2 条箭头、出口桩扇入 4 条箭头(见上)。
  - 一条 transition 型的边(l1_native → l2_compile)：灰虚线，代表 §23.9
    讲的"两级咬合"，不是函数调用。
  - split_symbol()：真实符号名太长(如 unified_attention_with_output)
    在节点宽度内装不下时，在离中点最近的下划线处拆两行，不加省略号。

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
LANES = ["第1级 · 单算子 dispatch (构造期)", "第2级 · 整图 dispatch (首次前向)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, [§编号,...])
NODES = [
    ("l1_init",     0, 0, 0, "CustomOp.__init__",
     "构造期实例化,触发 dispatch_forward", ["§23.1"]),
    ("l1_dispatch", 0, 1, 0, "dispatch_forward",
     "按 enabled 结果选定 forward_method", ["§23.1"]),
    ("l1_gate",     0, 2, 0, "enabled() / default_on()",
     "Inductor 时默认 none,否则 all", ["§23.2"]),
    ("l1_cuda",     0, 3, 0, "forward_cuda",
     "预编译融合 kernel,对编译器不透明", ["§23.3"]),
    ("l1_native",   0, 3, 1, "forward_native",
     "纯 torch 算子串,留给 Inductor 融合", ["§23.3"]),

    ("l2_wrap",     1, 0, 0, "support_torch_compile",
     "推断动态维,决定 do_not_compile", ["§23.4"]),
    ("l2_call",     1, 1, 0, "__call__",
     "首次前向:标动态维,触发 VllmBackend", ["§23.5"]),
    ("l2_split",    1, 2, 0, "split_graph",
     "在 splitting_ops(attention)处切图", ["§23.6"]),
    ("l2_compile",  1, 3, 0, "PiecewiseBackend",
     "规整段送 Inductor 编译+包 CUDA graph", ["§23.7"]),
    ("l2_attn",     1, 3, 1, "unified_attention_with_output",
     "不透明算子,eager 执行,天然是切点", ["§23.8"]),
]
# (src_id, dst_id, style) —— style 省略即 "main"(蓝实线,真实调用)；
# "transition" = 灰虚线,章内咬合,非函数调用。
EDGES = [
    ("l1_init", "l1_dispatch"),
    ("l1_dispatch", "l1_gate"),
    ("l1_gate", "l1_cuda"), ("l1_gate", "l1_native"),
    ("l2_wrap", "l2_call"),
    ("l2_call", "l2_split"),
    ("l2_split", "l2_compile"), ("l2_split", "l2_attn"),
    ("l1_native", "l2_compile", "transition"),
]
# 入口/出口不是"一对一"：两个独立触发点扇出，四条真实返回路径扇入。
ENTRY_IDS = ["l1_init", "l2_wrap"]
EXIT_IDS = ["l1_cuda", "l1_native", "l2_compile", "l2_attn"]

# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("第2级:整图怎么切(推荐)",   [(0, "§23.4"), (1, "§23.5"), (2, "§23.6"), (3, "§23.7")], True),
    ("第1级:单算子怎么定",       [(0, "§23.1"), (1, "§23.1"), (2, "§23.2"), (3, "§23.3")], False),
    ("attention 怎么进图(回债)", [(2, "§23.6"), (3, "§23.8")], False),
]
LEGEND = [
    ("#22c55e", "入口:从上层调用进入(两个独立触发点)"),
    ("#3b82f6", "章内主线调用边"),
    ("#f97316", "出口:返回上层(四条真实返回路径)"),
    ("#94a3b8", "章内咬合(§23.9,非函数调用)"),
]
TITLE = "第 23 章 · CustomOp 两级 dispatch 剖面(源码走线 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_TRANSITION = "#94a3b8"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 230, 62
TITLE_SIZE, TITLE_LINE_H, SUB_SIZE = 12, 13, 10
COL_GAP, ROW_GAP = 42, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 78, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 34  # 左右各留:接口桩 + 一段箭头
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
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11) + 26

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

# 入口接口桩:扇出到 ENTRY_IDS(本章两个独立触发点:构造期 / 首次前向)
entry_pts = [(NODE_XY[i][0], NODE_XY[i][1] + NODE_H / 2) for i in ENTRY_IDS]
ey = sum(p[1] for p in entry_pts) / len(entry_pts)
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
for ex, epy in entry_pts:
    L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{epy:.1f}" '
             f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')

# 出口接口桩:扇入自 EXIT_IDS(本章四条真实"返回调用方"路径)
exit_pts = [(NODE_XY[i][0] + NODE_W, NODE_XY[i][1] + NODE_H / 2) for i in EXIT_IDS]
xy = sum(p[1] for p in exit_pts) / len(exit_pts)
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
for xx, xpy in exit_pts:
    L.append(f'<line x1="{xx:.1f}" y1="{xpy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
             f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(main=主线蓝;transition=灰虚线,章内咬合非调用)
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

# transition 边:l1_native(col3,lane0 第2行) 正下方就是 l2_compile(col3,lane1
# 第1行)——同列,一条竖直虚线足够,不需要绕过其它节点。
for src, dst, _style in transition_edges:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W / 2, y1 + NODE_H)
    p2 = (x2 + NODE_W / 2, y2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_TRANSITION}" stroke-width="2" stroke-dasharray="7,5" '
              f'marker-end="url(#mTrans)"/>')
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    L += badge(mx, my, "§23.9")

# 节点(圆角框 + 真实符号名[必要时拆两行] + 一行短语 + 右上角 § 徽标)
for nid, lane, col, row, symbol, phrase, secs in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    title_lines = split_symbol(symbol, NODE_W - 26, TITLE_SIZE)
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
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.82:.1f}" text-anchor="middle" '
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
