#!/usr/bin/env python3
"""ch12 本章地图——BlockPtrAnalysis 物化剖面(源码走线 + § 讲解站牌)。

改自 .claude/skills/svg-diagram/references/example-chapter-map.py 模板,不可变部分
(配色/徽标胶囊/入口绿-出口橙-主线蓝/cjk_text_width)保持原样;可变部分(LANES/
NODES/EDGES/ROUTES)按 ch12 定稿改写。相对模板做了两处泛化(数据驱动,非新增语义):
  1. ENTRIES/EXITS 从单一 "entry"/"exit" 硬编码 id 泛化成列表——本章确有两个入口
     (tt.addptr 链走 BlockData 词汇表;block_ptr 链走 rewriteMakeTensorPtrOp)、三个
     出口(load/store/atomic 三个 converter),每个入口/出口各自在自己的行上挂一个
     独立接口桩+箭头,而不是拼一个假的单入单出。
  2. NODE_W/BADGE_W 从写死常量改成按本章最长符号/短语/徽标实际算出来的宽度
     (cjk_text_width 逐字符估算 + 固定内边距)——章里最长符号
     rewriteAddPtrToUnstrucMemAcc 有 29 个半角字符,套用模板原 190px 会溢出,
     故按内容动态算,不是新引入魔数。

用法:python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):逐字符判定——
    全角(ord>0x2E80)按 1.0×size,半角按 0.58×size,求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["状态与代数镜像(ch11 镜像)", "落地核心：结构化 ↔ 非结构化", "load/store/atomic 转换器"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, §编号)
# 注:blockdata/parse_mirror 同列不同行(纵向堆叠)——为把总列数压到 4,
# 换 5 列会因 NODE_W(按最长符号 rewriteAddPtrToUnstrucMemAcc 算)撑破 1500 画布预算。
NODES = [
    ("blockdata",       0, 0, 0, "BlockData / MemAccType",        "状态载体升级 + 访存结构化度三态", "§12.1"),
    ("parse_mirror",    0, 0, 1, "parseAddPtr",                    "递归还原 (offset,sizes,strides)，镜像 ch11", "§12.2"),
    ("rewrite_addptr",  1, 1, 0, "rewriteAddPtr",                  "落地总装：分岔判断 + 零 stride 修复", "§12.5"),
    ("parse_recast",    1, 1, 1, "parseReinterpretCast",           "读回 BlockData（createCastOp 逆映射）", "§12.4"),
    ("make_tensor_ptr", 1, 1, 2, "rewriteMakeTensorPtrOp",         "block_ptr → 两级 recast + 转置维序", "§12.10"),
    ("create_cast",     1, 2, 0, "createCastOp",                   "三元组铸成 reinterpret_cast（本章心脏）", "§12.3"),
    ("gather_fallback", 1, 2, 1, "rewriteAddPtrToUnstrucMemAcc",   "嵌套 scf.for，逐元素 gather 回退", "§12.6"),
    ("load_conv",       2, 3, 0, "LoadConverter",                  "alloc + memref.copy + to_tensor", "§12.7"),
    ("store_conv",      2, 3, 1, "StoreConverter",                 "materialize_in_destination 写回", "§12.8"),
    ("atomic_conv",     2, 3, 2, "AtomicRMWConverter",             "硬件原子算子，非 linalg.generic", "§12.9"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝
    ("blockdata", "parse_mirror"),
    ("parse_mirror", "rewrite_addptr"),
    ("parse_mirror", "parse_recast"),
    ("rewrite_addptr", "create_cast"),
    ("rewrite_addptr", "gather_fallback"),
    ("make_tensor_ptr", "create_cast"),
    ("create_cast", "load_conv"),
    ("create_cast", "store_conv"),
    ("create_cast", "atomic_conv"),
    ("gather_fallback", "load_conv"),
]
# 入口/出口节点(泛化自模板单一 "entry"/"exit"——本章两入三出,见文件头说明)
ENTRIES = ["blockdata", "make_tensor_ptr"]
EXITS = ["load_conv", "store_conv", "atomic_conv"]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
# 列号复用 NODES 里的列;col0 同时挂着 §12.1(BlockData/MemAccType)与 §12.2
# (parseAddPtr)两站,路线上只取该列里承前启后的那一站作代表徽标。
ROUTES = [
    ("结构化访存主线",     [(0, "§12.2"), (1, "§12.5"), (2, "§12.3"), (3, "§12.7")], True),
    ("非结构化 gather 回退", [(0, "§12.2"), (1, "§12.5"), (2, "§12.6"), (3, "§12.7")], False),
    ("block_ptr 转置路径",  [(1, "§12.10"), (2, "§12.3"), (3, "§12.8")], False),
]
LEGEND = [
    ("#22c55e", "入口：上一遍分析的三元组 / block_ptr 进入本章"),
    ("#3b82f6", "章内主线调用边"),
    ("#f97316", "出口：交还 memref/tensor 给外层管线"),
]
TITLE = "第 12 章 · BlockPtrAnalysis 物化剖面（三元组 → reinterpret_cast → load/store/atomic）"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
# NODE_W 按本章最长符号/短语实际宽度算(rewriteAddPtrToUnstrucMemAcc 29 字符会撑破
# 模板默认的 190px),不是写死的数。
_SYMBOL_FONT, _PHRASE_FONT = 13, 10.5
_NODE_TEXT_PAD = 30
_node_text_widths = []
for _n in NODES:
    _node_text_widths.append(cjk_text_width(_n[4], _SYMBOL_FONT))
    _node_text_widths.append(cjk_text_width(_n[5], _PHRASE_FONT))
NODE_W = max(190, max(_node_text_widths) + _NODE_TEXT_PAD)
NODE_H = 58
COL_GAP, ROW_GAP = 42, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
# BADGE_W 按本章最长 § 标签(§12.10,6 字符)实际宽度算,不是写死的数。
_all_sec_labels = [n[6] for n in NODES] + [sec for _, stops, _ in ROUTES for _, sec in stops]
BADGE_W = max(46, max(cjk_text_width(s, 11) for s in _all_sec_labels) + 14)
BADGE_H = 20

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
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BADGE_W:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
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
# 图例(>2 种语义色必须画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
for color, label in LEGEND:
    L.append(f'<rect x="{_lx:.1f}" y="{_ly - 11:.1f}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20:.1f}" y="{_ly:.1f}" font-family="sans-serif" font-size="11.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11.5) + 34

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w:.1f}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w:.1f}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w:.1f}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩(泛化:每个 ENTRIES/EXITS 节点各自在自己的行上挂一个独立接口桩,
# 本章两入(tt.addptr 链 / block_ptr 链)三出(load/store/atomic 三个 converter),
# 不强行拼成单入单出)
for eid in ENTRIES:
    ex, ey = NODE_XY[eid]
    ey += NODE_H / 2
    L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
             f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
    L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
    L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
             f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
for xid in EXITS:
    xx, xy = NODE_XY[xid]
    xy += NODE_H / 2
    sx = w - EDGE_MARGIN - STUB_W
    L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
             f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
    L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
    L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
             f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝)。多条边汇入同一节点时,终点 y 各偏移(间距 16px),
# 否则重合的终点在视觉上看不出"汇合"、像一条线断头。
# 同列纵向堆叠的两个节点(如 blockdata→parse_mirror,列号相同,只是同列不同行)
# 若仍按"右边出、左边进"连线,线会从上节点右边斜穿两个节点内部再切进下节点左边——
# 对同列(x1==x2)的边改走"下边出、上边进"的竖直连线,不穿节点。
_dst_total = {}
for _, dst in EDGES:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    same_col = abs(x1 - x2) < 1e-6
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    y_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
    if same_col:
        p1 = (x1 + NODE_W / 2, y1 + NODE_H)
        p2 = (x2 + NODE_W / 2 + y_offset, y2)
    else:
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角 § 徽标)
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W:.1f}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{_SYMBOL_FONT}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{_PHRASE_FONT}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
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
print(f"wrote {out}  ({w:.0f}x{h:.0f}, NODE_W={NODE_W:.0f}, BADGE_W={BADGE_W:.0f})")
