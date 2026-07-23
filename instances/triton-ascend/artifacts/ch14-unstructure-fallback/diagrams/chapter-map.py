#!/usr/bin/env python3
"""ch14「本章地图」——Unstructure 兜底剖面：pass 主体的判定改写与 OffsetAnalysis
的四态分析如何配合，把「结构化装不下的访存」标量化成 scf.for 逐点循环
（kind=deep，skip_impl=true；本章为纯 C++ MLIR pass，无精简版，图上只呈现真实
源码符号）。

本章是**编号标题章**（`## 14.1`…`## 14.9`），按契约用 §N.M 徽标。全章 9 节里
挑 7 个直接对应一段可指认源码的核心机制节点，两条泳道对应两个真实源码文件：
  Lane0  third_party/ascend/lib/TritonToUnstructure/UnstructureConversionPass.cpp
         —— pass 主体：入口 §14.1 / 判定改写 §14.5 / 标量化 codegen §14.6 /
            scalarLike 快路径 §14.8
  Lane1  third_party/ascend/{include,lib}/TritonToUnstructure/OffsetAnalysis.{h,cpp}
         —— 偏移分析：四态格 §14.2 / 格的 meet §14.3 / transfer function §14.4
（§14.7「代价」不建独立节点——它复用 §14.5 的对齐闸代码与 §14.6 的 codegen 结果做
量化对照，没有新符号；图上用一条选读路线的文字提示指向它，不单占节点/画布预算。
§14.9 小结不建节点。）

两条泳道之间的边（entry→lattice、transfer→gate）画的是**本章讲解顺序的前向流**，
不是字面单跳函数调用——lattice/combine/transfer 三者是 `parse` 递归按 defining-op
分派出去的兄弟分支（叶子用 transfer function 直接赋值，二元 op 用 combineInfo 汇
合），源码里没有 lattice→combine→transfer 这条串行调用链；图上按「格式→汇合→发证
规则」的讲解顺序排布，帮读者建立概念脉络，四条边都落在列间空白，不会穿过其他节点。
gate→codegen 与 gate→splat 是 matchAndRewrite 的两条真分支（前者是主路，后者是
isScalarLike() 命中时提前返回的快路径）——源码顺序是先判 scalarLike 走 splat、
不命中才继续到对齐闸与 codegen，图上仍从 gate 一点两支，不需要「无因果」注记
（这是真分支决策，不是无关结果的默认汇聚）。

跨章标注（exp-2026-07-18-04 硬规则：目标章号 > 本章号用「预告」，< 本章号用「回指」）：
  入口桩（绿）："回指 ch13" —— ch13 < ch14，紧邻上一章《MaskAnalysis》收尾于结构化
                路径的终点，本章接手它的失败分支。
  出口桩（橙）："预告 ch15" —— ch15 > ch14，正文小结明确预告下一章从
                「把多个网格实例折成一条 blockify 循环」讲起，本章标下的
                DiscreteMemAccess 标记正是下游那条优化 pass 的认领入口。

底部阅读路线复刻正文 hook 段原话（"只想知道『什么写法会掉坑』，直接跳 §14.4；想量化
代价，看 §14.7；想跟全程，按序读"）+ 一条 scalarLike vs gather 的选读跳转。

不可变视觉语言（全书统一，来自 example-chapter-map.py 模板 + ch10/ch11 的动态胶囊/
自适应换行/编号圆圈/monospace 路径行改法）：§徽标胶囊（圆角矩形 fill #eef2ff /
stroke #6366f1）、入口绿 #22c55e / 出口橙 #f97316 / 主线蓝 #3b82f6、
cjk_text_width() 逐字符宽度估算。

六项自查（渲染→Read PNG 亲眼看后如实记录）：见同目录 figure-manifest.json 该图 selfcheck。

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def mono_text_width(s, size):
    """monospace 路径行宽度估算：等宽字体每字符约 0.6×size(半角)。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.6) for ch in s)


_BREAK_AFTER = set("，；：、/ ,;")


def wrap_claim(text, max_w, size):
    """一句论点太长时换行——只在标点/斜杠/空格之后断行，不劈开一个标识符或中文词。
    贪心找"prefix 仍不超宽的最靠后一个合法断点"；找不到合法断点才整句照旧单行放行。"""
    breaks = [i for i, ch in enumerate(text) if ch in _BREAK_AFTER]
    best = None
    for i in breaks:
        if cjk_text_width(text[:i + 1], size) <= max_w:
            best = i
    if best is None:
        return [text]
    line1, line2 = text[:best + 1].rstrip(), text[best + 1:].lstrip()
    if cjk_text_width(line2, size) <= max_w:
        return [line1, line2]
    return [line1] + wrap_claim(line2, max_w, size)


# ---------------- DATA(可变：本章数据) ----------------
LANES = [
    "third_party/ascend/lib/TritonToUnstructure · pass 主体：入口/判定改写/codegen · UnstructureConversionPass.cpp",
    "third_party/ascend/{include,lib}/TritonToUnstructure · 偏移分析：四态格与 transfer function · OffsetAnalysis.{h,cpp}",
]

# (节点id, 泳道下标, 列, 泳道内行号, [符号行…], 省略前缀后的路径, 一句论点, §站牌)
NODES = [
    ("entry", 0, 0, 0,
     ["runOnOperation"],
     "…/UnstructureConversionPass.cpp",
     "串起 runParse(灌 offsetMap)→挂 4 个 converter pattern→CSE+Canonicalize 收尾",
     "§14.1"),
    ("lattice", 1, 1, 0,
     ["AxisInfo"],
     "…/OffsetAnalysis.h",
     "四态全序(声明序即偏序值 0~3)：unstructured / structured / scalarlike / scalar",
     "§14.2"),
    ("combine", 1, 2, 0,
     ["combineInfo"],
     "…/OffsetAnalysis.cpp",
     "二元 op 汇合逐维取 std::min：一处 unstructured 就拉低整维，scalarLike 标志取 &&",
     "§14.3"),
    ("transfer", 1, 3, 0,
     ["parseMakeRange", "parseLoad", "parseMulI"],
     "…/OffsetAnalysis.cpp",
     "好源头 arange→structured；坏源头：load 出的值当索引 / arange×arange→unstructured",
     "§14.4"),
    ("gate", 0, 4, 0,
     ["matchAndRewrite"],
     "…/UnstructureConversionPass.cpp",
     "早退：isStructured() 放行给结构化路径；否则查三类触发 + 32 字节对齐闸，判定是否标量化",
     "§14.5"),
    ("codegen", 0, 5, 0,
     ["UnstructuredMemAccessConverter"],
     "…/UnstructureConversionPass.cpp",
     "逐维扫描：structured 维保留[0:size]向量切片，非 structured 维建 scf.for 逐点访存",
     "§14.6"),
    ("splat", 0, 5, 1,
     ["splatAndLoadScenario"],
     "…/UnstructureConversionPass.cpp",
     "scalarLike 分支：extract 单指针→单次 load→splat 广播，O(1) 不进循环",
     "§14.8"),
]
NODE_ORDER = [n[0] for n in NODES]  # 阅读序①…⑦ = 本列表出现顺序
NODE_BY_ID = {n[0]: n for n in NODES}
ENTRY_NODE, EXIT_NODE = "entry", "codegen"  # 主线的真实起止(出口桩挂在标量化 codegen 上——
# 它产出的 DiscreteMemAccess 标记是下一章下游 pass 的认领入口)

EDGES_MAIN = [  # 主线，实线蓝——本章讲解顺序的前向流(章内自身语言，非字面调用边，
    # 详见文件头注释：lattice/combine/transfer 是 parse 递归的兄弟分支，非串行调用链)
    ("entry", "lattice"),
    ("lattice", "combine"),
    ("combine", "transfer"),
    ("transfer", "gate"),
    ("gate", "codegen"),
    ("gate", "splat"),  # matchAndRewrite 的两条真分支之一(scalarLike 快路径)，非无因果汇聚
]
EDGES_SIDE = []  # 本章无旁支

# 路由用的可路由站牌(顺序 = 底部路线条的左→右物理槽位序，与图面列序一致)
STATION_ORDER = ["§14.1", "§14.2", "§14.3", "§14.4", "§14.5", "§14.6", "§14.8"]
ROUTES = [  # (路线名, [§站牌…]按阅读顺序, 是否高亮：True=实线蓝/False=虚线灰)
    ("全览：分析先行→判定→标量化 codegen→快路径(按序读)", STATION_ORDER, True),
    ("跳读：只想知道『什么写法会掉坑』(呼应正文 hook『直接跳 §14.4』)", ["§14.4"], False),
    ("跳读：想量化代价(呼应正文 hook『看 §14.7』——复用本图 §14.5/§14.6 的证据)", ["§14.5", "§14.6"], False),
    ("跳读：scalarLike 为什么不是 gather", ["§14.4", "§14.8"], False),
]
LEGEND = [
    ("#22c55e", "入口(回指 ch13)：结构化路径解析失败/不适用时，本章的兜底 pass 才接管这次访存"),
    ("#3b82f6", "主线：pass 主体与偏移分析之间的讲解前向流"),
    ("#f97316", "出口(预告 ch15)：标量化打上的 DiscreteMemAccess 标记，交给下一章的 blockify 优化 pass 认领"),
]
TITLE = "第 14 章 · Unstructure 兜底剖面：四态分析 → 判定改写 → 标量化 codegen"
SUBNOTE = "节点路径省略公共前缀 third_party/ascend/(以 … 代替)；完整路径见正文行内夹注"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_NODE_PATH = "#7c3aed"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数) ----------------
NODE_W = 205
COL_GAP, ROW_GAP = 20, 22
EDGE_MARGIN, STUB_W, STUB_H = 10, 50, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 14
LANE_LABEL_H, BAND_PAD = 22, 13
TOP_PAD, TITLE_H, SUBNOTE_H, LEGEND_H, BOTTOM_PAD = 12, 26, 16, 3 * 14.5 + 12, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_H, BADGE_FONT, BADGE_PAD_X = 20, 11, 8
ROUTE_BADGE_FONT, ROUTE_BADGE_PAD_X = 10.5, 9
CLAIM_FONT = 9.0
SYM_FONT, SYM_LINE_H = 10.6, 13
ORD_R = 9

# 每个节点的论点先按 NODE_W 预算换行一遍，取全章最多的行数统一定 NODE_H；
# 符号行数同理取全章最大值——同一行号跨泳道对齐用的是同一个 NODE_H。
CLAIM_MAXW = NODE_W - 14
_CLAIM_LINES = {n[0]: wrap_claim(n[6], CLAIM_MAXW, CLAIM_FONT) for n in NODES}
_max_claim_lines = max(len(v) for v in _CLAIM_LINES.values())
_max_sym_lines = max(len(n[4]) for n in NODES)
SYM_TOP = 34
PATH_Y = SYM_TOP + _max_sym_lines * SYM_LINE_H  # 路径行基线
CLAIM_TOP = PATH_Y + 14                         # 首行论点基线
NODE_H = CLAIM_TOP + (_max_claim_lines - 1) * 11.5 + 10

n_cols = max(n[2] for n in NODES) + 1
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_lane = [0] * len(LANES)
for _id, lane, col, row, *_ in NODES:
    rows_per_lane[lane] = max(rows_per_lane[lane], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_lane]
band_top, _cum = [], TOP_PAD + TITLE_H + SUBNOTE_H + LEGEND_H
for bh in band_h:
    band_top.append(_cum)
    _cum += bh
lanes_bottom = _cum

NODE_XY = {}
for nid, lane, col, row, *_ in NODES:
    x = COLX[col]
    y = band_top[lane] + LANE_LABEL_H + BAND_PAD + row * (NODE_H + ROW_GAP)
    NODE_XY[nid] = (x, y)
NODE_COL = {n[0]: n[2] for n in NODES}

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD
assert w <= 1500 and w / h <= 2.6, f"画布预算超标：{w}x{h}, {w / h:.2f}:1"


def badge(cx, cy, text, font=BADGE_FONT, pad_x=BADGE_PAD_X):
    """§ 徽标胶囊，居中挂在 (cx,cy)，宽度按 cjk_text_width 动态算。"""
    bw = cjk_text_width(text, font) + 2 * pad_x
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="{font}" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ], bw


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN), ("Side", C_ROUTE_DIM))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题 + 省略前缀的说明
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 17}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + TITLE_H + 11}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.5" fill="{C_NODE_SUB}">{esc(SUBNOTE)}</text>')

# 图例(3 种语义色必须画图例)
for li, (color, label) in enumerate(LEGEND):
    _row_y = TOP_PAD + TITLE_H + SUBNOTE_H + 13 + li * 14.5
    L.append(f'<rect x="{PAD_L}" y="{_row_y - 10.5}" width="12" height="12" rx="3" fill="{color}"/>')
    L.append(f'<text x="{PAD_L + 18}" y="{_row_y}" font-family="sans-serif" font-size="10" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="12" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                 f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩(跨章标注：目标章号 > 本章号用「预告」，< 本章号用「回指」)
ex, ey = NODE_XY[ENTRY_NODE]; ey += NODE_H / 2
xx, xy = NODE_XY[EXIT_NODE]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.5" font-weight="bold" fill="#166534">{esc("回指 ch13")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.5" font-weight="bold" fill="#9a3412">{esc("预告 ch15")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 主线(实线蓝)——gate 一点两支(codegen/splat)是两条真分支，终点各自的 y 已按各自
# 节点(不同行)天然错开，无需额外偏移；其余边都是单源单宿，直接连边缘中点。
for src, dst in EDGES_MAIN:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    p1 = (xs_ + NODE_W, ys_ + NODE_H / 2)
    p2 = (xd, yd + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
             f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
# 旁支(虚线灰，非因果顺序)——本章为空，占位保留结构一致性
for src, dst in EDGES_SIDE:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    p1, p2 = (xs_ + NODE_W / 2, ys_ + NODE_H), (xd + NODE_W / 2, yd)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
             f'stroke="{C_ROUTE_DIM}" stroke-width="1.6" stroke-dasharray="6,4" '
             f'marker-end="url(#mSide)"/>')

# 节点(圆角框 + 序号圆圈 + 符号 + 路径 + 论点 + 右上角 § 徽标)
for oi, nid in enumerate(NODE_ORDER):
    nid_, lane, col, row, syms, path, claim, sec = NODE_BY_ID[nid]
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H:.1f}" rx="11" '
             f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<circle cx="{x + ORD_R + 4:.1f}" cy="{y + ORD_R + 4:.1f}" r="{ORD_R}" fill="{C_MAIN}"/>')
    L.append(f'<text x="{x + ORD_R + 4:.1f}" y="{y + ORD_R + 7.5:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#ffffff">{oi + 1}</text>')
    sym_w_budget = NODE_W - 20
    sym_size = SYM_FONT
    while max(cjk_text_width(s, sym_size) for s in syms) > sym_w_budget and sym_size > 7.5:
        sym_size -= 0.2
    for si, s in enumerate(syms):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + SYM_TOP + si * SYM_LINE_H:.1f}" '
                 f'text-anchor="middle" font-family="sans-serif" font-size="{sym_size:.1f}" '
                 f'font-weight="bold" fill="{C_NODE_TITLE}">{esc(s)}</text>')
    path_size = 8.0
    while mono_text_width(path, path_size) > NODE_W - 14 and path_size > 6.0:
        path_size -= 0.2
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + PATH_Y:.1f}" text-anchor="middle" '
             f'font-family="monospace" font-size="{path_size:.1f}" '
             f'fill="{C_NODE_PATH}">{esc(path)}</text>')
    for ci, cline in enumerate(_CLAIM_LINES[nid]):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + CLAIM_TOP + ci * 11.5:.1f}" '
                 f'text-anchor="middle" font-family="sans-serif" font-size="{CLAIM_FONT}" '
                 f'fill="{C_NODE_SUB}">{esc(cline)}</text>')
    _bw_station = cjk_text_width(sec, BADGE_FONT) + 2 * BADGE_PAD_X
    badge_svg, _bw = badge(x + NODE_W - 6 - _bw_station / 2, y, sec)
    L += badge_svg

# 底部阅读路线
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(胶囊=图上 § 站牌；实线蓝=推荐 / 虚线灰=次要)")}</text>')
_name_w = max(cjk_text_width(r[0], 11.5) for r in ROUTES)
SLOT_L = 16 + _name_w + 14
SLOT_R = w - EDGE_MARGIN - 6
SLOT_W = (SLOT_R - SLOT_L) / len(STATION_ORDER)
_max_badge_w = max(cjk_text_width(s, ROUTE_BADGE_FONT) + 2 * ROUTE_BADGE_PAD_X for s in STATION_ORDER)
assert _max_badge_w <= SLOT_W, f"站牌胶囊 {_max_badge_w:.0f}px 放不进槽位 {SLOT_W:.0f}px"
SLOT_CX = [SLOT_L + i * SLOT_W + SLOT_W / 2 for i in range(len(STATION_ORDER))]

for ri, (rname, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="11.5" '
             f'fill="{C_NODE_TITLE}">{esc(rname)}</text>')
    idxs = [STATION_ORDER.index(s) for s in stops]
    dash = '' if hi else ' stroke-dasharray="6,4"'
    if len(idxs) > 1:
        L.append(f'<line x1="{SLOT_CX[idxs[0]]:.1f}" y1="{ry:.1f}" x2="{SLOT_CX[idxs[-1]]:.1f}" '
                 f'y2="{ry:.1f}" stroke="{C_MAIN if hi else C_ROUTE_DIM}" '
                 f'stroke-width="{3 if hi else 1.5}"{dash}/>')
    for i, s in zip(idxs, stops):
        L += badge(SLOT_CX[i], ry, s, ROUTE_BADGE_FONT, ROUTE_BADGE_PAD_X)[0]

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  ({w}x{h}, aspect {w / h:.2f}:1)")
