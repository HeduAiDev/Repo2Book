#!/usr/bin/env python3
"""ch10「本章地图」——分水岭剖面：ttir_to_linalg 把指针张量逆向还原成结构化 memref
的源码剖面图（kind=deep，非 primer；这是源码剖面图，不是概念地图）。

本章是**编号标题章**（`## 10.1`…`## 10.7` + `## 小结`），按契约用 §N.M 徽标，
逐一对应正文真实标题：
  §10.1 分水岭在哪里：ttir_to_linalg 的一条管线
  §10.2 目标形态：结构化三元组与 memref
  §10.3 逆向侦探：PtrAnalysis 把指针算术还原成三元组
  §10.4 落地时刻：三元组铸成 memref.reinterpret_cast
  §10.5 访存换轨：load/store 落成对 memref 的搬运
  §10.6 namedOps 的真实语义：别把逐元素 arith 摊成 linalg.generic
  §10.7 桥一趟注解：TritonToAnnotation
（`## 小结` 不建独立节点，折进出口接口桩的说明文字——同 ch04/ch05/ch06/ch08 惯例。）

剖面组织 = 真实源码目录（3 条泳道，上→下）：
  Lane0 third_party/ascend/backend                        —— §10.1 装配入口（唯一节点）
  Lane1 third_party/ascend/lib/TritonToStructured         —— §10.2 目标形态、§10.3 PtrAnalysis
        （同泳道第 2 行挂 §10.7 TritonToAnnotation——它实际源码目录是
        lib/TritonToAnnotation，与 PtrAnalysis 不同目录，但排在真实管线里
        「第一次 add_triton_to_structure 之后」，故画成从 §10.3 下方引出的
        **虚线灰色旁支**，不并入 lib/TritonToStructured 的主线序列——避免误
        导「它是 TritonToStructured 的一部分」）
  Lane2 third_party/ascend/lib/TritonToLinalg              —— §10.4 落地、§10.5 访存换轨、§10.6 namedOps

主线（实线蓝，读者默认顺读路径）：§10.1→§10.2→§10.3→§10.4→§10.5→§10.6。
旁支（虚线灰，同管线内独立挂载、非因果顺序——遵守「关系语义显式标注」硬规则：
      图上与图例都写明「不产出/消费三元组，读者可跳过」，不让默认的向下箭头
      隐含它是主线的下一步）：§10.3 → §10.7。
阅读顺序用左上角序号圆圈①…⑦ 显式给出（旁支节点位置在版面上偏下，不是纯粹
「左上→右下」栅格序，序号圆圈消除歧义——同 ch08 惯例）。

底部四条阅读路线复刻正文 hook 段的原话（"只想看全景地形，读 §10.1 的管线图就够；
想弄懂「指针怎么变回结构」这件核心事，直奔 §10.3；关心那个总被误读的 namedOps
开关，跳 §10.6"）+ 一条从头顺读全通道。

跨章标注（exp-2026-07-18-04 硬规则：目标章号 > 本章号用「预告」，< 本章号用「回指」）：
  入口桩（绿）："回指 ch09" —— ch09 < ch10，本章开篇明确承接原理篇 ch09。
  出口桩（橙）："预告 ch11" —— ch11 > ch10，正文小结明确预告下一章钻进同一把刀的更难例子。

IR 算子名一律「方言 let name + ODS 助记符」，不从 C++ 类名倒推（Book Bible『IR 算子名
的写法约定』）：ascend.annotation（非 triton.ascend.annotation——那是 C++ 命名空间
triton::ascend::AnnotationOp 的路径，TritonAscendOps.td:L47 的真实助记符是
`ascend.annotation`）；tt.addptr/tt.load/tt.store/tt.splat（TritonDialect.td:L7
`let name = "tt"`）；memref.reinterpret_cast、bufferization.to_tensor、
bufferization.materialize_in_destination 用上游 MLIR 内建方言前缀，均已按 §10.1-§10.7
正文逐字核对（见本文件同目录 figure-manifest.json 的 selfcheck）。

源码路径显示时省略公共前缀 `third_party/ascend/`（用 `…/` 代替，仅版式收窄，完整
路径见正文行内夹注——路径信息不失真，只是不在窄节点里重复 20 个字符的公共段）。

不可变视觉语言（全书统一，来自 example-chapter-map.py 模板 + ch08 的动态胶囊/自适应
换行改法）：§徽标胶囊（圆角矩形 fill #eef2ff / stroke #6366f1）、入口绿 #22c55e /
出口橙 #f97316 / 主线蓝 #3b82f6、路线条（高亮=实线蓝 / 次要=虚线灰 #94a3b8）、
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
    "third_party/ascend/backend · 装配层",
    "third_party/ascend/lib/TritonToStructured · 逆向指针分析",
    "third_party/ascend/lib/TritonToLinalg · 落地 / 访存换轨 / namedOps",
]

# (节点id, 泳道下标, 列, 泳道内行号, [符号行…], 省略前缀后的路径, 一句论点, §站牌)
NODES = [
    ("pipeline", 0, 0, 0,
     ["add_stages → ttir_to_linalg", "named_ops=True"],
     "…/backend/compiler.py",
     "18 趟 pass 依序挂载(11 必挂+可选 7)，add_triton_to_structure 出现两次，是分水岭触发点",
     "§10.1"),
    ("target", 1, 1, 0,
     ["PtrState{source,offset,stateInfo}", "StridedLayoutAttr"],
     "…/include/TritonToStructured/PtrAnalysis.h",
     "把 4 路各算各的门牌号，折成 (offset,sizes,strides) 三标量，类型化身是 memref 的 StridedLayoutAttr",
     "§10.2"),
    ("ptranalysis", 1, 2, 0,
     ["visitOperand", "PtrState::addState"],
     "…/lib/TritonToStructured/PtrAnalysis.cpp",
     "按 defining-op 后序递归下潜，make_range/splat/add 各自还原贡献，不认识就整体 failure",
     "§10.3"),
    ("materialize", 2, 3, 0,
     ["BlockDataParser::rewriteAddPtr", "createCastOp"],
     "…/lib/TritonToLinalg/BlockPtrAnalysis.cpp",
     "同构递归再解析一遍，createCastOp 铸出 memref.reinterpret_cast，换掉 tt.addptr",
     "§10.4"),
    ("loadstore", 2, 4, 0,
     ["LoadConverter / StoreConverter", "bufferization.to_tensor"],
     "…/lib/TritonToLinalg/LoadStoreConverter.cpp",
     "tt.load 落成 alloc+copy+to_tensor，tt.store 落成 bufferization.materialize_in_destination",
     "§10.5"),
    ("namedops", 2, 5, 0,
     # [FIX-ROUND-2]:populateElementwiseToLinalgConversionPatterns 单行在窄节点里
     # 即使收缩到字号下限仍超框(overflow)——拆成两行(在驼峰边界断,不劈开子词)
     ["namedOps", "populateElementwise", "ToLinalgConversionPatterns"],
     "…/lib/TritonToLinalg/TritonToLinalgPass.cpp",
     "为真时张量 arith 判合法，不加载摊平 pattern，语义是别摊成 linalg.generic",
     "§10.6"),
    # 旁支：同一管线里的独立 pass，画在 §10.3 正下方(同列不同行)，虚线灰连边——
    # 不是主线的下一步，语义见图例第 4 条与本文件顶部注释。
    ("annotation", 1, 2, 1,
     ["ascend.annotation", "annotation.mark"],
     "…/lib/TritonToAnnotation/TritonToAnnotation.cpp",
     "同管线里旁挂的轻量 pass，把注解改写成 annotation.mark 并转发属性，不碰指针、不产三元组",
     "§10.7"),
]
NODE_ORDER = [n[0] for n in NODES]  # 阅读序①…⑦ = 本列表出现顺序
ENTRY_NODE, EXIT_NODE = "pipeline", "namedops"  # 主线的真实起止(旁支不是终点)

EDGES_MAIN = [  # 主线，实线蓝
    ("pipeline", "target"), ("target", "ptranalysis"),
    ("ptranalysis", "materialize"), ("materialize", "loadstore"), ("loadstore", "namedops"),
]
EDGES_SIDE = [("ptranalysis", "annotation")]  # 旁支，虚线灰(非因果顺序，仅示意同管线挂载)

STATION_ORDER = [n[7] for n in NODES]  # 站序槽位，供底部路线复用
ROUTES = [  # (路线名, [§站牌…]按阅读顺序, 是否高亮：True=实线蓝/False=虚线灰)
    ("从头顺读(全通道)", STATION_ORDER, True),
    ("只想看全景地形", ["§10.1"], False),
    ("直奔核心：指针→结构", ["§10.3"], False),
    ("只查 namedOps 误区", ["§10.6"], False),
]
LEGEND = [
    ("#22c55e", "入口(回指 ch09)：原理篇立好 Linalg 为什么值得，本章接着讲怎么做到"),
    ("#3b82f6", "主线：指针张量 → PtrAnalysis 逆向还原三元组 → memref.reinterpret_cast → 结构化访存/namedOps"),
    ("#f97316", "出口(预告 ch11)：下一章钻进同一把刀 PtrAnalysis，啃 iter-arg/rem/div/mask 更难的例子"),
    ("#94a3b8", "灰虚线：同一 pass 管线内独立挂载的旁支(§10.7)——不产出/消费三元组，非因果顺序，可跳过"),
]
TITLE = "第 10 章 · 分水岭剖面：ttir_to_linalg 把指针张量逆向还原成结构化 memref"
SUBNOTE = "节点路径省略公共前缀 third_party/ascend/(以 … 代替)；完整路径见正文行内夹注"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_NODE_PATH = "#7c3aed"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数) ----------------
NODE_W = 205
COL_GAP, ROW_GAP = 20, 22
EDGE_MARGIN, STUB_W, STUB_H = 10, 50, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 14
LANE_LABEL_H, BAND_PAD = 22, 13
TOP_PAD, TITLE_H, SUBNOTE_H, LEGEND_H, BOTTOM_PAD = 12, 26, 16, 72, 16
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
SYM_TOP = 34                                    # 第一行符号的基线偏移(须清空左上角序号圆圈
                                                 # y∈[+4,+22] 与右上角§徽标 y∈[-10,+10] 的
                                                 # 竖直范围,不能只按水平预算收窄字体——[FIX-ROUND-1]:
                                                 # SYM_TOP=15 时长符号行(如 add_stages→ttir_to_linalg、
                                                 # BlockDataParser::rewriteAddPtr)会压在§徽标/序号圆圈上
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
NODE_BY_ID = {n[0]: n for n in NODES}
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

# 图例(4 种语义色/线型必须画图例)
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
         f'font-family="sans-serif" font-size="9.5" font-weight="bold" fill="#166534">{esc("回指 ch09")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.5" font-weight="bold" fill="#9a3412">{esc("预告 ch11")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')


def edge_points(src, dst):
    """通用连边锚点：同列(旁支，不同行)走竖直(下边中点→上边中点)；
    否则按列先后走水平(右边中点→左边中点，或反向)。"""
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    if NODE_COL[src] == NODE_COL[dst]:
        return (xs_ + NODE_W / 2, ys_ + NODE_H), (xd + NODE_W / 2, yd)
    if NODE_COL[dst] > NODE_COL[src]:
        return (xs_ + NODE_W, ys_ + NODE_H / 2), (xd, yd + NODE_H / 2)
    return (xs_, ys_ + NODE_H / 2), (xd + NODE_W, yd + NODE_H / 2)


# 主线(实线蓝)
for src, dst in EDGES_MAIN:
    p1, p2 = edge_points(src, dst)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
             f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
# 旁支(虚线灰，非因果顺序——见图例第 4 条)
for src, dst in EDGES_SIDE:
    p1, p2 = edge_points(src, dst)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
             f'stroke="{C_ROUTE_DIM}" stroke-width="1.6" stroke-dasharray="6,4" '
             f'marker-end="url(#mSide)"/>')
    mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
    L.append(f'<rect x="{mx - 46:.1f}" y="{my - 8:.1f}" width="92" height="16" rx="8" '
             f'fill="#f1f5f9" stroke="{C_ROUTE_DIM}" stroke-width="1"/>')
    L.append(f'<text x="{mx:.1f}" y="{my + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="9" fill="{C_NODE_SUB}">{esc("旁挂·非主线")}</text>')

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
