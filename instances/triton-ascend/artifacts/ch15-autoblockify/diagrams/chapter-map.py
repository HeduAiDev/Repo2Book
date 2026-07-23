#!/usr/bin/env python3
"""ch15「本章地图」——AutoBlockify 剖面：一次 pass 执行里 no-op 门 → 造载体 →
逐 op 下推 → 批处理化/blockify 循环降级 → 收益，两条泳道对应两组真实源码文件
（kind=deep, skip_impl=true；纯 C++ MLIR pass，无精简版，图上只呈现真实源码符号
+ 夹具常量）。

本章是**自然标题章**（`## autoBlockifySize：折叠粒度，与那扇 no-op 门` 这类，
无 `## N.M` 编号），按契约禁用 §N.M 徽标，站牌一律用标题词本身。

**节点排布按 pass 真实调用序，不是正文行文顺序**——这一点正文开篇选读指引里
自己点破了："出于讲解顺手，守门 checkBlockifiable 的细节被安排在「造载体/下推」
之后展开；但它在真实调用里其实最先跑……别被小节排布误导成它更靠后"
（third_party/ascend/lib/AutoBlockify/AutoBlockify.cpp:L286-L311 印证：
runOnOperation 逐 FuncOp 先 checkBlockifiable 守门，通过才 preProcess）。
若图按行文标题顺序画（no-op 门→造载体→下推→守门→…），恰好复刻了正文自己提醒
读者别踩的那个坑；因此本图的主线边（entry→gate→carrier→propagate→landing→
benefit）取 dossier.json `data_flow` 记录的真实调用序，与「正文按序读」路线
（选读路线①，仍按行文标题顺序列出全部站牌，两者站牌集合相同、只是排列顺序
不同）分开表达，互不矛盾。

两条泳道：
  Lane0  third_party/ascend/lib/AutoBlockify/AutoBlockify.cpp
         —— pass 主体：no-op 门 / 守门 checkBlockifiable / 造载体 preProcess /
            逐 op 下推驱动 / cast 落地收尾
  Lane1  third_party/ascend/lib/AutoBlockify/{Utils,RewriteOperation}.cpp
         —— 批处理化机制：前导维张量化 / blockify 循环 / 尾块与 mask

「前导维批处理化」「blockify 循环」是 propagate（matchAndRewrite）的两条真分支
（同一 cast 的不同 user 各自命中 rewrite* 或 handleBlockifyLoop），是真实分派、
非无关默认汇聚，不需要「无因果」注记。「尾块与 mask」则是两支细节的汇总节点
（循环上界数学属 blockify 循环、mask 广播数学属批处理化 load/store 落地）——
两支并非先后因果，已在该节点论点文字里显式注明「汇总前两支细节」。

跨章标注（exp-2026-07-18-04 硬规则：目标章号 > 本章号用「预告」，< 本章号用
「回指」）：
  入口桩（绿）："回指 ch10" —— 第 10 章《分水岭》已点过 add_auto_blockify 是
                ttir_to_linalg 管线打头的第 1 趟 pass，本章接手细讲。
  出口桩（橙）："预告 ch16" —— 折完网格粒度后，下一章问"每个算子该落 cube
                还是 vector 核"，是另一套数据流分析。

底部阅读路线复刻正文 hook 段原话（"只想知道『折叠后 IR 长什么样』，直接跳到
前导维批处理化看夹具前后对照；想跟完整机制，按序读"）。

不可变视觉语言（全书统一，来自 example-chapter-map.py 模板 + ch14 的动态胶囊/
自适应换行/编号圆圈/monospace 路径行改法）：§徽标胶囊改用自然标题词（本章
无编号，禁 §N.M）、入口绿 #22c55e / 出口橙 #f97316 / 主线蓝 #3b82f6、
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
    "third_party/ascend/lib/AutoBlockify/AutoBlockify.cpp · pass 主体：no-op 门/守门/造载体/下推驱动/收尾落地",
    "third_party/ascend/lib/AutoBlockify/{Utils,RewriteOperation}.cpp · 批处理化机制：前导维张量化/blockify 循环/尾块与 mask",
]

# (节点id, 泳道下标, 列, 泳道内行号, [符号行…], 省略前缀后的路径, 一句论点, 站牌(自然标题词，禁 §N.M))
NODES = [
    ("entry", 0, 0, 0,
     ["runOnOperation"],
     "…/AutoBlockify.cpp",
     "size==1 直接返回(no-op)；size<=0 告警失败，否则逐 FuncOp 继续",
     "no-op 门"),
    ("gate", 0, 1, 0,
     ["checkBlockifiable"],
     "…/AutoBlockify.cpp",
     "沿 program-id 的 use-def 链递归查：碰硬拒绝算子整函数放弃，碰 if 打标转循环",
     "守门"),
    ("carrier", 0, 2, 0,
     ["preProcess"],
     "…/AutoBlockify.cpp",
     "拍平 3 维网格成 logicalBlockId；splat+range 造 blockifiedId，包成双输入 cast 载体",
     "造载体"),
    ("propagate", 0, 3, 0,
     ["PropagateUnrealizedCastDown", "matchAndRewrite"],
     "…/AutoBlockify.cpp",
     "贪婪重写沿载体逐 user 分派：能批处理走前导维张量化，不能的转 blockify 循环",
     "逐 op 下推"),
    ("leading_dim", 1, 4, 0,
     ["getExpandedType", "rewriteSplat"],
     "…/{Utils,RewriteOperation}.cpp",
     "size 恒拼到 shape 最前：tensor<8> 变 tensor<5x8>，一条向量指令跨 5 个实例",
     "批处理化"),
    ("blockify_loop", 1, 4, 1,
     ["createBlockifyLoop", "handleBlockifyLoop"],
     "…/{Utils,RewriteOperation}.cpp",
     "折不成张量的 region op 转一条 scf.for，逐 iv 切片喂回原 op",
     "blockify 循环"),
    ("tail_mask", 1, 5, 0,
     ["createMask"],
     "…/Utils.cpp",
     "循环上界兜尾块；载体 mask 广播后与用户 mask 相与兜越界 lane (汇总前两支细节)",
     "尾块与 mask"),
    ("landing", 0, 6, 0,
     ["UnrealizedConversionCastOp"],
     "…/AutoBlockify.cpp",
     "残留终态 cast 按输入种类落地为 constant/broadcast/splat；FuncOp 打 auto_blockify_size 属性",
     "cast 落地"),
    ("benefit", 0, 7, 0,
     ["auto_blockify_size = 5"],
     "unittest/…/auto_blockify.mlir",
     "G=6,size=5 时：调度块数 6→2；tensor<8>→tensor<5x8>，一条 store 顶 5 条",
     "收益量化"),
]
NODE_ORDER = [n[0] for n in NODES]  # 阅读序①…⑨ = 本列表出现顺序(= pass 真实调用序)
NODE_BY_ID = {n[0]: n for n in NODES}
ENTRY_NODE, EXIT_NODE = "entry", "benefit"

EDGES_MAIN = [  # 主线，实线蓝——pass 真实调用序(dossier.json data_flow)
    ("entry", "gate"),
    ("gate", "carrier"),
    ("carrier", "propagate"),
    ("propagate", "leading_dim"),   # matchAndRewrite 的真分支①：能张量化的 user
    ("propagate", "blockify_loop"), # matchAndRewrite 的真分支②：已在循环体内的 user
    ("propagate", "landing"),       # 贪婪重写收敛后，控制权回到 runOnOperation 做落地
    ("landing", "benefit"),
]
EDGES_SIDE = [  # 虚线灰——两支细节汇总进"尾块与 mask"，非先后因果(节点论点已注明)
    ("leading_dim", "tail_mask"),
    ("blockify_loop", "tail_mask"),
]

STATION_ORDER = [  # 底部路线的可路由站牌，按物理列从左到右排列
    "no-op 门", "守门", "造载体", "逐 op 下推",
    "批处理化", "blockify 循环", "尾块与 mask", "cast 落地", "收益量化",
]
ROUTES = [  # (路线名, [站牌…]按阅读顺序, 是否高亮：True=实线蓝/False=虚线灰)
    ("全览：按调用序读全部环节", STATION_ORDER, True),
    ("跳读：折叠后 IR 长什么样", ["批处理化"], False),
]
LEGEND = [
    ("#22c55e", "入口(回指 ch10)：管线里 add_auto_blockify 挂为第 1 趟 pass 的位置"),
    ("#3b82f6", "主线：本 pass 一次执行的真实调用序(no-op 门→守门→造载体→下推→落地)"),
    ("#f97316", "出口(预告 ch16)：折完网格粒度，下一章问每个算子该落哪个核"),
]
TITLE = "第 15 章 · AutoBlockify 剖面：no-op 门 → 造载体 → 逐 op 下推 → 批处理化/循环降级 → 收益"
SUBNOTE = "节点路径省略公共前缀 third_party/ascend/lib/AutoBlockify/(以 … 代替)；完整路径见正文行内夹注。节点排布按 pass 真实调用序，非正文行文顺序(见文件头注释)"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_NODE_PATH = "#7c3aed"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数) ----------------
NODE_W = 150
COL_GAP, ROW_GAP = 14, 20
EDGE_MARGIN, STUB_W, STUB_H = 8, 48, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 12
LANE_LABEL_H, BAND_PAD = 22, 12
TOP_PAD, TITLE_H, SUBNOTE_H, LEGEND_H, BOTTOM_PAD = 12, 26, 24, 3 * 14.5 + 12, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 38
BADGE_H, BADGE_FONT, BADGE_PAD_X = 18, 9.5, 6
ROUTE_BADGE_FONT, ROUTE_BADGE_PAD_X = 9.5, 7
CLAIM_FONT = 8.3
SYM_FONT, SYM_LINE_H = 9.6, 12
ORD_R = 8

# 每个节点的论点先按 NODE_W 预算换行一遍，取全章最多的行数统一定 NODE_H；
# 符号行数同理取全章最大值——同一行号跨泳道对齐用的是同一个 NODE_H。
CLAIM_MAXW = NODE_W - 12
_CLAIM_LINES = {n[0]: wrap_claim(n[6], CLAIM_MAXW, CLAIM_FONT) for n in NODES}
_max_claim_lines = max(len(v) for v in _CLAIM_LINES.values())
_max_sym_lines = max(len(n[4]) for n in NODES)
SYM_TOP = 30
PATH_Y = SYM_TOP + _max_sym_lines * SYM_LINE_H  # 路径行基线
CLAIM_TOP = PATH_Y + 13                         # 首行论点基线
NODE_H = CLAIM_TOP + (_max_claim_lines - 1) * 10.5 + 9

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

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD
assert w <= 1500 and w / h <= 2.6, f"画布预算超标：{w}x{h}, {w / h:.2f}:1"


def badge(cx, cy, text, font=BADGE_FONT, pad_x=BADGE_PAD_X):
    """站牌胶囊，居中挂在 (cx,cy)，宽度按 cjk_text_width 动态算(本章自然标题，禁 §N.M)。"""
    bw = cjk_text_width(text, font) + 2 * pad_x
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 3.5:.1f}" text-anchor="middle" font-family="sans-serif" '
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
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 16}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
_subnote_lines = wrap_claim(SUBNOTE, w - 2 * PAD_L, 9.0)
for si, sline in enumerate(_subnote_lines[:2]):
    L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + TITLE_H + 9 + si * 11:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="9.0" fill="{C_NODE_SUB}">{esc(sline)}</text>')

# 图例(3 种语义色必须画图例)
for li, (color, label) in enumerate(LEGEND):
    _row_y = TOP_PAD + TITLE_H + SUBNOTE_H + 12 + li * 14.5
    L.append(f'<rect x="{PAD_L}" y="{_row_y - 10}" width="11" height="11" rx="3" fill="{color}"/>')
    L.append(f'<text x="{PAD_L + 16}" y="{_row_y}" font-family="sans-serif" font-size="9.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="14" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="10.5" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
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
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 3.5:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9" font-weight="bold" fill="#166534">{esc("回指 ch10")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 3.5:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9" font-weight="bold" fill="#9a3412">{esc("预告 ch16")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 主线(实线蓝)——propagate 一点两支(leading_dim/blockify_loop)是 matchAndRewrite 的两条
# 真分支，终点各自的 y 已按各自节点(不同行)天然错开，无需额外偏移。
for src, dst in EDGES_MAIN:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    p1 = (xs_ + NODE_W, ys_ + NODE_H / 2)
    p2 = (xd, yd + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
             f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
# 旁支(虚线灰，两支细节汇总进"尾块与 mask"，非先后因果——节点论点已注明)
_dst_total = {}
for _, dst in EDGES_SIDE:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst in EDGES_SIDE:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    p1 = (xs_ + NODE_W, ys_ + NODE_H / 2)
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    y_offset = (i - (n - 1) / 2) * 14 if n > 1 else 0
    p2 = (xd, yd + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
             f'stroke="{C_ROUTE_DIM}" stroke-width="1.6" stroke-dasharray="6,4" '
             f'marker-end="url(#mSide)"/>')

# 节点(圆角框 + 序号圆圈 + 符号 + 路径 + 论点 + 右上角站牌胶囊)
for oi, nid in enumerate(NODE_ORDER):
    nid_, lane, col, row, syms, path, claim, sec = NODE_BY_ID[nid]
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H:.1f}" rx="10" '
             f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<circle cx="{x + ORD_R + 4:.1f}" cy="{y + ORD_R + 4:.1f}" r="{ORD_R}" fill="{C_MAIN}"/>')
    L.append(f'<text x="{x + ORD_R + 4:.1f}" y="{y + ORD_R + 7:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="9" font-weight="bold" fill="#ffffff">{oi + 1}</text>')
    sym_w_budget = NODE_W - 16
    sym_size = SYM_FONT
    while max(cjk_text_width(s, sym_size) for s in syms) > sym_w_budget and sym_size > 6.5:
        sym_size -= 0.2
    for si, s in enumerate(syms):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + SYM_TOP + si * SYM_LINE_H:.1f}" '
                 f'text-anchor="middle" font-family="sans-serif" font-size="{sym_size:.1f}" '
                 f'font-weight="bold" fill="{C_NODE_TITLE}">{esc(s)}</text>')
    path_size = 7.6
    while mono_text_width(path, path_size) > NODE_W - 12 and path_size > 5.8:
        path_size -= 0.2
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + PATH_Y:.1f}" text-anchor="middle" '
             f'font-family="monospace" font-size="{path_size:.1f}" '
             f'fill="{C_NODE_PATH}">{esc(path)}</text>')
    for ci, cline in enumerate(_CLAIM_LINES[nid]):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + CLAIM_TOP + ci * 10.5:.1f}" '
                 f'text-anchor="middle" font-family="sans-serif" font-size="{CLAIM_FONT}" '
                 f'fill="{C_NODE_SUB}">{esc(cline)}</text>')
    _bw_station = cjk_text_width(sec, BADGE_FONT) + 2 * BADGE_PAD_X
    badge_svg, _bw = badge(x + NODE_W - 4 - _bw_station / 2, y, sec)
    L += badge_svg

# 底部阅读路线
L.append(f'<text x="14" y="{routes_top + 14:.1f}" font-family="sans-serif" font-size="10.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(胶囊=图上站牌；实线蓝=推荐 / 虚线灰=次要)")}</text>')
_name_w = max(cjk_text_width(r[0], 9.5) for r in ROUTES)
SLOT_L = 14 + _name_w + 12
SLOT_R = w - EDGE_MARGIN - 6
SLOT_W = (SLOT_R - SLOT_L) / len(STATION_ORDER)
_max_badge_w = max(cjk_text_width(s, ROUTE_BADGE_FONT) + 2 * ROUTE_BADGE_PAD_X for s in STATION_ORDER)
assert _max_badge_w <= SLOT_W, f"站牌胶囊 {_max_badge_w:.0f}px 放不进槽位 {SLOT_W:.0f}px"
SLOT_CX = [SLOT_L + i * SLOT_W + SLOT_W / 2 for i in range(len(STATION_ORDER))]

for ri, (rname, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="14" y="{ry + 3.5:.1f}" font-family="sans-serif" font-size="9.5" '
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
