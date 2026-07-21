#!/usr/bin/env python3
"""ch06「本章地图」——跨 GM 的四个 mem_ops 索引搬运 + 片上 vec_ops 词汇 +
三种写法与落点分野 的源码剖面图。

本章是**自然标题章**(正文全是 `## 接缝在哪：...` 这类自然标题，无 `## N.M`
编号)——按契约禁用 `§N.M` 徽标，站牌改用正文标题词本身(逐字是正文标题里出现过
的真实子串，供自查与 linter 逐一核对)：
  “接缝在哪”“三级映射”“反向搬运”“不查越界”“片上词汇表”“两条路”
  “只排末维”“决策树”“三种写法”“落点表”——十个站牌对应正文十个自然标题小节。
  “小结：一层可选的加速词汇”一节不建独立节点，改由出口接口桩的说明文字收尾
  (同 ch04/ch05 惯例)。

剖面(蛇形三带，十站递进，每节点标『真实符号 + 规范源码路径 + 一句论点』)：
  Lane0 左→右 ①接缝(pybind 只导出 5 档，没有 GM) → ②gather 的三级映射
        → ③scatter/index_put 反向 → ④index_select_simd 不查越界
  Lane1 右→左 ⑤片上切片/取标量(落上游 tensor 方言) → ⑥flip 双路径
        → ⑦sort 只排末维 → ⑧cast 决策树
  Lane2 左→右 ⑨同一件事的三种写法 → ⑩落点表(方言分野在 C++ 侧钉死)

蛇形(boustrophedon)排布的理由：十站单排会把画布拉到 2400px 宽、远超
lint_chapter_map 的「宽 ≤1500 且宽高比 ≤2.6:1」预算。折成三带后跨带的走线是
一段短竖线(贴在最右/最左列中心)，不必横穿泳道标签。阅读顺序由每个节点左上角
的序号圆圈 ①…⑩ 显式给出，不依赖「左上→右下」的默认约定。

模板：.claude/skills/svg-diagram/references/example-chapter-map.py；不可变视觉语言
(站牌徽标胶囊 / 入口绿-出口橙-主线蓝 / 高亮实线蓝-次要虚线灰 / cjk_text_width)
照搬同书 ch05 版，只改 DATA 与三处结构：①节点头部允许两行符号(本章多个站点是
一族算子，如 scatter_ub_to_out + index_put)，行数全章统一以保持跨带对齐；
②跨带走线的竖直附着分支(同列上下相邻)；③底部阅读路线的胶囊按「站序槽位」等分
排布而非按节点列排布——蛇形版图里列号已不等于阅读序，按槽位才能让同一站在各条
路线里上下对齐。

六项自查(渲染→Read PNG 亲眼看后如实记录)：见 figure-manifest.json 该图 selfcheck。

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


# ---------------- DATA(可变：本章数据) ----------------
# 泳道文字要短：跨带竖线落在首/末列中心，标签太长会被竖线压到(下方有断言兜底)。
LANES = ["跨 GM 的接缝 · 四个 mem_ops",
         "片上 vec_ops · 结构与类型",
         "收口 · 写法与落点"]  # 上→下
LANE_DIR = [+1, -1, +1]  # 蛇形：+1 左→右，-1 右→左

# (节点id, 泳道下标, 列, 泳道内行号, [符号行…], 规范源码路径, 一句论点, 站牌=正文自然标题词)
NODES = [
    ("seam", 0, 0, 0,
     ["AddressSpace", "(pybind 枚举)"],
     "third_party/ascend/ascend_ir.cc",
     ".td 定义 7 档、pybind 只导出 5 档；Zero 与 GM 不进 Python，跨 GM 的带索引访问只能交给 mem_ops",
     "接缝在哪"),
    ("gather", 0, 1, 0,
     ["gather_out_to_ub"],
     "third_party/ascend/language/cann/extension/mem_ops.py",
     "index 值只定 dim 轴坐标，其余维来自格子自身位置；越界格子在第二级被摘掉，由 other 顶上",
     "三级映射"),
    ("scatter", 0, 2, 0,
     ["scatter_ub_to_out", "index_put"],
     "third_party/ascend/language/cann/extension/mem_ops.py",
     "镜像的反向搬运：越界的不写回、直接丢弃；写回数 + 丢弃数 = index 元素总数",
     "反向搬运"),
    ("simd", 0, 3, 0,
     ["index_select_simd"],
     "third_party/ascend/language/cann/extension/mem_ops.py",
     "四个里唯一没有 index_boundary 的：粒度升到整条连续段，docstring 明写不检查越界",
     "不查越界"),
    ("slice", 1, 3, 0,
     ["insert_slice / extract_slice", "get_element"],
     "third_party/ascend/language/cann/extension/vec_ops.py",
     "两端都在片上、不过缝：落点是上游 tensor 方言，不是昇腾方言——这条分野是设计意图",
     "片上词汇表"),
    ("flip", 1, 2, 0,
     ["flip"],
     "third_party/ascend/language/cann/extension/vec_ops.py",
     "同一个 API 两条路：SIMD 一条 create_flip 就完事；SIMT 没有这条指令，退成 log2(n) 轮 xor 换位",
     "两条路"),
    ("sort", 1, 1, 0,
     ["sort"],
     "third_party/ascend/language/cann/extension/vec_ops.py",
     "只排末维 + dtype 白名单；int8/int16 在出口自动挂一条 overflow_mode=saturate 提示",
     "只排末维"),
    ("cast", 1, 0, 0,
     ["cast", "ascend_cast_impl"],
     "third_party/ascend/language/cann/extension/vec_ops.py",
     "另写一份决策树：bf16/fp16 借道 fp32、自递归深度 ≤ 2；只认 trunc/saturate，文档却拼成 sautrate",
     "决策树"),
    ("three", 2, 0, 0,
     ["手写 / 内建 / 交给编译器", "ascend.indirect_load"],
     "third_party/ascend/unittest/pytest_ut/test_index_select.py",
     "同一件事三种写法结果一致，差别只在前端发出的算子条数、以及最终落到哪个 IR 算子",
     "三种写法"),
    ("landing", 2, 1, 0,
     ["create_gather_out_to_ub", "ascend.gather_out_to_ub"],
     "third_party/ascend/triton_ascend.cc",
     "昇腾方言与上游 tensor 方言的分野在 C++ 侧钉死；返回类型由 index 的 shape 拼出",
     "落点表"),
]
EDGES = [  # (src_id, dst_id) —— 章内递进主线，统一主线蓝
    ("seam", "gather"), ("gather", "scatter"), ("scatter", "simd"),
    ("simd", "slice"),                      # 跨带：右列竖直下行
    ("slice", "flip"), ("flip", "sort"), ("sort", "cast"),
    ("cast", "three"),                      # 跨带：左列竖直下行
    ("three", "landing"),
]
# 站序槽位 = NODES 的顺序(①…⑩)；路线里的站牌按该槽位对齐
STATION_ORDER = [n[7] for n in NODES]
# (路线名, [站牌…] 按阅读顺序, 是否高亮：True=实线蓝/False=虚线灰)
ROUTES = [
    ("从头顺读（全览）", STATION_ORDER, True),
    ("只想会用：索引搬运", ["接缝在哪", "三级映射", "反向搬运", "不查越界"], False),
    ("只关心片上算子", ["片上词汇表", "两条路", "只排末维", "决策树"], False),
    ("鸟瞰：分野在哪", ["接缝在哪", "三种写法", "落点表"], False),
]
LEGEND = [
    ("#22c55e", "入口：上一章讲完 buffer 语言管「数据在几楼」，本章讲语言层剩下的半边——带索引的搬运与片上算子"),
    ("#3b82f6", "章内主线：接缝 → 四个 mem_ops(3 带越界上界、1 不带) → 片上 vec_ops → 三种写法 → 落点分野"),
    ("#f97316", "出口：小结把这批内建定性为「可选的加速词汇」；下一章接自定义算子与 libdevice"),
]
TITLE = "第 6 章 · 昇腾内建算子：索引搬运、向量算子与定制 cast 的源码剖面"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_NODE_PATH = "#7c3aed"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数) ----------------
NODE_W = 232
COL_GAP, ROW_GAP = 32, 20
EDGE_MARGIN, STUB_W, STUB_H = 10, 46, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 22
LANE_LABEL_H, BAND_PAD = 24, 14
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 62, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 42
BADGE_H = 20
BADGE_FONT = 11
BADGE_PAD_X = 13  # 徽标左右各留的内边距(动态宽度=文本宽+2×BADGE_PAD_X)
CLAIM_FONT = 9.2
SYM_FONT, SYM_LINE_H = 12.0, 14
ORD_R = 9  # 序号圆圈半径

_BREAK_AFTER = set("，；：、/ ,;")


def wrap_claim(text, max_w, size):
    """一句论点太长时换行——只在标点/斜杠/空格之后断行，不允许劈开一个标识符
    (如 index_select_simd)或一个中文词。贪心找"prefix 仍不超宽的最靠后一个
    合法断点"；找不到合法断点才整句照旧单行放行。"""
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
    more = wrap_claim(line2, max_w, size)
    return [line1] + more


# 每个节点的论点先按 NODE_W 预算换行一遍，取全章最多的行数统一定 NODE_H；
# 符号行数同理取全章最大值——同一行号跨泳道对齐用的是同一个 NODE_H，
# 节点内容多寡不能各自决定框高，否则同一带里矮框高框错位、背景条被撑破。
CLAIM_MAXW = NODE_W - 16
_CLAIM_LINES = {n[0]: wrap_claim(n[6], CLAIM_MAXW, CLAIM_FONT) for n in NODES}
_max_claim_lines = max(len(v) for v in _CLAIM_LINES.values())
_max_sym_lines = max(len(n[4]) for n in NODES)
SYM_TOP = 26                                   # 第一行符号的基线偏移(须 > BADGE_H/2 + 字高，否则被顶边胶囊压住)
PATH_Y = SYM_TOP + _max_sym_lines * SYM_LINE_H  # 路径行基线
CLAIM_TOP = PATH_Y + 15                        # 首行论点基线
NODE_H = CLAIM_TOP + (_max_claim_lines - 1) * 12 + 10

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
NODE_LANE = {n[0]: n[1] for n in NODES}
NODE_COL = {n[0]: n[2] for n in NODES}
ORDER_OF = {n[0]: i + 1 for i, n in enumerate(NODES)}

# 跨带竖线附着在框宽中心：它只穿过**目标泳道**的标签行，故只核目标带的标签宽度
# (泳道标签从 x=16 起排，须在竖线左侧留出空隙，否则标签被走线压住——no_overlap 一项)。
# 中心附着同时避开右对齐的站牌胶囊(胶囊左缘恒在框宽中点右侧，下方断言兜底)。
CROSS_FRAC = 0.5
for _s, _d in EDGES:
    if NODE_LANE[_s] == NODE_LANE[_d]:
        continue
    _line_x = COLX[NODE_COL[_d]] + NODE_W * CROSS_FRAC
    _lb = LANES[NODE_LANE[_d]] + " ←"
    assert 16 + cjk_text_width(_lb, 13) + 12 <= _line_x, (
        f"泳道标签『{_lb}』会被 x={_line_x:.0f} 的跨带走线压住——请缩短标签")
    _bw_d = cjk_text_width(NODE_BY_ID[_d][7], BADGE_FONT) + 2 * BADGE_PAD_X
    assert NODE_W * CROSS_FRAC + 6 <= NODE_W - 8 - _bw_d, (
        f"跨带走线会压到站牌『{NODE_BY_ID[_d][7]}』的胶囊")

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD
assert w <= 1500 and w / h <= 2.6, f"画布预算超标：{w}x{h}, {w / h:.2f}:1"


def badge(cx, cy, text):
    """站牌徽标胶囊，居中挂在 (cx,cy)。宽度按 cjk_text_width 动态算(自然标题
    站牌比 §N.M 长得多，模板里的定宽 BADGE_W 会把长站牌文字挤出胶囊)。
    胶囊样式/配色/圆角高度仍是模板的不可变视觉语言。"""
    bw = cjk_text_width(text, BADGE_FONT) + 2 * BADGE_PAD_X
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="{BADGE_FONT}" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ], bw


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
# 图例(>2 种语义色必须画图例)；三条说明偏长，纵向各占一行堆叠，避免横排挤出画布
for li, (color, label) in enumerate(LEGEND):
    _row_y = TOP_PAD + TITLE_H + 14 + li * 14
    L.append(f'<rect x="{PAD_L}" y="{_row_y - 11}" width="12" height="12" rx="3" fill="{color}"/>')
    L.append(f'<text x="{PAD_L + 18}" y="{_row_y}" font-family="sans-serif" font-size="10.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')

# 泳道背景 + 标签(带流向箭头指示蛇形方向) + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    arrow = "→" if LANE_DIR[i] > 0 else "←"
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name + " " + arrow)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                 f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩：入口挂在第一站左侧，出口挂在末站右侧
_first, _last = NODES[0][0], NODES[-1][0]
ex, ey = NODE_XY[_first]; ey += NODE_H / 2
xx, xy = NODE_XY[_last]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#166534">{esc("读者")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#9a3412">{esc("下一章")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝)：同带内按该带流向做左右附着；跨带同列走竖直附着(下边中点 → 上边中点)。
for src, dst in EDGES:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    if NODE_LANE[src] != NODE_LANE[dst]:
        p1 = (xs_ + NODE_W * CROSS_FRAC, ys_ + NODE_H)
        p2 = (xd + NODE_W * CROSS_FRAC, yd)
    elif LANE_DIR[NODE_LANE[src]] > 0:
        p1 = (xs_ + NODE_W, ys_ + NODE_H / 2)
        p2 = (xd, yd + NODE_H / 2)
    else:
        p1 = (xs_, ys_ + NODE_H / 2)
        p2 = (xd + NODE_W, yd + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
             f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 序号圆圈 + 真实符号 + 规范源码路径 + 一句论点 + 右上角站牌徽标)
for nid, lane, col, row, syms, path, claim, station in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H:.1f}" rx="12" '
             f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    # 阅读序号：蛇形排布下「左上→右下」的默认约定不成立，序号显式给出看图顺序
    L.append(f'<circle cx="{x + ORD_R + 4:.1f}" cy="{y + ORD_R + 4:.1f}" r="{ORD_R}" '
             f'fill="{C_MAIN}"/>')
    L.append(f'<text x="{x + ORD_R + 4:.1f}" y="{y + ORD_R + 7.5:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#ffffff">'
             f'{ORDER_OF[nid]}</text>')
    # 符号行：字号按最宽一行自适应缩，保证不越框(序号圆圈占左上角，故留出 2×ORD_R 余量)
    sym_w_budget = NODE_W - 16 - 2 * ORD_R
    sym_size = SYM_FONT
    while max(cjk_text_width(s, sym_size) for s in syms) > sym_w_budget and sym_size > 8.5:
        sym_size -= 0.3
    for si, s in enumerate(syms):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + SYM_TOP + si * SYM_LINE_H:.1f}" '
                 f'text-anchor="middle" font-family="sans-serif" font-size="{sym_size:.1f}" '
                 f'font-weight="bold" fill="{C_NODE_TITLE}">{esc(s)}</text>')
    path_size = 8.3
    while mono_text_width(path, path_size) > NODE_W - 16 and path_size > 6.0:
        path_size -= 0.3
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + PATH_Y:.1f}" text-anchor="middle" '
             f'font-family="monospace" font-size="{path_size:.1f}" '
             f'fill="{C_NODE_PATH}">{esc(path)}</text>')
    for ci, cline in enumerate(_CLAIM_LINES[nid]):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + CLAIM_TOP + ci * 12:.1f}" '
                 f'text-anchor="middle" font-family="sans-serif" font-size="{CLAIM_FONT}" '
                 f'fill="{C_NODE_SUB}">{esc(cline)}</text>')
    # 徽标右对齐钉在框内：居中挂角会让长站牌探出框外压到相邻节点
    _bw_station = cjk_text_width(station, BADGE_FONT) + 2 * BADGE_PAD_X
    badge_svg, _bw = badge(x + NODE_W - 8 - _bw_station / 2, y, station)
    L += badge_svg

# 底部「阅读路线」：模板要求的多条读法条(高亮=实线蓝 / 次要=虚线灰)。
# 蛇形版图里列号 ≠ 阅读序，故胶囊按「站序槽位」等分排布：同一站在各条路线里
# 上下对齐，缺席的站留空——一眼看出某条路线跳过了哪几站。
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线（胶囊=图上站牌，按 ①→⑩ 站序排；实线蓝=推荐 / 虚线灰=次要）")}</text>')
_name_w = max(cjk_text_width(r[0], 12.0) for r in ROUTES)
SLOT_L = 16 + _name_w + 14
SLOT_R = w - EDGE_MARGIN - 6
SLOT_W = (SLOT_R - SLOT_L) / len(STATION_ORDER)
_max_badge_w = max(cjk_text_width(s, BADGE_FONT) + 2 * BADGE_PAD_X for s in STATION_ORDER)
assert _max_badge_w <= SLOT_W, f"站牌胶囊 {_max_badge_w:.0f}px 放不进槽位 {SLOT_W:.0f}px"
SLOT_CX = [SLOT_L + i * SLOT_W + SLOT_W / 2 for i in range(len(STATION_ORDER))]

for ri, (rname, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12.0" '
             f'fill="{C_NODE_TITLE}">{esc(rname)}</text>')
    idxs = [STATION_ORDER.index(s) for s in stops]
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{SLOT_CX[idxs[0]]:.1f}" y1="{ry:.1f}" x2="{SLOT_CX[idxs[-1]]:.1f}" '
             f'y2="{ry:.1f}" stroke="{C_MAIN if hi else C_ROUTE_DIM}" '
             f'stroke-width="{3 if hi else 1.5}"{dash}/>')
    for i, s in zip(idxs, stops):
        L += badge(SLOT_CX[i], ry, s)[0]

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  ({w}x{h}, aspect {w / h:.2f}:1)")
