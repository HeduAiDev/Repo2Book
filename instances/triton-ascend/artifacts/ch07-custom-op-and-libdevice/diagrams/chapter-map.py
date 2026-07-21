#!/usr/bin/env python3
"""ch07「本章地图」——自定义算子注册 + libdevice 数学库的源码剖面图。

本章是**自然标题章**(chapter.md 无 `## N.M` 编号,只有自然标题,如"注册这道门：
八条断言，一次抄写，一次入表"/"从名字到 IR：`al.custom` 一次调用的六步")——按
illustrator 契约,**禁用 §N.M 徽标**,站牌一律改用标题词本身的真实子串(如
"注册这道门"取自标题「注册这道门：八条断言……」,"从名字到 IR"取自标题
「从名字到 IR：`al.custom`……」)。

两条泳道 = 本章两条独立主线(chapter.md 按此顺序讲):
  泳道 0:自定义算子——register_custom_op 注册 → custom_semantic 六步调用
         (内部分支:_get_op_class 命中/未命中→_builtin_custom_op 哑类兜底)
         → _index_select 真实样例(__init__ 十一断言 + arg_type 动态定型)
         → _make_attrs(core/pipe/mode 翻三条 hivm 属性)→ 最终 emit 的 hivm.custom。
         列序是真实调用顺序(_index_select 先实例化,_make_attrs 才用它建属性)——
         **这与正文四个自然标题的先后顺序不完全相同**:正文顺序是"注册这道门"(L19)
         →"从名字到 IR"(L94)→"core/pipe/mode"(L195)→"真实的注册样例"(L274),
         即 core/pipe/mode 一节在前、_index_select 样例一节在后,与调用顺序里
         _index_select(实例化)先于 _make_attrs 恰好相反。上一轮盲审 FAIL 就是
         因为底部"阅读路线"误把这两站按图上物理列序排、没按正文真实顺序排。
         处置:上方主流程图**保留**调用顺序(不改,这是真实调用顺序,不是错误);
         下方"阅读路线"改按正文小节先后独立排序(不再机械对齐物理列),并在
         该行标题里明写"按正文小节先后"以区别于主流程图的"调用顺序",
         避免两行隐含同一顺序却互相矛盾。
  泳道 1:libdevice 数学库——libdevice.py 四类形态 → extern_elementwise(基座
         dispatch,dtype 元组查表)→ __init__.py(覆盖 + 复用拼出 al.libdevice)。
         libdevice.py 里"从不碰菜单"/"@jit 组合"两类函数不经过 dispatch、直接
         被 __init__.py 挂进命名空间——画一条次要(虚线灰)旁路边,为避免与同排的
         extern_elementwise 节点重叠,用二次贝塞尔曲线下绕过去(唯一一条跨列旁路边)。
         extdisp/libdev 两条边汇入 __init__.py 是"两类实现独立汇入,非顺序因果",
         写进该节点 phrase 明示,不靠汇聚箭头隐含顺序。

模板:.claude/skills/svg-diagram/references/example-chapter-map.py 的不可变视觉语言
(徽标胶囊/入口绿-出口橙-主线蓝/次要虚线灰/cjk_text_width)照搬,只改 DATA;
沿用 ch10 chapter-map.py 的两个通用文本适配工具(fit_size/wrap_symbol,本章
register_custom_op/_builtin_custom_op/extern_elementwise 等符号名较长)与
"按两端实际坐标通用判定边路由"的写法;新增:①支持每条边 secondary(次要虚线灰)
标记;②支持多个入口/出口桩(两条泳道各自的调用方/emit 终点);③"跨列同排"边
一律走二次贝塞尔下绕,避免直线穿过中间节点。

六项自查(渲染→Read PNG 亲眼看后如实记录):见 figure-manifest.json 该图 selfcheck。

用法:python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算:全角(ord>0x2E80)按 1.0×size,半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def fit_size(text, max_w, sizes):
    """从大到小试字号,返回第一个能让 text 单行塞进 max_w 的字号;都不行则返回最小字号。"""
    for size in sizes:
        if cjk_text_width(text, size) <= max_w:
            return size
    return sizes[-1]


def wrap_symbol(text, max_w, sizes):
    """符号名较长时的通用换行:先试单行从大到小的字号;仍塞不下,在 '_'/'.' 边界
    二分成两行(挑一个让两行里"更长的那行"最短的切点),用最小字号。返回 (lines, size)。"""
    for size in sizes:
        if cjk_text_width(text, size) <= max_w:
            return [text], size
    size = sizes[-1]
    candidates = ([i + 1 for i, c in enumerate(text) if c == '_']
                  + [i + 1 for i, c in enumerate(text) if c == '.'])
    if not candidates:
        candidates = [len(text) // 2]
    best = None
    for idx in candidates:
        if idx <= 0 or idx >= len(text):
            continue
        a, b = text[:idx], text[idx:]
        wd = max(cjk_text_width(a, size), cjk_text_width(b, size))
        if best is None or wd < best[0]:
            best = (wd, a, b)
    if best is None:
        return [text], size
    return [best[1], best[2]], size


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["自定义算子:register_custom_op 与调用", "libdevice 数学函数库"]

FONT_SIZES = (12.5, 11.5, 10.5, 9.5, 8.5)

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌——自然标题摘要,禁用 §N.M)
NODES = [
    ("register", 0, 0, 0, "register_custom_op", "八条断言→写入注册表(只增不改)", "注册这道门"),
    ("csem",      0, 1, 0, "custom_semantic",    "六步:查表→实例化→emit",         "从名字到 IR"),
    ("getop",     0, 1, 1, "_get_op_class",       "查注册表,命中→真实类",         "从名字到 IR"),
    ("dummy",     0, 1, 2, "_builtin_custom_op",  "未命中+__builtin_前缀→现造哑类", "从名字到 IR"),
    ("idxsel",    0, 2, 0, "_index_select",       "__init__十一断言+arg_type动态定型", "真实的注册样例"),
    ("attrs",     0, 3, 0, "_make_attrs",         "core/pipe/mode→三条hivm属性",   "core / pipe / mode"),
    ("emitir",    0, 4, 0, "hivm.custom",         "由 _builder.create_custom_op 建出", "从名字到 IR"),

    ("libdev",    1, 0, 0, "libdevice.py",        "37函数四类形态,__hmf_共66处",   "四类形态的拼装"),
    ("extdisp",   1, 1, 0, "extern_elementwise",  "dtype元组查表,查不到就报错",     "菜单的边界"),
    ("nsinit",    1, 2, 0, "__init__.py",         "汇入(非顺序因果):dispatch结果+直连纯IR/@jit", "命名空间是拼出来的"),
]
# (src_id, dst_id, secondary) —— secondary=True 画次要虚线灰(非主干路径)
EDGES = [
    ("register", "csem", False),
    ("csem", "getop", False),
    ("getop", "dummy", True),      # 未命中的兜底分支,次要
    ("csem", "idxsel", False),
    ("idxsel", "attrs", False),
    ("attrs", "emitir", False),

    ("libdev", "extdisp", False),
    ("extdisp", "nsinit", False),
    ("libdev", "nsinit", True),    # 从不碰菜单/@jit 组合两类,跳过 dispatch 直连命名空间
]
# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("自定义算子路线",
     [(0, "注册这道门"), (1, "从名字到 IR"), (3, "core / pipe / mode"), (2, "真实的注册样例")], True),
    ("libdevice 路线",
     [(0, "四类形态的拼装"), (1, "菜单的边界"), (2, "命名空间是拼出来的")], False),
]
LEGEND = [
    ("#22c55e", "入口:用户 kernel 源码(写类 / 调 al.custom / 调 al.libdevice 函数)"),
    ("#3b82f6", "章内主线调用/数据流"),
    ("#94a3b8", "次要分支(免注册兜底 / 跳过菜单直连纯 IR)"),
    ("#f97316", "出口:emit 完成的 IR(ttadapter / ttir 阶段)"),
]
TITLE = "自定义算子框架与 Ascend libdevice —— 源码剖面(third_party/ascend/language/cann/)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_SECONDARY = "#94a3b8"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 190, 70
COL_GAP, ROW_GAP = 42, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
PAD_L = PAD_R = EDGE_MARGIN + max(STUB_W, 100) + 32  # 100 留给最长的 stub 标签(IR(ttadapter))
LANE_LABEL_H, BAND_PAD = 24, 26
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


def badge_w(text):
    """站牌胶囊宽度——按文字自适应,不用固定 BADGE_W 截断(避免中文站牌被裁)。"""
    return max(BADGE_W, cjk_text_width(text, 11) + 14)


def stub_w(text):
    """入口/出口接口桩宽度——按文字自适应(避免长英文标签如 IR(ttadapter) 溢出固定 STUB_W)。"""
    return max(STUB_W, cjk_text_width(text, 10.5) + 16)


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy)——自然标题摘要,非 §N.M。"""
    bw = badge_w(text)
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN), ("Secondary", C_SECONDARY))
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
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="10.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 10.5) + 26

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

# 入口/出口接口桩(两条泳道各自一对):入口挂各泳道首节点,出口挂各泳道尾节点
ENTRY_STUBS = [("register", "kernel 源码"), ("libdev", "kernel 源码")]
EXIT_STUBS = [("emitir", "IR(ttadapter)"), ("nsinit", "IR(ttir)")]
for nid, label in ENTRY_STUBS:
    ex, ey = NODE_XY[nid]; ey += NODE_H / 2
    sw = stub_w(label)
    L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{sw:.1f}" height="{STUB_H}" '
             f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
    L.append(f'<text x="{EDGE_MARGIN + sw / 2:.1f}" y="{ey + 4:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#166534">{esc(label)}</text>')
    L.append(f'<line x1="{EDGE_MARGIN + sw:.1f}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
             f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
for nid, label in EXIT_STUBS:
    xx, xy = NODE_XY[nid]; xy += NODE_H / 2
    sw = stub_w(label)
    sx = w - EDGE_MARGIN - sw
    L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{sw:.1f}" height="{STUB_H}" '
             f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
    L.append(f'<text x="{sx + sw / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc(label)}</text>')
    L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
             f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边:按两端实际坐标通用判定——同列(x 相同)按 y 前后竖直附着;不同列且相邻
# (|Δcol|<=1)走对角附着;不同列且跨列(|Δcol|>=2,同排 y 相同)会与中间节点共线,
# 改走二次贝塞尔下绕(本章仅 libdev→nsinit 一条,跳过中间的 extdisp)。
for src, dst, secondary in EDGES:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    color = C_SECONDARY if secondary else C_MAIN
    marker = "mSecondary" if secondary else "mMain"
    dash = ' stroke-dasharray="6,4"' if secondary else ''
    same_row = abs((ys_ - yd)) < 0.01
    col_src = round((xs_ - PAD_L) / (NODE_W + COL_GAP))
    col_dst = round((xd - PAD_L) / (NODE_W + COL_GAP))
    if same_row and abs(col_dst - col_src) >= 2:
        p1 = (xs_ + NODE_W, ys_ + NODE_H / 2)
        p2 = (xd, yd + NODE_H / 2)
        dip = NODE_H / 2 + 24
        mx, my = (p1[0] + p2[0]) / 2, max(p1[1], p2[1]) + dip
        L.append(f'<path d="M{p1[0]:.1f},{p1[1]:.1f} Q{mx:.1f},{my:.1f} {p2[0]:.1f},{p2[1]:.1f}" '
                  f'fill="none" stroke="{color}" stroke-width="2"{dash} marker-end="url(#{marker})"/>')
        continue
    if xs_ == xd:
        if yd > ys_:
            p1 = (xs_ + NODE_W / 2, ys_ + NODE_H)
            p2 = (xd + NODE_W / 2, yd)
        else:
            p1 = (xs_ + NODE_W / 2, ys_)
            p2 = (xd + NODE_W / 2, yd + NODE_H)
    else:
        p1 = (xs_ + NODE_W, ys_ + NODE_H / 2)
        p2 = (xd, yd + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{color}" stroke-width="2"{dash} marker-end="url(#{marker})"/>')

# 节点(圆角框 + 真实符号名(必要时自动换行/缩字号) + 一行短语 + 右上角站牌)
SYM_MAXW = NODE_W - 14
PHR_MAXW = NODE_W - 12
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_lines, sym_size = wrap_symbol(symbol, SYM_MAXW, FONT_SIZES)
    cx = x + NODE_W / 2
    if len(sym_lines) == 1:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.36:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
    else:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.28:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.28 + sym_size + 2:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[1])}</text>')
    phr_size = fit_size(phrase, PHR_MAXW, (10.5, 9.5, 8.5, 7.5))
    L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.86:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{phr_size}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    bw = badge_w(sec)
    L += badge(x + NODE_W - bw / 2 + 8, y, sec)

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点(自然标题摘要,非 §N.M)
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(按正文小节先后顺序排列,区别于上方主流程图按代码调用顺序排列;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    # 站牌沿路线的左右位置按 stops 的**列表顺序**(=正文小节先后)均匀排开,不是
    # 直接照搬 col 值——本章"自定义算子路线"的正文顺序(core/pipe/mode 先于
    # 真实的注册样例)与上方主流程图的物理列序(_index_select 先于 _make_attrs,
    # 即调用顺序)相反,若仍按 col 值取横坐标,两个后段站牌会原地对调回错误顺序。
    # 用 min/max 列的物理跨度定端点,再按列表顺序等分,视觉顺序才等于叙事顺序。
    cols_in_route = [c for c, _ in stops]
    x_lo = COLX[min(cols_in_route)] + NODE_W / 2
    x_hi = COLX[max(cols_in_route)] + NODE_W / 2
    n = len(stops)
    xs_positions = [x_lo + i * (x_hi - x_lo) / (n - 1) for i in range(n)] if n > 1 else [x_lo]
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{xs_positions[0]:.1f}" y1="{ry:.1f}" x2="{xs_positions[-1]:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for (col, sec), bx in zip(stops, xs_positions):
        L += badge(bx, ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  ({w}x{h}, aspect {w / h:.2f}:1)")
