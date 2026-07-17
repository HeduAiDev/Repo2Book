#!/usr/bin/env python3
"""ch16《CodeGenerator 翻译框架——把 Python AST 逐节点翻成 tt.* IR 的那台机器》
—— 本章地图(源码剖面图)。

本章是自然标题章(chapter.md 无 `## N.M` 编号,标题用 §1..§7 单层记号 + 自然语言,
如"§5 调用怎么分诊：visit_Call 三分派")——按契约禁用 §N.M 徽标(带小数点的两级
编号)。本章的 §N 是单层编号、字面就出现在标题文本里,站牌沿用这套章节自己的
记号(如"§5"),与 ch14/ch15 是同一条"自然标题"规则下的同款呈现(先例见
ch14/ch15 的 chapter-map.py 文件头说明)。

剖面(3 条泳道,按讲解顺序从上到下):
  ① 入口与分派骨架 + 报错路径(§1, §7):ast_to_ttir(拆料造 CodeGenerator)→
     visit(分派外壳,钉 loc、按类型分派);shell 另分一条虚线红旁支到
     CompilationError——非本类异常在 visit 里被就地包成它,是异常路径而非
     主线数据流,故单独用红色虚线画在 shell 正下方,不占主线宽度。
  ② 两个世界·名字与运算符下降(§2–§4):`_is_constexpr`/`_is_triton_tensor`
     二分(本章唯一主线心智模型,配色特殊——靛紫,呼应它"贯穿全器"的地位)→
     name_lookup(三级查找+constexpr 全局守卫)→ _apply_binary_method(反射
     分派到 tensor.__add__)。
  ③ 调用分诊 f4 命门 + 函数建 IR 性能命门(§5–§6):visit_Call(f4 回收——三/
     四出口分诊)→ call_JitFunction(JITFunction 内联)→ visit_FunctionDef
     (建 tt.func,constexpr 跳 idx)→ set_arg_attr(divisibility 落 IR,访存
     向量化源头)。

配色:绿 #22c55e=入口(被 ch14 compile()/make_ir 调用);蓝 #3b82f6=章内主线
调用边/常规节点;靛紫 #7c3aed(填充 #ede9fe)=本章唯一主线心智模型节点(§2 的
constexpr↔tensor 二分,全章 4 处分派都是它的推论);红 #ef4444 虚线(填充
#fee2e2)=异常路径(非主线数据流,标"旁路");橙 #f97316=出口(generator.module
经 ast_to_ttir 交回 ch14 compile())。5 种语义色均入图例。

布局手法:沿用 ch15 的"每条泳道局部列从 0 重新起笔 + 折角连接线跨泳道"手法
(未继续 ch14 的全局共享列号——本章 3 条泳道节点数不等 2/3/4,局部列更省宽度、
更符合"每层各自讲完再翻到下一层"的阅读体验);沿用 ch14 的 badge_w 自适应
胶囊宽度(§ 记号本身很短,不需要但保持同款实现)与 wrap_symbol 长符号自动换行。
新增:同泳道内的竖直"branch"边(shell→error),不参与折行逻辑,画法独立。

阅读路线(4 条,直接对应正文开篇"选读指引"段的四句话,而非另编路线充数):
  1. 通读全程 §1→§7(推荐,实线蓝)
  2. 只抓主线心智模型——§2 就够(虚线灰,quick jump)
  3. 回收第 1 章"三岔口"模型——跳 §5(虚线灰)
  4. 只看性能落点——看 §6(虚线灰)
路线用独立等距时间轴(不复用节点局部列坐标,同 ch15 手法),因为跨泳道的局部
列坐标系不共享,没法直接借用节点 x 做路线定位。

六项自查(渲染→Read PNG 亲眼看后如实记录):见同目录 figure-manifest.json 该图
selfcheck 字段。

用法:python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算:全角(ord>0x2E80)按 1.0×size,半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def wrap_symbol(text, max_w, sizes):
    """符号名/短语较长时的通用换行:先试单行从大到小的字号;仍塞不下,在
    ' '/'_'/'('/','/'/' 边界二分成两行(挑一个让两行里"更长的那行"最短的切
    点),用最小字号。返回 (lines, size)。"""
    for size in sizes:
        if cjk_text_width(text, size) <= max_w:
            return [text], size
    size = sizes[-1]
    candidates = ([i + 1 for i, c in enumerate(text) if c == ' ']
                  + [i + 1 for i, c in enumerate(text) if c == '_']
                  + [i for i, c in enumerate(text) if c == '(']
                  + [i + 1 for i, c in enumerate(text) if c == '/']
                  + [i + 1 for i, c in enumerate(text) if c == ','])
    if not candidates:
        candidates = [len(text) // 2]
    best = None
    for idx in candidates:
        if idx <= 0 or idx >= len(text):
            continue
        a, b = text[:idx], text[idx:]
        w = max(cjk_text_width(a, size), cjk_text_width(b, size))
        if best is None or w < best[0]:
            best = (w, a, b)
    if best is None:
        return [text], size
    return [best[1], best[2]], size


# ---------------- DATA(本章数据) ----------------
LANES = [
    "入口与分派骨架 · 报错路径(§1, §7)",
    "两个世界 · 名字与运算符下降(§2–§4)",
    "调用分诊 f4 命门 · 函数建 IR 性能命门(§5–§6)",
]

FONT_SIZES = (12.5, 11.5, 10.5, 9.5, 8.5)

# (节点id, 泳道下标, 泳道内局部列, 泳道内行号, 真实符号名, 一行短语, §编号, kind)
# kind: "normal" 常规 / "core" 本章唯一主线心智模型(靛紫高亮) / "branch" 异常旁路(红)
NODES = [
    ("entry", 0, 0, 0, "ast_to_ttir",
     "入口:拆 constants/arg_types/fn_attrs,new CodeGenerator", "§1", "normal"),
    ("shell", 0, 1, 0, "visit",
     "分派外壳:钉 MLIR loc,super().visit 按类型分派", "§1", "normal"),
    ("error", 0, 1, 1, "CompilationError",
     "异常旁路:非本类异常被就地包成带源码摘录的它", "§7", "branch"),

    ("dichotomy", 1, 0, 0, "_is_constexpr / _is_triton_tensor",
     "全章判据:编译期折叠(不建 op)↔ 运行期建 SSA op", "§2", "core"),
    ("lookup", 1, 1, 0, "name_lookup",
     "local→global(只放行 constexpr)→builtin 三级查找", "§3", "normal"),
    ("binop", 1, 2, 0, "_apply_binary_method",
     "反射分派到 tensor.__add__(_builder=...)", "§4", "normal"),

    ("call", 2, 0, 0, "visit_Call",
     "f4 命门:static/JITFunction/builtin/纯 Python 四出口", "§5", "normal"),
    ("calljit", 2, 1, 0, "call_JitFunction",
     "constexpr 抽 constants,tensor 走 handle,mangle 内联", "§5", "normal"),
    ("funcdef", 2, 2, 0, "visit_FunctionDef",
     "建 tt.func:constexpr 跳 idx,5 参数→4 IR 位", "§6", "normal"),
    ("divis", 2, 3, 0, "set_arg_attr",
     "divisibility 落成 tt.divisibility,访存向量化源头", "§6", "normal"),
]
NODE_BY_ID = {n[0]: n for n in NODES}

# (src_id, dst_id, kind, label) —— kind: "main"=同泳道主线蓝实线 /
# "wrap"=跨泳道折角连接线(蓝虚线,标"续下一行") / "branch"=同泳道竖直旁路(红虚线)
EDGES = [
    ("entry", "shell", "main", None),
    ("shell", "dichotomy", "wrap", None),
    ("dichotomy", "lookup", "main", None),
    ("lookup", "binop", "main", None),
    ("binop", "call", "wrap", None),
    ("call", "calljit", "main", None),
    ("calljit", "funcdef", "main", None),
    ("funcdef", "divis", "main", None),
    ("shell", "error", "branch", "包异常"),
]

# 阅读路线:直接对应正文开篇"选读指引"的四句话,不另编路线充数。
# 单站路线额外给一个"锚定局部列"(anchor_col):把该站徽标直接画在图上对应节点
# 真实所在的局部列 x 下方——本章 3 条泳道都用同一套局部列刻度(col0/1/2/3 在
# 每条泳道里 x 都相同,因为每条泳道都从 PAD_L 起笔),于是"只读 §2"的徽标能
# 垂直对齐到上方 dichotomy 节点正下方,而不是随手扔在行尾——比隔行随机独立
# 定位更能让读者一眼对上"跳去图上哪一块"(比第一轮"单站徽标一律居右"版本更
# 清楚,那一版把 §2/§5/§6 都挤在行尾同一列,看不出跟哪个节点对应)。
ROUTES = [
    ("通读全程(按 §1→§7 顺序读完整台机器)",
     [("§1", None), ("§2", None), ("§3", None), ("§4", None),
      ("§5", None), ("§6", None), ("§7", None)], True),
    ("只抓主线心智模型——读 §2 就够",
     [("§2", 0)], False),
    ("回收第 1 章「三岔口」模型——跳 §5 看它怎么讲透",
     [("§5", 0)], False),
    ("只关心性能落点——看 §6(constexpr 不占位 + divisibility 落 IR)",
     [("§6", 2)], False),
]
LEGEND = [
    ("#22c55e", "入口:被 ch14 compile() 的 make_ir 阶段调用"),
    ("#3b82f6", "章内主线调用边 / 常规节点"),
    ("#7c3aed", "靛紫:本章唯一主线心智模型(constexpr↔tensor 二分)"),
    ("#ef4444", "红虚线:异常旁路(非主线数据流)"),
    ("#f97316", "出口:generator.module 经 ast_to_ttir 交回 ch14"),
]
TITLE = "第 16 章 · CodeGenerator 剖面——AST 逐节点分派翻成 tt.* IR 的那台机器"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
# 本章专属:核心心智模型节点(靛紫)+ 异常旁路节点(红)
C_CORE_FILL, C_CORE_STROKE = "#ede9fe", "#7c3aed"
C_BRANCH_FILL, C_BRANCH_STROKE = "#fee2e2", "#ef4444"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 196, 64
COL_GAP, ROW_GAP = 34, 18
EDGE_MARGIN, STUB_W, STUB_H = 14, 78, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 30
LANE_LABEL_H, BAND_PAD = 22, 14
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 32, 48, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_W, BADGE_H = 40, 20
WRAP_GAP = 26  # 折行处(跨泳道连接线)额外留白

# 每条泳道各自的局部列数/行数(不跨泳道共享列号 —— 见文件头说明)
cols_per_lane = [0] * len(LANES)
rows_per_lane = [0] * len(LANES)
for _id, lane, col, row, *_ in NODES:
    cols_per_lane[lane] = max(cols_per_lane[lane], col + 1)
    rows_per_lane[lane] = max(rows_per_lane[lane], row + 1)
n_cols_max = max(cols_per_lane)

# 画布宽度按最宽的一条泳道定(局部列坐标系,所有泳道共用同一套 x 刻度基准)
w = PAD_L + n_cols_max * NODE_W + (n_cols_max - 1) * COL_GAP + PAD_R

band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP
          for r in rows_per_lane]
band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for i, bh in enumerate(band_h):
    if i > 0:
        _cum += WRAP_GAP
    band_top.append(_cum)
    _cum += bh
lanes_bottom = _cum

NODE_XY = {}
for nid, lane, col, row, *_ in NODES:
    x = PAD_L + col * (NODE_W + COL_GAP)
    y = band_top[lane] + LANE_LABEL_H + BAND_PAD + row * (NODE_H + ROW_GAP)
    NODE_XY[nid] = (x, y)

routes_top = lanes_bottom + 8
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge_w(text):
    """站牌胶囊宽度——按文字自适应,不用固定 BADGE_W 截断(本章 §N 很短,通常
    用不到扩宽,但保持与 ch14 同款实现,行为一致)。"""
    return max(BADGE_W, cjk_text_width(text, 11) + 14)


def badge(cx, cy, text):
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
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN),
                        ("Branch", C_BRANCH_STROKE))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(5 色,两行摆放避免单行过宽)
_legend_rows = [LEGEND[:3], LEGEND[3:]]
_ly = TOP_PAD + TITLE_H + 14
for row_items in _legend_rows:
    _lx = PAD_L
    for color, label in row_items:
        L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
        L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11" '
                 f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
        _lx += 20 + cjk_text_width(label, 11) + 26
    _ly += 18

# 泳道背景 + 标签
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 4:.1f}" font-family="sans-serif" '
             f'font-size="12" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')

# 入口/出口接口桩:入口挂 ast_to_ttir(被 ch14 compile()/make_ir 调用),
# 出口挂 set_arg_attr(本章主线终点:divisibility 落完 IR 属性后,
# generator.module 经 ast_to_ttir 交回 ch14)
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["divis"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#166534">{esc("compile()")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#9a3412">{esc("compile()")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边:main 同泳道横向;wrap 跨泳道折角(同 ch15 手法);branch 同泳道竖直旁路(新增)
for src, dst, kind, label in EDGES:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    if kind == "main":
        p1 = (xs_ + NODE_W, ys_ + NODE_H / 2)
        p2 = (xd, yd + NODE_H / 2)
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    elif kind == "wrap":
        p1 = (xs_ + NODE_W, ys_ + NODE_H / 2)
        mid_x = xs_ + NODE_W + 20
        p2 = (xd + NODE_W / 2, yd)
        path = (f'M {p1[0]:.1f},{p1[1]:.1f} L {mid_x:.1f},{p1[1]:.1f} '
                f'L {mid_x:.1f},{p2[1] - 14:.1f} L {p2[0]:.1f},{p2[1] - 14:.1f} '
                f'L {p2[0]:.1f},{p2[1]:.1f}')
        L.append(f'<path d="{path}" fill="none" stroke="{C_MAIN}" stroke-width="2" '
                  f'stroke-dasharray="5,3" marker-end="url(#mMain)"/>')
        L.append(f'<text x="{mid_x + 6:.1f}" y="{(p1[1] + p2[1] - 14) / 2:.1f}" font-family="sans-serif" '
                  f'font-size="9.5" fill="{C_ROUTE_DIM}">{esc("续下一行")}</text>')
    else:  # branch:同泳道内竖直旁路(shell 下方直落 error)
        p1 = (xs_ + NODE_W / 2, ys_ + NODE_H)
        p2 = (xd + NODE_W / 2, yd)
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_BRANCH_STROKE}" stroke-width="2" stroke-dasharray="6,4" '
                  f'marker-end="url(#mBranch)"/>')
        if label:
            L.append(f'<text x="{p1[0] + 8:.1f}" y="{(p1[1] + p2[1]) / 2:.1f}" '
                      f'font-family="sans-serif" font-size="9.5" fill="{C_BRANCH_STROKE}">{esc(label)}</text>')

# 节点(圆角框 + 真实符号名(必要时自动换行/缩字号) + 一行短语 + 右上角 §N 站牌)
SYM_MAXW = NODE_W - 16
PHR_MAXW = NODE_W - 14
for nid, lane, col, row, symbol, phrase, sec, kind in NODES:
    x, y = NODE_XY[nid]
    if kind == "core":
        fill, stroke = C_CORE_FILL, C_CORE_STROKE
    elif kind == "branch":
        fill, stroke = C_BRANCH_FILL, C_BRANCH_STROKE
    else:
        fill, stroke = C_NODE_FILL, C_NODE_STROKE
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    sym_lines, sym_size = wrap_symbol(symbol, SYM_MAXW, FONT_SIZES)
    cx = x + NODE_W / 2
    if len(sym_lines) == 1:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.32:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
    else:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.24:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.24 + sym_size + 2:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[1])}</text>')
    phr_lines, phr_size = wrap_symbol(phrase, PHR_MAXW, (9.5, 9, 8.5, 8))
    py0 = y + NODE_H * 0.58
    for k, pl in enumerate(phr_lines):
        L.append(f'<text x="{cx:.1f}" y="{py0 + k * (phr_size + 3):.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{phr_size}" fill="{C_NODE_SUB}">{esc(pl)}</text>')
    bw = badge_w(sec)
    L += badge(x + NODE_W - bw / 2 + 6, y, sec)

# 底部阅读路线:多站的通读路线用独立等距时间轴(各泳道局部列坐标系不共享,
# 直接借用会让多站路线定位失真;同 ch15 手法);单站的选读跳转路线改用
# anchor_col——把徽标钉在该节点真实所在的局部列 x 下方(见 ROUTES 注释)。
def _col_center(col):
    return PAD_L + col * (NODE_W + COL_GAP) + NODE_W / 2


L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌,对应正文同号小节;实线蓝=通读 / 虚线灰=选读跳转)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    row_top = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H
    name_y = row_top + 13
    ry = row_top + ROUTE_ROW_H - 15
    L.append(f'<text x="16" y="{name_y:.1f}" font-family="sans-serif" font-size="11" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    n_stops = len(stops)
    name_w = cjk_text_width(name, 11) + 28
    line_x0 = 16 + name_w + BADGE_W / 2
    route_x1 = w - PAD_R
    if n_stops > 1:
        route_x = [line_x0 + i * (route_x1 - line_x0) / (n_stops - 1) for i, _ in enumerate(stops)]
    else:
        sec0, anchor_col = stops[0]
        route_x = [_col_center(anchor_col) if anchor_col is not None else route_x1]
    dash = '' if hi else ' stroke-dasharray="6,4"'
    if n_stops > 1:
        L.append(f'<line x1="{route_x[0]:.1f}" y1="{ry:.1f}" x2="{route_x[-1]:.1f}" y2="{ry:.1f}" '
                  f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    else:
        L.append(f'<line x1="{line_x0:.1f}" y1="{ry:.1f}" x2="{route_x[0]:.1f}" y2="{ry:.1f}" '
                  f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for i, (sec, _anchor) in enumerate(stops):
        L += badge(route_x[i], ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  ({w:.0f}x{h:.0f}, ratio={w / h:.2f}:1)")
