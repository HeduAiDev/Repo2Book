#!/usr/bin/env python3
"""本章地图 — ch18《双语桥：libtriton 的 pybind11 绑定层》源码剖面图。

本章把 python/src/ 这道 Python↔C++ 接缝拆到底，按 chapter.md 实际的四个 `##`
一级小节（不含小结）分四条并列泳道，每条泳道对应一个真实 .cc/.h 文件：
  ir.cc          —— create_* 的双语绑定链 + TritonOpBuilder 底座（本章第①②节）
  main.cc        —— PYBIND11_MODULE 一次 import 装配整个 _C 扩展（第③节）
  passes.cc/.h   —— add_* pass 挂载口，只挂不跑（第④节）
  interpreter.cc —— 按掩码聚散(gather/scatter)一瞥（第⑤节）

■ 不可变(全书统一视觉语言):
  1. 站牌胶囊:圆角矩形(pill),fill #eef2ff / stroke #6366f1;贴节点右上角。
  2. 入口/出口接口桩:绿 #22c55e(入口) / 橙 #f97316(出口)。
  3. 节点间主线边 = 蓝 #3b82f6。
  4. 底部路线条:高亮=实线蓝(粗)/次要=虚线灰 #94a3b8(细)。
  5. >2 种语义色须画图例。
  6. 文本宽度估算一律用 cjk_text_width(),不用半角系数硬乘 len(s)。

■ 本章特有(自然标题章,无 §N.M 编号——按 illustrator 契约:禁用 §N.M 徽标,
  站牌改用标题词本身,逐字取自 chapter.md 真实 `## ...` 标题的子串):
  - 5 个站牌文本分别取自 5 个真实 `## ` 标题冒号前的那一截(verbatim 子串)：
    "create_* 的双语绑定链"(该标题本身无冒号,整段照抄)、"TritonOpBuilder"
    (取自 `## TritonOpBuilder：\`_builder\` 的 C++ 真身`)、"PYBIND11_MODULE"
    (取自 `## PYBIND11_MODULE：一次 import 装配整个 _C 扩展`)、"passes.cc"
    (取自 `## passes.cc：pass 从 Python 被挂上的那个口`)、"interpreter.cc"
    (取自 `## interpreter.cc：按掩码聚散的一瞥`)。"小结"一节不设站牌/节点
    (它是收束陈述,不是一个可定位的机制讲解站)。
  - 节点符号名全部是 dossier.json code_spine/mechanisms 或正文内嵌代码块里
    逐字出现的真实标识符(create_make_range/create_get_program_id/
    TritonOpBuilder/create<OpTy>()/PYBIND11_MODULE/INIT_BACKEND/
    ADD_PASS_WRAPPER_0/init_triton_passes/load/store)，不杜撰。
  - 4 条泳道内部的蓝色边是**真实的源码调用/包含关系**(非仅阅读顺序)：
    create_make_range/create_get_program_id 两个 lambda 都调用 self.create<OpTy>()
    这个 TritonOpBuilder 的模板方法(ir.cc:L96-99)；PYBIND11_MODULE 函数体最后
    一行就是 FOR_EACH_P(INIT_BACKEND,…)；ADD_PASS_WRAPPER_0 宏在各分组函数里
    展开出 add_* 绑定，这些分组再被 init_triton_passes 逐个 def_submodule；
    load 与 store 是 interpreter.cc 里紧邻的一对对称实现(gather/scatter)。
  - 列(col)在四条泳道间复用(同一 x 坐标在不同泳道对应不同真实节点，是
    并列关系而非同一调用链的延伸)——与 ch09 chapter-map 先例一致，这样
    4 条独立泳道不必线性顺排导致画布超宽。
  - 底部两条路线直接复用章节开篇「只想认路」选读指引原句的信息结构：
    "只想认路"=读完 create_* 绑定链 + PYBIND11_MODULE 两节即拼出全貌(高亮实线)；
    "按需延伸"=passes.cc 与 interpreter.cc 两节是往两侧的延伸,可跳读(虚线灰)。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录):
  claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
  arrows_attached=True     cjk_rendered=True         reading_order_clear=True

用法: python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算:全角(ord>0x2E80)按 1.0×size,半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def fit_size(text, max_w, base, min_size):
    """按 max_w 反解一个不超出的字号(单行,不换行)。"""
    unit = cjk_text_width(text, 1.0)
    if unit <= 0:
        return base
    return max(min_size, min(base, max_w / unit))


def badge_width(text, font_size):
    """站牌胶囊宽度——badge 文本恒为 bold 且本章多是全大写+下划线的宏/类名
    (PYBIND11_MODULE/ADD_PASS_WRAPPER_0 这类),比 cjk_text_width 校准基准的常规
    大小写混排字重更宽,经渲染实测(Read PNG 发现 PYBIND11_MODULE 溢出胶囊右边缘)
    按 1.22 倍系数放宽,重渲后不再溢出。"""
    return cjk_text_width(text, font_size) * 1.22 + BADGE_PAD_X * 2


# ==================== DATA(可变:本章数据) ====================
LANES = [
    "ir.cc：create_* 绑定链 + TritonOpBuilder 底座",
    "main.cc：PYBIND11_MODULE 入口装配",
    "passes.cc / passes.h：pass 挂载口",
    "interpreter.cc：按掩码聚散",
]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌文本[取自真实标题子串])
NODES = [
    ("create_make_range", 0, 0, 0, "create_make_range",
     "Python 名↔C++ lambda 的完整绑定样板",       "create_* 的双语绑定链"),
    ("create_get_program_id", 0, 0, 1, "create_get_program_id",
     "lambda 先校验 axis 越界再建 op",             "create_* 的双语绑定链"),
    ("triton_op_builder", 0, 1, 0, "TritonOpBuilder",
     "_builder 的 C++ 真身,记住 lastLoc",          "TritonOpBuilder"),
    ("create_template", 0, 2, 0, "create<OpTy>()",
     "取 lastLoc→builder->create<OpTy>(loc,…)",    "TritonOpBuilder"),

    ("pybind11_module", 1, 1, 0, "PYBIND11_MODULE",
     "声明模块 libtriton,逐个 def_submodule",       "PYBIND11_MODULE"),
    ("init_backend", 1, 2, 0, "INIT_BACKEND",
     "CMake 注入的后端元组逐个展开注册",           "PYBIND11_MODULE"),

    ("add_pass_wrapper", 2, 0, 0, "ADD_PASS_WRAPPER_0",
     "pass 名↔pm.addPass(builder()) 的 lambda",    "passes.cc"),
    ("init_triton_passes", 2, 1, 0, "init_triton_passes",
     "6 个子分组按 pipeline 阶段各挂一次",         "passes.cc"),

    ("interp_load", 3, 2, 0, "load",
     "按 mask[i] 逐元素 gather",                    "interpreter.cc"),
    ("interp_store", 3, 3, 0, "store",
     "按 mask[i] 逐元素 scatter （对称）",           "interpreter.cc"),
]

EDGES = [  # (src_id, dst_id) —— 真实源码调用/包含关系,统一主线蓝
    ("create_make_range", "triton_op_builder"),
    ("create_get_program_id", "triton_op_builder"),
    ("triton_op_builder", "create_template"),
    ("pybind11_module", "init_backend"),
    ("add_pass_wrapper", "init_triton_passes"),
    ("interp_load", "interp_store"),
]

# (路线名, [(列, 站牌文本), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("只想认路:两节拼出全貌",
     [(0, "create_* 的双语绑定链"), (1, "PYBIND11_MODULE")], True),
    ("按需延伸:两节旁支",
     [(1, "passes.cc"), (3, "interpreter.cc")], False),
]
LEGEND = [
    ("#22c55e", "入口:Python 前端调用 / import triton 触发绑定"),
    ("#3b82f6", "章内源码调用/挂载/配对关系(逐节讲解见正文)"),
    ("#f97316", "出口:回到 Python （ir.value / numpy 数组继续使用）"),
]
TITLE = "第 18 章 · python/src/ 双语桥 剖面(源码走线 + 讲解站牌)"

# ==================== 不可变:配色 ====================
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ==================== 几何常量(全计算,零魔数) ====================
NODE_W, NODE_H = 200, 64
COL_GAP, ROW_GAP = 34, 18
EDGE_MARGIN, STUB_W, STUB_H = 14, 66, 30
LANE_LABEL_H, BAND_PAD = 22, 10
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 32, 24, 14
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_H, BADGE_PAD_X = 20, 10
BADGE_FONT = 11
ROUTE_NAME_X, ROUTE_NAME_FONT = 16, 12

# 左边距须同时满足两件事:①给入口接口桩留够空间;②路线区第一列的站牌不能压住左侧
# 路线名文字——按路线名文字的真实渲染宽度 + 落在第 0 列的站牌的真实渲染宽度反算
# 所需净空(两段文字各自的宽度都来自 cjk_text_width,不用手写魔数)。
_stub_pad = EDGE_MARGIN + STUB_W + 26
_route_name_w = max((cjk_text_width(name, ROUTE_NAME_FONT) for name, _, _ in ROUTES), default=0)
_col0_badge_texts = [t for _, stops, _ in ROUTES for c, t in stops if c == 0]
_col0_badge_w = max((cjk_text_width(t, BADGE_FONT) + BADGE_PAD_X * 2 for t in _col0_badge_texts), default=0)
_route_gap = 30
_route_col0_clear = ROUTE_NAME_X + _route_name_w + _route_gap + _col0_badge_w / 2 - NODE_W / 2
PAD_L = max(_stub_pad, _route_col0_clear)
PAD_R = _stub_pad

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


def badge_width(text):
    """站牌胶囊宽度——badge 文本恒为 bold 大写居多(PYBIND11_MODULE/ADD_PASS_WRAPPER_0
    这类全大写+下划线标识符),比 cjk_text_width 校准基准的常规大小写混排字重更宽,
    经渲染实测(Read PNG 发现 PYBIND11_MODULE 溢出胶囊)按 1.22 倍系数放宽,不再溢出。"""
    return cjk_text_width(text, BADGE_FONT) * 1.22 + BADGE_PAD_X * 2


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy)——宽度按文本动态算(本章站牌是完整词组,非定长短码)。"""
    bw = badge_width(text)
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.1"/>',
        f'<text x="{cx:.1f}" y="{cy + 3.8:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="{BADGE_FONT}" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
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
         f'font-size="14.5" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')

# 图例(3 种语义色 → 必须画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 13
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11) + 30

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩
ex, ey = NODE_XY["create_make_range"]; ey += NODE_H / 2
xx, xy = NODE_XY["interp_store"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.2"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.2"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 章内真实调用/包含边,多条边汇入同一节点时终点 y 错开
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
    y_offset = (i - (n - 1) / 2) * 14 if n > 1 else 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌),字号按文本长度自适应收缩避免溢出
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_size = fit_size(symbol, NODE_W - 18, 13, 8.5)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.40:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{sym_size:.1f}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    ph_size = fit_size(phrase, NODE_W - 16, 10.5, 7.8)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{ph_size:.1f}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    bw = cjk_text_width(sec, BADGE_FONT) + BADGE_PAD_X * 2
    L += badge(x + NODE_W - bw / 2 + 10, y, sec)

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上讲解站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="{ROUTE_NAME_X}" y="{ry + 4:.1f}" font-family="sans-serif" '
              f'font-size="{ROUTE_NAME_FONT}" fill="{C_NODE_TITLE}">{esc(name)}</text>')
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
print(f"wrote {out}  ({w:.0f}x{h:.0f}, ratio {w / h:.2f}:1)")
