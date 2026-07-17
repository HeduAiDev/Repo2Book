#!/usr/bin/env python3
"""ch19 本章地图——源码剖面图。

本章是 Part V 开篇的「识字课」：不重讲语义，只教怎么从 `.td`/C++ 定义认出一行
dump 长什么样。四条泳道 = 章内四层代码(方言注册 / 算子定义 / 类型系统 /
trait 语义)，圆角节点 = 真实符号名 + 一行短语，右上角挂站牌(本章自然标题，
无 `## N.M` 编号，故站牌用标题词而非 §N.M)，左右各一个接口桩(入口=打开
TRITON_KERNEL_DUMP 看到一段真实 dump，出口=认字完成、转下一章 ttg 层布局)，
底部两条阅读路线复用同一批站牌。

[FIX-ROUND-2] 上一轮盲审 FAIL:站牌顺序与正文实际标题顺序不一致(把「方言
注册口」错画成开篇后第 2 站、且「类型系统」错排在「tt.*词汇表」之前)。本轮
按正文真实标题顺序(一行 dump 三元组 L41 → tt.*词汇表 L162 → tt 层的类型
系统 L246 → trait 的性能承诺 L350 → 方言注册口与枚举字符串 L451 → 小结
L519)重排列号:entry col0(开篇) → make_range/load col1(拆一行dump) →
reduce/make_tensor_ptr col2(tt.*词汇表) → ptr_type/memdesc col3(类型系统)
→ tt_op/tensor_size/verify_layout col4(trait性能承诺) → dialect col5(方言
注册口，正文里是倒数第二节，画在 exit 前一列) → exit col6(小结)。

走线严格左→右单向递增列号，同一泳道内的相邻节点(如 make_range/load、
tensor_size/verify_layout)彼此并列、不互相连边，只各自向下一列节点收敛，
避免"同列回绕"的走线穿过节点本体。这条主线正是正文结尾「小结:识字之后」
七步的图形版，「阅读路线(完整通读)」的站牌顺序现在与正文标题顺序逐一对应。

■ 不可变(照搬模板视觉语言,只改 DATA 与几何常量):站牌胶囊 / 入口绿
  #22c55e-出口橙#f97316-主线蓝#3b82f6 / 高亮路线实线蓝、次要虚线灰 /
  cjk_text_width() 宽度估算。
■ 本章为自然标题(无 `## N.M` 编号),站牌一律用标题词本身,禁用 §N.M。
■ 几何常量(NODE_W/COL_GAP/PAD 等)按本章 7 列节点数据调小,以满足画布预算
  (宽 ≤1500 且宽高比 ≤2.6:1)——这是"可变"的布局参数,不是共享视觉语言。
■ 长符号名(如 VerifyTensorLayoutsTrait)按估算宽度动态缩小标题字号,避免
  文字越界(title_font_size())。

[自查记录见文件末尾注释：Read PNG 后逐项如实记录，不能凭想象填。]
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):全角按 1.0×size,半角按
    0.58×size,求和——中文标签若按半角系数算会算短,导致下一个图例压上来。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def title_font_size(symbol, max_w, base=13, floor=10):
    """长标识符(如 VerifyTensorLayoutsTrait)按估算宽度动态缩小字号,不许
    文字越界——符号名全 ASCII,按半角 0.58 系数估算。"""
    est = cjk_text_width(symbol, base)
    if est <= max_w:
        return base
    return max(floor, max_w / (cjk_text_width(symbol, 1.0) or 1))


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["方言注册层", ".td 算子定义层", "类型系统层", "trait 语义层"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌(标题词,自然标题章无 §))
# 列号 = 正文标题出现顺序:0 开篇 → 1 拆一行dump(L41) → 2 tt.*词汇表(L162)
# → 3 类型系统(L246) → 4 trait性能承诺(L350) → 5 方言注册口(L451，正文倒数
# 第二节) → 6 小结(L519)。
NODES = [
    ("entry", 0, 0, 0, "TRITON_KERNEL_DUMP",
     "把降级链逐层落盘的观察窗口", "开篇:认字动机"),
    ("make_range", 1, 1, 0, "TT_MakeRangeOp",
     "最小样本:两属性无操作数", "拆一行 dump"),
    ("load", 1, 1, 1, "TT_LoadOp",
     "复杂样本:可选操作数+默认属性", "拆一行 dump"),
    ("reduce", 1, 2, 0, "TT_ReduceOp",
     "带内联 region,通用合并算法", "tt.* 词汇表"),
    ("make_tensor_ptr", 1, 2, 1, "TT_MakeTensorPtrOp",
     "块指针构造口:打包父张量信息", "tt.* 词汇表"),
    ("ptr_type", 2, 3, 0, "PointerType",
     "标量类型:两种寻址模态之源", "类型系统"),
    ("memdesc", 2, 3, 1, "MemDescType",
     "唯一带 encoding 字段", "类型系统"),
    ("tt_op", 3, 4, 0, "TT_Op",
     "全体算子基类,自动挂 2 trait", "trait 性能承诺"),
    ("tensor_size", 3, 4, 1, "TensorSizeTrait",
     "上限 2^20,防寄存器爆炸", "trait 性能承诺"),
    ("verify_layout", 3, 4, 2, "VerifyTensorLayoutsTrait",
     "布局合法性统一闸门", "trait 性能承诺"),
    ("dialect", 0, 5, 0, "Triton_Dialect",
     "tt 前缀源+后端登记口", "方言注册口"),
    ("exit", 0, 6, 0, "TTGIR",
     "转下一章:ttg 层填 layout", "小结:识字之后"),
]
EDGES = [  # (src_id, dst_id) —— 章内讲解走线,统一主线蓝;src 列号恒 < dst 列号
    ("entry", "make_range"), ("entry", "load"),
    ("make_range", "reduce"), ("load", "reduce"), ("load", "make_tensor_ptr"),
    ("reduce", "ptr_type"), ("make_tensor_ptr", "memdesc"),
    ("ptr_type", "tt_op"), ("memdesc", "tensor_size"), ("memdesc", "verify_layout"),
    ("tt_op", "dialect"), ("tensor_size", "dialect"), ("verify_layout", "dialect"),
    ("dialect", "exit"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("完整通读(识字七步)", [(0, "开篇:认字动机"), (1, "拆一行 dump"), (2, "tt.* 词汇表"),
                       (3, "类型系统"), (4, "trait 性能承诺"), (5, "方言注册口"),
                       (6, "小结:识字之后")], True),
    ("按需查表(跳读)", [(2, "tt.* 词汇表"), (4, "trait 性能承诺")], False),
]
LEGEND = [("#22c55e", "入口:打开 TRITON_KERNEL_DUMP 看到的一段 dump"),
          ("#3b82f6", "章内讲解走线"),
          ("#f97316", "出口:认字完成,转下一章 ttg 布局")]
TITLE = "第 19 章 · tt.* 方言词汇表:从 .td 定义到 dump 认字(源码剖面图)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数;本章 7 列,调小以合画布预算) ----------------
NODE_W, NODE_H = 175, 60
COL_GAP, ROW_GAP = 20, 18
EDGE_MARGIN, STUB_W, STUB_H = 10, 42, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 18
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_H = 20
TITLE_MAX_W = NODE_W - 24  # 符号名文字可用宽度(留左右各 12px 内边距)

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
    """站牌胶囊,居中挂在 (cx,cy)——节点用它贴右上角,路线图例用它居中挂线上。
    宽度按 cjk_text_width() 估算(本章站牌是中文标题词,非 §N.M 短数字)。"""
    bw = cjk_text_width(text, 11) + 16
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

# 入口/出口接口桩
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["exit"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#166534">{esc("读者")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("下一章")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用/走线边(主线蓝);多条边汇入同一节点时终点 y 各偏移,看得出"汇合"
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

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌)
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    fsz = title_font_size(symbol, TITLE_MAX_W)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.4:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{fsz:.1f}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W / 2, y, sec)

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
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
print(f"wrote {out}  ({w}x{h}, aspect {w/h:.2f}:1)")
