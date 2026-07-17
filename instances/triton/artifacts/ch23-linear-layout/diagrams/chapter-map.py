#!/usr/bin/env python3
"""ch23《LinearLayout：一个抽象统一所有布局》—— 本章地图（源码剖面图）。

本章标题是 `## §1`～`## §11`（单数字 § 记法，无 `N.M` 小数点），不匹配
lint_chapter_map 的 `§N.M` 徽标正则——沿用 ch22 chapter-map.py 已验证过的精确
先例：站牌直接用正文的 `§1`～`§11` 记法（固定宽度小胶囊），不做成 ch20 那种
自然标题的动态宽度站牌。

本章叙事形状（primer 四段式）：动机（§1 转换代码 O(K²) 爆炸）→ 核心换向定义
（§2 硬件位置→逻辑索引）→ 展开三级：bases 压缩（§3）→ xor 线性律 4×4 顶悟填表
（§4，核心顶悟）→ GF(2) 数学背景（§5）→ 矩阵引擎三件套 getMatrix/compose/
invertAndCompose（§6-§8）→ 秩（§9）→ Four Russians 背景盒（§10，可略读的旁支）
→ 落地 toLinearLayout 统一收编（§11，出口）。11 个节点，三条语义带，沿用
ch20/ch02 验证过的「三泳道局部列 + 折角续行」几何（避免单行 11 节点把画布拉爆）。

■ 不可变（同 example-chapter-map.py / ch20/ch22 chapter-map.py）：入口绿
  #22c55e·出口橙#f97316·主线蓝#3b82f6 / 路线高亮实线蓝-次要虚线灰 /
  cjk_text_width() 宽度估算 / >2 种语义色画图例 / § 徽标固定宽度胶囊（ch22 先例）。

■ 本章专属改动：
  1. 三条泳道各自局部列号（不共用 COLX），泳道间以折角虚线「续下一行」相接
     （ch20 手法）——本章是一条主叙事线，§10 是可跳过的背景盒但仍在主链上。
  2. 两条阅读路线：①全通读（§1→…→§11 全部 11 站）；②跳过背景盒的精简路线
     （同样 §1→…→§11，但跳过 §10 Four Russians）——给读者显式的选读指引。
     路线站牌用独立的底部横向布局（固定胶囊宽度累加摆放），不复用节点网格
     的局部列坐标（两条路线跨泳道，节点局部列本就不可比）。

■ 六项自查（渲染→Read PNG 亲眼看后如实记录）：
    claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
    arrows_attached=True     cjk_rendered=True         reading_order_clear=True

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["动机与核心定义(bases + 顶悟)", "GF(2) 矩阵引擎", "秩、背景盒与落地"]

# (节点id, 泳道下标, 泳道内局部列, 真实符号/概念名, 一行短语, §编号)
NODES = [
    ("explosion",      0, 0, "转换代码 O(K²)",
     "每对布局各写一套转换代码,种类一多按平方增长", "§1"),
    ("direction",      0, 1, "L: 硬件位置→逻辑索引",
     "换方向保函数性:一对多的一侧放进定义域(broadcast)", "§2"),
    ("bases",          0, 2, "bases 基向量",
     "只需给出 2 的幂次输入点上的取值,其余全推出", "§3"),
    ("xor4x4",         0, 3, "4×4 swizzle 填表",
     "xor 线性律 L(a⊕b)=L(a)⊕L(b),亲手验算认出 (t,w⊕t)", "§4"),
    ("gf2",            1, 0, "GF(2):加=xor 乘=and",
     "L(a)=Ba,bases 就是矩阵 B 的列(定理而非约定)", "§5"),
    ("getmatrix",      1, 1, "getMatrix",
     "把布局压成比特矩阵,每列一个 base、每行一 uint64_t", "§6"),
    ("compose",        1, 2, "compose",
     "O∘L:把 this 的每个 base 喂给 outer 求值即得复合", "§7"),
    ("invertcompose",  1, 3, "invertAndCompose",
     "拼接 [outer|this] 做一次 f2reduce RREF:左半单位阵=求逆", "§8"),
    ("rank",           2, 0, "getMatrixRank",
     "同一个 RREF 调用:数非零行=秩,判满射/单射", "§9"),
    ("fourrussians",   2, 1, "Four Russians / f2reduce",
     "分块查表削一个对数因子,统一抽象在编译期也跑得起(背景盒)", "§10"),
    ("unify",          2, 2, "toLinearLayout",
     "把 Blocked/Shared/MMA 全部折叠成一个 LinearLayout", "§11"),
]
# (src_id, dst_id, is_wrap) —— 主线蓝;is_wrap=True 的两条走折角连接线(跨泳道换行)
EDGES = [
    ("explosion", "direction", False), ("direction", "bases", False),
    ("bases", "xor4x4", False),
    ("xor4x4", "gf2", True),
    ("gf2", "getmatrix", False), ("getmatrix", "compose", False),
    ("compose", "invertcompose", False),
    ("invertcompose", "rank", True),
    ("rank", "fourrussians", False), ("fourrussians", "unify", False),
]
# 两条阅读路线:①全通读(11 站全部);②跳过背景盒的精简路线(同序但少 §10)
ROUTE_FULL = ["§1", "§2", "§3", "§4", "§5", "§6", "§7", "§8", "§9", "§10", "§11"]
ROUTE_SKIP = ["§1", "§2", "§3", "§4", "§5", "§6", "§7", "§8", "§9", "§11"]
ROUTES = [
    ("全通读(含 Four Russians 背景盒)", ROUTE_FULL, True),
    ("精简路线(跳过 §10 背景盒,不影响主线理解)", ROUTE_SKIP, False),
]
LEGEND = [("#22c55e", "入口:从上一章 GF(2) 前瞻而来"),
          ("#3b82f6", "章内讲解顺序/依赖链"),
          ("#f97316", "出口:带比特矩阵读法进入下一章")]
TITLE = "第 23 章 · LinearLayout 剖面:bases → xor 线性律 → GF(2) 矩阵引擎 → 统一落地"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 190, 66
COL_GAP, ROW_GAP = 34, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 96, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 14
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_W, BADGE_H = 30, 18
WRAP_GAP = 30  # 折行处(跨泳道连接线)额外留白

# 每条泳道各自的局部列数(不跨泳道共享列号)
cols_per_lane = [0] * len(LANES)
for _id, lane, col, *_ in NODES:
    cols_per_lane[lane] = max(cols_per_lane[lane], col + 1)
n_cols_max = max(cols_per_lane)

w_lanes = PAD_L + n_cols_max * NODE_W + (n_cols_max - 1) * COL_GAP + PAD_R

# 底部阅读路线所需宽度:两条路线站牌按各自实际(固定)宽度累加布局,
# 取站牌数更多的一条(全通读,11 站)反推所需宽度。
ROUTE_GAP = 12
_route_name_w = max(cjk_text_width(name, 12) for name, _, _ in ROUTES) + 28
_route_start = 16 + _route_name_w
w_route_required = _route_start + len(ROUTE_FULL) * (BADGE_W + ROUTE_GAP)

w = max(w_lanes, w_route_required)

rows_per_lane = [1] * len(LANES)  # 每条泳道本章都只用 1 行
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_lane]
band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for i, bh in enumerate(band_h):
    if i > 0:
        _cum += WRAP_GAP
    band_top.append(_cum)
    _cum += bh
lanes_bottom = _cum

NODE_XY = {}
for nid, lane, col, *_ in NODES:
    x = PAD_L + col * (NODE_W + COL_GAP)
    y = band_top[lane] + LANE_LABEL_H + BAND_PAD
    NODE_XY[nid] = (x, y)
NODE_BY_ID = {n[0]: n for n in NODES}
NODE_LANE = {n[0]: n[1] for n in NODES}
BADGE_BY_ID = {n[0]: n[4] for n in NODES}
LANE_BOTTOM = [band_top[i] + band_h[i] for i in range(len(LANES))]  # 每条泳道自己的下边界

routes_top = lanes_bottom + 8
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
    """§ 徽标胶囊,固定宽度,居中挂在 (cx,cy)——节点用它贴右上角,路线用它居中挂线上。"""
    bx, by = cx - BADGE_W / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BADGE_W}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 3.8:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="10" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


_SOFT_BREAK_PUNCT = set("/-,;:()=，、；：（）")


def _is_break_char(ch):
    """可在其后安全换行的字符:空白、CJK(含全角标点)、以及英文短语常见软分隔符。"""
    return ch.isspace() or ord(ch) > 0x2E80 or ch in _SOFT_BREAK_PUNCT


def wrap_lines(phrase, font_size, max_w, max_lines=2):
    """贪心按宽度换行,最多 max_lines 行;每行末尾回退到最近安全断点,避免断词。"""
    lines = []
    remaining = phrase
    while remaining:
        if len(lines) == max_lines - 1 or cjk_text_width(remaining, font_size) <= max_w:
            lines.append(remaining)
            remaining = ""
            break
        acc, cut = 0.0, len(remaining)
        for i, ch in enumerate(remaining):
            acc += font_size * (1.0 if ord(ch) > 0x2E80 else 0.58)
            if acc > max_w:
                cut = i
                break
        safe_cut = cut
        while safe_cut > 1 and not _is_break_char(remaining[safe_cut - 1]):
            safe_cut -= 1
        if safe_cut <= 1:
            safe_cut = max(cut, 1)
        lines.append(remaining[:safe_cut].rstrip())
        remaining = remaining[safe_cut:].lstrip()
    return lines


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
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11) + 30

# 泳道背景 + 标签
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩
ex, ey = NODE_XY["explosion"]; ey += NODE_H / 2
xx, xy = NODE_XY["unify"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#166534">{esc("读者入口")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("下一章:转换算子层")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用/讲解顺序边(主线蓝);跨泳道折行的两条画成折角虚线
for src, dst, is_wrap in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    if not is_wrap:
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        p2 = (x2, y2 + NODE_H / 2)
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    else:
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        mid_x = x1 + NODE_W + 20
        p2 = (x2 + NODE_W / 2, y2)
        # 折角转向点固定在"源节点所在泳道"与"下一泳道"之间的 WRAP_GAP 空白带正中央
        # (而非按节点 y 坐标简单取中点)——避免转向点/标签蹭到相邻泳道背景带的
        # 下边缘(此前版本的做法在 lint_diagram_geometry 的 text-rect 检查里被
        # 判定为"插进了泳道背景框"，改成对齐留白带几何后不再触发)。
        turn_y = LANE_BOTTOM[NODE_LANE[src]] + WRAP_GAP / 2
        path = (f'M {p1[0]:.1f},{p1[1]:.1f} L {mid_x:.1f},{p1[1]:.1f} '
                f'L {mid_x:.1f},{turn_y:.1f} L {p2[0]:.1f},{turn_y:.1f} '
                f'L {p2[0]:.1f},{p2[1]:.1f}')
        L.append(f'<path d="{path}" fill="none" stroke="{C_MAIN}" stroke-width="2" '
                  f'stroke-dasharray="5,3" marker-end="url(#mMain)"/>')
        L.append(f'<text x="{mid_x + 6:.1f}" y="{turn_y + 3:.1f}" font-family="sans-serif" '
                  f'font-size="9.5" fill="{C_ROUTE_DIM}">{esc("续下一行")}</text>')

# 节点(圆角框 + 真实符号名 + 一行短语,最多两行 + 右上角 § 徽标)
for nid, lane, col, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.30:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    max_w = NODE_W - 16
    phrase_lines = wrap_lines(phrase, 9.3, max_w, max_lines=2)
    if len(phrase_lines) == 1:
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.58:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="9.3" fill="{C_NODE_SUB}">{esc(phrase_lines[0])}</text>')
    else:
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.56:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="9.3" fill="{C_NODE_SUB}">{esc(phrase_lines[0])}</text>')
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.78:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="9.3" fill="{C_NODE_SUB}">{esc(phrase_lines[1])}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 6, y, sec)

# 底部阅读路线:两条(全通读 / 跳过背景盒),站牌固定宽度累加摆放
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=推荐通读 / 虚线灰=可跳过的精简路线)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="11.5" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    stop_x = [_route_start + i * (BADGE_W + ROUTE_GAP) + BADGE_W / 2 for i in range(len(stops))]
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{stop_x[0]:.1f}" y1="{ry:.1f}" x2="{stop_x[-1]:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for cx, sec in zip(stop_x, stops):
        L += badge(cx, ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  ({w:.0f}x{h:.0f}, ratio={w / h:.2f}:1)")
