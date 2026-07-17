#!/usr/bin/env python3
"""ch38《对照落地：AMD HIP 后端》本章地图——源码剖面图。

本章是一章「逐面对照」章（自然标题，无 ## N.M 编号），叙事骨架是「同一张
BaseBackend 契约表，NVIDIA 与 AMD 各填一份」。因此本图不是单链路的时间流程图，
而是一张**两泳道对照图**：
  - 泳道 0「契约面(共享)」：唯一的 BaseBackend 入口节点，两条填法从这里分叉。
  - 泳道 1「NVIDIA 填法」/ 泳道 2「AMD 填法」：同样 4 个对照列（编译选项/
    add_stages 五段骨架/make_ttgir pass/工具链末端），AMD 泳道在第 5 列多出
    一个「专属特化的接缝」节点——NVIDIA 没有这一列，用列数的不对称本身
    表达「AMD 多一个后端专属钩子，NVIDIA 不需要」。
  - 两条填法最终收敛回同一个出口桩（compile() 契约收敛），呼应正文小结
    「六个面，六种同一位置两种填法，编译总控一行不改」。

自然标题章规则：本章 chapter.md 没有 `## N.M` 编号标题，只有自然标题，
所以站牌**不用 §N.M**，改用标题词本身的原样子串（如「HIPOptions vs
CUDAOptions」「add_stages 五段骨架」「make_ttgir」「工具链末端」
「专属特化的接缝」「契约面」——逐一可在 narrative/chapter.md 的实际
`## ...` 标题里找到原样子串）。

■ 不可变（沿用全书本章地图视觉语言，换章节数据时不要动这些，只改 DATA）：
  1. §/标题徽标：圆角胶囊(pill)，fill #eef2ff / stroke #6366f1 / 文字靛蓝深色。
  2. 入口/出口：入口箭头=绿 #22c55e，出口箭头=橙 #f97316。
  3. 节点间调用边（主线）=蓝 #3b82f6。
  4. 底部路线条：高亮/推荐路线=实线蓝(粗)，其余路线=虚线中性灰(细)。
  5. >2 种语义色需画图例——本图 3 色(绿/蓝/橙)，画图例。
  6. 文本宽度估算一律用 cjk_text_width()，不用半角系数直乘 len(s)。

■ 可变（换章节时改这些）：LANES/NODES/EDGES/ROUTES/TITLE/LEGEND；
  以及本章专属的两处适配——① 标题徽标改用自然标题原样子串而非 §N.M；
  ② 徽标宽度按文本动态计算（不用模板里固定的 46px），因为本章徽标文本比
  §20.4 这类短编号长得多（如「HIPOptions vs CUDAOptions」26 字符），
  固定宽度会让文字溢出胶囊；③ 对照列的徽标只画一次、挂在 NVIDIA/AMD 两个
  同列节点之间的分隔带上（而非两节点各自重复一遍）——因为该徽标标记的是
  同一个章节小节，两侧节点在讲同一段落，不是各自独立的小节。

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变：本章数据) ----------------
LANES = ["契约面(共享)", "NVIDIA 填法 · 回指 ch36/ch37", "AMD 填法 · 本章主角"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语)
# 徽标(第7个元素)只在需要"个别节点专属徽标"时给(entry/amd_hook)；
# 两泳道共享的对照列徽标改走 COL_BADGES(见下)，不重复画在每个节点上。
NODES = [
    ("entry",    0, 0, 0, "BaseBackend",             "6 必填 + 2 可覆写钩子的契约表", "契约面"),
    ("nv_opt",   1, 1, 0, "CUDAOptions",              "warp 恒 32，无 warp_size 字段", None),
    ("nv_stg",   1, 2, 0, "add_stages",               "ttir/ttgir/llir + ptx/cubin", None),
    ("nv_ttg",   1, 3, 0, "add_accelerate_matmul",    "零参，门控 capability 分档", None),
    ("nv_tool",  1, 4, 0, "make_ptx / make_cubin",    "ptxas 一步出 cubin", None),
    ("amd_opt",  2, 1, 0, "HIPOptions",               "专属旋钮 + warp_size 按 gfx 算", None),
    ("amd_stg",  2, 2, 0, "add_stages",               "ttir/ttgir/llir + amdgcn/hsaco", None),
    ("amd_ttg",  2, 3, 0, "add_accelerate_matmul",    "三参调 mfma，门控 matrix core 探测", None),
    ("amd_tool", 2, 4, 0, "make_amdgcn / make_hsaco", "assemble_amdgcn + ld.lld 出 hsaco", None),
    ("amd_hook", 2, 5, 0, "HIPAttrsDescriptor",       "覆写钩子加 pointer_range=32", "专属特化的接缝"),
]
# 对照列共享徽标：(列号, 徽标文字) —— 挂在泳道 0/1 分隔带上，NVIDIA/AMD
# 两个同列节点共用同一个徽标（同一节小节讲两种填法，不是两个独立小节）。
COL_BADGES = [
    (1, "HIPOptions vs CUDAOptions"),
    (2, "add_stages 五段骨架"),
    (3, "make_ttgir"),
    (4, "工具链末端"),
]
EDGES = [  # (src_id, dst_id) —— 调用边，统一主线蓝
    ("entry", "nv_opt"), ("entry", "amd_opt"),
    ("nv_opt", "nv_stg"), ("nv_stg", "nv_ttg"), ("nv_ttg", "nv_tool"),
    ("amd_opt", "amd_stg"), ("amd_stg", "amd_ttg"), ("amd_ttg", "amd_tool"), ("amd_tool", "amd_hook"),
]
# 收敛边：两条填法末端各自汇入同一个出口桩(不经过某个节点，直接連出口)
CONVERGE_TO_EXIT = ["nv_tool", "amd_hook"]

# (路线名, [(列, 徽标文字), ...] 按阅读顺序, 是否高亮：True=实线蓝/False=虚线灰)
ROUTES = [
    ("AMD 填法(本章主线)", [(1, "HIPOptions vs CUDAOptions"), (2, "add_stages 五段骨架"),
                              (3, "make_ttgir"), (4, "工具链末端"), (5, "专属特化的接缝")], True),
    ("NVIDIA 填法(回指 ch36/ch37)", [(1, "HIPOptions vs CUDAOptions"), (2, "add_stages 五段骨架"),
                                        (3, "make_ttgir"), (4, "工具链末端")], False),
]
LEGEND = [("#22c55e", "入口：compile() 拿到某个 BaseBackend 子类"),
          ("#3b82f6", "章内主线：六面填空逐层展开"),
          ("#f97316", "出口：两份填法收敛回同一契约")]
TITLE = "第 38 章 · BaseBackend 配对脊柱剖面：NVIDIA / AMD 两份填法对照"
ENTRY_STUB_LABEL = "compile()"
EXIT_STUB_LABEL = "契约收敛"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#fff7ed"]  # 泳道背景交替，仅装饰，非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数) ----------------
NODE_W, NODE_H = 205, 62
COL_GAP, ROW_GAP = 16, 20
EDGE_MARGIN, STUB_W, STUB_H = 12, 50, 24
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 20
LANE_LABEL_H, BAND_PAD = 22, 18
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 32, 24, 16
COLBADGE_GAP = 30  # 泳道0(契约面)与泳道1(NVIDIA)之间额外留白，放对照列共享徽标
ROUTE_HEAD_H, ROUTE_ROW_H = 20, 40
BADGE_H = 20

n_cols = max(n[2] for n in NODES) + 1
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_lane = [0] * len(LANES)
for _id, lane, col, row, *_ in NODES:
    rows_per_lane[lane] = max(rows_per_lane[lane], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_lane]

band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for i, bh in enumerate(band_h):
    if i == 1:
        _cum += COLBADGE_GAP  # 泳道0→泳道1 之间插入对照列徽标带
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


def badge(cx, cy, text, size=11):
    """§/标题徽标胶囊，居中挂在 (cx,cy)。宽度按文本动态算(cjk_text_width)，
    本章徽标文本远比 §20.4 这类短编号长，固定宽度会溢出，故不写死 BADGE_W。"""
    bw = cjk_text_width(text, size) + 18
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="{size}" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
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
    L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
             f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 对照列共享徽标：挂在泳道0/泳道1之间的留白带上，居中于该列
_badge_band_cy = (band_top[0] + band_h[0] + band_top[1]) / 2
for col, text in COL_BADGES:
    cx = COLX[col] + NODE_W / 2
    parts, _ = badge(cx, _badge_band_cy, text, size=11.5)
    L += parts

# 入口接口桩(compile() 调用契约面)
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
         f'fill="#166534">{esc(ENTRY_STUB_LABEL)}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')

# 出口接口桩(两份填法收敛回同一个契约调用点)——桩的 y 取契约面(entry)所在行，
# 与入口桩上下对称；两条收敛边从 nv_tool / amd_hook 各自斜线汇入。
xy_exit_y = ey
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy_exit_y - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy_exit_y + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
         f'fill="#9a3412">{esc(EXIT_STUB_LABEL)}</text>')
for i, src in enumerate(CONVERGE_TO_EXIT):
    x1, y1 = NODE_XY[src]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    y_off = (i - (len(CONVERGE_TO_EXIT) - 1) / 2) * 10
    p2 = (sx, xy_exit_y + y_off)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝)
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

# 节点(圆角框 + 真实符号名 + 一行短语 + 可选单独徽标)
for nid, lane, col, row, symbol, phrase, *rest in NODES:
    own_badge = rest[0] if rest else None
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    if "/" in symbol:  # 长符号名(两个 make_* 并列)拆两行，避免溢出节点宽度
        top, bot = [p.strip() for p in symbol.split("/", 1)]
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.34:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(top)}</text>')
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.56:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc("/ " + bot)}</text>')
        sub_y = y + NODE_H * 0.82
    else:
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.4:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
        sub_y = y + NODE_H * 0.72
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{sub_y:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    if own_badge:
        parts, _ = badge(x + NODE_W - 8, y, own_badge)
        L += parts

# 底部阅读路线
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上对照列站牌；实线蓝=推荐 / 虚线灰=次要)")}</text>')
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
        parts, _ = badge(COLX[col] + NODE_W / 2, ry, sec, size=10.5)
        L += parts

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w:.0f}x{h:.0f}  ratio={w/h:.3f}")
