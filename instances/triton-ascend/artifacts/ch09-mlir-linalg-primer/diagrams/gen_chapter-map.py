#!/usr/bin/env python3
"""ch09《MLIR 与 Linalg》本章地图 —— 概念脉络图(primer 原理章,非源码调用链)。

本章是自然标题章(无 `## N.M` 编号),按契约禁用 §N.M 徽标——站牌改用从真实标题
里摘出的关键词。四条泳道 = 本章四段概念脉络(不是代码分层):
  A. MLIR 论文 · 造 IR 的基础设施 (arXiv:2002.11054)
  B. Linalg 论文 · 结构化算子基础 (arXiv:2202.03293)
  C. Linalg 论文 · 变换与设计哲学
  D. 落地对位 · ttadapter → HFusion → HIVM

每个节点的「symbol」字段是本章正文里逐字出现过的真实记号(IR 算子名/论文术语),
防杜撰门禁按这个字符串去 chapter.md 里找原样子串。

■ 不可变(全书视觉语言):入口绿 #22c55e / 出口橙 #f97316 / 主线蓝 #3b82f6;
  路线条 高亮=蓝实线 / 次要=灰虚线;站牌用胶囊(pill),配色沿用既有 badge 语言。
■ 可变:泳道数/节点排布/路线;站牌文字(自然标题章,不用 §N.M)。

用法:python3 gen_chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算:全角(ord>0x2E80)按 1.0×size,半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------

LANES = [
    "MLIR 论文 · 造 IR 的基础设施",
    "Linalg 论文 · 结构化算子基础",
    "Linalg 论文 · 变换与设计哲学",
    "落地对位",
]

# (节点id, 泳道下标, 列, 真实记号(symbol,正文逐字可核), 站牌(摘自真实标题的关键词),
#  phrase_lines: 1~2 行短语)
NODES = [
    ("n1", 0, 0, "progressive lowering", "一层→渐进下降",
     ["渐进式下降 + 维持高层语义", "——一对孪生原则"]),
    ("n2", 0, 1, "affine.for", "六件套",
     ["Op 挂 region、region 装 block", "方言可任意层级共存"]),
    ("n3", 0, 2, "Canonicalization", ".td + 可扩展 pass",
     [".td 声明式定义,C++ 由生成器产出", "trait/interface 让 pass 跨方言复用"]),

    ("n4", 1, 0, "structured code generation", "结构化 codegen 起因",
     ["把 tiling/fusion/向量化", "搬到张量层,不等退化成循环"]),
    ("n5", 1, 1, "linalg.conv_1d_nwc_wcf", "索引表达式",
     ["配料单写在算子身上:", "O[n,w,f]=I[n,w+kw,c]·K[kw,c,f]"]),
    ("n6", 1, 2, "iteration domain", "隐式迭代域",
     ["边界从操作数形状反解", "IR 里不写一个字的循环边界"]),
    ("n7", 1, 3, "tensor.extract_slice", "求像即子集",
     ["像 = 索引函数(迭代域)", "halo = 像宽 − tile 宽"]),

    ("n8", 2, 0, "linalg.generic", "named 与 generic",
     ["具名算子只是省略了算子体", "语义同归 linalg.generic"]),
    ("n9", 2, 1, "destination-passing style", "outs 不是顺手传",
     ["outs 是 bufferization 的", "一等操作数,张量仍不可变"]),
    ("n10", 2, 2, "vector.contract", "变换栈",
     ["tiling→padding→向量化→", "bufferization→向量渐进下降"]),
    ("n11", 2, 3, "Profitability", "三问依附抽象层级",
     ["Legality / Applicability /", "Profitability 依附哪层抽象"]),

    ("n12", 3, 3, "ttadapter", "落到本书:三方对位",
     ["HFusion=Linalg 扩展集(仅 named op)", "HIVM 感知硬件细节,ttadapter 接入"]),
]

# 章内叙事主线(同一泳道内 L→R;跨泳道换行)——全部主线蓝
EDGES_STRAIGHT = [("n1", "n2"), ("n2", "n3"),
                  ("n4", "n5"), ("n5", "n6"), ("n6", "n7"),
                  ("n8", "n9"), ("n9", "n10"), ("n10", "n11")]
EDGES_ELBOW = [("n3", "n4"), ("n7", "n8"), ("n11", "n12")]  # 跨泳道换行连接

# 底部阅读路线:(路线名, [(列, 站牌短标签), ...], 是否高亮)
ROUTES = [
    ("顺序精读(默认)", [(0, "一层→渐进下降"), (3, "落地对位")], True),
    ("只要结论", [(0, "结构化 codegen 起因"), (3, "三问依附抽象层级")], False),
    ("想把数学吃透", [(1, "索引表达式"), (2, "变换栈")], False),
]

LEGEND = [("#22c55e", "入口:开始读本章"), ("#3b82f6", "章内叙事主线"), ("#f97316", "出口:下一章 ch10 分水岭")]
TITLE = "第 9 章 · MLIR 与 Linalg:编译基础设施概念地图(非源码调用链)"
SUBTITLE = "MLIR (arXiv:2002.11054) 与 Linalg (arXiv:2202.03293) 两篇论文的心智模型 + 昇腾落地对位"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#fdf4e3"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 210, 76
COL_GAP = 40
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32
LANE_LABEL_H, BAND_PAD = 22, 16
TOP_PAD, TITLE_H, SUBTITLE_H, LEGEND_H, BOTTOM_PAD = 14, 26, 20, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_PAD_X, BADGE_H = 10, 20

n_cols = max(n[2] for n in NODES) + 1
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

NODE_BY_ID = {n[0]: n for n in NODES}
lane_names_by_idx = {i: name for i, name in enumerate(LANES)}

band_top, _cum = [], TOP_PAD + TITLE_H + SUBTITLE_H + LEGEND_H
band_h = [LANE_LABEL_H + BAND_PAD * 2 + NODE_H for _ in LANES]  # 每条泳道恰好 1 行
for bh in band_h:
    band_top.append(_cum)
    _cum += bh
lanes_bottom = _cum

NODE_XY = {}
for nid, lane, col, symbol, badge_label, lines in NODES:
    x = COLX[col]
    y = band_top[lane] + LANE_LABEL_H + BAND_PAD
    NODE_XY[nid] = (x, y)

routes_top = lanes_bottom + 10
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge_width(text):
    return cjk_text_width(text, 11) + BADGE_PAD_X * 2


def badge(cx, cy, text):
    """站牌胶囊——宽度按文字自适应(cjk_text_width),居中挂在 (cx,cy)。"""
    bw = badge_width(text)
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ], bw


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题 + 副标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="15" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + TITLE_H + 14}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="11" fill="{C_NODE_SUB}">{esc(SUBTITLE)}</text>')

# 图例(3 种语义色 → 必须画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + SUBTITLE_H + 16
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
ex, ey = NODE_XY["n1"]; ey += NODE_H / 2
xx, xy = NODE_XY["n12"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
          f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("开始读")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
          f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
          f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("下一章")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
          f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 同泳道直线边(右边缘 → 左边缘)
for src, dst in EDGES_STRAIGHT:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    p2 = (x2, y2 + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 跨泳道折线边(elbow:src 下沿中点 → 中间水平 → dst 上沿中点)
for src, dst in EDGES_ELBOW:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    sx0 = x1 + NODE_W / 2; sy0 = y1 + NODE_H
    dx0 = x2 + NODE_W / 2; dy0 = y2
    mid_y = (sy0 + dy0) / 2
    pts = f"{sx0:.1f},{sy0:.1f} {sx0:.1f},{mid_y:.1f} {dx0:.1f},{mid_y:.1f} {dx0:.1f},{dy0:.1f}"
    L.append(f'<polyline points="{pts}" fill="none" stroke="{C_MAIN}" stroke-width="2" '
              f'stroke-dasharray="5,4" marker-end="url(#mMain)"/>')

# 节点(圆角框 + symbol + 1~2 行短语 + 右上角站牌胶囊)
for nid, lane, col, symbol, badge_label, lines in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + 22:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    for i, line in enumerate(lines):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + 42 + i * 15:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="9.8" fill="{C_NODE_SUB}">{esc(line)}</text>')
    # 站牌贴节点右上角,但右缘不得越过下一列节点左缘(留 ≥10px 视觉间隙)——
    # 长标签(如「结构化 codegen 起因」)默认锚点会探进车道间隙压住相邻框,故夹紧。
    bw_badge = badge_width(badge_label)
    BADGE_GAP_MIN = 10
    cx_default = x + NODE_W - 8
    cx_cap = x + NODE_W + COL_GAP - BADGE_GAP_MIN - bw_badge / 2
    cx_badge = min(cx_default, cx_cap)
    b_svg, _bw = badge(cx_badge, y, badge_label)
    L += b_svg

# 底部阅读路线
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
          f'font-weight="bold" fill="{C_LANE_LABEL}">'
          f'{esc("阅读路线(标号=图上站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
ROUTE_GAP_MIN = 12  # 路线行内相邻元素(标签文字/徽标/徽标)之间的最小视觉间隙
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    name_w = cjk_text_width(name, 12)
    # 逐站牌顺序摆放:默认对齐节点列坐标,但右缘不得早于「前一元素右缘 + 最小间隙」——
    # 防止路线名文字长(如「顺序精读(默认)」)把第一个徽标往左压出重叠。
    prev_right = 16 + name_w + ROUTE_GAP_MIN
    stop_cx = []
    for col, sec in stops:
        bw = badge_width(sec)
        desired_cx = COLX[col] + NODE_W / 2
        min_cx = prev_right + ROUTE_GAP_MIN + bw / 2
        cx = max(desired_cx, min_cx)
        stop_cx.append(cx)
        prev_right = cx + bw / 2
    x_first, x_last = stop_cx[0], stop_cx[-1]
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for (col, sec), cx in zip(stops, stop_cx):
        bsvg, _ = badge(cx, ry, sec)
        L += bsvg

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w:.0f}x{h:.0f})")
