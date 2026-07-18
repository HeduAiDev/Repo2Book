#!/usr/bin/env python3
"""ch01「本章地图」——三支柱源码剖面图(自然标题章,禁用 §N.M 徽标,站牌改用标题词本身)。

本章 chapter.md 全程是自然标题(`## 支柱一：fork，不是插件` / `### ttadapter 内部…`等,
无 `## N.M` 编号),故站牌不用 §N.M,改用各标题里的真实关键词(如"支柱一""分叉""pass 链"
"接进 libtriton""合流")——每个站牌都是对应标题的原样子串,可逐一对回正文。

剖面(左→右主脊 + 下方支撑细节泳道):
  add_kernel(用户核,三支柱合流一节完整展示)→ supports_target(支柱一挂载点)
  → add_stages(支柱二主脊)→ make_ttir(分叉前共同祖先)→ ttir_to_linalg(分叉点,
  指针张量→结构化 memref)→ npu_compile_A2_A3(闭源 bishengir 出 .npubin)
  → CoreType(支柱三,落地达芬奇双核)。
  下方支撑泳道:load_dialects(支柱一方言证据,挂在 mount 下)、add_triton_to_linalg
  (pass 链收官,挂在 ttadapt 下)、init_triton_ascend(C++ 总装点,挂在 pass 链下)、
  AddressSpace(支柱三内存层级,挂在 hw 下)。

模板见 .claude/skills/svg-diagram/references/example-chapter-map.py;不可变视觉语言
(入口绿#22c55e/出口橙#f97316/主线蓝#3b82f6/>2色配图例/cjk_text_width)照搬,只改 DATA。
因本章站牌是变长词语(非固定 "§N.M"),badge() 改为按 cjk_text_width 动态算宽度
(仍是圆角胶囊+同色语义,形状/配色不可变,只有宽度公式因内容变长而不再是常量);
支撑泳道与主脊之间的边全部落在同一列(纵向),节点内文字超宽时按比例自动缩字号
以保证不溢出——两处都是公式算,零手写魔数。

六项自查(渲染→Read PNG 亲眼看后如实记录):见 figure-manifest.json 该图 selfcheck。

用法:python3 gen_chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算:全角(ord>0x2E80)按 1.0×size,半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def fit_font_size(text, max_width, base_size, min_size=8.0):
    """若 text 在 base_size 下超出 max_width,按比例缩小字号(下限 min_size),
    保证文字不溢出节点框——公式算,不针对单个节点手调。"""
    w = cjk_text_width(text, base_size)
    if w <= max_width:
        return base_size
    return max(min_size, base_size * max_width / w)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["主脊：入口挂载 → 三段下降链 → 落地硬件", "支撑细节：证据 · pass 链 · 总装 · 内存层级"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌——正文自然标题的原样子串)
NODES = [
    ("entry",     0, 0, 0, "add_kernel",           "@triton.jit 核,与基座逐字同构",      "合流"),
    ("mount",     0, 1, 0, "supports_target",       "只认 target.backend=='npu'",         "支柱一"),
    ("stages",    0, 2, 0, "add_stages",             "登记 ttir→ttadapter→npubin 三段",   "支柱二"),
    ("ttir",      0, 3, 0, "make_ttir",              "GPU/NPU 分叉前的共同祖先",           "分叉"),
    ("ttadapt",   0, 4, 0, "ttir_to_linalg",         "分叉点:指针张量→结构化 memref",      "分叉"),
    ("npubin",    0, 5, 0, "npu_compile_A2_A3",      "闭源 bishengir 编出 .npubin",        "支柱二"),
    ("hw",        0, 6, 0, "CoreType",               "CUBE/VECTOR 双核类别",               "支柱三"),

    ("dialects",  1, 1, 0, "load_dialects",          "注册 HIVM/annotation/scope 方言",    "支柱一"),
    ("passchain", 1, 4, 0, "add_triton_to_linalg",   "pass 链收官,吐出结构化 Linalg",       "pass 链"),
    ("init",      1, 4, 1, "init_triton_ascend",     "总装点,拼进 libtriton.ascend",        "接进 libtriton"),
    ("addr",      1, 6, 0, "AddressSpace",           "UB/L1/L0A/L0B/L0C 显式内存",          "支柱三"),
]
EDGES = [  # (src_id, dst_id) —— 主脊调用边,统一主线蓝(同一泳道,列相邻,水平箭头)
    ("entry", "mount"), ("mount", "stages"), ("stages", "ttir"),
    ("ttir", "ttadapt"), ("ttadapt", "npubin"), ("npubin", "hw"),
]
SUPPORT_EDGES = [  # (src_id, dst_id) —— 支撑细节引用边,虚线靛蓝,全部同列纵向
    ("mount", "dialects"), ("ttadapt", "passchain"), ("passchain", "init"), ("hw", "addr"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("先立心智模型：三支柱总览", [(1, "支柱一"), (2, "支柱二"), (4, "分叉"), (6, "支柱三")], True),
    ("端到端核对（合流一节的 vector-add 例）", [(2, "支柱二"), (4, "分叉"), (6, "支柱三")], False),
]
LEGEND = [
    ("#22c55e", "入口:用户 @triton.jit 核"),
    ("#3b82f6", "主线:三段下降调用"),
    ("#6366f1", "支撑:细节/证据引用"),
    ("#f97316", "出口:落地达芬奇双核"),
]
TITLE = "鸟瞰篇 · 三支柱源码剖面：fork 挂载 → 三段下降链 → 达芬奇双核落地"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN, C_SUPPORT = "#22c55e", "#f97316", "#3b82f6", "#6366f1"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 136, 56
COL_GAP, ROW_GAP = 22, 18
EDGE_MARGIN, STUB_W, STUB_H = 10, 50, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_H, BADGE_FONT = 20, 11

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


def badge_width(text, font_size=BADGE_FONT):
    return cjk_text_width(text, font_size) + 16


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy)——宽度按文字动态算(本章站牌是变长标题词,
    非固定 §N.M),形状(圆角胶囊)与配色仍是不可变视觉语言。"""
    bw = badge_width(text)
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="{BADGE_FONT}" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN), ("Support", C_SUPPORT))
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
    _lx += 20 + cjk_text_width(label, 11.5) + 26

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

# 入口/出口接口桩:入口挂 entry(最左,用户核),出口挂 hw(主脊最右,落地硬件)
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["hw"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("用户核")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("落地硬件")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 主脊调用边(主线蓝,同泳道列相邻,水平居中附着)
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    p2 = (x2, y2 + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 支撑细节引用边(虚线靛蓝,全部同列纵向:上节点底心 → 下节点顶心)
for src, dst in SUPPORT_EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W / 2, y1 + NODE_H)
    p2 = (x2 + NODE_W / 2, y2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_SUPPORT}" stroke-width="1.6" stroke-dasharray="5,4" '
              f'marker-end="url(#mSupport)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌;文字超宽自动缩字号)
inner_w = NODE_W - 16
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_fs = fit_font_size(symbol, inner_w, 13)
    phr_fs = fit_font_size(phrase, inner_w, 10.5)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{sym_fs:.1f}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.74:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{phr_fs:.1f}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    bw = badge_width(sec)
    L += badge(x + NODE_W - bw / 2 + 6, y, sec)

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌,取自正文自然标题;实线蓝=推荐 / 虚线灰=次要)")}</text>')
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
print(f"wrote {out}  ({w:.0f}x{h:.0f}, aspect {w / h:.2f}:1)")
