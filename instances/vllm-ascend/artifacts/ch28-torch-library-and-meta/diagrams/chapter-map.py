#!/usr/bin/env python3
"""第 28 章「本章地图」——torch.library 三条注册线源码剖面图。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py 模板改写：
不可变机制(esc/cjk_text_width/badge/配色/图例规则)原样保留，只改下面的 DATA。

节点预算 8(entry/real_reg/meta_reg/entry2/direct_reg/py_meta/table/exit) ≤ 12。
本章标题为编号标题(## 28.1 ... ## 28.10)，站牌用 §28.N。

设计要点：
- 两个入口(两条互相独立的 import 路径，都发生在模型前向之前)：
  entry = `import vllm_ascend.vllm_ascend_C`（触发 C++ 两套注册：真实现 + C++ meta）；
  entry2 = `vllm_ascend/ops/register_custom_ops.py`（经 ops/__init__.py 的另一条 import 路径，
  触发纯 Python 的 direct_register_custom_op × 10）。这是源码里明写的事实(§28.8「另一条 import
  路径」)，不是虚构的调用关系。
- 泳道内两行(row0=真算/主注册 track，row1=meta/推形状 track)对齐展现「同一个位置，两份实现」
  的核心立意：real_reg(真实现)与 meta_reg(C++ meta)同列上下；direct_reg(纯 Python，一次调用
  内同时挂 impl+fake)与 py_meta(Python 兜底 meta)同列上下。
- 四条注册产物(real_reg/meta_reg/direct_reg/py_meta)最终都汇入同一个 table 节点(`_C_ascend`/
  `vllm` 两个命名空间的派发表)，再到 exit(torch.compile / ACLGraph 捕获图)——对应正文 §28.8
  「三条线，一张表」的收束句。
- 缺口「63−57=6」折进 meta_reg 的短语里，不单独占一个节点(避免为一句数字事实多开一个节点
  挤占预算)；六个缺口算子的清单与「为什么断」的因果链改由底部路线 2/3 的 §28.3/§28.5 站牌
  指向正文，读者按图跳读。
- 底部三条路线覆盖 §28.2/28.3/28.4/28.5/28.6/28.7/28.8/28.9/28.10 共 9 节(仅 §28.1 立意段
  未单独设站——它是全章前提，不对应具体注册代码位置，已有独立图 fig24-1 讲透，此处不重复)。

[FIX-ROUND-1](首版渲染后 Read PNG 发现两处几何问题，本轮修复并重新渲染+Read PNG 复核，
下方六项自查是复核后的结果)：
- entry2 首版放在 col2(与 direct_reg 同列、entry2 在其正上方)——这让 entry2→direct_reg
  这条边的起点(entry2 右边框)在终点(direct_reg 左边框)的右上方，边线必须先"倒退"穿过
  direct_reg 自身的框顶再折向其左边框，实际渲染中这段边线明显斜切过 direct_reg 节点内部，
  箭头也被节点遮住看不见。改成 entry2 放 col1(与 real_reg/meta_reg 同列、lane0，entry 右侧
  一列)，entry2→direct_reg 变回"列 1→列 2"的标准相邻列前进边，不再穿框。
- real_reg/meta_reg(col1)→table(col3) 跨列 2，中间恰好隔着 direct_reg/py_meta 所在的 col2——
  直线连接会斜穿过 col2 节点框(渲染中肉眼可见线段切过 direct_reg/py_meta 内部)。改为三段折线
  (先在列间空档竖直下探到 table 所在泳道的高度——此时已低于所有 col2 节点的纵向范围——再
  横向穿过 col2、最后水平进入 table 左边)，几何上保证不与任何节点框相交。

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算——全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["加载触发(两条独立 import 路径)", "C++ / Python 注册实现", "运行期汇总"]  # 泳道,上→下

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, §编号)
NODES = [
    ("entry",      0, 0, 0, "vllm_ascend.vllm_ascend_C",              "import .so,触发全部 C++ 注册",           "§28.8"),
    ("real_reg",   1, 1, 0, "TORCH_LIBRARY_EXPAND",                    "63 个真实现绑 PrivateUse1，真算",         "§28.2"),
    ("meta_reg",   1, 1, 1, "TORCH_LIBRARY_\nIMPL_EXPAND",               "57 个 C++ meta 绑 Meta\n63−57=6 缺口",  "§28.4"),
    ("entry2",     0, 1, 0, "vllm_ascend/ops/\nregister_custom_ops.py", "经 ops/__init__.py 的另一条 import 路径", "§28.7"),
    ("direct_reg", 1, 2, 0, "direct_register_custom_op",                "10 个 torch.ops.vllm.*,各配 impl+fake", "§28.7"),
    ("py_meta",    1, 2, 1, "register_meta_if_necessary",                "紧随其后,查表缺才补 3 个",               "§28.6"),
    ("table",      2, 3, 0, "_C_ascend · vllm 命名空间",                  "两个命名空间汇入同一张派发表",           "§28.8"),
    ("exit",       2, 4, 0, "torch.compile / ACLGraph",                  "捕获图:备齐 meta 的算子进图,6 个不进",   "§28.10"),
]
EDGES = [  # (src_id, dst_id) —— 调用/注册边,统一主线蓝
    ("entry", "real_reg"), ("entry", "meta_reg"),
    ("meta_reg", "py_meta"),
    ("entry2", "direct_reg"),
    ("real_reg", "table"), ("meta_reg", "table"), ("direct_reg", "table"), ("py_meta", "table"),
    ("table", "exit"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("主线:通读顺序(推荐)", [(0, "§28.8"), (1, "§28.2"), (2, "§28.7"), (3, "§28.8"), (4, "§28.10")], True),
    ("meta 暗线:何处会断", [(0, "§28.3"), (1, "§28.4"), (2, "§28.6")], False),
    ("跳读:缺口实证 + 精简版交叉验证", [(3, "§28.5"), (4, "§28.9")], False),
]
LEGEND = [("#22c55e", "入口:从上层调用进入"), ("#3b82f6", "章内主线调用边"), ("#f97316", "出口:返回上层")]
TITLE = "第 28 章 · torch.library 三条注册线剖面(源码走线 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
# entry2 符号名较长，机械换行(内容不变，仅切在路径分隔符 "/" 之后)；meta_reg 的符号
# TORCH_LIBRARY_IMPL_EXPAND 与短语首版渲染后 Read PNG 发现明显溢出节点边框、压住汇入
# 箭头([FIX-ROUND-1] 见下)，同样机械换行(切在下划线边界/括号前，不改文字内容)。
# NODE_W 190→200、NODE_H 74→90，让符号/短语各自最多两行都留有边距。
NODE_W, NODE_H = 200, 90
COL_GAP, ROW_GAP = 40, 22
EDGE_MARGIN, STUB_W, STUB_H = 16, 68, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
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


def badge(cx, cy, text):
    """§ 徽标胶囊,居中挂在 (cx,cy) —— 节点用它贴右上角,路线legend用它居中挂线上。"""
    bx, by = cx - BADGE_W / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BADGE_W}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
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

# 入口/出口接口桩(给入口/出口箭头一个可附着的框,兼表达"调用方在画布外")
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["exit"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝,画在节点下面这条先画后画都行,这里先画边再画节点盖住端点毛刺)
# 多条边汇入同一节点时,终点 y 各偏移(间距 16px),否则重合的终点在视觉上看不出"汇合"。
NODE_COL = {n[0]: n[2] for n in NODES}
# 列间空档的中点 x——恰好落在两列节点框之间的空白区,任何节点框都不会占用这段 x,
# 竖直线走这条 x 保证不会穿框(见 [FIX-ROUND-1])。
GAP_X = [COLX[c] + NODE_W + COL_GAP / 2 for c in range(n_cols - 1)]
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
    col_skip = NODE_COL[dst] - NODE_COL[src] >= 2
    if col_skip:
        # 跨列(中间还隔着别的列的节点框)——直线连接会斜穿中间列节点。改走三段折线:
        # 先水平到本列右侧空档 → 在空档里竖直下探到目标行高(target 所在泳道更低,已避开
        # 中间列节点的纵向范围)→ 最后水平进入目标左边。三段各自都不落在任何节点框内。
        gx = GAP_X[NODE_COL[src]]
        pts = f"{p1[0]:.1f},{p1[1]:.1f} {gx:.1f},{p1[1]:.1f} {gx:.1f},{p2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}"
        L.append(f'<polyline points="{pts}" fill="none" '
                  f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    else:
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名(1~2 行) + 短语(1~2 行) + 右上角 § 徽标)
SYMBOL_1LINE_Y, SYMBOL_2LINE_Y1, SYMBOL_2LINE_Y2 = 34, 24, 40
PHRASE_1LINE_Y, PHRASE_2LINE_Y1, PHRASE_2LINE_Y2 = 71, 66, 80
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_lines = symbol.split("\n")
    sym_ys = [y + SYMBOL_1LINE_Y] if len(sym_lines) == 1 else [y + SYMBOL_2LINE_Y1, y + SYMBOL_2LINE_Y2]
    for line, ly in zip(sym_lines, sym_ys):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ly:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(line)}</text>')
    phrase_lines = phrase.split("\n")
    phrase_ys = [y + PHRASE_1LINE_Y] if len(phrase_lines) == 1 else [y + PHRASE_2LINE_Y1, y + PHRASE_2LINE_Y2]
    for line, ly in zip(phrase_lines, phrase_ys):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ly:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(line)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 8, y, sec)

# 底部阅读路线:复用列坐标 COLX,§ 徽标与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
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
print(f"wrote {out}: {w:.0f}x{h:.0f}")
