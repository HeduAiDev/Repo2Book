#!/usr/bin/env python3
"""第 32 章「本章地图」——投机解码源码剖面图。

改自 .claude/skills/svg-diagram/references/example-chapter-map.py（不可变项照办：
§徽标胶囊样式 / 入口绿#22c55e-出口橙#f97316-主线蓝#3b82f6 / 高亮实线蓝-次要虚线灰 /
cjk_text_width() 宽度估算）。本章数据流是单向线性管线（proposer 产草稿 → 摊平 index →
一次目标前向 → rejection sampling 两条 kernel → 还原变长输出），无多态分流，故不需要
5 态判定那种模板结构；改用「两个入口（n-gram / EAGLE 系）→ 单一主链 → 两条 kernel
分支（greedy / random）→ 单一出口」的拓扑。

■ 相对模板的必要改动（本章数据决定，非任意改动）：
  1. 两个真实入口符号（NgramProposer / SpecDecodeBaseProposer.propose 分属两个
     proposer 家族，都在 §32.1 讨论），改用一根跨两行的入口桩 + 两条箭头分别射向
     两个节点，而不是模板默认的单入口单箭头——因为本章确实有两条独立的草稿来源。
  2. 4 个符号名过长（22~30 字符），按下划线/驼峰边界机械换行成两行（内容不变，
     不做任何截断/改写），节点框相应加高（NODE_H 58→94）以容纳"两行符号+一行短语"。

■ 六项自查记录（渲染→Read PNG 亲眼看后如实记录）：
  claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
  arrows_attached=True     cjk_rendered=True         reading_order_clear=True
  见 figure-manifest.json 的 blind_review 记录首轮发现与修复细节。

用法：python3 gen_chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(本章数据) ----------------
LANES = ["草稿产出层 · spec_decode", "摊平/调度/还原层 · worker + rejection_sampler", "kernel 执行层 · Triton"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名(str 或多行 list), 一行短语, §编号)
NODES = [
    ("ngram",     0, 0, 0, "NgramProposer",
     "KMP 最长后缀,抄历史草稿", "§32.1"),
    ("eagle",     0, 0, 1, ["SpecDecodeBase", "Proposer.propose"],
     "EAGLE/MTP 链式采草稿", "§32.1"),
    ("metadata",  1, 1, 0, ["_calc_spec_decode", "_metadata"],
     "累积和建三组 index", "§32.2"),
    ("forward",   1, 2, 0, ["RejectionSampler", ".forward"],
     "切 bonus/target logits", "§32.3"),
    ("dispatch",  1, 3, 0, "rejection_sample",
     "按 greedy/random 分派", "§32.4"),
    ("greedy",    2, 4, 0, ["rejection_greedy", "_sample_kernel"],
     "draft==argmax,全中补bonus", "§32.4"),
    ("recovered", 2, 4, 1, ["sample_recovered", "_tokens_kernel"],
     "残差(p-q)+,Gumbel-max采样", "§32.5"),
    ("random",    2, 5, 1, ["rejection_random", "_sample_kernel"],
     "以 min(1,p/q) 接受草稿", "§32.4"),
    ("parse",     1, 6, 0, "parse_output",
     "过滤-1,还原变长list", "§32.6"),
]
EDGES = [  # (src_id, dst_id) —— 调用/数据依赖边，统一主线蓝
    ("ngram", "metadata"), ("eagle", "metadata"),
    ("metadata", "forward"),
    ("forward", "dispatch"),
    ("dispatch", "greedy"), ("dispatch", "recovered"),
    ("recovered", "random"),
    ("greedy", "parse"), ("random", "parse"),
]
ENTRY_IDS = ["ngram", "eagle"]
EXIT_IDS = ["parse"]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("随机采样链(残差恢复)",
     [(0, "§32.1"), (1, "§32.2"), (2, "§32.3"), (3, "§32.4"), (4, "§32.5"), (5, "§32.4"), (6, "§32.6")], True),
    ("greedy 捷径(不采残差)",
     [(3, "§32.4"), (6, "§32.6")], False),
]
LEGEND = [("#22c55e", "入口:proposer 产出草稿"), ("#3b82f6", "章内主线调用/数据依赖边"), ("#f97316", "出口:parse_output 返回引擎")]
TITLE = "第 32 章 · 投机解码剖面：proposer → 摊平 index → rejection sampling"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数;NODE_W/H 按本章符号长度收窄/加高) ----------------
NODE_W, NODE_H = 164, 94
COL_GAP, ROW_GAP = 22, 20
EDGE_MARGIN, STUB_W, STUB_H = 14, 56, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32
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

# 入口接口桩:本章有两个真实入口符号(n-gram / EAGLE 系两个 proposer 家族都在 §32.1
# 讨论),用一根跨两行的桩 + 两条箭头分别射向两个节点(而非模板默认单入口单箭头)。
_entry_ys = [NODE_XY[nid][1] + NODE_H / 2 for nid in ENTRY_IDS]
_entry_top = min(NODE_XY[nid][1] for nid in ENTRY_IDS)
_entry_bot = max(NODE_XY[nid][1] + NODE_H for nid in ENTRY_IDS)
_entry_mid = (_entry_top + _entry_bot) / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{_entry_top:.1f}" width="{STUB_W}" height="{_entry_bot - _entry_top:.1f}" '
         f'rx="16" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{_entry_mid + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
for nid, ey in zip(ENTRY_IDS, _entry_ys):
    ex = NODE_XY[nid][0]
    L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
             f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')

# 出口接口桩(单一出口:parse_output)
for nid in EXIT_IDS:
    xx, xy = NODE_XY[nid]
    xy += NODE_H / 2
    sx = w - EDGE_MARGIN - STUB_W
    L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
             f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
    L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
    L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
             f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝);多条边汇入同一节点时终点 y 各偏移,让"汇合"可见
# 特例:ngram→metadata 若走直线对角线,会斜穿 eagle 节点(同列下一行)右上角的
# §32.1 徽标——两行节点纵向紧邻时,上一行发往右下方节点的直线天然会扫过下一行
# 节点的顶角。改成"水平→垂直→水平"的折线,垂直段设在 eagle 徽标右边缘(x=274)
# 之外(x=282),绕开徽标再拐入 metadata。
_ELBOW_BEND_X = {("ngram", "metadata"): NODE_XY["eagle"][0] + NODE_W + 16}

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
    bend_x = _ELBOW_BEND_X.get((src, dst))
    if bend_x is None:
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    else:
        pts = f'{p1[0]:.1f},{p1[1]:.1f} {bend_x:.1f},{p1[1]:.1f} {bend_x:.1f},{p2[1]:.1f} {p2[0]:.1f},{p2[1]:.1f}'
        L.append(f'<polyline points="{pts}" fill="none" '
                  f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名[单行或两行机械换行] + 一行短语 + 右上角 § 徽标)
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_lines = symbol if isinstance(symbol, list) else [symbol]
    cx = x + NODE_W / 2
    if len(sym_lines) == 1:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.38:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
    else:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.28:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.48:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[1])}</text>')
    L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.80:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
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
print(f"wrote {out} ({w:.0f}x{h:.0f}, aspect {w/h:.2f}:1)")
