#!/usr/bin/env python3
"""第 36 章「本章地图」——投机解码提议侧(vllm_ascend/spec_decode/)源码剖面图。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py 模板改写：
不可变机制(esc/cjk_text_width/badge/配色/图例规则/entry-exit 接口桩)原样保留，
仅在末尾新增一条「循环反馈」曲线(本章特有:提议→验证→prepare_inputs→回灌下一步提议，
不是单入单出的直线管线，而是一个环)，配色不复用绿/橙/蓝三色语义，另开一色 + 图例项。

节点预算 9(entry / suffix / ngram / ngram_gpu / medusa / heavy_entry / base_core / exit /
prepare_inputs) ≤ 12。本章标题为编号标题(## 35.1 … ## 35.9)，站牌用 §35.N。

设计要点：
- 工厂层(entry)一次 if-elif 分发到 5 条分支：4 条薄壳(suffix/ngram/ngram_gpu/medusa，
  各自真实类)直接把 draft_token_ids 交给验证侧(exit=AscendRejectionSampler，全部
  proposer 共用同一个验证出口，§36.8)；第 5 条走重量级——heavy_entry(以
  AscendEagleProposer 为代表，draft_model 与它同构，见 §36.5 原文)先构造再调用
  AscendSpecDecodeBaseProposer(base_core，2043 行重写核心，§36.6)，也汇入同一个
  exit。
- exit 之后只有重量级路径继续：AscendRejectionSampler 给出的拒绝计数喂给
  prepare_inputs(§36.7，纯 host 端索引收缩)，收缩后的结果要回灌给下一步的
  propose()——这是一个环，不是继续往右的直线，所以用一条单独的青色虚线曲线
  从 prepare_inputs 弧线绕回 base_core，而不是塞进主线蓝的 EDGES 列表(那样会跟
  正向的 base_core→exit 边在同一行反向重叠，视觉上看不出"汇合"反而像穿模)。
- 薄壳节点符号偏长(Ascend+Xxx+Proposer 常 20-28 字符)，统一用 "\n" 在 "Ascend"
  后手工换行——不是逐个心算宽度，是让全部 8 个 Ascend* 符号维持同一种两行视觉
  节奏(唯一例外 prepare_inputs，本身不到 Ascend 前缀，够短，单行)。

六项自查记录见文件末尾 [SELF-CHECK] 注释(渲染→Read PNG 亲眼看后如实记录)。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算——全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["工厂层(配置分发)", "薄壳层(继承 vLLM 父类)", "重量级层(昇腾自建 base)"]  # 泳道,上→下

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名(可含 "\n" 机械换行,不改变拼写), 一行短语, §编号)
NODES = [
    ("entry",          0, 0, 0, "get_spec_\ndecode_method",       "按 method 分 8 支",     "§36.1"),
    ("suffix",         1, 1, 0, "Ascend\nSuffixDecodingProposer", "一行转发父类",           "§36.2"),
    ("ngram",          1, 1, 1, "Ascend\nNgramProposer",          "筛请求+调父类算法",       "§36.3"),
    ("ngram_gpu",      1, 1, 2, "Ascend\nNgramProposerNPU",       "GPU kernel 全 stub",    "§36.3"),
    ("medusa",         1, 1, 3, "Ascend\nMedusaProposer",         "gather 末位隐藏态",       "§36.4"),
    ("heavy_entry",    2, 1, 0, "Ascend\nEagleProposer",          "draft_model 同构",      "§36.5"),
    ("base_core",      2, 2, 0, "Ascend\nSpecDecodeBaseProposer", "ACLGraph+并行组+MLA",   "§36.6"),
    ("exit",           0, 3, 0, "Ascend\nRejectionSampler",       "接受最长前缀+1 bonus",   "§36.8"),
    ("prepare_inputs", 2, 4, 0, "prepare_inputs",                 "按拒绝数收缩输入",        "§36.7"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝;循环反馈边另画,不在这里
    ("entry", "suffix"), ("entry", "ngram"), ("entry", "ngram_gpu"), ("entry", "medusa"),
    ("entry", "heavy_entry"),
    ("heavy_entry", "base_core"),
    ("suffix", "exit"), ("ngram", "exit"), ("ngram_gpu", "exit"), ("medusa", "exit"),
    ("base_core", "exit"),
    ("exit", "prepare_inputs"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("重量级路径(真跑前向)",
     [(0, "§36.1"), (1, "§36.5"), (2, "§36.6"), (3, "§36.8"), (4, "§36.7")], True),
    ("薄壳路径(近零成本)",
     [(0, "§36.1"), (1, "§36.2"), (3, "§36.8")], False),
]
LEGEND = [
    ("#22c55e", "入口:配置阶段构造 proposer"),
    ("#3b82f6", "章内调用边"),
    ("#f97316", "出口:交给验证侧(ch33)"),
    ("#0891b2", "循环:prepare_inputs 回灌下一步"),
]
TITLE = "第 36 章 · 投机解码提议侧剖面(工厂分发→薄壳/重量级→循环反馈)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_LOOP = "#0891b2"  # 本章新增语义色:循环反馈边(prepare_inputs → 下一步 propose)
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 190, 90  # NODE_H 加高以容纳 2 行符号(Ascend\nXxxProposer)
COL_GAP, ROW_GAP = 42, 22
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_W, BADGE_H = 46, 20
LOOP_GAP = 64  # 泳道底部到阅读路线之间留出的空当,专给循环反馈曲线走线(不可与 8px 旧值混用)

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

routes_top = lanes_bottom + LOOP_GAP
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
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN), ("Loop", C_LOOP))
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

# 节点(圆角框 + 真实符号名(1~2 行) + 一行短语(始终锚在节点下半区) + 右上角 § 徽标)
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
                  f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(line)}</text>')
    phrase_lines = phrase.split("\n")
    phrase_ys = [y + PHRASE_1LINE_Y] if len(phrase_lines) == 1 else [y + PHRASE_2LINE_Y1, y + PHRASE_2LINE_Y2]
    for line, ly in zip(phrase_lines, phrase_ys):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ly:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(line)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 8, y, sec)

# 循环反馈曲线(本章特有,不在通用 EDGES 里):prepare_inputs 收缩完的输入回灌给
# 下一步的 propose()，也就是绕回 base_core(AscendSpecDecodeBaseProposer)。
# 画在泳道下方、阅读路线上方的空当(LOOP_GAP)里，两端分别贴在两个节点的下边框,
# 与正向的 base_core→exit 主线边(在泳道内部、走的是右侧边而非下边框)完全分离,
# 不会重叠/穿模。
pi_x, pi_y = NODE_XY["prepare_inputs"]
pi_cx, pi_bottom = pi_x + NODE_W / 2, pi_y + NODE_H
bc_x, bc_y = NODE_XY["base_core"]
bc_cx, bc_bottom = bc_x + NODE_W / 2, bc_y + NODE_H
loop_dip = lanes_bottom + LOOP_GAP * 0.62
L.append(f'<path d="M {pi_cx:.1f},{pi_bottom:.1f} C {pi_cx:.1f},{loop_dip:.1f} '
          f'{bc_cx:.1f},{loop_dip:.1f} {bc_cx:.1f},{bc_bottom:.1f}" fill="none" '
          f'stroke="{C_LOOP}" stroke-width="2" stroke-dasharray="7,4" marker-end="url(#mLoop)"/>')
L.append(f'<text x="{(pi_cx + bc_cx) / 2:.1f}" y="{loop_dip + 15:.1f}" text-anchor="middle" '
          f'font-family="sans-serif" font-size="11" fill="{C_LOOP}">'
          f'{esc("下一步 propose:收缩后的输入回灌")}</text>')

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

# [SELF-CHECK](渲染→Read PNG 亲眼看后如实记录 —— 见下方 ROUND 记录，随实际渲染结果更新)
