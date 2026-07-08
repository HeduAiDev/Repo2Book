#!/usr/bin/env python3
"""第 10 章(Logprobs 装配与字节回退修正)——本章地图:源码剖面图。

真实调用链(两段纵向折行,画布预算:宽 ≤1500 且宽高比 ≤2.6:1):
  上段「入口 · 初始化 · 双路装配」——update_from_output(§10.1,process_outputs 循环内
    的分派口,接绿色调用方箭头)按两个 if 分派到 _update_sample_logprobs(§10.3)/
    _update_prompt_logprobs(§10.4)两条泳道;from_new_request(§10.2)是同段内的一次性
    前置步骤(容器/cumulative 初值由它建好,供两条泳道后续读写),用灰色虚线 + 说明文字
    与主链区分——它不是 update_from_output 触发的调用,不该画成同色实线主线。
  下段「修正 · 写入 · 下游」——两条泳道汇入 _correct_decoded_token(§10.5,本章技术核心,
    由 _verify_tokens 触发),再到 append_logprobs_for_next_position(§10.7,rank 链 +
    flat/nested 分叉)→ FlatLogprobs(§10.6,写入承接的扁平结构)→ _new_completion_output
    (§10.8,下游取用,接橙色出口箭头)。
两段之间的桥接带画两条跨段主线蓝箭头(sample→correct、prompt→correct),汇入同一节点
时按 x 方向错开,避免视觉上看不出"汇合"。

用法: python3 gen_chapter-map.py → 同目录 chapter-map.svg

[FIX-ROUND-2](自查阶段:渲染→Read PNG 后发现、修复、重渲重核,替换第一轮记录):
  第一轮 PREREQ_EDGES 的连线公式直接搬了跨段桥接的"src 下沿→dst 上沿"写法,但
  init 实际在 entry 正下方一行(init 是 src、entry 是 dst,dst 在 src 上方)——公式
  方向反了,画出的线从 init 下沿一路穿过 init 和 entry 两个框体(被节点矩形遮住大半,
  只露出行间距那一小段),说明文字"一次性前置，非本次调用触发"又被同一时刻画出的
  entry→prompt 主线斜边压穿。本轮改为按实际相对位置手写 p1=src 上沿中点、
  p2=dst 下沿中点(线完全落在两节点间的行间距内),并把说明文字缩短为"一次性前置"
  (原文字太长,即使方向修对了也会紧贴主线斜边),重渲染后 Read PNG 复核:虚线箭头
  完整落在行间距内、不再穿框,说明文字与主线斜边净空清楚,现在的 no_overlap=True
  是修复后重新核对的结果。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):逐字符判定——
    全角(ord>0x2E80)按 1.0×size,半角(ASCII/拉丁等)按 0.58×size,求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(本章数据) ----------------
LANES = ["入口 · 初始化 · 双路装配", "修正 · 写入 · 下游取用"]  # 折成上下两段

# (节点id, 段下标(0=上段/1=下段), 段内列, 段内行号, 真实符号名, 一行短语, § 站牌)
NODES = [
    ("entry",   0, 0, 0, "update_from_output",
     "process_outputs 循环内的分派口，两个 if 各表一路", "§10.1"),
    ("init",    0, 0, 1, "from_new_request",
     "按 3 个三元分支建容器；cumulative 初值 0.0", "§10.2"),
    ("sample",  0, 1, 0, "_update_sample_logprobs",
     "numpy 已 tolist；cumulative += logprobs[0]", "§10.3"),
    ("prompt",  0, 1, 1, "_update_prompt_logprobs",
     "torch 张量自行 Pythonize；不维护 cumulative", "§10.4"),
    ("correct", 1, 0, 0, "_correct_decoded_token",
     "被 _verify_tokens 触发；上下文增长 1..4 补全字符", "§10.5"),
    ("append",  1, 1, 0, "append_logprobs_for_next_position",
     "rank 链 (rank, 1..K)；按格式分叉写入", "§10.7"),
    ("flat",    1, 2, 0, "FlatLogprobs",
     "6 条原生列表 + 区间索引，对象数 O(1)", "§10.6"),
    ("exit",    1, 3, 0, "_new_completion_output",
     "DELTA 切尾 -len(token_ids)；cumulative 不切", "§10.8"),
]
# 主链调用边(蓝实线):段内 = 左→右;跨段(sample/prompt → correct)= 桥接带竖向。
EDGES = [
    ("entry", "sample"), ("entry", "prompt"),
    ("sample", "correct"), ("prompt", "correct"),
    ("correct", "append"), ("append", "flat"), ("flat", "exit"),
]
# 前置边(灰虚线,非调用关系——容器初始化早于、且不由 update_from_output 触发):
# 只画 1 条,与主链视觉区分,旁边配简短说明文字(init 在 entry 正下方一行,边只占
# 行间距那一小段竖直空当,不与任何主线交叉)。
PREREQ_EDGES = [("init", "entry", "一次性前置")]
# (路线名, [§站牌,...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
READING_ORDER = ["§10.1", "§10.2", "§10.3", "§10.4", "§10.5", "§10.6", "§10.7", "§10.8"]
ROUTES = [
    ("全程精读（按章节顺序）", READING_ORDER, True),
    ("只看字节回退这一段（本章技术核心）", ["§10.1", "§10.5"], False),
    ("只看两种存储格式（flat vs nested）", ["§10.2", "§10.6"], False),
]
LEGEND = [
    ("#22c55e", "入口：process_outputs 循环调用进入"),
    ("#3b82f6", "章内主线：真实符号调用/数据流"),
    ("#f97316", "出口：写入 CompletionOutput 后返回上层"),
]
TITLE = "第 10 章 · Logprobs 装配剖面：双路分派 → 字节回退修正 → 下游取用"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
C_PREREQ_CAPTION = "#64748b"

# ---------------- 几何常量(全计算,零魔数) ----------------
BADGE_FONT_SIZE = 11
BADGE_PAD_X = 14
BADGE_H = 20


def badge_width(text):
    return max(46.0, cjk_text_width(text, BADGE_FONT_SIZE) + BADGE_PAD_X * 2)


NODE_H = 70
COL_GAP, ROW_GAP = 30, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 46
INTER_LANE_GAP = 170  # 桥接带:两段之间的空白,专放跨段箭头 + 简短说明

_SYMBOL_FONT, _PHRASE_FONT = 13, 10.5
_NODE_TEXT_PAD = 20
NODE_W = max(
    190,
    max(cjk_text_width(sym, _SYMBOL_FONT) for *_, sym, _, _ in NODES) + _NODE_TEXT_PAD,
    max(cjk_text_width(ph, _PHRASE_FONT) for *_, ph, _ in NODES) + _NODE_TEXT_PAD,
)
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 16

n_cols = max(n[2] for n in NODES) + 1
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_band = [0] * len(LANES)
for _id, band, col, row, *_ in NODES:
    rows_per_band[band] = max(rows_per_band[band], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_band]

band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for i, bh in enumerate(band_h):
    if i > 0:
        _cum += INTER_LANE_GAP
    band_top.append(_cum)
    _cum += bh
lanes_bottom = _cum

NODE_XY = {}
for nid, band, col, row, *_ in NODES:
    x = COLX[col]
    y = band_top[band] + LANE_LABEL_H + BAND_PAD + row * (NODE_H + ROW_GAP)
    NODE_XY[nid] = (x, y)
NODE_BY_ID = {n[0]: n for n in NODES}

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
    """§ 徽标胶囊,居中挂在 (cx,cy)——宽度按文字自适应。"""
    bw = badge_width(text)
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="{BADGE_FONT_SIZE}" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.1f} {h:.1f}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN), ("Prereq", C_ROUTE_DIM))
) + '</defs>')
L.append(f'<rect width="{w:.1f}" height="{h:.1f}" fill="white"/>')

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

# 泳道背景 + 标签
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w:.1f}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w:.1f}" y2="{band_top[i]:.1f}" '
              f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
    L.append(f'<line x1="0" y1="{band_top[i] + band_h[i]:.1f}" x2="{w:.1f}" y2="{band_top[i] + band_h[i]:.1f}" '
              f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩
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

# 主链调用边:段内(band 相同)= 左中→右中;跨段(band 不同)= 上段下沿中点→下段上沿中点。
# 多条边汇入同一 dst 时,按方向(段内 y / 跨段 x)错开,否则重合的端点看不出"汇合"。
_dst_total = {}
for src, dst in EDGES:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst in EDGES:
    src_band = NODE_BY_ID[src][1]
    dst_band = NODE_BY_ID[dst][1]
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    if src_band == dst_band:
        y_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        p2 = (x2, y2 + NODE_H / 2 + y_offset)
    else:
        x_offset = (i - (n - 1) / 2) * 24 if n > 1 else 0
        p1 = (x1 + NODE_W / 2, y1 + NODE_H)   # 上段下沿中点(本章跨段边均上→下)
        p2 = (x2 + NODE_W / 2 + x_offset, y2)  # 下段上沿中点,多条汇入时按 x 错开
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 前置边(灰虚线,非调用关系):src(init) 在 dst(entry) 正下方一行,边从 src 上沿中点
# 竖直向上指到 dst 下沿中点,整条边都落在两节点之间的行间距里,不与任何主线交叉。
for src, dst, cap in PREREQ_EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W / 2, y1)             # src(init) 上沿中点
    p2 = (x2 + NODE_W / 2, y2 + NODE_H)    # dst(entry) 下沿中点
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_ROUTE_DIM}" stroke-width="1.5" stroke-dasharray="5,4" '
              f'marker-end="url(#mPrereq)"/>')
    L.append(f'<text x="{p1[0] + 10:.1f}" y="{(p1[1] + p2[1]) / 2 + 4:.1f}" font-family="sans-serif" '
              f'font-size="10.5" font-style="italic" fill="{C_PREREQ_CAPTION}">{esc(cap)}</text>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角 § 徽标)
for nid, band, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W:.1f}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{_SYMBOL_FONT}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{_PHRASE_FONT}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    bw = badge_width(sec)
    L += badge(x + NODE_W - bw / 2 + 8, y, sec)

# 底部阅读路线:8 个站牌按 READING_ORDER 均匀分布在整个画布宽度上(独立于图上节点的
# 段内列号——init 在段内列 0、correct 也在列 0,若借列号两个不同站牌会叠在同一 x)。
_route_label_w = max(cjk_text_width(name, 12) for name, *_ in ROUTES)
_route_left = 16 + _route_label_w + 24
_n_stops = len(READING_ORDER)
_route_x = {name: _route_left + i * (w - PAD_R - _route_left) / (_n_stops - 1)
            for i, name in enumerate(READING_ORDER)}

L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first, x_last = _route_x[stops[0]], _route_x[stops[-1]]
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for sec in stops:
        L += badge(_route_x[sec], ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w:.0f}x{h:.0f}, NODE_W={NODE_W:.0f})")
