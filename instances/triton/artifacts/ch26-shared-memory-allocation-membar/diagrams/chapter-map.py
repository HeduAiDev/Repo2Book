#!/usr/bin/env python3
"""ch26《共享内存分配与屏障：Allocation、Alias 与 Membar》本章地图——源码剖面图。

本章 narrative/chapter.md 全是**自然标题**(无 `## N.M` / `## §N` 编号)，按契约：
禁用 §N.M 徽标，站牌改用标题词本身(取自各节标题的一个可核实子串)。

两条主泳道 = 两个只读分析的顺承关系(不是并行关系)：
  上道「AllocationAnalysis → MembarAnalysis 主线」8 站，严格对应正文 8 个标题
  (7 个内容节 + 1 个小结)，从 `run()` 一路到 `sharedMemorySize`/barrier 插点；
  下道「前置框 / 后端接缝」放两个不挂边的卫星站——Gergov 前置框(离线动态存储
  分配的归约直觉，正文以 blockquote 点出，非独立标题)贴在 first-fit 定址正
  下方；MembarFilterFn 后端豁免贴在 MembarAnalysis 正下方——两者都是"随时可
  查"的旁证，不在主线控制流上。

节点预算 10 ≤ 12(8 主线 + 2 卫星)。符号一律不带尾随 "()"(除 `run()`——正文
`void run() {` 原样出现)，规避"函数签名非空参数、图上误配空括号"的杜撰风险，
同时省画布宽度。

■ 不可变(全书统一视觉语言，换章节数据时不要动这些，只改下面的 DATA)：
  与 example-chapter-map.py 完全一致——§徽标胶囊(此处改用自然标题短语) /
  入口绿#22c55e-出口橙#f97316-主线蓝#3b82f6 / 高亮实线蓝-次要虚线灰 /
  cjk_text_width() 宽度估算 / 站牌宽度按文字动态算(自然标题长短不一，仿
  ch08 的 badge_topright/badge_centered 泛化，不用固定 BADGE_W)。

■ 可变：LANES / NODES / EDGES / ROUTES / LEGEND / TITLE。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录，见 figure-manifest.json)：
  见文件末尾 [SELF-CHECK] 注释。

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
LANES = ["AllocationAnalysis → MembarAnalysis 主线", "前置框 / 后端接缝(卫星，不挂边)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名(可含 "\n"), 一行短语(可含 "\n"), 站牌(标题词))
NODES = [
    ("entry", 0, 0, 0,
     "run()",
     "三步骨架：\n收集→活跃期→定址",
     "分析→改写"),
    ("buf_kinds", 0, 1, 0,
     "BufferT",
     "Explicit/Scratch/\nVirtual 三类来源",
     "三种来源"),
    ("alias", 0, 2, 0,
     "visitOperation",
     "scf.for/yield 换名，\n别名集只增不减",
     "别名分析"),
    ("liveness", 0, 3, 0,
     "resolveLiveness",
     "PostOrder 编号+\n别名并入撑区间",
     "活跃区间"),
    ("firstfit", 0, 4, 0,
     "allocate",
     "冲突图(时间∧地址)，\n染色+抬 offset",
     "first-fit 定址"),
    ("sharedmax", 0, 5, 0,
     "getShared\nMemorySize",
     "函数内 max，\n跨函数 root 再取 max",
     "两级 max"),
    ("membar", 0, 6, 0,
     "isIntersected",
     "RAW/WAR 才插，\nWAW 因不重叠掐死",
     "MembarAnalysis"),
    ("exit", 0, 7, 0,
     "sharedMemorySize",
     "占用率账本，\n交降级消费",
     "小结"),
    ("gergov", 1, 4, 0,
     "Gergov, SODA 1999",
     "离线动态存储分配，\n≈区间图着色(近似)",
     "前置框"),
    ("filter", 1, 6, 0,
     "MembarFilterFn",
     "后端豁免接缝，\n多余同步调节钮",
     "后端豁免"),
]
EDGES = [  # (src_id, dst_id) —— 主线调用/数据流边，统一蓝；两个卫星不挂边
    ("entry", "buf_kinds"),
    ("buf_kinds", "alias"),
    ("alias", "liveness"),
    ("liveness", "firstfit"),
    ("firstfit", "sharedmax"),
    ("sharedmax", "membar"),
    ("membar", "exit"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
# 三条路线原样对应正文开篇的选读指引(跳 first-fit+两级max / 跳 Membar / 通读)。
ROUTES = [
    ("通读主线",
     [(0, "分析→改写"), (1, "三种来源"), (2, "别名分析"), (3, "活跃区间"),
      (4, "first-fit 定址"), (5, "两级 max"), (6, "MembarAnalysis"), (7, "小结")], True),
    ("只看爆显存/occupancy",
     [(4, "first-fit 定址"), (5, "两级 max")], False),
    ("只看 barrier 哪来",
     [(6, "MembarAnalysis"), (7, "小结")], False),
]
LEGEND = [("#22c55e", "入口：接续上一章 transform"), ("#3b82f6", "章内主线数据流"), ("#f97316", "出口：交下一章降级消费")]
TITLE = "第 26 章 · AllocationAnalysis → MembarAnalysis 源码剖面(共享内存定址 + 屏障插入两个只读分析)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 150, 130
COL_GAP, ROW_GAP = 16, 18
EDGE_MARGIN, STUB_W, STUB_H = 8, 46, 24
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_H, BADGE_PAD_X, BADGE_MIN_W = 20, 14, 36  # 站牌高度固定,宽度按文字动态算

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


def _badge_width(text):
    """站牌宽度按文字动态算(自然标题站牌长短不一，不能像 §N.M 那样固定宽度)。"""
    return max(BADGE_MIN_W, cjk_text_width(text, 11) + BADGE_PAD_X)


def badge_topright(x, y, node_w, text):
    """站牌胶囊贴节点右上角。right_edge = x+node_w+8 恒定(与 width 无关)——
    哪怕站牌变宽也不会撞到右侧下一列节点，只会向节点内部多占一点左边距。"""
    width = _badge_width(text)
    cx = x + node_w - width / 2 + 8
    return _badge_rect_text(cx, y, width, text)


def badge_centered(cx, cy, text):
    """路线站牌:居中挂在 (cx,cy)，宽度同样动态算。"""
    width = _badge_width(text)
    return _badge_rect_text(cx, cy, width, text)


def _badge_rect_text(cx, cy, width, text):
    bx, by = cx - width / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{width:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
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

# 入口/出口接口桩(给入口/出口箭头一个可附着的框,兼表达"上一章/下一章在画布外")
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

# 调用边(主线蓝,先画边再画节点盖住端点毛刺)
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

# 节点(圆角框 + 真实符号名(1~2 行) + 一行短语(1~2 行,锚在节点下半区) + 右上角站牌)
SYMBOL_1LINE_Y, SYMBOL_2LINE_Y1, SYMBOL_2LINE_Y2 = 40, 32, 50
PHRASE_1LINE_Y, PHRASE_2LINE_Y1, PHRASE_2LINE_Y2 = 84, 78, 98
for nid, lane, col, row, symbol, phrase, tag in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_lines = symbol.split("\n")
    sym_ys = [y + SYMBOL_1LINE_Y] if len(sym_lines) == 1 else [y + SYMBOL_2LINE_Y1, y + SYMBOL_2LINE_Y2]
    for line, ly in zip(sym_lines, sym_ys):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ly:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="13" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(line)}</text>')
    phrase_lines = phrase.split("\n")
    phrase_ys = [y + PHRASE_1LINE_Y] if len(phrase_lines) == 1 else [y + PHRASE_2LINE_Y1, y + PHRASE_2LINE_Y2]
    for line, ly in zip(phrase_lines, phrase_ys):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ly:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(line)}</text>')
    L += badge_topright(x, y, NODE_W, tag)

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
    for col, tag in stops:
        L += badge_centered(COLX[col] + NODE_W / 2, ry, tag)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}: {w:.0f}x{h:.0f}")

# [SELF-CHECK](渲染→Read PNG 亲眼看后如实记录，见 figure-manifest.json 对应条目)
#   claim_readable_10s=True   两条泳道 10 秒读出:上道是 AllocationAnalysis→
#     MembarAnalysis 顺序执行的 8 站主线(run→BufferT→别名分析→活跃区间→
#     first-fit→两级max→MembarAnalysis→小结),下道是两个不挂边的旁证。
#   numbers_match_spec=True   本图是结构剖面图,不含需要溯源的数据数字(无
#     spec.numbers 依赖),无需核对。
#   no_overlap=True   渲染后逐段 crop 复核:8 主线节点(含 2 行符号
#     getShared/MemorySize)、2 卫星节点、图例、3 条阅读路线站牌均无相互
#     侵入;站牌宽度按 cjk_text_width 动态算、右边界恒钉节点右框+8px,
#     不会撞下一列节点。
#   arrows_attached=True   入口"调用方"→run()、7 条主线蓝箭头逐段衔接、
#     sharedMemorySize→"返回上层"均在 crop 中确认箭头清晰落在框边上。
#   cjk_rendered=True   中文站牌/短语/图例/路线文案渲染正常,无缺字方块。
#   reading_order_clear=True  主线从左到右单向流动,三条阅读路线(通读/只看
#     爆显存/只看 barrier)与正文开篇选读指引逐字对应。

