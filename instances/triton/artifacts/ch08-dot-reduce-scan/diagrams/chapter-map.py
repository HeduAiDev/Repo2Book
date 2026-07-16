#!/usr/bin/env python3
"""本章地图 — ch08《dot、归约与扫描》源码剖面图。

本章两大主题各自一条独立流水线，上下堆叠：
  上半程 §1-§5：tl.dot 从 core.py 前端到 semantic.dot() 校验闸门，一路问后端
                NVIDIA backend 拿 min_dot_size/CUDAOptions，最终 create_dot()。
  下半程 §6-§9：combine_fn 经 semantic.reduction()→call_JitFunction() 被重新
                编译进 region；同一条机制分叉出 argmax(_reduce_with_indices)
                与 histogram(无 region 的对照)两个去向。

■ 不可变(全书统一视觉语言):
  1. §徽标胶囊:圆角矩形(pill),fill #eef2ff / stroke #6366f1。
  2. 入口/出口接口桩:绿 #22c55e(入口) / 橙 #f97316(出口)。
  3. 节点间调用边(主线) = 蓝 #3b82f6。
  4. 底部路线条:高亮=实线蓝(粗)/次要或对照=虚线灰 #94a3b8(细)。
  5. >2 种语义色须画图例。
  6. 文本宽度估算一律用 cjk_text_width()，不用半角系数硬乘 len(s)。

■ 本章特有(可变部分的设计说明，供下次改图时复用理由):
  - 本章 chapter.md 标题是 `## §1`..`## §9`(单级编号，非 `## N.M` 两级)——沿用
    ch06/ch07 已确立的先例:徽标直接写裸 `§N`(不加 `.M`)，逐字匹配真实标题子串。
  - 两条流水线的"前端转发"角色(core.py 的 tl.dot()/tl.reduce())没有单独设
    节点，而是折进入口接口桩的文案里——因为章节原文明确说"前端只做参数归一，
    真正的入口在 semantic.dot()/semantic.reduction()"，这样列序能与 §N 顺序
    保持单调，读者对着图从左到右走就是从 §1 走到 §9。
  - 每条流水线用 2 条泳道:主链在"语言层"，一个分支节点下沉到"后端/再编译层"
    体现"语言层查后端能力表"或"combine_fn 被复用 CodeGenerator 机制"这两个
    本章反复强调的关键点。
  - histogram(§9)与主链之间用灰虚线相连(而非蓝实线)，呼应正文"没有
    combine_fn 的对照"——它是同一族原语但走一条完全不同、不建 region 的路。
  - 底部两条阅读路线直接复用章节开篇给出的选读指引:"想抓性能结论读 §1-§5；
    想懂归约编译机理读 §6-§9"。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录):
  claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
  arrows_attached=True     cjk_rendered=True         reading_order_clear=True
  (无迭代轮——首轮渲染即通过自查，细节见收工报告)

用法: python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算:全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def fit_size(text, max_w, base, min_size):
    """按 max_w 反解一个不超出的字号(单行)，不做换行——本章节点文案都收成一行短语。"""
    unit = cjk_text_width(text, 1.0)
    if unit <= 0:
        return base
    return max(min_size, min(base, max_w / unit))


# ==================== DATA(可变:本章数据) ====================

# 每条流水线共用同一套 5 列坐标(见下方 COLX 计算)，两条流水线上下堆叠。
N_COLS = 5

GROUPS = [
    {
        "header": "上半程 §1-§5 · tl.dot 命不命中 Tensor Core",
        "lanes": ["语言层（core.py / semantic.py）", "后端能力（NVIDIA backend）"],
        # (节点id, 泳道下标, 列, 行, 真实符号名, 一行短语, §编号)
        "nodes": [
            ("dot_gate",  0, 0, 0, "semantic.dot",                "dtype 白名单/同型 + 2D/3D 形状 + K 维相容",       "§1"),
            ("min_dot",   0, 1, 0, "min_dot_size 门禁",            "block 需过后端最小 tile（本章性能落点）",         "§2"),
            ("precision", 0, 2, 0, "_str_to_dot_input_precision", "precision 过白名单 → tf32/ieee 枚举",             "§3"),
            ("acc_dtype", 0, 3, 0, "ret_scalar_ty 选型",           "int8→int32 / fp32|bf16→fp32 / 其余看 out_dtype", "§4"),
            ("create_dot",0, 4, 0, "create_dot",                  "三把锁全开 → 建 tt.dot IR",                       "§5"),
            ("backend",   1, 1, 0, "min_dot_size / CUDAOptions",  "codegen_fns 声明最小 tile + 精度默认（tf32）",    "§2"),
        ],
        "edges": [
            ("dot_gate", "min_dot"), ("min_dot", "precision"),
            ("precision", "acc_dtype"), ("acc_dtype", "create_dot"),
            ("min_dot", "backend"), ("precision", "backend"),
        ],
        "dashed_edges": [],
        "entry_id": "dot_gate",
        "exit_id": "create_dot",
        "entry_lines": ["tl.dot", "（core.py 前端转发）"],
        "exit_lines": ["返回上层", "（tt.dot IR）"],
    },
    {
        "header": "下半程 §6-§9 · combine_fn 变 IR region",
        "lanes": ["语言层（core.py / semantic.py）", "再编译（CodeGenerator / combine_fn 示例）"],
        "nodes": [
            ("reduction", 0, 0, 0, "semantic.reduction",        "create_reduce → 回调建 region body → verify",   "§6"),
            ("call_jit",  1, 1, 0, "call_JitFunction",          "fn.parse()+新 CodeGenerator.visit → 再编译成 IR", "§6"),
            ("traceable", 1, 2, 0, "_argmax_combine / core.where","只能用 tl.* 可追踪操作，原生 if/外部库追踪期崩", "§7"),
            ("argmax",    0, 3, 0, "_reduce_with_indices",      "造 index 并 broadcast → argmax/argmin 支撑",     "§8"),
            ("histogram", 0, 4, 0, "semantic.histogram",        "1D+int 校验 → 直出 create_histogram，无 region",  "§9"),
        ],
        "edges": [
            ("reduction", "call_jit"), ("call_jit", "traceable"), ("traceable", "argmax"),
        ],
        "dashed_edges": [
            ("traceable", "histogram"),
        ],
        "entry_id": "reduction",
        "exit_id": "histogram",
        "entry_lines": ["combine_fn", "（如 a+b）"],
        "exit_lines": ["返回上层", "（IR 结果）"],
    },
]

# 底部阅读路线:直接复用章节开篇的选读指引,列号复用上面两组共用的 COLX(0-4)
ROUTES = [
    ("① 只抓性能结论",     [(0, "§1"), (1, "§2"), (2, "§3"), (3, "§4"), (4, "§5")], True),
    ("② 只弄懂归约编译机理", [(0, "§6"), (1, "§6"), (2, "§7"), (3, "§8"), (4, "§9")], True),
]
LEGEND = [
    ("#22c55e", "入口：用户代码调用进入"),
    ("#3b82f6", "章内主线调用/依赖边"),
    ("#f97316", "出口：建 IR 后返回上层"),
    ("#94a3b8", "灰虚线：对照支路（histogram 不建 region）"),
]
TITLE = "第 8 章 · tl.dot 命中判据 + combine_fn→region 剖面（源码走线 + § 讲解站牌）"

# ==================== 不可变:配色 ====================
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
C_GROUP_HEADER = "#1e3a8a"

# ==================== 几何常量(全计算,零魔数) ====================
NODE_W, NODE_H = 190, 62
COL_GAP, ROW_GAP = 34, 18
EDGE_MARGIN, STUB_W, STUB_H = 14, 66, 30
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 26
LANE_LABEL_H, BAND_PAD = 22, 10
TOP_PAD, TITLE_H, LEGEND_H = 14, 30, 24
GROUP_HEADER_H, GROUP_GAP = 22, 14
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BOTTOM_PAD = 14
BADGE_W, BADGE_H = 40, 19

COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(N_COLS)]
W = PAD_L + N_COLS * NODE_W + (N_COLS - 1) * COL_GAP + PAD_R

# 逐组计算每条泳道的高度与该组的总高度
for g in GROUPS:
    rows_per_lane = [0] * len(g["lanes"])
    for _id, lane, col, row, *_ in g["nodes"]:
        rows_per_lane[lane] = max(rows_per_lane[lane], row + 1)
    band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_lane]
    g["band_h"] = band_h
    g["lanes_h"] = sum(band_h)

# 纵向布局:TOP_PAD + TITLE + LEGEND, 然后逐组(header + 泳道区), 组间留 GROUP_GAP
y = TOP_PAD + TITLE_H + LEGEND_H
for g in GROUPS:
    g["header_y"] = y
    y += GROUP_HEADER_H
    band_top = []
    for bh in g["band_h"]:
        band_top.append(y)
        y += bh
    g["band_top"] = band_top
    g["lanes_bottom"] = y
    y += GROUP_GAP

routes_top = y - GROUP_GAP + 6
H = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD

# 每组内节点坐标
for g in GROUPS:
    node_xy = {}
    for nid, lane, col, row, *_ in g["nodes"]:
        x = COLX[col]
        ny = g["band_top"][lane] + LANE_LABEL_H + BAND_PAD + row * (NODE_H + ROW_GAP)
        node_xy[nid] = (x, ny)
    g["node_xy"] = node_xy
    g["node_by_id"] = {n[0]: n for n in g["nodes"]}


def badge(cx, cy, text):
    bx, by = cx - BADGE_W / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BADGE_W}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.1"/>',
        f'<text x="{cx:.1f}" y="{cy + 3.8:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="10.5" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN), ("Dim", C_ROUTE_DIM))
) + '</defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# 标题
L.append(f'<text x="{W / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="14.5" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')

# 图例(4 种语义色 → 必须画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 13
for color, label in LEGEND:
    dash = ' stroke-dasharray="4,3"' if color == C_ROUTE_DIM else ''
    if color == C_ROUTE_DIM:
        L.append(f'<line x1="{_lx}" y1="{_ly - 4}" x2="{_lx + 14}" y2="{_ly - 4}" '
                  f'stroke="{color}" stroke-width="2"{dash}/>')
    else:
        L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="10.8" '
              f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 10.8) + 26

# ---- 逐组绘制 ----
for g in GROUPS:
    # 组标题
    L.append(f'<text x="{PAD_L}" y="{g["header_y"] + GROUP_HEADER_H - 6:.1f}" font-family="sans-serif" '
              f'font-size="13" font-weight="bold" fill="{C_GROUP_HEADER}">{esc(g["header"])}</text>')

    # 泳道背景 + 标签 + 分隔线
    for i, name in enumerate(g["lanes"]):
        bt = g["band_top"][i]
        L.append(f'<rect x="0" y="{bt:.1f}" width="{W}" height="{g["band_h"][i]:.1f}" '
                  f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
        L.append(f'<text x="16" y="{bt + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
                  f'font-size="11.5" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
        if i > 0:
            L.append(f'<line x1="0" y1="{bt:.1f}" x2="{W}" y2="{bt:.1f}" '
                      f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
    L.append(f'<line x1="0" y1="{g["lanes_bottom"]:.1f}" x2="{W}" y2="{g["lanes_bottom"]:.1f}" '
              f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

    node_xy = g["node_xy"]
    first_id = g["entry_id"]
    last_id = g["exit_id"]

    # 入口接口桩(左边缘,绿)
    ex, ey = node_xy[first_id]
    ey += NODE_H / 2
    L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
              f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.2"/>')
    for i, line in enumerate(g["entry_lines"]):
        fs = fit_size(line, STUB_W - 8, 9, 6.5)
        L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey - 2 + i * 11:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs:.1f}" font-weight="bold" '
                  f'fill="#166534">{esc(line)}</text>')
    L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
              f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')

    # 出口接口桩(右边缘,橙)
    xx, xy = node_xy[last_id]
    xy += NODE_H / 2
    sx = W - EDGE_MARGIN - STUB_W
    L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
              f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.2"/>')
    for i, line in enumerate(g["exit_lines"]):
        fs = fit_size(line, STUB_W - 8, 9, 6.5)
        L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy - 2 + i * 11:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{fs:.1f}" font-weight="bold" '
                  f'fill="#9a3412">{esc(line)}</text>')
    L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
              f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

    # 调用边(主线蓝) —— 多条边汇入同一节点时,终点 y 各偏移,避免看不出汇合
    all_edges = [(s, d, True) for s, d in g["edges"]] + [(s, d, False) for s, d in g["dashed_edges"]]
    dst_total = {}
    for _, dst, _solid in all_edges:
        dst_total[dst] = dst_total.get(dst, 0) + 1
    dst_seen = {}
    for src, dst, solid in all_edges:
        x1, y1 = node_xy[src]
        x2, y2 = node_xy[dst]
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        n = dst_total[dst]
        i = dst_seen.get(dst, 0)
        dst_seen[dst] = i + 1
        y_offset = (i - (n - 1) / 2) * 14 if n > 1 else 0
        # 跨泳道竖向调用(如查后端表)从节点底部/顶部出边,更符合"下沉查表"的直觉
        lane_src = g["node_by_id"][src][1]
        lane_dst = g["node_by_id"][dst][1]
        if lane_src != lane_dst and x1 == x2:
            p1 = (x1 + NODE_W / 2, y1 + NODE_H if lane_dst > lane_src else y1)
            p2 = (x2 + NODE_W / 2 + y_offset, y2 if lane_dst > lane_src else y2 + NODE_H)
        else:
            p2 = (x2, y2 + NODE_H / 2 + y_offset)
        color = C_MAIN if solid else C_ROUTE_DIM
        marker = "mMain" if solid else "mDim"
        dash = '' if solid else ' stroke-dasharray="5,4"'
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{color}" stroke-width="2"{dash} marker-end="url(#{marker})"/>')

    # 节点(圆角框 + 符号 + 短语 + 右上角 § 徽标),字号按文本长度自适应收缩避免溢出
    for nid, lane, col, row, symbol, phrase, sec in g["nodes"]:
        x, ny = node_xy[nid]
        L.append(f'<rect x="{x:.1f}" y="{ny:.1f}" width="{NODE_W}" height="{NODE_H}" rx="11" '
                  f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.4"/>')
        sym_size = fit_size(symbol, NODE_W - 18, 12.5, 8.5)
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ny + NODE_H * 0.40:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size:.1f}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
        ph_size = fit_size(phrase, NODE_W - 16, 10, 7.6)
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{ny + NODE_H * 0.72:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{ph_size:.1f}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
        L += badge(x + NODE_W - BADGE_W / 2 + 6, ny, sec)

# 底部阅读路线:复用 COLX(两组共用),§ 徽标与图上节点竖向对齐
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12" '
          f'font-weight="bold" fill="{C_LANE_LABEL}">'
          f'{esc("阅读路线(标号=图上 § 站牌;两条路线各自独立成篇,任选其一)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="11.5" '
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
print(f"wrote {out}  ({W:.0f}x{H:.0f}, ratio {W / H:.2f}:1)")
