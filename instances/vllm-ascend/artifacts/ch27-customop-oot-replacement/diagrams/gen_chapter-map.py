#!/usr/bin/env python3
"""第 27 章「本章地图」——CustomOp 的 OOT 顶替剖面图。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py 模板改写：
不可变机制(esc/cjk_text_width/badge/配色/图例规则)原样保留，只改 DATA + 几何常量
(参照 ch20 chapter-map.py 的压宽手法：缩小 NODE_W/COL_GAP/桩宽 + 长符号手工换行)。

节点预算 9(entry/register_oot/new/dispatch/silu/rmsnorm/fused/fallback/exit) ≤ 12。
本章标题为编号标题(## 27.1 … ## 27.9)，站牌用 §27.N。

设计要点：
- 三条泳道对应本章的三个时间阶段(不是同一次调用栈内的连续帧，是「注册期→构造期→
  前向期」三个不同时机)：
  1. 注册期(worker 初始化，全程只跑一次)——entry(register_ascend_customop) →
     register_oot(CustomOp.register_oot 写入全局 op_registry_oot)。
  2. 构造期(模型代码写 RMSNorm(...)/SiluAndMul(...) 构造算子实例时)——
     new(CustomOp.__new__ 换身) → dispatch(dispatch_forward 换头)。
  3. 前向期(forward_oot 内，两个标本各自的二分)——silu(AscendSiluAndMul，只覆一行)
     与 rmsnorm(AscendRMSNorm，二次二分) → fused/fallback(融合 kernel vs 原子算子回退)。
  三条泳道之间的边(register_oot→new、dispatch→silu/rmsnorm)表达的是「前一阶段的产物
  是后一阶段查表/绑定的依据」，不是字面同一次调用栈里的连续函数调用——这点在此说明，
  避免误读成"注册期直接同步调用了构造期"。
- exit 节点用真实符号 `forward`(CustomOp.forward，转调 _forward_method 并把结果
  返回给上层调用者)做收束站——这是"换头"绑定的值最终被使用的地方，比虚构一个
  "输出合并函数"更贴题也更可核。
- fused/fallback 两个节点是 §27.6/§27.7 合起来讲的「第二层二分」的两个出口：
  enable_custom_op() 为真走 npu_add_rms_norm_bias(1 颗融合 kernel)，为假走
  npu_add_rms_norm(原子算子回退)。二者本身没有互相调用关系，只是同一个 if 的两支，
  故不画二者之间的边，只各自单独连到 exit。
- 阅读路线里第 3 条把 col5(fused/fallback 所在列)的站牌标成 §27.7 而非 §27.6——
  意图是把"为什么会走这条分支"的追问指向 enable_custom_op() 的详解节(§27.7)，
  而不是重复 fused/fallback 节点本身已经挂着的 §27.6。第 4 条路线同理，把
  col1(register_oot 所在列)标成 §27.8，指向"为什么全程只生效一次"的幂等闸详解节。
  这是有意为之的「站牌 ≠ 节点自身徽标」用法，路线站牌的作用是给读者指路，不是
  重复标注节点身份。

[FIX-ROUND-2](仅缩短两条路线名文字,不改路线覆盖的 §站点/顺序/高亮):首轮渲染后 Read PNG
发现第 1 条("主链:一次注册→换身→换头")和第 4 条("为何全程只跑一次(幂等闸)")路线名过长，
在 col0 badge(§27.1,左边缘 x≈153,label 起点 x=16,可用宽度仅 137px)前挤压重叠——
按 cjk_text_width 估算原文本分别宽 140.9px/145.9px,与徽标视觉相压。本轮删掉两条路线名
里的冗余字("一次"/"全程")压到 116.9px/121.9px，留出安全间距，重渲染后 Read PNG 复核确认
不再重叠；其余路线(3/4 条起点在 col3/col4,可用宽度 695/881px)本就充裕，未改。

用法: python3 gen_chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算——全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["注册期(worker 初始化,一次性)", "构造期(模型实例化时)", "前向期(forward_oot 内,标本二分)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名(可含 "\n" 机械换行,不改变拼写), 一行短语(可含 "\n"), §编号)
NODES = [
    ("entry",    0, 0, 0, "register_ascend_\ncustomop()", "worker 初始化调一次,\n建注册表",             "§27.1"),
    ("regoot",   0, 1, 0, "CustomOp.\nregister_oot",      "写入全局 op_registry_oot,\n打 .name",         "§27.2"),
    ("new",      1, 2, 0, "CustomOp.__new__",             "查表换身:\n实例化 Ascend 子类",               "§27.3"),
    ("dispatch", 1, 3, 0, "dispatch_forward",             "OOT 平台→绑定\nforward_oot (换头)",           "§27.4"),
    ("silu",     2, 4, 0, "AscendSiluAndMul",             "forward_oot:\n一行 npu_swiglu 顶替",          "§27.5"),
    ("rmsnorm",  2, 4, 1, "AscendRMSNorm",                "forward_oot 二次二分:\nenable_custom_op()",  "§27.6"),
    ("fused",    2, 5, 1, "npu_add_rms_\nnorm_bias",      "真→1 颗融合 kernel",                          "§27.6"),
    ("fallback", 2, 5, 2, "npu_add_\nrms_norm",           "假→多颗原子算子回退",                         "§27.6"),
    ("exit",     2, 6, 0, "forward",                      "转调 _forward_method,\n结果返回上层",         "§27.4"),
]
EDGES = [  # (src_id, dst_id) —— 主线蓝;跨泳道边表达"前一阶段产物是后一阶段查表/绑定依据"
    ("entry", "regoot"),
    ("regoot", "new"),
    ("new", "dispatch"),
    ("dispatch", "silu"), ("dispatch", "rmsnorm"),
    ("rmsnorm", "fused"), ("rmsnorm", "fallback"),
    ("silu", "exit"), ("fused", "exit"), ("fallback", "exit"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("主链:注册→换身→换头",             [(0, "§27.1"), (1, "§27.2"), (2, "§27.3"), (3, "§27.4")], True),
    ("标本一:只覆 forward_oot(SiluAndMul)", [(3, "§27.4"), (4, "§27.5")], False),
    ("标本二:融合 vs 回退(RMSNorm)",       [(4, "§27.6"), (5, "§27.7")], False),
    ("为何只跑一次(幂等闸)",             [(0, "§27.1"), (1, "§27.8")], False),
]
LEGEND = [("#22c55e", "入口:从上层调用进入"), ("#3b82f6", "章内主线调用边"), ("#f97316", "出口:返回上层")]
TITLE = "第 27 章 · CustomOp 顶替剖面(换身 + 换头 + 二次二分)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
# 本章符号名偏长——不靠加宽节点装下整行，改成机械换行(见 NODES 里的 "\n")；
# NODE_W 只需装下"半个符号名"，NODE_H 加高以容纳最多 2 行符号 + 最多 2 行短语。
NODE_W, NODE_H = 160, 90
COL_GAP, ROW_GAP = 26, 22
EDGE_MARGIN, STUB_W, STUB_H = 10, 54, 24
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

# 调用边(主线蓝)。多条边汇入同一节点时,终点 y 各偏移(间距 16px),否则重合的终点
# 在视觉上看不出"汇合"、像一条线断头。
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

# 节点(圆角框 + 真实符号名(1~2 行) + 一行短语(1~2 行,始终锚在节点下半区) + 右上角 § 徽标)
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

# [SELF-CHECK](渲染→Read PNG 亲眼看后如实记录，见 figure-manifest.json)
