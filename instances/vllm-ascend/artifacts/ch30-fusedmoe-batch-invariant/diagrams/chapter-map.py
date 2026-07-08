#!/usr/bin/env python3
"""第 30 章「本章地图」——AscendFusedMoE 源码剖面图(换头顶替 + 通信四选一 + 批不变正交开关)。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py 模板改写：
不可变机制(esc/cjk_text_width/badge/配色/图例规则/多行文本渲染)取自 ch20 的加固版
(chapter-map.py，多行 symbol/phrase 排布 + 动态边偏移)，原样保留，只改下面的 DATA。

节点预算 11 ≤ 12。本章标题为编号标题(## 30.1 .. ## 30.8)，站牌用 §30.N。

设计要点：
- 四条泳道对应本章四类机制，从上到下：
  ①"顶替与建表(构造期,一次性)"——AscendFusedMoE(§30.1,换头+调 setup_moe_comm_method)
    → _MoECommMethods(§30.4,枚举→CommImpl 注册表；ch16 选枚举、本章建实例，回收 f10)。
    这条边(init→registry)是真实调用关系，但整条泳道不接主线——它只在算子构造时跑一次，
    不是每拍前向都走。
  ②"forward 骨架(每拍前向)"——forward(§30.2,薄入口)→moe_comm_method.prepare(§30.2,
    通信前置)→quant_method.apply(§30.3,select_experts 选专家→打包)→…→
    moe_comm_method.finalize(§30.2,通信后置,出口)。entry 与 exit 同处这条泳道，
    工作在下层泳道③完成，呼应"entry/exit 同层、下层干活"的 fork-join 视觉语法。
  ③"token 重分发:四选一(§30.5/30.6)"——apply 分叉到四种 *CommImpl 的 token 派发实现，
    同列不同行(与 ch03 技法层同列不同行的约定一致，表达"运行时四选一"而非"顺序执行
    四遍")：TokenDispatcherWithAllGather / TokenDispatcherWithMC2 /
    TokenDispatcherWithAll2AllV(回收 ch06 的 all_to_all 形状代数) / FusedMC2CommImpl
    (覆写 fused_experts，三步融一算子)，四者都汇入 exit。
  ④"可复现保证:批不变(§30.7,与 MoE 正交)"——init_batch_invariance 卫星节点，不接
    任何调用边(章内原文明说"这一节和 MoE 正交")，独立摆在最底层泳道。
- 底部三条阅读路线：构造期建表(次要,虚线灰)/ 每拍前向主线(推荐,实线蓝)/
  FusedMC2 特例(次要,虚线灰)——批不变正交、不占路线，只留卫星节点+图注说明。

六项自查记录(渲染→Read PNG 亲眼看后如实记录，见文件末尾 [SELF-CHECK] 注释)。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算——全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["顶替与建表(构造期,一次性)", "forward 骨架(每拍前向)", "token 重分发:四选一", "可复现保证:批不变(正交)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名(可含 "\n" 机械换行,不改变拼写), 一行短语, §编号)
NODES = [
    ("init",        0, 0, 0, "AscendFusedMoE",
     "__init__ 换 quant_method/runner,\n调 setup_moe_comm_method", "§30.1"),
    ("registry",    0, 1, 0, "_MoECommMethods",
     "枚举→CommImpl 注册表\n(ch16 选枚举,本章建实例)", "§30.4"),
    ("entry",       1, 0, 0, "forward",
     "薄入口,委托 runner.forward", "§30.2"),
    ("prepare",     1, 1, 0, "moe_comm_method\n.prepare",
     "通信前置:pad/切片/all-gather", "§30.2"),
    ("apply",       1, 2, 0, "quant_method.apply",
     "select_experts 选专家→\n打包 fused_experts_input", "§30.3"),
    ("dispatch_ag", 2, 3, 0, "TokenDispatcherWith\nAllGather",
     "本地 npu_moe_init_routing,\n不跨卡 all_to_all", "§30.5"),
    ("dispatch_mc2",2, 3, 1, "TokenDispatcherWithMC2",
     "npu_moe_distribute_dispatch\n融合原语", "§30.5"),
    ("dispatch_a2a",2, 3, 2, "TokenDispatcherWith\nAll2AllV",
     "async_all_to_all 不等长\n重分发(回收 ch06)", "§30.5"),
    ("fused_mc2",   2, 3, 3, "FusedMC2CommImpl",
     "dispatch_ffn_combine\n三步融一算子", "§30.6"),
    ("exit",        1, 4, 0, "moe_comm_method\n.finalize",
     "通信后置,按类型决定\n是否 reduce,返回上层", "§30.2"),
    ("batchinv",    3, 0, 0, "init_batch_invariance",
     "正交开关:关漂移源+替换\naten 算子,逐位可复现", "§30.7"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝
    ("init", "registry"),
    ("entry", "prepare"), ("prepare", "apply"),
    ("apply", "dispatch_ag"), ("apply", "dispatch_mc2"),
    ("apply", "dispatch_a2a"), ("apply", "fused_mc2"),
    ("dispatch_ag", "exit"), ("dispatch_mc2", "exit"),
    ("dispatch_a2a", "exit"), ("fused_mc2", "exit"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("构造期建表(一次性)", [(0, "§30.1"), (1, "§30.4")], False),
    ("每拍前向主线(四选一取其一)", [(0, "§30.2"), (1, "§30.2"), (2, "§30.3"), (3, "§30.5"), (4, "§30.2")], True),
    ("FusedMC2 特例:融合旁路", [(2, "§30.3"), (3, "§30.6"), (4, "§30.2")], False),
]
LEGEND = [("#22c55e", "入口:从上层调用进入"), ("#3b82f6", "章内主线调用边"), ("#f97316", "出口:返回上层")]
TITLE = "第 30 章 · AscendFusedMoE 源码剖面(换头顶替 + 通信四选一 + 批不变正交开关)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
# 本章符号名/短语偏长——用机械换行(见 NODES 里的 "\n")；NODE_H 加高以容纳
# 最多 2 行符号 + 最多 2 行短语。
NODE_W, NODE_H = 185, 90
COL_GAP, ROW_GAP = 30, 22
EDGE_MARGIN, STUB_W, STUB_H = 12, 60, 26
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

# 调用边(主线蓝)。多条边汇入同一节点时,终点 y 各偏移——间距在 16px 与
# "(NODE_H-14)/(n-1)" 之间取更小值:边数少时保持默认 16px 手感,边数多时
# (本图 exit 汇入 4 条)自动收窄间距,确保偏移量落在节点框内,不会悬空。
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
    if n > 1:
        spacing = min(16.0, (NODE_H - 14) / (n - 1))
        y_offset = (i - (n - 1) / 2) * spacing
    else:
        y_offset = 0
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

# [SELF-CHECK](渲染→Read PNG 亲眼看后如实记录，见下方注释——首轮记录，若有 FIX-ROUND
# 会追加在此处并替换判定，不覆盖历史记录原文)
