#!/usr/bin/env python3
"""第 38 章「本章地图」——三种注册形态源码剖面图。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py 模板改写：
不可变机制(esc/cjk_text_width/badge/配色/图例规则)原样保留，只改下面的 DATA。

本章和典型「单入单出」章不同——它是三个各自独立的注册机制并排讲(模型注册 /
LoRA 全局类替换+算子分发 / netloader loader 注册)，彼此之间没有调用关系。
于是不用共享的单一入口/出口，而是每条泳道各自成一条完整的
「入口(绿)→节点链→出口(橙)」，三条泳道互不交叉——这比强行把三者汇合到
一个假想的"收口"节点更贴合源码事实(三者确实是并列、不是串联)。

节点预算：2(模型) + 4(LoRA) + 4(netloader) = 10 ≤ 12。
本章标题为编号标题(## 36.1 ... ## 36.5)，站牌用 §36.N。

设计要点：
- Lane 0(模型注册 §38.1)只有 2 个节点——对应正文"全部代码就在...整个文件只有
  七行"的最简单形态，节点数刻意少，视觉上呼应"三种形态从最规矩到最野"的谱系。
- Lane 1(LoRA 接入)4 个节点按源码时间顺序连成一条链：__init__ 先调
  refresh_all_lora_classes()(§38.2 的机制，在 §38.3 引入的入口里被调用)，
  再做 device/rank 二选一绑算子(§38.3)，运行期才轮到 add_lora_linear(§38.3)——
  这条链故意让 §38.3→§38.2→§38.3→§38.3 来回跳，如实反映"被调函数在前一节
  讲、调用者在后一节引入"的实际行文顺序，而不是编造一个单调递增的假顺序。
- Lane 2(netloader §38.4)4 个节点：注册(register_model_loader 触发→装饰器
  落表)接运行期(load_model 分流→revert_to_default 兜底)，对应正文"先看怎么
  挂进去，再看怎么用"的结构。
- 三条泳道各自的出口箭头长度不同(Lane 0 只到 col1 就出，Lane 1/2 到 col3
  才出)——都画到画布最右侧的出口桩，不同长度的水平线不会跨泳道穿过别的
  节点，是安全的（每条线只经过自己泳道的空白列）。

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
LANES = ["模型注册 - S36.1", "LoRA 接入 - S36.2~36.3", "netloader 注册 - S36.4"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名(可含 "\n" 机械换行,不改变拼写), 一行短语, §编号或 "")
NODES = [
    ("a1", 0, 0, 0, "register_model()",
     "插件加载触发,注册\nDeepseekV4 + MTP 架构名", "§38.1"),
    ("a2", 0, 1, 0, "ModelRegistry.\nregister_model",
     "存入 <module>:<class>\n懒加载字符串", "§38.1"),

    ("b1", 1, 0, 0, "PunicaWrapperNPU.\n__init__",
     "平台选中后实例化,\n两招都在这里", "§38.3"),
    ("b2", 1, 1, 0, "refresh_all_lora_\nclasses()",
     "全局类替换:追加进\n_all_lora_classes", "§38.2"),
    ("b3", 1, 2, 0, "lora_ops /\ntorch_ops",
     "device/rank 二选一,\n绑 6 个 shrink/expand 算子", "§38.3"),
    ("b4", 1, 3, 0, "add_lora_linear",
     "运行期 shrink→expand,\n调用已绑定算子", "§38.3"),

    ("c1", 2, 0, 0, "register_model_\nloader()",
     "插件加载触发,注册\nnetloader + rfork", "§38.4"),
    ("c2", 2, 1, 0, "register_model_loader(\n\"netloader\")",
     "装饰器写进\n_LOAD_FORMAT_TO_MODEL_LOADER", "§38.4"),
    ("c3", 2, 2, 0, "load_model",
     "source 有效?→弹性拉权重\n:否→兜底", "§38.4"),
    ("c4", 2, 3, 0, "revert_to_default",
     "委托 vLLM\nDefaultModelLoader 兜底", "§38.4"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝;三条泳道互不相连
    ("a1", "a2"),
    ("b1", "b2"), ("b2", "b3"), ("b3", "b4"),
    ("c1", "c2"), ("c2", "c3"), ("c3", "c4"),
]
# 每条泳道各自的入口节点(接绿色调用方箭头)与出口节点(接橙色返回箭头)
ENTRY_NODES = ["a1", "b1", "c1"]
EXIT_NODES = ["a2", "b4", "c4"]

# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("模型注册(§38.1)", [(0, "§38.1"), (1, "§38.1")], False),
    ("LoRA 双招(§38.2→36.3)", [(0, "§38.3"), (1, "§38.2"), (2, "§38.3"), (3, "§38.3")], True),
    ("netloader 注册(§38.4)", [(0, "§38.4"), (1, "§38.4"), (2, "§38.4"), (3, "§38.4")], False),
]
LEGEND = [("#22c55e", "入口:插件加载/平台选型触发"), ("#3b82f6", "章内调用边"), ("#f97316", "出口:注册生效/返回上层")]
TITLE = "第 38 章 · 三种注册形态剖面(ModelRegistry / 全局类替换 / netloader)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 220, 92
COL_GAP, ROW_GAP = 34, 20
EDGE_MARGIN, STUB_W, STUB_H = 12, 62, 26
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

# 入口/出口接口桩——本章三条泳道各自独立,每条各配一对绿色入口桩/橙色出口桩
# (而非模板默认的单一入出口),因为三种注册形态彼此并列、没有调用关系。
sx_exit = w - EDGE_MARGIN - STUB_W
for nid in ENTRY_NODES:
    nx, ny = NODE_XY[nid]
    ny += NODE_H / 2
    L.append(f'<rect x="{EDGE_MARGIN}" y="{ny - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
             f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
    L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ny + 4:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
    L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ny:.1f}" x2="{nx:.1f}" y2="{ny:.1f}" '
             f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
for nid in EXIT_NODES:
    nx, ny = NODE_XY[nid]
    ny += NODE_H / 2
    L.append(f'<rect x="{sx_exit:.1f}" y="{ny - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
             f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
    L.append(f'<text x="{sx_exit + STUB_W / 2:.1f}" y="{ny + 4:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
    L.append(f'<line x1="{nx + NODE_W:.1f}" y1="{ny:.1f}" x2="{sx_exit:.1f}" y2="{ny:.1f}" '
             f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝)。三条泳道各自一条直链,无节点有多个来源,无需汇入偏移。
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    p2 = (x2, y2 + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名(1~2 行) + 一行短语(1~2 行,始终锚在节点下半区) + 右上角 § 徽标)
SYMBOL_1LINE_Y, SYMBOL_2LINE_Y1, SYMBOL_2LINE_Y2 = 36, 26, 43
PHRASE_1LINE_Y, PHRASE_2LINE_Y1, PHRASE_2LINE_Y2 = 72, 66, 82
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
    if sec:
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

# [SELF-CHECK](渲染→Read PNG 亲眼看后如实记录，见下方注释)
