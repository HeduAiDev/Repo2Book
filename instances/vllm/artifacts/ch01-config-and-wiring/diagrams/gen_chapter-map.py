#!/usr/bin/env python3
"""第 1 章「本章地图」——一个请求的源码剖面(两个使用面 → 共享请求管线)。

改写自 .claude/skills/svg-diagram/references/example-chapter-map.py。本章是
全书导读/心智模型章(无精简版 companion)，源码剖面的真实骨架是: 两个使用面
(离线 `LLM` / 服务 `AsyncLLM`)各自构造，但从配置装配开始就走同一条共享管线
(`create_engine_config` → `InputProcessor` → `EngineCore.step` →
`OutputProcessor`)——这正是 §1.3"同一个内核,两种驱动"的字面意思。

■ 不可变(全书统一视觉语言,抄自模板,未改动):
  1. §徽标胶囊 badge()；2. 入口=绿#22c55e/出口=橙#f97316 接口桩；
  3. 章内主线调用边=蓝#3b82f6；4. 底部路线条(高亮=实线蓝/次要=虚线灰)；
  5. >2 种语义色画图例；6. cjk_text_width() 做宽度估算。

■ 本章新增(仅本章需要，未改动上面的不可变部分):
  - 本章两个真实入口(`LLM`/`AsyncLLM`)在源码里没有共同的"派发函数"——用户
    在写代码时就二选一，不是运行期由某个真实符号分流。模板原版只支持单一
    entry 节点喂给左侧入口桩。这里把入口桩泛化成"支持多个真实入口节点"：
    桩本身仍是同一个绿色"调用方"接口桩(样式、颜色、marker 均未改)，只是
    改成对 ENTRY_IDS 列表里的每个节点各画一条箭头(桩的高度按纵向跨度自适应)。
    EXIT_IDS 同理泛化(本章只有一个出口节点，退化成与模板等价的单箭头)。
  - 两条底部路线里各自还有两条"只看某一段"的单站路线(如"只看两个使用面
    怎么分工"只标 §1.3 一站)——单站路线画出的连线长度为 0(起止点重合)，
    不会画出可见线段，只留一个悬浮徽标，这是预期效果(纯粹的"跳读指路牌"，
    不代表一段可见调用路径)。

用法: python3 gen_chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(本章数据) ----------------
LANES = ["使用面(两条真实入口)", "共享请求管线(装配 → 内核 → 输出)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, §编号)
NODES = [
    ("entry_llm",   0, 0, 0, "LLM",
     "__init__建同步引擎（离线面）", "§1.3"),
    ("entry_async", 0, 0, 1, "AsyncLLM",
     "__init__建异步门面（服务面）", "§1.3"),
    ("config",      1, 1, 0, "create_engine_config",
     "装配成 VllmConfig", "§1.4"),
    ("input_proc",  1, 2, 0, "InputProcessor",
     "prompt → EngineCoreRequest", "§1.1"),
    ("engine_core", 1, 3, 0, "EngineCore.step",
     "schedule→execute→update", "§1.2"),
    ("output_proc", 1, 4, 0, "OutputProcessor",
     "→ RequestOutput", "§1.1"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝
    ("entry_llm", "config"), ("entry_async", "config"),
    ("config", "input_proc"),
    ("input_proc", "engine_core"),
    ("engine_core", "output_proc"),
]
ENTRY_IDS = ["entry_llm", "entry_async"]
EXIT_IDS = ["output_proc"]

# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("请求主线(两面共享)", [(0, "§1.3"), (1, "§1.4"), (2, "§1.1"), (3, "§1.2"), (4, "§1.1")], True),
    ("只看两个使用面怎么分工", [(0, "§1.3")], False),
    ("只看配置怎么装配成 VllmConfig", [(1, "§1.4")], False),
]
LEGEND = [("#22c55e", "入口:从上层调用进入"), ("#3b82f6", "章内主线调用边"), ("#f97316", "出口:返回上层")]
TITLE = "第 1 章 · 一个请求的源码剖面(两个使用面 → 共享请求管线)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 190, 58
COL_GAP, ROW_GAP = 42, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
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


def stub_and_arrows(ids, side):
    """入口/出口接口桩的泛化版:给一批节点 id 共用同一个接口桩(按纵向跨度自适应
    桩高),每个 id 各画一条箭头。side='left' 是绿色调用方入口,'right' 是橙色
    返回上层出口——颜色/marker/桩样式与模板单入口版完全一致,只是循环多画。"""
    ys = [NODE_XY[i][1] + NODE_H / 2 for i in ids]
    y_top, y_bot = min(ys), max(ys)
    y_mid = (y_top + y_bot) / 2
    stub_h = max(STUB_H, (y_bot - y_top) + STUB_H)
    out = []
    if side == "left":
        bx = EDGE_MARGIN
        color, fill, text_fill, label, marker = C_ENTRY, "#dcfce7", "#166534", "调用方", "mEntry"
    else:
        bx = w - EDGE_MARGIN - STUB_W
        color, fill, text_fill, label, marker = C_EXIT, "#ffedd5", "#9a3412", "返回上层", "mExit"
    out.append(f'<rect x="{bx:.1f}" y="{y_mid - stub_h / 2:.1f}" width="{STUB_W}" height="{stub_h:.1f}" '
               f'rx="{STUB_H / 2}" fill="{fill}" stroke="{color}" stroke-width="1.3"/>')
    out.append(f'<text x="{bx + STUB_W / 2:.1f}" y="{y_mid + 4:.1f}" text-anchor="middle" '
               f'font-family="sans-serif" font-size="11" font-weight="bold" fill="{text_fill}">{esc(label)}</text>')
    for i in ids:
        nx, ny = NODE_XY[i]
        ny_mid = ny + NODE_H / 2
        if side == "left":
            out.append(f'<line x1="{bx + STUB_W:.1f}" y1="{ny_mid:.1f}" x2="{nx:.1f}" y2="{ny_mid:.1f}" '
                       f'stroke="{color}" stroke-width="2" marker-end="url(#{marker})"/>')
        else:
            out.append(f'<line x1="{nx + NODE_W:.1f}" y1="{ny_mid:.1f}" x2="{bx:.1f}" y2="{ny_mid:.1f}" '
                       f'stroke="{color}" stroke-width="2" marker-end="url(#{marker})"/>')
    return out


L += stub_and_arrows(ENTRY_IDS, "left")
L += stub_and_arrows(EXIT_IDS, "right")

# 调用边(主线蓝,画在节点下面这条先画后画都行,这里先画边再画节点盖住端点毛刺)
# 多条边汇入同一节点时,终点 y 各偏移(间距 16px,如 2 条即 ±8px),
# 否则重合的终点在视觉上看不出"汇合"、像一条线断头。
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

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角 § 徽标)
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
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
print(f"wrote {out}  viewBox=0 0 {w:.1f} {h:.1f}  aspect={w / h:.2f}")
