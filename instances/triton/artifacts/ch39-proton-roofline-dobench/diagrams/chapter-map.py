#!/usr/bin/env python3
"""ch39 本章地图——度量三工具的源码剖面图。
本章为自然标题(正文无 `## N.M` 编号,只有 proton / roofline viewer / do_bench
三个自然标题小节),故禁用 §N.M 徽标——站牌一律改用小节的标题词本身。

三条泳道 = 三件量东西的工具:
  proton(挂钩采集 flops/bytes) → roofline viewer(派生 util 判瓶颈)
  do_bench(独立计时:量准实测时间)
proton 落盘 hatchet json 喂给 viewer(唯一一条跨泳道边);do_bench 是独立
的一条计时线,不喂 viewer(viewer 的实测时间来自厂商 tracer,见正文)。

不可变(套用 references/example-chapter-map.py 的视觉语言):
  入口绿 #22c55e / 出口橙 #f97316 / 主线蓝 #3b82f6;徽标胶囊 fill #eef2ff
  stroke #6366f1;高亮路线=实线蓝、次要=虚线灰;cjk_text_width() 估宽。
可变:LANES / NODES / EDGES / ROUTES(本章数据)。徽标改为按标题词自适应宽度。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知宽度估算:全角(ord>0x2E80)按 1.0×size,半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["proton:挂钩采集 flops/bytes", "roofline viewer:派生判瓶颈", "do_bench:计时秒表"]

# (id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 标题词站牌)
NODES = [
    # proton 泳道(lane0)——注册钩子 + 每次发射惰性算 flops/bytes
    ("reg",   0, 0, 0, "register_triton_hook",          "两次赋值挂类级钩子",       "注入"),
    ("meta",  0, 0, 1, "launch_metadata",               "钩子未挂即返回 None",       "惰性账本"),
    ("lazy",  0, 1, 1, "LazyDict",                       "get() 时才算 flops/bytes",  "惰性"),
    ("hook",  0, 1, 0, "TritonHook.enter",               "发射前后各回调一次",       "类级槽位"),
    # roofline viewer 泳道(lane1)——读 json,两面屋顶取 max 派生 util
    ("raw",    1, 2, 0, "get_raw_metrics",               "拆调用树 + device_info",    "主管线"),
    ("flops",  1, 2, 1, "get_min_time_flops",            "算力屋顶 F÷peak",           "两面屋顶"),
    ("bytes",  1, 2, 2, "get_min_time_bytes",            "带宽屋顶 B÷peak",           "两面屋顶"),
    ("derive", 1, 3, 1, "derive_metrics",                "util=max(两屋顶)÷实测",     "判瓶颈"),
    # do_bench 泳道(lane2)——独立计时:估时/冲 L2/取中位
    ("bench",  2, 1, 0, "do_bench",                       "估时·定次数·逐轮打点",     "五段计时"),
    ("cache",  2, 2, 0, "get_empty_cache_for_benchmark",  "256MB 每轮冲冷 L2",         "冲 L2"),
    ("stat",   2, 3, 0, "_summarize_statistics",          "取中位/分位抗抖动",         "取中位数"),
]
EDGES = [  # (src, dst) —— 调用/数据边,统一主线蓝
    ("reg", "hook"), ("meta", "lazy"), ("lazy", "hook"),   # proton 内部
    ("hook", "raw"),                                        # 落盘 hatchet json → viewer(唯一跨泳道边)
    ("raw", "derive"), ("flops", "derive"), ("bytes", "derive"),  # viewer 内部
    ("bench", "cache"), ("cache", "stat"),                 # do_bench 内部
]
# 多入口:用户发起度量的落点;多出口:两条独立的度量产物
ENTRY_TARGETS = ["reg", "meta", "bench"]
EXIT_SOURCES = [("derive", "判瓶颈→优化方向"), ("stat", "可比时间数字")]
# 跨泳道 json 边单独标注
JSON_EDGE = ("hook", "raw", "hatchet json")

# (路线名, [(列, 标题词), ...] 按阅读顺序, 是否高亮)
ROUTES = [
    ("度量全链(推荐按序读)", [(0, "注入"), (1, "类级槽位"), (2, "主管线"), (3, "判瓶颈")], True),
    ("只量时间 → do_bench", [(1, "五段计时"), (2, "冲 L2"), (3, "取中位数")], False),
    ("只判瓶颈 → viewer", [(2, "主管线"), (3, "判瓶颈")], False),
]
LEGEND = [("#22c55e", "入口:发起度量"), ("#3b82f6", "工具内主线 / json 交接"), ("#f97316", "出口:度量结果")]
TITLE = "第 39 章 · 度量三工具:proton 挂钩 → viewer 判瓶颈 · do_bench 计时"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 216, 60
COL_GAP, ROW_GAP = 46, 22
EDGE_MARGIN, STUB_W, STUB_H = 16, 78, 28
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 34
LANE_LABEL_H, BAND_PAD = 26, 14
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 28, 18
ROUTE_HEAD_H, ROUTE_ROW_H = 24, 46
BADGE_H = 21

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

routes_top = lanes_bottom + 10
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
    """标题词徽标胶囊,居中挂在 (cx,cy),宽度按文本自适应(本章站牌是标题词非 §N.M)。"""
    bw = max(40, cjk_text_width(text, 11) + 18)
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


def sym_font(symbol):
    """长符号名自适应字号,避免越出节点框(NODE_W)。"""
    target = NODE_W - 18
    for fs in (13, 12, 11, 10):
        if cjk_text_width(symbol, fs) <= target:
            return fs
    return 10


def edge_pts(src, dst):
    """按几何关系给边的起止点:同列→竖直(下:底→顶 / 上:顶→底);否则水平(右→左)。"""
    sx, sy = NODE_XY[src]
    dx, dy = NODE_XY[dst]
    scx, scy = sx + NODE_W / 2, sy + NODE_H / 2
    dcx, dcy = dx + NODE_W / 2, dy + NODE_H / 2
    if abs(dx - sx) < 1:  # 同列:竖直边
        if dy > sy:
            return (scx, sy + NODE_H), (dcx, dy)
        return (scx, sy), (dcx, dy + NODE_H)
    return (sx + NODE_W, scy), (dx, dcy)


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
# 图例
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
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 7:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                 f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口接口桩(左)——一个桩,分出多条绿箭头到各入口节点
entry_cy = (band_top[0] + lanes_bottom) / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{entry_cy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{entry_cy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
for tid in ENTRY_TARGETS:
    tx, ty = NODE_XY[tid]
    tcy = ty + NODE_H / 2
    L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{entry_cy:.1f}" x2="{tx:.1f}" y2="{tcy:.1f}" '
             f'stroke="{C_ENTRY}" stroke-width="1.8" marker-end="url(#mEntry)"/>')

# 出口接口桩(右)——两条橙箭头汇入
exit_cy = (NODE_XY["derive"][1] + NODE_XY["stat"][1]) / 2 + NODE_H / 2
sx_exit = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx_exit:.1f}" y="{exit_cy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx_exit + STUB_W / 2:.1f}" y="{exit_cy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("度量结果")}</text>')
for sid, lbl in EXIT_SOURCES:
    px, py = NODE_XY[sid]
    pcy = py + NODE_H / 2
    L.append(f'<line x1="{px + NODE_W:.1f}" y1="{pcy:.1f}" x2="{sx_exit:.1f}" y2="{exit_cy:.1f}" '
             f'stroke="{C_EXIT}" stroke-width="1.8" marker-end="url(#mExit)"/>')
    # 出口边标签(读者语,非脚手架)
    mlx = (px + NODE_W + sx_exit) / 2
    L.append(f'<text x="{mlx:.1f}" y="{(pcy + exit_cy) / 2 - 5:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="10" fill="{C_EXIT}">{esc(lbl)}</text>')

# 调用/数据边(主线蓝);多条汇入同一节点时,水平边终点 y 各偏移防重合
_dst_total, _dst_seen = {}, {}
for _s, _d in EDGES:
    _dst_total[_d] = _dst_total.get(_d, 0) + 1
for src, dst in EDGES:
    (x1, y1), (x2, y2) = edge_pts(src, dst)
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    if abs(NODE_XY[dst][0] - NODE_XY[src][0]) >= 1 and n > 1:
        y2 += (i - (n - 1) / 2) * 14
    L.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
             f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# json 交接边的读者语标签——挂在斜边右侧的空白区(靠 start 锚点向右延伸,避开 LazyDict 框)
_js, _jd, _jl = JSON_EDGE
(jx1, jy1), (jx2, jy2) = edge_pts(_js, _jd)
L.append(f'<text x="{(jx1 + jx2) / 2 + 12:.1f}" y="{(jy1 + jy2) / 2 - 6:.1f}" text-anchor="start" '
         f'font-family="sans-serif" font-size="10" fill="{C_MAIN}">{esc(_jl)}</text>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 顶边居中标题词徽标)
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
             f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.44:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="{sym_font(symbol)}" font-weight="bold" '
             f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.74:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W / 2, y, sec)

# 底部阅读路线
L.append(f'<text x="16" y="{routes_top + 16:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上标题词站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
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
print(f"wrote {out}  ({w}x{h}, ratio {w / h:.2f})")
