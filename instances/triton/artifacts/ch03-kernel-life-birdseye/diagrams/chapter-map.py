#!/usr/bin/env python3
"""ch03「本章地图」——一个 kernel 从 add_kernel[grid] 到 GPU 发射的源码剖面图。

本章是自然标题章(`## §1`…`## §9`,无 `## N.M` 编号)——站牌用正文实际使用的
§1…§9 标记(无小数点,不触发 lint_chapter_map 的 §N.M 徽标核对——与 ch01 chapter-map
同一处理口径)。

剖面(左→右一条主脊柱,自上而下三条泳道):
  第 0 泳道(主脊柱,时间顺序):
    add_kernel[grid](...)(§1,host 触发)
    → JITFunction.run(§2,查缓存)
    → compile()(§3,五级 for 循环心脏)
    → 五级降级 ttir→cubin(§4,每级一个 make_*)
    → kernel.run(§8,发射)
    → 定位地图(§9,本章 hook 收口:症状→旋钮层→章节)
  第 1 泳道(三个性能旋钮,各自挂在对应脊柱节点正下方):
    key/缓存键(§2 旋钮①编译期特化)、num_warps/#blocked(§4 旋钮②优化 pass)、
    非 warmup 发射段(§8 旋钮③发射开销)
  第 2 泳道(坐标系 · 无卡断裂线 · 双语接缝,均为独立参考卡片,无出边——
    这三点是本章新增的"地图"而非脊柱本身,故不接线,避免与第 1 泳道的竖直
    知识边视觉打架):
    make_ir vs make_ttir(§5 坐标系)、ptxas / load_binary(§6 断裂线)、
    四道双语接缝(§7 接缝时间轴)

模板:.claude/skills/svg-diagram/references/example-chapter-map.py;不可变视觉语言
(§徽标胶囊 / 入口绿-出口橙-主线蓝 / 高亮实线蓝-次要虚线灰 / cjk_text_width)照搬,
只改 DATA;边路由沿用 ch01 chapter-map.py 的列感知路由(同列跨泳道→竖直附着,
不同列→右到左对角附着),第 1 泳道的三条竖直边正是靠这个路由与主脊柱节点对齐。

六项自查(渲染→Read PNG 亲眼看后如实记录):见 figure-manifest.json 该图 selfcheck。

用法:python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算:全角(ord>0x2E80)按 1.0×size,半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["主脊柱:查缓存 → 五级降级 → 发射 → 定位", "三个性能旋钮", "坐标系 · 无卡断裂线 · 双语接缝"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, § 站牌)
NODES = [
    ("entry",    0, 0, 0, "add_kernel[grid](...)",     "host 触发,交给 JITFunction.run",     "§1"),
    ("run",      0, 1, 0, "JITFunction.run",            "绑参→算缓存键→查缓存(miss 才编译)",  "§2"),
    ("compile",  0, 2, 0, "compile()",                  "五级 stages 的 for 循环心脏",         "§3"),
    ("lower",    0, 3, 0, "五级降级 ttir→cubin",         "add_stages 注册,逐级调 make_*",       "§4"),
    ("launch",   0, 4, 0, "kernel.run",                 "下发 grid,GPU 上异步执行",            "§8"),
    ("locator",  0, 5, 0, "定位地图",                    "症状→旋钮层→章节 对照表",             "§9"),
    ("knob1",    1, 1, 0, "缓存键 key",                  "签名+特化位+常量·旋钮①",              "§2"),
    ("knob2",    1, 3, 0, "num_warps / #blocked",       "make_ttgir 贴布局·旋钮②优化 pass",     "§4"),
    ("knob3",    1, 4, 0, "非 warmup 发射段",            "每次调用都过,缓存不省·旋钮③",         "§8"),
    ("coord",    2, 2, 0, "make_ir vs make_ttir",       "追踪期 56 行 vs 内联后 38 行·坐标系",  "§5"),
    ("fracture", 2, 3, 0, "ptxas / load_binary",        "cubin 仍 headless·断裂线在此",         "§6"),
    ("seams",    2, 4, 0, "四道双语接缝",                "跨 4 次语言/进程边界",                 "§7"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝;第 2 泳道三张参考卡片刻意不接边
    ("entry", "run"), ("run", "compile"), ("compile", "lower"), ("lower", "launch"), ("launch", "locator"),
    ("run", "knob1"), ("lower", "knob2"), ("launch", "knob3"),
]
# (路线名, [(列, § 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("全程走读:编译脊柱",
     [(0, "§1"), (1, "§2"), (2, "§3"), (3, "§4"), (4, "§8"), (5, "§9")], True),
    ("只查对照表:直接跳定位地图",
     [(1, "§2"), (3, "§4"), (4, "§8"), (5, "§9")], False),
    ("断点地图:坐标系→无卡线→双语缝",
     [(2, "§5"), (3, "§6"), (4, "§7")], False),
]
LEGEND = [
    ("#22c55e", "入口:host 调用 add_kernel[grid]"),
    ("#3b82f6", "章内主线:查缓存→五级降级→发射"),
    ("#f97316", "出口:§9 定位地图,你的下一步旋钮"),
]
TITLE = "第 3 章 · 一个 kernel 的一生:add_kernel[grid] 到 GPU 发射的源码剖面（§1–§9 讲解站牌）"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 180, 58
COL_GAP, ROW_GAP = 30, 20
EDGE_MARGIN, STUB_W, STUB_H = 12, 56, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 30
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_W, BADGE_H = 40, 20

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
    """§ 徽标胶囊,居中挂在 (cx,cy)。"""
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
    _lx += 20 + cjk_text_width(label, 11.5) + 30

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

# 入口/出口接口桩:入口挂 entry(最左),出口挂 locator(最右)
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["locator"]; xy += NODE_H / 2
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
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("你的下一步")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝):列感知路由——两端不同列走右→左附着(对角仍落源框右侧,不穿
# 正下方堆叠框);同列跨泳道走竖直 底心↔顶心 附着(第 1 泳道三条旋钮边正是此类)。
for src, dst in EDGES:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    col_s, col_d = NODE_BY_ID[src][2], NODE_BY_ID[dst][2]
    lane_s, lane_d = NODE_BY_ID[src][1], NODE_BY_ID[dst][1]
    if col_s != col_d:
        p1 = (xs_ + NODE_W, ys_ + NODE_H / 2)
        p2 = (xd, yd + NODE_H / 2)
    elif lane_d > lane_s:
        p1 = (xs_ + NODE_W / 2, ys_ + NODE_H)
        p2 = (xd + NODE_W / 2, yd)
    else:
        p1 = (xs_ + NODE_W / 2, ys_)
        p2 = (xd + NODE_W / 2, yd + NODE_H)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角 § 徽标)
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 6, y, sec)

# 底部阅读路线:复用列坐标 COLX,§ 站牌与图上节点对齐成竖向落点
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
print(f"wrote {out}  ({w}x{h}, aspect {w / h:.2f}:1)")
