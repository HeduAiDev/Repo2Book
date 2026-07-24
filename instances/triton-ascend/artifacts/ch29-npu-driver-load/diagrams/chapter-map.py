#!/usr/bin/env python3
"""ch29 本章地图:NPU 驱动装载剖面——driver.py(Python 驱动层)+ npu_utils.cpp
(C++ 扩展)两层结构,从 triton 核心的 _init_handles 到最终返回可调用句柄。

模板来源:.claude/skills/svg-diagram/references/example-chapter-map.py(§徽标胶囊/
入口绿-出口橙-主线蓝/cjk_text_width() 不可变,几何常量按本章节点数/画布预算调过)。

六项自查记录(渲染→Read PNG 亲眼看后如实记录):
  claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
  arrows_attached=True     cjk_rendered=True         reading_order_clear=True
  (无 spec.numbers——本图是源码剖面图/站牌图,不含数值型 claim,numbers_match_spec
  按"图上无数字断言"判定为 True。)

用法:python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):逐字符判定——
    全角(ord>0x2E80)按 1.0×size,半角按 0.58×size,求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(本章数据) ----------------
LANES = ["triton 核心（章外）", "NPU 驱动层 · driver.py", "C++ 扩展 · npu_utils.cpp"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, §编号)
NODES = [
    ("entry",      0, 0, 0, "_init_handles",    "首次访问触发,换句柄",      "§29.3"),
    ("jit",        1, 1, 0, "NPUUtils",         "即时编译加载 npu_utils.so", "§29.2"),
    ("driver_ctr", 1, 1, 1, "NPUDriver",        "triton 认的 driver 对象",   "§29.7"),
    ("loadbin",    1, 2, 0, "load_binary",      "rsplit 拆出 mix_mode",      "§29.3"),
    ("hw_probe",   2, 2, 0, "getArch",          "+getAiCoreNum 探硬件规格",  "§29.8"),
    ("lkb",        2, 3, 0, "loadKernelBinary", "解包六元组,转调 C++",       "§29.3"),
    ("regk",       2, 4, 0, "registerKernel",   "三步 rt* 短路注册",         "§29.4"),
    ("magic",      2, 5, 0, "devbin.magic",     "aiv 用 AIVEC,余用 ELF",     "§29.5"),
    ("stub",       2, 5, 1, "registered_names", "同名 kernel 编号防撞",      "§29.6"),
    ("exit",       0, 6, 0, "装载完成",         "回填句柄,下一章发射",      "§29.9"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝
    ("entry", "jit"), ("entry", "driver_ctr"),
    ("driver_ctr", "hw_probe"),
    ("jit", "loadbin"),
    ("loadbin", "lkb"),
    ("lkb", "regk"),
    ("regk", "magic"), ("regk", "stub"),
    ("magic", "exit"), ("stub", "exit"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("全程（推荐）",
     [(0, "§29.3"), (1, "§29.2"), (2, "§29.3"), (3, "§29.3"), (4, "§29.4"), (6, "§29.9")], True),
    ("跳读:直奔装载核心",
     [(0, "§29.3"), (4, "§29.4"), (5, "§29.5"), (6, "§29.9")], False),
    ("旁支:驱动与硬件探测",
     [(1, "§29.7"), (2, "§29.8")], False),
]
LEGEND = [("#22c55e", "入口:triton 核心首次调用"), ("#3b82f6", "章内跨层调用边"),
          ("#f97316", "出口:回填句柄给 triton 核心")]
TITLE = "第 29 章 · NPU 驱动装载剖面（driver.py 与 npu_utils.cpp 两层结构）"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数;本章节点较多,按 10 节点/7 列调窄) ----------------
NODE_W, NODE_H = 160, 58
COL_GAP, ROW_GAP = 26, 18
EDGE_MARGIN, STUB_W, STUB_H = 14, 62, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 28  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 22, 10
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 12, 30, 24, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 20, 34
BADGE_W, BADGE_H = 44, 19

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
        f'<text x="{cx:.1f}" y="{cy + 3.8:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="10.5" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 16}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 13
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 10}" width="13" height="13" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 18}" y="{_ly}" font-family="sans-serif" font-size="10.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 18 + cjk_text_width(label, 10.5) + 26

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="14" y="{band_top[i] + LANE_LABEL_H - 5:.1f}" font-family="sans-serif" '
             f'font-size="12" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩(给入口/出口箭头一个可附着的框,兼表达"调用方/下一章在画布外")
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["exit"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 3.8:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#166534">{esc("上一章")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 3.8:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#9a3412">{esc("下一章")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝);多条边汇入同一节点时终点 y 各偏移,避免看不出"汇合"
_dst_total = {}
for _, dst in EDGES:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
_src_total = {}
for src, _ in EDGES:
    _src_total[src] = _src_total.get(src, 0) + 1
_src_seen = {}
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    n_out = _src_total[src]
    i_out = _src_seen.get(src, 0)
    _src_seen[src] = i_out + 1
    y_out_offset = (i_out - (n_out - 1) / 2) * 14 if n_out > 1 else 0
    p1 = (x1 + NODE_W, y1 + NODE_H / 2 + y_out_offset)
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    y_offset = (i - (n - 1) / 2) * 14 if n > 1 else 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角 § 徽标)
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="12" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="9.8" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 7, y, sec)

# 底部阅读路线:复用列坐标 COLX,§ 徽标与图上节点对齐成竖向落点
L.append(f'<text x="14" y="{routes_top + 14:.1f}" font-family="sans-serif" font-size="11.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="14" y="{ry + 3.8:.1f}" font-family="sans-serif" font-size="11" '
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
print(f"wrote {out}  canvas={w:.0f}x{h:.0f}  ratio={w/h:.2f}")
