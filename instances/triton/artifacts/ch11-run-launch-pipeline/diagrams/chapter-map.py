#!/usr/bin/env python3
"""ch11「本章地图」——JITFunction.run 一次 launch 的源码剖面图
(python/triton/runtime/jit.py + python/triton/compiler/compiler.py)。

本章是自然标题章(chapter.md 无 `## N.M` 编号,标题用 ①②③④⑤⑥ 记号 + 自然语言,
如"① 边界一跳：driver.active 取环境 + make_backend")——按契约禁用 §N.M 徽标,
站牌一律改用标题词本身的短摘要(如"边界一跳"取自标题「① 边界一跳：...」)。

剖面(单泳道,run() 六段主脊 + 前后各一个收口节点,横向一条主时间线):
  driver.active(边界一跳,需真设备) → create_binder(惰性 binder) → self.cache[device](拼键查cache)
  →(未命中才展开)compile(慢路径,条件分支,橙色) → grid_0/1/2(规范化 grid)
  → kernel.run(跨语言发射,需真设备) → _init_handles(惰性设备句柄,需真设备) → 三档开销判据(收官)
  cache 命中时直接从「拼键查cache」跳到「规范化 grid」,不经过 compile 分支——
  用一条弧形跳过边(越过 compile 节点上方)表达"命中直达",与"未命中→compile→回填"
  两条边共同汇入 grid_0/1/2 节点(色彩区分:蓝实线=命中直达/主线,橙虚线=未命中支路)。

配色(不可变三色 + 本章新增两色,均配图例):
  绿 #22c55e = 入口(用户代码调用 fn[grid](args));蓝 #3b82f6 = 主线调用边/常规节点;
  橙 #f97316 = 出口(返回上层/下一章);橙红 #b45309(浅橙填充 #fef3c7) = 本章新增语义,
  未命中才展开的编译慢路径(条件分支节点);红 #b91c1c(浅红填充 #fee2e2) = 本章新增语义,
  该节点需要真实 GPU 设备(host 无 GPU 在此断裂),沿用 fig-ch11-launch-spine /
  fig-ch11-driver-boundary 等本章其余插图已用过的同一红色语言,视觉语言跨图统一。

模板:.claude/skills/svg-diagram/references/example-chapter-map.py 的不可变视觉语言
(徽标胶囊/入口绿-出口橙-主线蓝/cjk_text_width)照搬,只改 DATA;沿用 ch10
chapter-map.py 的 fit_size/wrap_symbol 长符号自适应换行工具(本章符号名有长有短,
如 self.cache[device] / grid_0, grid_1, grid_2)。

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


def fit_size(text, max_w, sizes):
    """从大到小试字号,返回第一个能让 text 单行塞进 max_w 的字号;都不行则返回最小字号。"""
    for size in sizes:
        if cjk_text_width(text, size) <= max_w:
            return size
    return sizes[-1]


def wrap_symbol(text, max_w, sizes):
    """符号名较长时的通用换行:先试单行从大到小的字号;仍塞不下,在 '_'/'('/','
    边界二分成两行(挑一个让两行里"更长的那行"最短的切点),用最小字号。返回 (lines, size)。"""
    for size in sizes:
        if cjk_text_width(text, size) <= max_w:
            return [text], size
    size = sizes[-1]
    candidates = ([i + 1 for i, c in enumerate(text) if c == ' ']
                  + [i + 1 for i, c in enumerate(text) if c == '_']
                  + [i for i, c in enumerate(text) if c == '(']
                  + [i + 2 for i, c in enumerate(text) if c == ','])
    if not candidates:
        candidates = [len(text) // 2]
    best = None
    for idx in candidates:
        if idx <= 0 or idx >= len(text):
            continue
        a, b = text[:idx], text[idx:]
        w = max(cjk_text_width(a, size), cjk_text_width(b, size))
        if best is None or w < best[0]:
            best = (w, a, b)
    if best is None:
        return [text], size
    return [best[1], best[2]], size


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["JITFunction.run 一次 launch 的主脊——python/triton/runtime/jit.py + compiler.py"]

FONT_SIZES = (12.5, 11.5, 10.5, 9.5, 8.5)

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌(自然标题摘要,禁用§N.M),
#  needs_device: 是否需要真实 GPU 设备, is_branch: 是否为未命中才展开的条件分支)
NODES = [
    ("env",     0, 0, 0, "driver.active",           "取 device/stream/target + make_backend",
     "边界一跳", True, False),
    ("binder",  0, 1, 0, "create_binder",           "惰性建 binder,调它得 5 元组",
     "惰性 binder", False, False),
    ("cache",   0, 2, 0, "self.cache[device]",      "拼 key,查内存 cache",
     "拼键查 cache", False, False),
    ("slow",    0, 3, 0, "compile",                 "未命中才现造整套编译输入",
     "慢路径", False, True),
    ("grid",    0, 4, 0, "grid_0, grid_1, grid_2",  "callable grid 求值,补齐三维",
     "规范化 grid", False, False),
    ("emit",    0, 5, 0, "kernel.run",              "跨语言交给 C++ launcher",
     "跨语言发射", True, False),
    ("lazy",    0, 6, 0, "_init_handles",           "首次读 .run 才装 cubin 到设备",
     "惰性设备句柄", True, False),
    ("cost",    0, 7, 0, "三档开销判据",             "命中µs / 热编ms / 冷编s,发射受限的尺子",
     "三档开销", False, False),
]
NODE_BY_ID = {n[0]: n for n in NODES}

# (src_id, dst_id, 边样式: "main"=蓝实线常规调用 / "skip"=蓝实线命中直达(跨过 slow 节点)
#  / "branch"=橙虚线未命中支路, 边上小字标注或 None)
EDGES = [
    ("env", "binder", "main", None),
    ("binder", "cache", "main", None),
    ("cache", "slow", "branch", "未命中"),
    ("slow", "grid", "branch", "回填后"),
    ("cache", "grid", "skip", "命中直达"),
    ("grid", "emit", "main", None),
    ("emit", "lazy", "main", None),
    ("lazy", "cost", "main", None),
]

# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("全程走读(推荐):按章节顺序从边界一跳到发射收口",
     [(0, "边界一跳"), (1, "惰性 binder"), (2, "拼键查 cache"), (4, "规范化 grid"),
      (5, "跨语言发射"), (6, "惰性设备句柄")], True),
    ("cache 命中直达:③ 拼键查 cache 之后直接跳到 ⑤,不展开 ④",
     [(2, "拼键查 cache"), (4, "规范化 grid")], False),
    ("未命中慢路径:④ 查完 cache 才现造的编译派单",
     [(2, "拼键查 cache"), (3, "慢路径"), (4, "规范化 grid")], False),
    ("只看性能结论:跳到全章最后一节读判据,不逐段跟读",
     [(0, "边界一跳"), (7, "三档开销")], False),
]
LEGEND = [
    ("#22c55e", "入口:用户代码 fn[grid](args) 调用 run()"),
    ("#3b82f6", "章内主线调用边 / 常规节点"),
    ("#b45309", "橙红:未命中才展开的编译慢路径(条件分支)"),
    ("#b91c1c", "红:该步需要真实 GPU 设备(host 无 GPU 在此断裂)"),
    ("#f97316", "出口:返回调用方 / 下一章(driver 子系统·ch12,compile 内部·ch14)"),
]
TITLE = "第 11 章 · run() 一次 launch 的脊柱剖面——从缓存查询到编译再到内核发射"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
# 本章新增两色(需图例,已在 LEGEND 登记)
C_BRANCH_FILL, C_BRANCH_STROKE = "#fef3c7", "#b45309"
C_DEVICE_FILL, C_DEVICE_STROKE = "#fee2e2", "#b91c1c"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 138, 112
COL_GAP, ROW_GAP = 26, 20
EDGE_MARGIN, STUB_W, STUB_H = 12, 52, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 22
LANE_LABEL_H, BAND_PAD = 24, 16
ARC_RISE = 58  # 弧形"命中直达"跳过边所需的额外顶部空间
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 36, 46, 18
ROUTE_HEAD_H, ROUTE_ROW_H = 24, 52
BADGE_W, BADGE_H = 74, 20

n_cols = max(n[2] for n in NODES) + 1
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_lane = [0] * len(LANES)
for _id, lane, col, row, *_ in NODES:
    rows_per_lane[lane] = max(rows_per_lane[lane], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + ARC_RISE + r * NODE_H + max(0, r - 1) * ROW_GAP
          for r in rows_per_lane]
band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for bh in band_h:
    band_top.append(_cum)
    _cum += bh
lanes_bottom = _cum

NODE_XY = {}
for nid, lane, col, row, *_ in NODES:
    x = COLX[col]
    y = band_top[lane] + LANE_LABEL_H + BAND_PAD + ARC_RISE + row * (NODE_H + ROW_GAP)
    NODE_XY[nid] = (x, y)

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge_w(text):
    """站牌胶囊宽度——按文字自适应,不用固定 BADGE_W 截断(避免中文站牌被裁)。"""
    return max(BADGE_W, cjk_text_width(text, 11) + 14)


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy)——自然标题摘要,非 §N.M。"""
    bw = badge_w(text)
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN),
                        ("Branch", C_BRANCH_STROKE))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例;本章 5 色,两行摆放避免单行过宽)
_legend_rows = [LEGEND[:3], LEGEND[3:]]
_ly = TOP_PAD + TITLE_H + 14
for row_items in _legend_rows:
    _lx = PAD_L
    for color, label in row_items:
        L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
        L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11" '
                 f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
        _lx += 20 + cjk_text_width(label, 11) + 28
    _ly += 18

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩:入口挂 env(最左,driver 边界一跳),出口挂 cost(最右,三档开销收官)
ex, ey = NODE_XY["env"]; ey += NODE_H / 2
xx, xy = NODE_XY["cost"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#166534">{esc("用户代码")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("下一章")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边:main=蓝实线直连;skip=蓝实线弧形跳过(越过 slow 节点上方);branch=橙虚线
for src, dst, kind, label in EDGES:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    if kind == "skip":
        p1 = (xs_ + NODE_W / 2, ys_)
        p2 = (xd + NODE_W / 2, yd)
        arc_y = ys_ - ARC_RISE + 10
        path = f'M {p1[0]:.1f},{p1[1]:.1f} C {p1[0]:.1f},{arc_y:.1f} {p2[0]:.1f},{arc_y:.1f} {p2[0]:.1f},{p2[1]:.1f}'
        L.append(f'<path d="{path}" fill="none" stroke="{C_MAIN}" stroke-width="2.2" '
                  f'marker-end="url(#mMain)"/>')
        if label:
            L.append(f'<text x="{(p1[0] + p2[0]) / 2:.1f}" y="{arc_y - 6:.1f}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="11" font-weight="bold" '
                     f'fill="{C_MAIN}">{esc(label)}</text>')
    elif kind == "branch":
        p1 = (xs_ + NODE_W, ys_ + NODE_H / 2 - 12)
        p2 = (xd, yd + NODE_H / 2 - 12)
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_BRANCH_STROKE}" stroke-width="2" stroke-dasharray="6,4" '
                  f'marker-end="url(#mBranch)"/>')
        if label:
            L.append(f'<text x="{(p1[0] + p2[0]) / 2:.1f}" y="{p1[1] - 6:.1f}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="10.5" fill="{C_BRANCH_STROKE}">{esc(label)}</text>')
    else:
        p1 = (xs_ + NODE_W, ys_ + NODE_H / 2 + 12)
        p2 = (xd, yd + NODE_H / 2 + 12)
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名(必要时自动换行/缩字号) + 一行短语 + 右上角站牌 + 需真设备⚡角标)
SYM_MAXW = NODE_W - 14
PHR_MAXW = NODE_W - 12
for nid, lane, col, row, symbol, phrase, sec, needs_device, is_branch in NODES:
    x, y = NODE_XY[nid]
    fill, stroke = C_NODE_FILL, C_NODE_STROKE
    if is_branch:
        fill, stroke = C_BRANCH_FILL, C_BRANCH_STROKE
    elif needs_device:
        fill, stroke = C_DEVICE_FILL, C_DEVICE_STROKE
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    sym_lines, sym_size = wrap_symbol(symbol, SYM_MAXW, FONT_SIZES)
    cx = x + NODE_W / 2
    if len(sym_lines) == 1:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.30:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
    else:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.24:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.24 + sym_size + 2:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[1])}</text>')
    phr_lines, phr_size = wrap_symbol(phrase, PHR_MAXW, (10.5, 9.5, 8.5, 8))
    py0 = y + NODE_H * 0.60
    for k, pl in enumerate(phr_lines):
        L.append(f'<text x="{cx:.1f}" y="{py0 + k * (phr_size + 3):.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{phr_size}" fill="{C_NODE_SUB}">{esc(pl)}</text>')
    bw = badge_w(sec)
    L += badge(x + NODE_W - bw / 2 + 10, y, sec)
    if needs_device:
        L.append(f'<circle cx="{x + 2:.1f}" cy="{y + 2:.1f}" r="10" '
                  f'fill="{C_DEVICE_STROKE}" stroke="white" stroke-width="1.5"/>')
        L.append(f'<text x="{x + 2:.1f}" y="{y + 6:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" font-weight="bold" '
                  f'fill="white">{esc("!")}</text>')

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点(自然标题摘要,非 §N.M)
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌文字,对应正文对应小节;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    row_top = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H
    name_y = row_top + 13
    ry = row_top + ROUTE_ROW_H - 20
    L.append(f'<text x="16" y="{name_y:.1f}" font-family="sans-serif" font-size="11.5" '
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
