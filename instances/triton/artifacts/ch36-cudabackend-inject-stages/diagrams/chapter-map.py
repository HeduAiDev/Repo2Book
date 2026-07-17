#!/usr/bin/env python3
"""ch36「本章地图」——CUDABackend 把五段 stages 注入编译管线的源码剖面图
(third_party/nvidia/backend/compiler.py + python/triton/backends/compiler.py
 + python/triton/compiler/compiler.py + third_party/nvidia/triton_nvidia.cc)。

本章是自然标题章(chapter.md 无 `## N.M` 编号,七个标题全是自然语言,如
"一块新卡怎么接进来：BaseBackend 契约面")——按契约禁用 §N.M 徽标,站牌一律
取自标题词本身的短摘要(如 "BaseBackend 契约面" 取自第一个标题的后半段,
"配对脊柱"/"双语接缝" 取自末两个标题的前两字)。

剖面(三条泳道,自上而下):
  ① 契约面(顶,细):BaseBackend 抽象基类——parse_options/add_stages/
     load_dialects 三个 @abstractmethod,下探一条灰虚线到它在本章最想讲的
     那个具体方法 add_stages(回收 f1)。
  ② CUDABackend 落地实现(主脊,中,单行六列):按 compile() 里真实的调用先后
     顺序排列——CUDABackend(self.capability=target.arch)→parse_options
     (按 capability 拼 fp8 清单,回收 f6)→CUDAOptions(组装,__post_init__
     校验 num_warps 2 的幂)→add_stages(五个 lambda 钉进 stages 字典)→
     load_dialects(Python 薄封装)→make_ttgir(本章主角:按 capability//10
     分档注入 pass,串起 ch27-31)。左端挂入口(compile() 驱动,ch14 已讲),
     右端挂出口(编译产物落地,交给你的 kernel)。
  ③ C++ 双语接缝(底,细):init_triton_nvidia——pybind 把 load_dialects 与
     nvidia.passes.ttnvgpuir.* 暴露给 Python,两条紫虚线分别从 load_dialects
     与 make_ttgir 下探到这里(Python 编排顺序,C++ 提供 pass 实现)。

配色(不可变三色 + 本章新增两色,均配图例):
  绿 #22c55e=入口(compile() 驱动,ch14 已讲透);蓝 #3b82f6=主脊调用边
  (compile() 内真实先后调用序);橙 #f97316=出口(编译产物落地,回到你的
  kernel);灰 #94a3b8 虚线=BaseBackend 抽象契约→具体方法(回收 f1,非运行时
  数据流,是"定义→实现"的填充关系);紫 #8b5cf6 虚线=Python→C++ 的 pybind
  双语接缝(load_dialects/nvidia.passes.* 均在此落地)。

底部阅读路线复用章首那句"只想看 pass 序列怎么按卡分档,直接跳...;想跟全程,
按序读"——外加一条专讲 capability 两把尺(m4)的路线,把"粗档控 pass /
细阈控 dtype"这条设计不变量落回图上两个已有节点,不必为它单开节点。

模板:.claude/skills/svg-diagram/references/example-chapter-map.py 的不可变
视觉语言(徽标胶囊/入口绿-出口橙-主线蓝/cjk_text_width)照搬,只改 DATA;
跨泳道虚线边画法参照 ch14 chapter-map.py 的 down/up 边,改为本章的
contract(灰,契约面→主脊)与 cpp(紫,主脊→C++侧)两种。

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


def wrap_symbol(text, max_w, sizes):
    """符号名/短语较长时的通用换行:先试单行从大到小的字号;仍塞不下,
    在 ' '/'_'/'('/','/'/' 边界二分成两行(挑一个让两行里"更长的那行"最短
    的切点),用最小字号。返回 (lines, size)。"""
    for size in sizes:
        if cjk_text_width(text, size) <= max_w:
            return [text], size
    size = sizes[-1]
    candidates = ([i + 1 for i, c in enumerate(text) if c == ' ']
                  + [i + 1 for i, c in enumerate(text) if c == '_']
                  + [i for i, c in enumerate(text) if c == '(']
                  + [i + 1 for i, c in enumerate(text) if c == ',']
                  + [i + 1 for i, c in enumerate(text) if c == '/'])
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
LANES = [
    "契约面:BaseBackend 抽象基类(回收 f1)",
    "CUDABackend 落地实现(主脊 · 按 compile() 真实调用先后排列)",
    "C++ 双语接缝:pybind 暴露(Python 编排顺序,C++ 提供 pass 实现)",
]

FONT_SIZES = (12.5, 11.5, 10.5, 9.5, 8.5)
PHRASE_SIZES = (9.5, 9, 8.5, 8)

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌(取自本章标题词))
NODES = [
    ("cudabackend", 1, 0, 0, "CUDABackend",
     "BaseBackend 的 NVIDIA 实现,self.capability=target.arch", "配对脊柱"),
    ("parse_options", 1, 1, 0, "parse_options",
     "按 capability 拼 fp8 清单(回收 f6)", "按卡拼 fp8 清单"),
    ("cudaoptions", 1, 2, 0, "CUDAOptions",
     "frozen dataclass,校验 num_warps 是 2 的幂", "按卡拼 fp8 清单"),
    ("add_stages", 1, 3, 0, "add_stages",
     "五个 lambda 按序钉进 stages 字典", "钉进 stages 字典"),
    ("load_dialects", 1, 4, 0, "load_dialects",
     "Python 薄封装,转调 nvidia.load_dialects(ctx)", "双语接缝"),
    ("make_ttgir", 1, 5, 0, "make_ttgir",
     "按 capability//10 分档注入 pass,串起 ch27-31", "pass 串成真实序列"),
    ("base_backend", 0, 3, 0, "BaseBackend",
     "抽象契约:parse_options/add_stages/load_dialects", "BaseBackend 契约面"),
    ("init_nvidia", 2, 4, 0, "init_triton_nvidia",
     "pybind 暴露 load_dialects/nvidia.passes.* 给 Python", "双语接缝"),
]
NODE_BY_ID = {n[0]: n for n in NODES}

# (src_id, dst_id, 边样式: "main"=蓝实线,主脊内 compile() 真实调用先后序 /
#  "contract"=灰虚线,契约面→具体方法(定义→实现,非运行时数据流) /
#  "cpp"=紫虚线,Python→C++ 的 pybind 双语接缝, 边上小字标注或 None)
EDGES = [
    ("cudabackend", "parse_options", "main", None),
    ("parse_options", "cudaoptions", "main", "组装"),
    ("cudaoptions", "add_stages", "main", "闭包捕获"),
    ("add_stages", "load_dialects", "main", None),
    ("load_dialects", "make_ttgir", "main", "稍后调用"),
    ("base_backend", "add_stages", "contract", "抽象契约(回收 f1)"),
    ("load_dialects", "init_nvidia", "cpp", "转调(pybind)"),
    ("make_ttgir", "init_nvidia", "cpp", "nvidia.passes.*"),
]

# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("按序读完整落地实现:契约→拼配置→注册→双语→pass 序列",
     [(0, "配对脊柱"), (1, "按卡拼 fp8 清单"), (3, "钉进 stages 字典"),
      (4, "双语接缝"), (5, "pass 串成真实序列")], True),
    ("只看真实 pass 序列怎么按卡分档:直接跳 make_ttgir",
     [(5, "pass 串成真实序列")], True),
    ("capability 两把尺:粗档(pass)在 make_ttgir / 细阈(dtype)在 parse_options",
     [(1, "按卡拼 fp8 清单"), (5, "pass 串成真实序列")], False),
]
LEGEND = [
    ("#22c55e", "入口:compile() 驱动(ch14 已讲透)"),
    ("#3b82f6", "主脊调用边:compile() 内真实先后调用序"),
    ("#f97316", "出口:编译产物落地,回到你的 kernel"),
    ("#94a3b8", "灰虚线:BaseBackend 抽象契约→具体方法(回收 f1)"),
    ("#8b5cf6", "紫虚线:Python→C++ 的 pybind 双语接缝"),
]
TITLE = "CUDABackend 剖面 · 把五段 stages 注入编译管线(Part VIII 开篇)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
C_CONTRACT = "#94a3b8"
C_CPP = "#8b5cf6"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 172, 84
COL_GAP, ROW_GAP = 46, 18
EDGE_MARGIN, STUB_W, STUB_H = 10, 60, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 22
LANE_LABEL_H, BAND_PAD = 20, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 30, 42, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_H = 20

n_cols = max(n[2] for n in NODES) + 1
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_lane = [0] * len(LANES)
for _id, lane, col, row, *_ in NODES:
    rows_per_lane[lane] = max(rows_per_lane[lane], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP
          for r in rows_per_lane]
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

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge_w(text):
    """站牌胶囊宽度——按文字自适应,不用固定宽度截断。"""
    return max(40, cjk_text_width(text, 11) + 14)


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy)——本章自然标题词的短摘要(非虚构 §N.M)。"""
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
                        ("Contract", C_CONTRACT), ("Cpp", C_CPP))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14.5" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(5 色,两行摆放避免单行过宽)
_legend_rows = [LEGEND[:3], LEGEND[3:]]
_ly = TOP_PAD + TITLE_H + 14
for row_items in _legend_rows:
    _lx = PAD_L
    for color, label in row_items:
        L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
        L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="10.5" '
                 f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
        _lx += 20 + cjk_text_width(label, 10.5) + 24
    _ly += 17

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 4:.1f}" font-family="sans-serif" '
             f'font-size="11.5" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩:入口挂主脊最左(CUDABackend,被 ch14 compile() 驱动调用),
# 出口挂主脊最右(make_ttgir 之后,编译产物落地交给你的 kernel)
ex, ey = NODE_XY["cudabackend"]; ey += NODE_H / 2
xx, xy = NODE_XY["make_ttgir"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#166534">{esc("compile()")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#9a3412">{esc("你的 kernel")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边:main 走主脊同泳道水平线;contract/cpp 是跨泳道竖直虚线(定义→实现 /
# Python→C++),多条边汇入同一节点时终点沿竖直方向各偏移,避免视觉上完全重合。
_dst_total = {}
for _, dst, kind, _lab in EDGES:
    if kind in ("contract", "cpp"):
        _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst, kind, label in EDGES:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    if kind == "main":
        p1 = (xs_ + NODE_W, ys_ + NODE_H / 2)
        p2 = (xd, yd + NODE_H / 2)
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
        if label:
            L.append(f'<text x="{(p1[0] + p2[0]) / 2:.1f}" y="{p1[1] - 8:.1f}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="9" fill="{C_MAIN}">{esc(label)}</text>')
    else:
        n_ = _dst_total[dst]
        i = _dst_seen.get(dst, 0)
        _dst_seen[dst] = i + 1
        x_off = (i - (n_ - 1) / 2) * 22 if n_ > 1 else 0
        color = C_CONTRACT if kind == "contract" else C_CPP
        marker = "mContract" if kind == "contract" else "mCpp"
        p1 = (xs_ + NODE_W / 2 + x_off, ys_ + NODE_H)
        p2 = (xd + NODE_W / 2 + x_off, yd)
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{color}" stroke-width="1.8" stroke-dasharray="5,4" '
                  f'marker-end="url(#{marker})"/>')
        if label:
            mid_y = (p1[1] + p2[1]) / 2
            anchor = "end" if x_off < 0 else "start"
            tx = p1[0] + (-6 if x_off < 0 else 6)
            L.append(f'<text x="{tx:.1f}" y="{mid_y + 3:.1f}" text-anchor="{anchor}" '
                     f'font-family="sans-serif" font-size="9" fill="{color}">{esc(label)}</text>')

# 节点(圆角框 + 真实符号名(必要时自动换行) + 一行短语 + 右上角站牌)
SYM_MAXW = NODE_W - 16
PHR_MAXW = NODE_W - 14
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_lines, sym_size = wrap_symbol(symbol, SYM_MAXW, FONT_SIZES)
    cx = x + NODE_W / 2
    if len(sym_lines) == 1:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.30:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
    else:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.22:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.22 + sym_size + 2:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[1])}</text>')
    phr_lines, phr_size = wrap_symbol(phrase, PHR_MAXW, PHRASE_SIZES)
    py0 = y + NODE_H * 0.58
    for k, pl in enumerate(phr_lines):
        L.append(f'<text x="{cx:.1f}" y="{py0 + k * (phr_size + 3):.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{phr_size}" fill="{C_NODE_SUB}">{esc(pl)}</text>')
    bw = badge_w(sec)
    L += badge(x + NODE_W - bw / 2 + 8, y, sec)

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="11.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌,对应正文同名小节;实线蓝=推荐 / 虚线灰=次要跳读)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    row_top = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H
    name_y = row_top + 12
    ry = row_top + ROUTE_ROW_H - 15
    L.append(f'<text x="16" y="{name_y:.1f}" font-family="sans-serif" font-size="10.5" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first = COLX[stops[0][0]] + NODE_W / 2
    x_last = COLX[stops[-1][0]] + NODE_W / 2
    dash = '' if hi else ' stroke-dasharray="6,4"'
    if x_first != x_last:
        L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
                  f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for col, sec in stops:
        L += badge(COLX[col] + NODE_W / 2, ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  ({w}x{h}, aspect {w / h:.2f}:1)")
