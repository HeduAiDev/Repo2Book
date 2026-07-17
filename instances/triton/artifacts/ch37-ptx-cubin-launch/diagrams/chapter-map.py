#!/usr/bin/env python3
"""ch37「本章地图」——从 PTX 到 cubin 到发射的源码剖面图
(third_party/nvidia/backend/compiler.py + driver.c + driver.py)。

本章是自然标题章(chapter.md 无 `## N.M` 编号,标题就是真实函数/类名,如
"make_ptx：LLVM IR 出成 PTX 文本")——按契约禁用 §N.M 徽标,站牌直接摘标题冒号前的
函数/类名本身(如"make_ptx"取自标题「make_ptx：...」),与代码符号天然重合。

剖面(三泳道 = 三个源文件/两种语言,自左向右是数据流顺序):
  编译期(compiler.py,Python)make_ptx→make_cubin(唯一起子进程处)
  → 装载期(driver.c,C)loadBinary(cuModuleLoadData 装载+读回 n_regs/n_spills+48KB opt-in)
  → 发射期(driver.py,Python)make_launcher(现造 C 源)→compile_module_from_src(sha256 缓存
    编译 .so)→CudaLauncher(self.launch=mod.launch)→_init_handles(首次懒装载触发点)
    →_launch(按 num_ctas 分派 cuLaunchKernel/Ex)→CudaDriver(GPUDriver 落地,配对脊柱)。
loadBinary 与 CudaLauncher 两条边一起汇入 _init_handles(它既要读已装载的 module/function
句柄,也要靠 launcher_cls 建好的 launcher)——用两条主线蓝边表达"汇合",非分支。

本章"寄存器占用/spill/occupancy"一节(标题「寄存器占用、spill 与 occupancy」)是纯理论
推导,没有独立代码符号(锚点复用 make_cubin 的 -v 打印与 loadBinary 的读回),故不设独立
节点——改用底部第二条阅读路线(跳读:只看 make_cubin→loadBinary 两站)呼应正文自己给的
跳读指引("只想弄清寄存器/spill 怎么压占用率,直接跳...一节")。

needs_device 红色语义(沿用 ch11 已用过的同一红色语言,本书视觉语言跨图统一):
loadBinary 与 _init_handles/_launch 三站需要真实 GPU 设备——host 无卡时 make_ptx/
make_cubin(ptxas 是 CPU 程序)/make_launcher/compile_module_from_src/CudaLauncher(纯
生成+gcc 编译,无需设备)都能跑通,只有真正"碰显存/碰驱动"的三站需要真机(呼应正文
"注意 cubin/发射需真机 CUDA"的读者提示)。

模板:.claude/skills/svg-diagram/references/example-chapter-map.py 的不可变视觉语言
(徽标胶囊/入口绿-出口橙-主线蓝/cjk_text_width)照搬,只改 DATA;沿用 ch11 chapter-map.py
的 fit_size/wrap_symbol 长符号自适应换行工具(本章符号名长短不一,如 CudaDriver 与
compile_module_from_src()/_init_handles()→_launch())。

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
    """符号名较长时的通用换行:先试单行从大到小的字号;仍塞不下,在 '_'/'('/'→'/','
    边界二分成两行(挑一个让两行里"更长的那行"最短的切点),用最小字号。返回 (lines, size)。"""
    for size in sizes:
        if cjk_text_width(text, size) <= max_w:
            return [text], size
    size = sizes[-1]
    candidates = ([i + 1 for i, c in enumerate(text) if c == ' ']
                  + [i + 1 for i, c in enumerate(text) if c == '_']
                  + [i for i, c in enumerate(text) if c == '(']
                  + [i + 1 for i, c in enumerate(text) if c == '→']
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
LANES = [
    "编译期·产出机器码——third_party/nvidia/backend/compiler.py",
    "装载期·搬进显存——third_party/nvidia/backend/driver.c",
    "发射期·按签名焊 launcher——third_party/nvidia/backend/driver.py",
]

FONT_SIZES = (12, 11, 10, 9, 8.5)

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌(自然标题的函数/类名,
#  禁用§N.M), needs_device: 是否需要真实 GPU 设备)
NODES = [
    ("ptx", 0, 0, 0, "make_ptx",
     "LLVM IR 译 PTX + 改版本号/去 debug 标志", "make_ptx", False),
    ("cubin", 0, 1, 0, "make_cubin",
     "起 ptxas 子进程,-v 打印 n_regs/spill 到 stderr", "make_cubin", False),
    ("load", 1, 2, 0, "loadBinary",
     "cuModuleLoadData 装载 + 读回占用 + 48KB opt-in", "loadBinary", True),
    ("launcher", 2, 3, 0, "make_launcher",
     "按核签名现场生成一段 C 发射器源码", "make_launcher", False),
    ("compile_so", 2, 4, 0, "compile_module_from_src",
     "sha256 缓存命中免重编,否则编成 .so", "make_launcher", False),
    ("cudalauncher", 2, 5, 0, "CudaLauncher",
     "self.launch = mod.launch,后续调用直接透传", "make_launcher", False),
    ("init_handles", 2, 6, 0, "_init_handles",
     "首次懒装载:建 module/function + launcher", "发射路径", True),
    ("launch", 2, 7, 0, "_launch",
     "num_ctas==1→cuLaunchKernel / >1→…Ex", "发射路径", True),
    ("driver", 2, 8, 0, "CudaDriver",
     "GPUDriver 落地:launcher_cls + is_active", "配对脊柱", False),
]
NODE_BY_ID = {n[0]: n for n in NODES}

# (src_id, dst_id, 边上小字标注或 None) —— 章内阅读顺序 + 数据依赖,统一主线蓝
EDGES = [
    ("ptx", "cubin", None),
    ("cubin", "load", None),
    ("load", "launcher", None),
    ("launcher", "compile_so", None),
    ("compile_so", "cudalauncher", None),
    ("load", "init_handles", None),
    ("cudalauncher", "init_handles", None),
    ("init_handles", "launch", None),
    ("launch", "driver", None),
]

# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("全通读(推荐):按序读完整条编译→装载→发射链",
     [(0, "make_ptx"), (1, "make_cubin"), (2, "loadBinary"), (3, "make_launcher"),
      (6, "发射路径"), (8, "配对脊柱")], True),
    ("跳读:只看寄存器占用/spill 怎么压 occupancy(不逐段跟读)",
     [(1, "make_cubin"), (2, "loadBinary")], False),
]
LEGEND = [
    ("#22c55e", "入口:上一章 add_stages 注册好的段被调用"),
    ("#3b82f6", "章内主线调用边 / 阅读顺序"),
    ("#b91c1c", "红:该步需要真实 GPU 设备(host 无卡在此断裂)"),
    ("#f97316", "出口:下一章换成 AMD 后端的镜像链"),
]
TITLE = "从 PTX 到 cubin 到发射：源码剖面(ptxas 子进程 · 装载读回占用 · C launcher 现场生成)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
# 本章新增一色(需图例,已在 LEGEND 登记):该步需要真实 GPU 设备
C_DEVICE_FILL, C_DEVICE_STROKE = "#fee2e2", "#b91c1c"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 124, 76
COL_GAP, ROW_GAP = 16, 18
EDGE_MARGIN, STUB_W, STUB_H = 10, 48, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 12
LANE_LABEL_H, BAND_PAD = 22, 14
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 46, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 54
BADGE_W, BADGE_H = 70, 20

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
    """站牌胶囊宽度——按文字自适应,不用固定 BADGE_W 截断(避免长站牌被裁)。"""
    return max(BADGE_W, cjk_text_width(text, 11) + 14)


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy)——自然标题的函数/类名摘要,非 §N.M。"""
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
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14.5" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例;本章 4 色,两行摆放避免单行过宽)
_legend_rows = [LEGEND[:2], LEGEND[2:]]
_ly = TOP_PAD + TITLE_H + 14
for row_items in _legend_rows:
    _lx = PAD_L
    for color, label in row_items:
        L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
        L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11" '
                 f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
        _lx += 20 + cjk_text_width(label, 11) + 30
    _ly += 18

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 5:.1f}" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩:入口挂 ptx(最左,上一章交棒),出口挂 driver(最右,配对脊柱收口)
ex, ey = NODE_XY["ptx"]; ey += NODE_H / 2
xx, xy = NODE_XY["driver"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#166534">{esc("上一章")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("下一章")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝):多条边汇入同一节点时终点 y 各偏移,避免"看不出汇合"
_dst_total = {}
for _, dst, _lbl in EDGES:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst, lbl in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    y_offset = (i - (n - 1) / 2) * 14 if n > 1 else 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
    if lbl:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2 - 5
        lsz = fit_size(lbl, NODE_W + COL_GAP - 8, (9.5, 8.5, 7.5))
        L.append(f'<rect x="{mx - cjk_text_width(lbl, lsz) / 2 - 3:.1f}" y="{my - lsz - 1:.1f}" '
                  f'width="{cjk_text_width(lbl, lsz) + 6:.1f}" height="{lsz + 4:.1f}" '
                  f'fill="white" opacity="0.85"/>')
        L.append(f'<text x="{mx:.1f}" y="{my:.1f}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="{lsz}" fill="{C_NODE_SUB}">{esc(lbl)}</text>')

# 节点(圆角框 + 真实符号名[自适应换行] + 一行短语 + 右上角站牌;needs_device 用红色语义)
for nid, lane, col, row, symbol, phrase, sec, needs_device in NODES:
    x, y = NODE_XY[nid]
    fill = C_DEVICE_FILL if needs_device else C_NODE_FILL
    stroke = C_DEVICE_STROKE if needs_device else C_NODE_STROKE
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    lines, sym_size = wrap_symbol(symbol, NODE_W - 16, FONT_SIZES)
    if len(lines) == 1:
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.34:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(lines[0])}</text>')
    else:
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.24:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(lines[0])}</text>')
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.24 + sym_size + 2:.1f}" '
                  f'text-anchor="middle" font-family="sans-serif" font-size="{sym_size}" '
                  f'font-weight="bold" fill="{C_NODE_TITLE}">{esc(lines[1])}</text>')
    phrase_lines, phr_size = wrap_symbol(phrase, NODE_W - 14, (9.5, 8.5, 7.5))
    py0 = y + NODE_H * 0.68
    for k, pl in enumerate(phrase_lines):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{py0 + k * (phr_size + 2):.1f}" '
                  f'text-anchor="middle" font-family="sans-serif" font-size="{phr_size}" '
                  f'fill="{C_NODE_SUB}">{esc(pl)}</text>')
    L += badge(x + NODE_W - badge_w(sec) / 2 + 6, y, sec)

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    row_top = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H
    # 路线名独占一行(在线/站牌上方),避免长中文路线名与左侧站牌横向挤在同一行相撞
    L.append(f'<text x="16" y="{row_top + 13:.1f}" font-family="sans-serif" font-size="11.5" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    ry = row_top + ROUTE_ROW_H - 16
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
print(f"wrote {out}  size={w}x{h}  ratio={w/h:.2f}")
