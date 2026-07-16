#!/usr/bin/env python3
"""ch12「本章地图」——driver 抽象、后端发现、autotune 与磁盘缓存的源码剖面图
(python/triton/runtime/driver.py + python/triton/backends/{__init__,driver}.py
 + python/triton/runtime/autotuner.py + python/triton/runtime/cache.py
 + python/triton/compiler/compiler.py)。

本章是自然标题章(chapter.md 无 `## N.M` 编号,只有自然标题,如
"惰性代理：import triton 为什么不点火 CUDA")——按契约禁用 §N.M 徽标,
站牌一律改用标题词本身的关键词摘要(如"配对脊柱"取自 H2「配对脊柱：后端怎么被
发现、又怎么选出唯一一个」,"契约与断裂点"取自 H3「契约与断裂点：GPUDriver
桥到 torch.cuda」)。

剖面(三条泳道 = 章正文自己的"门后三间屋子"隐喻,每条泳道内部一行左→右主脊):
  ① driver 惰性发现与选择——_discover_backends(配对脊柱,扫目录建表)
     与 LazyProxy(惰性代理,首次访问触发)一起喂给 _create_driver(筛唯一
     is_active),实例化出的 driver 是 GPUDriver 子类,__init__ 桥接 torch.cuda
     ——即 ch11 标出的"真设备断裂点"(红色需真设备标记)。
  ② Autotuner:自动调优的完整操作面(性能杠杆,本章重点)——run() 按缓存键分流:
     miss 才展开 prune_configs → _bench(需真设备)→ hook,三者仅未命中才执行
     (橙色分支节点),取最优后与 hit 直达路径(蓝色弧形跳过边,同 ch11 的
     cache-hit-skip 视觉手法)在 config.all_kwargs()「旋钮下发」汇合。
  ③ 磁盘缓存:跨进程免重编(与①②内存态的两层缓存正交)——compile() 缓存键构造
     → get_group/put_group 命中判定(命中早返回 CompiledKernel,文字点出,不再
     画第二条跳过弧以免与②的弧形视觉冲突)→ FileCacheManager.put 仅未命中才
     执行(橙色)。

配色(不可变三色 + 本章沿用 ch11 已建立的两条语义色,保持全书视觉语言一致):
  绿 #22c55e = 入口;蓝 #3b82f6 = 主线调用边/常规节点(含 hit 跳过弧,复用同一蓝);
  橙 #f97316 = 出口;橙棕 #b45309(填充 #fef3c7) = "仅未命中(cache miss)才执行"
  的节点——本章有两层缓存(autotune 内存缓存、磁盘缓存),同一橙色统一标注两层
  各自的 miss-only 路径,是刻意的跨小节視覺呼应;红 #b91c1c(填充 #fee2e2) =
  该步需要真实 GPU 设备,沿用 ch11 figure 已用过的同一红色语言。

模板:.claude/skills/svg-diagram/references/example-chapter-map.py 的不可变视觉语言
(徽标胶囊/入口绿-出口橙-主线蓝/cjk_text_width)+ ch11 chapter-map.py 的
fit_size/wrap_symbol(长符号自适应换行)与 skip/branch 边画法照搬,只改 DATA、
扩成三泳道并让 ARC_RISE 按泳道独立取值(仅②需要跳过弧的额外顶部空间)。

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
    """符号名较长时的通用换行:先试单行从大到小的字号;仍塞不下,在 '_'/'('/','/' /'
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
LANES = [
    "第一间屋子:driver 惰性发现与选择——backends/ 目录 + LazyProxy",
    "第二间屋子(性能杠杆):Autotuner 自动调优的完整操作面",
    "第三间屋子:磁盘缓存——跨进程免重编(与①②的内存态缓存正交)",
]

FONT_SIZES = (12.5, 11.5, 10.5, 9.5, 8.5)

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌(自然标题关键词,禁用§N.M),
#  needs_device: 是否需要真实 GPU 设备, is_branch: 是否"仅未命中/miss 才执行")
NODES = [
    # ① driver 惰性发现与选择
    ("discover", 0, 0, 0, "_discover_backends", "扫 backends/<name>/,零改动纳入新后端(如 ascend/)",
     "配对脊柱", False, False),
    ("lazy", 0, 1, 0, "LazyProxy", "首次属性访问才触发 _init_fn 构造",
     "惰性代理", False, False),
    ("create", 0, 2, 0, "_create_driver", "从配对脊柱建的表里筛唯一 is_active() 的 driver",
     "选出唯一一个", False, False),
    ("gpudriver", 0, 3, 0, "GPUDriver.__init__", "桥接 torch.cuda——headless 断裂点",
     "契约与断裂点", True, False),
    # ② Autotuner
    ("run", 1, 0, 0, "Autotuner.run", "key + 张量 dtype 组缓存键,miss 才展开搜索",
     "缓存键", False, False),
    ("prune", 1, 1, 0, "prune_configs", "early_config_prune + perf_model top_k 裁剪候选",
     "prune_configs 裁剪", False, True),
    ("bench", 1, 2, 0, "_bench", "do_bench 取分位数(0.5,0.2,0.8),异常记 inf",
     "_bench 计时", True, True),
    ("hook", 1, 3, 0, "reset_to_zero / restore_value", "pre/post hook 保护被 kernel 改写的张量",
     "hook 同一张卷子", False, True),
    ("dispatch", 1, 4, 0, "config.all_kwargs()", "摊平旋钮,转给 fn.run(接回上一章脊柱)",
     "旋钮下发", False, False),
    # ③ 磁盘缓存
    ("key", 2, 0, 0, "compile() 缓存键构造", "triton_key+src+backend+options+env → sha256",
     "缓存键指纹", False, False),
    ("group", 2, 1, 0, "get_group / put_group", "hit 早返回 CompiledKernel;miss 才落盘+建索引",
     "group 校验", False, False),
    ("put", 2, 2, 0, "FileCacheManager.put", "tmp 目录 + os.replace 原子改名落盘",
     "原子落盘", False, True),
]
NODE_BY_ID = {n[0]: n for n in NODES}

# (src_id, dst_id, 边样式: "main"=蓝实线常规调用 / "skip"=蓝实线命中直达(跨过 miss 分支)
#  / "branch"=橙虚线仅未命中才执行的分支, 边上小字标注或 None)
EDGES = [
    # ①(discover 与 create 之间隔着 lazy,数据关系走 create 的 phrase 文案交代,
    #   不画跨列箭头以免线压在 lazy 节点身上)
    ("lazy", "create", "main", "触发"),
    ("create", "gpudriver", "main", "实例化"),
    # ②(run 的 miss 分支 prune→bench→hook→dispatch,与 hit 的直接跳过边并存)
    ("run", "prune", "branch", "未命中"),
    ("prune", "bench", "branch", "裁剪后"),
    ("bench", "hook", "branch", "装配"),
    ("hook", "dispatch", "branch", "取 min"),
    ("run", "dispatch", "skip", "hit:直接复用 best_config"),
    # ③
    ("key", "group", "main", "查询"),
    ("group", "put", "branch", "未命中"),
]

# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("顺序①先看 driver 怎么被发现、选出唯一一个",
     [(0, "配对脊柱"), (1, "惰性代理"), (2, "选出唯一一个"), (3, "契约与断裂点")], True),
    ("顺序②再看 Autotuner 完整操作面(本章性能杠杆)",
     [(0, "缓存键"), (1, "prune_configs 裁剪"), (2, "_bench 计时"), (3, "hook 同一张卷子"), (4, "旋钮下发")], True),
    ("顺序③最后看磁盘缓存(跨进程免重编,与①②正交)",
     [(0, "缓存键指纹"), (1, "group 校验"), (2, "原子落盘")], True),
    ("跳读:只想上手调优→直接看②,跳过①③",
     [(0, "缓存键"), (1, "prune_configs 裁剪"), (2, "_bench 计时"), (3, "hook 同一张卷子"), (4, "旋钮下发")], False),
]
LEGEND = [
    ("#22c55e", "入口:import triton / @triton.autotune 核调用"),
    ("#3b82f6", "章内主线调用边(含 hit 跳过弧)/ 常规节点"),
    ("#b45309", "橙:仅未命中(cache miss)才执行的节点(两层缓存各自的 miss 支路)"),
    ("#b91c1c", "红:该步需要真实 GPU 设备(host 无 GPU 在此断裂)"),
    ("#f97316", "出口:返回调用方 / 下一章"),
]
TITLE = "第 12 章 · 门后三间屋子——driver 发现与选择、Autotuner 操作面、磁盘缓存"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
# 本章沿用 ch11 已建立的两色(需图例,已在 LEGEND 登记)
C_BRANCH_FILL, C_BRANCH_STROKE = "#fef3c7", "#b45309"
C_DEVICE_FILL, C_DEVICE_STROKE = "#fee2e2", "#b91c1c"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 202, 106
COL_GAP, ROW_GAP = 48, 20
EDGE_MARGIN, STUB_W, STUB_H = 12, 60, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 26
LANE_LABEL_H, BAND_PAD = 22, 14
ARC_RISE = [0, 52, 0]  # 每条泳道各自的跳过弧顶部预留空间;仅②(run→dispatch)需要
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 46, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 46
BADGE_W, BADGE_H = 74, 20

n_cols = max(n[2] for n in NODES) + 1
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_lane = [0] * len(LANES)
for _id, lane, col, row, *_ in NODES:
    rows_per_lane[lane] = max(rows_per_lane[lane], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + ARC_RISE[i] + r * NODE_H + max(0, r - 1) * ROW_GAP
          for i, r in enumerate(rows_per_lane)]
band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for bh in band_h:
    band_top.append(_cum)
    _cum += bh
lanes_bottom = _cum

NODE_XY = {}
for nid, lane, col, row, *_ in NODES:
    x = COLX[col]
    y = band_top[lane] + LANE_LABEL_H + BAND_PAD + ARC_RISE[lane] + row * (NODE_H + ROW_GAP)
    NODE_XY[nid] = (x, y)

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge_w(text):
    """站牌胶囊宽度——按文字自适应,不用固定 BADGE_W 截断(避免中文站牌被裁)。"""
    return max(BADGE_W, cjk_text_width(text, 11) + 14)


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy)——自然标题关键词摘要,非 §N.M。"""
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
        _lx += 20 + cjk_text_width(label, 11) + 26
    _ly += 18

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 4:.1f}" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩:入口挂①的 discover(最左上,import triton 触发建表),
# 出口挂③的 put(最右下,磁盘缓存收口——章末转入下一章)
ex, ey = NODE_XY["discover"]; ey += NODE_H / 2
xx, xy = NODE_XY["put"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#166534">{esc("import")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("下一章")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边:main=蓝实线直连;skip=蓝实线弧形跳过(越过 miss 分支节点上方);branch=橙虚线
for src, dst, kind, label in EDGES:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    src_lane = NODE_BY_ID[src][1]
    if kind == "skip":
        rise = ARC_RISE[src_lane]
        p1 = (xs_ + NODE_W / 2, ys_)
        p2 = (xd + NODE_W / 2, yd)
        arc_y = ys_ - rise + 10
        path = f'M {p1[0]:.1f},{p1[1]:.1f} C {p1[0]:.1f},{arc_y:.1f} {p2[0]:.1f},{arc_y:.1f} {p2[0]:.1f},{p2[1]:.1f}'
        L.append(f'<path d="{path}" fill="none" stroke="{C_MAIN}" stroke-width="2.2" '
                  f'marker-end="url(#mMain)"/>')
        if label:
            L.append(f'<text x="{(p1[0] + p2[0]) / 2:.1f}" y="{arc_y - 6:.1f}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
                     f'fill="{C_MAIN}">{esc(label)}</text>')
    elif kind == "branch":
        p1 = (xs_ + NODE_W, ys_ + NODE_H / 2 - 12)
        p2 = (xd, yd + NODE_H / 2 - 12)
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_BRANCH_STROKE}" stroke-width="2" stroke-dasharray="6,4" '
                  f'marker-end="url(#mBranch)"/>')
        if label:
            L.append(f'<text x="{(p1[0] + p2[0]) / 2:.1f}" y="{p1[1] - 6:.1f}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="9.5" fill="{C_BRANCH_STROKE}">{esc(label)}</text>')
    else:
        p1 = (xs_ + NODE_W, ys_ + NODE_H / 2 + 12)
        p2 = (xd, yd + NODE_H / 2 + 12)
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
        if label:
            L.append(f'<text x="{(p1[0] + p2[0]) / 2:.1f}" y="{p1[1] - 6:.1f}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="9.5" fill="{C_MAIN}">{esc(label)}</text>')

# 节点(圆角框 + 真实符号名(必要时自动换行/缩字号) + 一行短语 + 右上角站牌 + 需真设备⚡角标)
SYM_MAXW = NODE_W - 16
PHR_MAXW = NODE_W - 14
for nid, lane, col, row, symbol, phrase, sec, needs_device, is_branch in NODES:
    x, y = NODE_XY[nid]
    fill, stroke = C_NODE_FILL, C_NODE_STROKE
    if is_branch:
        fill, stroke = C_BRANCH_FILL, C_BRANCH_STROKE
    if needs_device:
        fill, stroke = C_DEVICE_FILL, C_DEVICE_STROKE
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    sym_lines, sym_size = wrap_symbol(symbol, SYM_MAXW, FONT_SIZES)
    cx = x + NODE_W / 2
    if len(sym_lines) == 1:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.26:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
    else:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.20:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.20 + sym_size + 2:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[1])}</text>')
    phr_lines, phr_size = wrap_symbol(phrase, PHR_MAXW, (10, 9.5, 8.5, 8))
    py0 = y + NODE_H * 0.56
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

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点(自然标题关键词,非 §N.M)
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌文字,对应正文对应小节;实线蓝=推荐 / 虚线灰=次要跳读)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    row_top = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H
    name_y = row_top + 13
    ry = row_top + ROUTE_ROW_H - 18
    L.append(f'<text x="16" y="{name_y:.1f}" font-family="sans-serif" font-size="11" '
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
