#!/usr/bin/env python3
"""第 14 章「本章地图」——NPUWorker 生命周期源码剖面图。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py 模板改写：
不可变机制(esc/cjk_text_width/badge/配色/图例规则)原样保留，只改 DATA 与几何常量
（本章符号偏长，沿用 ch20 chapter-map.py 引入的"符号/短语可各拆 2 行"渲染扩展）。

节点预算 11(entry/init/initdev/detmem/compile/exec/exit 共 7 主线 + workerbase/workergpu
两个契约对照卫星 + profiler/xlite 两个横切卫星) ≤ 12。
本章标题为编号标题(## 14.1 ... ## 14.8 + 小结)，站牌用 §14.N。

设计要点：
- 主线(实边，蓝)是单入单出、无分叉的一条脊柱，对应"四步生命周期"外加首尾：
  NPUWorker(实例化/派生 WorkerBase) → __init__ → init_device/_init_device →
  determine_available_memory → compile_or_warm_up_model → execute_model →
  NPUModelRunner(出口:接棒真前向，下一章主角)。这条链不是纯粹的"调用关系"，而是
  "进程一生"的时间序——WorkerBase 契约规定这四步由外部引擎在不同阶段各调一次，
  execute_model 稳态里反复调——与 ch20 模板"调用边"的语义一致对待(同一条主线蓝)。
- 两个"契约对照"卫星 WorkerBase / Worker(GPU) 放在最上一条泳道、都挂在列 0(entry 正
  上方)，不挂调用边——代表"entry 节点(NPUWorker 派生 WorkerBase)背后的对照证据"，
  对应 §14.2 的论证(WorkerBase 是空抽象、Worker(GPU) 把设备层钉死在 cuda)。
- 两个"横切"卫星 TorchNPUProfilerWrapper / XliteWorker 放在最下一条泳道，分别贴在
  __init__ 列(profiler 在 __init__ 里置空)和 init_device 列(xlite 复用 _init_device
  的拆层接缝)正下方，但徽标是 §14.8——即"这条内容在主线的这个位置生根，但正文把它
  挪到 §14.8 横切小节统一交代"，不挂调用边(点名不展开，与正文一致)。
- exit 站牌用真实符号 `NPUModelRunner`——execute_model 派发的目标、下一章主角，
  比画一个泛化的"返回值"更真实可核。

节点内符号名普遍偏长(如 determine_available_memory()、compile_or_warm_up_model()、
TorchNPUProfilerWrapper)——沿用 ch20 引入的按 "\n" 机械换行(只切在真实的
下划线/驼峰边界上，不改变符号拼写本身，也不生造新词)：
  determine_available_memory() → "determine_" / "available_memory()"
  compile_or_warm_up_model()   → "compile_or_" / "warm_up_model()"
  init_device()/_init_device() → 两行分别是这两个真实同名方法（外层/内层）
  TorchNPUProfilerWrapper       → "TorchNPUProfiler" / "Wrapper"
每一半均可在 dossier.json 或正文中找到原样子串(如 "determine_available_memory()"
整串出现在 dossier data_flow 里，"available_memory()"/"determine_"均为其子串)。

六项自查记录(渲染→Read PNG 亲眼看后如实记录；见文件末尾)。

用法：python3 gen_chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算——全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["抽象契约对照", "NPUWorker 生命周期(主线)", "横切 / 平行路径"]  # 泳道,上→下

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名(可含 "\n" 机械换行,不改变拼写), 一行短语(可含 "\n"), §编号)
NODES = [
    ("workerbase", 0, 0, 0, "WorkerBase",        "抽象契约:四步方法\n体全 raise",            "§14.2"),
    ("workergpu",  0, 0, 1, "Worker(GPU)",        "init_device 锁死 cuda,\n非 cuda 即 raise",  "§14.2"),
    ("entry",      1, 0, 0, "NPUWorker",          "直接派生抽象 WorkerBase,\n与 Worker(GPU) 平级", "§14.2"),
    ("init",       1, 1, 0, "__init__",           "adapt_patch+ATB/customop\n注册,再 super().__init__", "§14.3"),
    ("initdev",    1, 2, 0, "init_device()\n_init_device()", "npu:N+torch_npu._inductor+\n快照+hccl 分布式", "§14.4"),
    ("detmem",     1, 3, 0, "determine_\navailable_memory()", "profile_run 量峰值→\n算 KV 可用显存", "§14.5"),
    ("compile",    1, 4, 0, "compile_or_\nwarm_up_model()",   "warmup_sizes→\ncapture_model→热 ATB", "§14.6"),
    ("exec",       1, 5, 0, "execute_model()",    "profile_memory→\n派发给 model_runner",      "§14.7"),
    ("exit",       1, 6, 0, "NPUModelRunner",     "接棒真前向\n(下一章主角)",                    "§14.7"),
    ("profiler",   2, 1, 0, "TorchNPUProfiler\nWrapper", "profiler 懒初始化,\nexecute_model 里 step()", "§14.8"),
    ("xlite",      2, 2, 0, "XliteWorker",        "override init_device,\n复用 _init_device", "§14.8"),
]
EDGES = [  # (src_id, dst_id) —— 主线蓝;workerbase/workergpu/profiler/xlite 是卫星,不挂边
    ("entry", "init"),
    ("init", "initdev"),
    ("initdev", "detmem"),
    ("detmem", "compile"),
    ("compile", "exec"),
    ("exec", "exit"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("通读主线(14.2→14.7)", [(0, "§14.2"), (1, "§14.3"), (2, "§14.4"), (3, "§14.5"), (4, "§14.6"), (5, "§14.7")], True),
    ("跳读:显存记账(快照顺序→KV预算)", [(2, "§14.4"), (3, "§14.5")], False),
    ("跳读:两个横切点(profiler/xlite)", [(1, "§14.8"), (2, "§14.8")], False),
]
LEGEND = [("#22c55e", "入口:从上层调用进入"), ("#3b82f6", "章内主线调用边"), ("#f97316", "出口:返回上层")]
TITLE = "第 14 章 · NPUWorker 生命周期源码剖面(重写证据 + 四步执行 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
# 本章符号名偏长——不靠加宽节点装下整行，改成机械换行(见 NODES 里的 "\n")；
# NODE_W 只需装下"半个符号名"，NODE_H 加高以容纳最多 2 行符号 + 最多 2 行短语。
NODE_W, NODE_H = 175, 92
COL_GAP, ROW_GAP = 14, 20
EDGE_MARGIN, STUB_W, STUB_H = 10, 46, 26
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

# 调用边(主线蓝,画在节点下面这条先画后画都行,这里先画边再画节点盖住端点毛刺)
# 多条边汇入同一节点时,终点 y 各偏移(间距 16px,如 2 条即 ±8px),
# 否则重合的终点在视觉上看不出"汇合"、像一条线断头。本章主线无分叉,单入单出,
# 但偏移逻辑保留以防将来改动 EDGES 引入多入边。
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

# [SELF-CHECK](渲染→Read PNG 亲眼看后如实记录，见 figure-manifest.json 的 blind_review/selfcheck)
