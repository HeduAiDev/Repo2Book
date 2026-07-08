#!/usr/bin/env python3
"""第 18 章「本章地图」——310P 全栈特化的源码剖面图。

立意:一个布尔总开关 is_310p() 点亮整条 310 栈。它先在平台层选中 worker_cls
(NPUWorker310)，由此装配出 310 版 runner/input batch/block table，沿着这条
装配链分岔出「四条主线」(§18.3 输入批与块表、§18.4 受限 KV cache、§18.5 KV
清零、§18.6 权重加载)；同一个布尔又在另外两处横切文件里被直接读取，分流出
「注意力后端选择」与「分布式通信补丁」(均属 §18.7)。六个终点各自独立生效，
执行完毕后控制权交还给正常的 vLLM 调用链——这正是本图右侧那根贯穿全高的
「返回上层」竖条要表达的：不是单点汇合，而是六路各自回归主线。

改自 .claude/skills/svg-diagram/references/example-chapter-map.py 模板。

■ 不可变(照抄模板，未改动):§徽标胶囊样式/配色、入口绿#22c55e·出口橙#f97316·
  主线蓝#3b82f6、路线条高亮实线蓝/次要虚线灰、legend 规则、cjk_text_width()。
■ 本章特有的模板外扩展(非不可变项，因图形结构本身与模板范例不同才需要):
  - 节点宽度 NODE_W 从模板默认 190 放宽到 280——本章源码符号名普遍偏长
    (如 `communication_adaptation_310p()` 32 字符)，190 会溢出节点框。
  - 出口不是单个 merge 节点，而是六个独立终点(四条主线 + 两处横切)各自
    执行完毕后独立返回——不强行画一个不存在的"汇合函数"来凑模板的单出口
    形状(那样会杜撰一个源码里没有的符号)。因此右侧"返回上层"桩画成贯穿
    主线层+横切层整个纵深的竖条，六条橙色线各自水平接入，不做人为汇合。

用法: python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):逐字符判定——
    全角(CJK 及其他东亚文字,ord>0x2E80)按 1.0×size,半角(ASCII/拉丁/半角标点等)
    按 0.58×size,求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(本章数据) ----------------
LANES = ["总开关层", "主线特化层", "横切补丁层"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, §编号)
NODES = [
    ("entry",      0, 0, 0, "is_310p()",                       "SOC 字符串判定芯片型号",              "§18.1"),
    ("dispatch",   0, 1, 0, "worker_cls",                      "选中 NPUWorker310,关掉 custom_ops",  "§18.1"),
    ("slot_map",   1, 2, 0, "compute_slot_mapping",             "CPU NumPy 算 slot,替代 Triton",       "§18.3"),
    ("kv_cache",   1, 2, 1, "initialize_kv_cache_tensors",      "FRACTAL_NZ+128×128 对齐,早失败拒 MLA", "§18.4"),
    ("kv_zero",    1, 2, 2, "zero_block_ids",                   "张量切片 zero_() 替代 Triton",         "§18.5"),
    ("weight",     1, 2, 3, "save_model",                       "单 part 保存+生成量化描述",            "§18.6"),
    ("attn_be",    2, 1, 0, "backend_map_310",                  "选中 310 专属后端,注释掉 MLA/SFA",    "§18.7"),
    ("dist_patch", 2, 1, 1, "communication_adaptation_310p()",  "all_gather 模拟 broadcast/all_reduce", "§18.7"),
]
EDGES = [  # (src_id, dst_id) —— 调用/装配边,统一主线蓝
    ("entry", "dispatch"),
    ("entry", "attn_be"),
    ("entry", "dist_patch"),
    ("dispatch", "slot_map"),
    ("dispatch", "kv_cache"),
    ("dispatch", "kv_zero"),
    ("dispatch", "weight"),
]
# 出口:六个独立终点,各自返回上层——不汇入单一 merge 节点(源码里没有这样一个函数)
EXIT_NODE_IDS = ["slot_map", "kv_cache", "kv_zero", "weight", "attn_be", "dist_patch"]

# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("主线·slot mapping",   [(0, "§18.1"), (1, "§18.1"), (2, "§18.3")], True),
    ("主线·KV cache 约束",  [(0, "§18.1"), (1, "§18.1"), (2, "§18.4")], False),
    ("主线·KV 清零",        [(0, "§18.1"), (1, "§18.1"), (2, "§18.5")], False),
    ("主线·权重加载",       [(0, "§18.1"), (1, "§18.1"), (2, "§18.6")], False),
    ("横切补丁(后端+分布式)", [(0, "§18.1"), (1, "§18.7")], False),
]
LEGEND = [("#22c55e", "入口:从上层调用进入"), ("#3b82f6", "章内主线调用/装配边"), ("#f97316", "出口:返回上层执行")]
TITLE = "第 18 章 · is_310p() 分流剖面(源码走线 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 280, 58  # NODE_W 放宽(见文件头说明):本章符号名偏长
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

# 入口接口桩(单点,贴 entry 节点)
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')

# 出口接口桩:六个独立终点各自返回,不汇入单一节点——竖条贯穿主线层+横切层,
# 六条橙色线各自水平接入自己的行高,不做人为汇合(见文件头说明)。
_exit_ys = [NODE_XY[nid][1] + NODE_H / 2 for nid in EXIT_NODE_IDS]
exit_bar_y0 = min(_exit_ys) - NODE_H / 2 - BAND_PAD
exit_bar_y1 = max(_exit_ys) + NODE_H / 2 + BAND_PAD
exit_bar_h = exit_bar_y1 - exit_bar_y0
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{exit_bar_y0:.1f}" width="{STUB_W}" height="{exit_bar_h:.1f}" '
         f'rx="14" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{(exit_bar_y0 + exit_bar_h / 2 + 4):.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
for nid in EXIT_NODE_IDS:
    nx, ny = NODE_XY[nid]
    yc = ny + NODE_H / 2
    L.append(f'<line x1="{nx + NODE_W:.1f}" y1="{yc:.1f}" x2="{sx:.1f}" y2="{yc:.1f}" '
             f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝)。多条边汇入同一节点时终点 y 各偏移,本章无共享终点,故 y_offset 恒为 0。
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
print(f"wrote {out}  ({w:.0f}x{h:.0f}, ratio {w/h:.2f}:1)")
