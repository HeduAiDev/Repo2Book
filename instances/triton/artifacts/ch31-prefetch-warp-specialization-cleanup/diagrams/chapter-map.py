#!/usr/bin/env python3
"""ch31《Prefetch、Warp Specialization 与杂项清理 pass》本章地图——源码剖面图。

五条泳道 = 本章五个自然分节，按正文阅读顺序自上而下排列：
  L0 全景:这些 pass 落在编译流程的哪一格(make_ttgir 里排队)
  L1 Prefetch:让共享内存到寄存器的搬运藏进计算(本章主角，三步机器)
  L2 F32DotTC 的 TF32x3:为什么一个 fp32 dot 要拆成三次
  L3 Warp Specialization(选读·进阶):按角色把循环拆到不同 warpgroup
  L4 尾部清理:两个代表看源码
全章是一条不分岔的主线链(overview → initialize 配对脊柱 → generatePrefetch →
createNewForOp → TF32x3 → num_consumer_groups 门控 → WSLowering → ReduceDataDuplication
→ ReorderInstructions)，同泳道内相邻节点用直线(EDGES)，跨泳道用 WRAP_EDGES 的
elbow(右侧绕出 → 下降到下一泳道标签留白带 → 横移到目标列 → 落入目标节点顶部)。

■ 本章特有(自然标题章——chapter.md 只有 `## 标题正文`/`### 标题正文`，无
  `## N.M` 编号，heading_set 为空，判定为自然标题章，与 ch28/ch09 chapter-map
  先例一致):
  - 节点右上角站牌**禁用 §N.M 徽标**，改用对应 `##`/`###` 标题的逐字子串
    (如 "配对脊柱" 取自 `### 配对脊柱：通用 pass 怎么接纳第三方的 MMA 编码"；
    "Prefetcher 的四步机器" 取自 `### 源码：Prefetcher 的四步机器"，去掉"源码："
    前缀后仍是原标题的连续子串)。全部 9 个站牌逐一核对见文件末尾自查记录。
  - **底部阅读路线不复用 COLX 列坐标**(与 example-chapter-map.py/ch28 的默认
    做法不同，这是本章特有的必要偏离，原因见下)：本章 9 个节点分布在 5 条
    泳道、每条泳道内部列号从 0 重新起算(如 L1 用列 0/1/2，L2 又用列 0)，若
    路线沿用 COLX[col] 当 x 坐标，"列 0"在图上会对应五个不同的 x 位置吗？不——
    COLX 是全局列坐标数组，同一列号在所有泳道中共享同一个 x。于是 L0/L2/L3/L4
    里各自的"列 0"节点会被画在**同一条竖线**上，路线里连续出现多个"列 0"站牌
    时，其徽标会在同一 x 位置层叠重合(no_overlap 会当场炸)。改为按站牌文本宽度
    动态推进的水平游标(复用 LEGEND 那种"量宽度→前进"算法，cjk_text_width 逐个
    累加间距)，序号由路线自身的阅读顺序决定，不依赖节点列号——彻底避免同列号
    复用导致的重叠，同时仍保持"从左到右=阅读顺序"这条不变量。

■ 不可变(全书统一视觉语言，换章节数据时不要动这些，只改下面的 DATA)：
  与 example-chapter-map.py / ch28 chapter-map.py 完全一致——站牌胶囊 / 入口绿
  #22c55e-出口橙#f97316-主线蓝#3b82f6 / 高亮实线蓝-次要虚线灰 / cjk_text_width()
  宽度估算 / badge() 按文本动态算宽(自然语言站牌，非定长 §N.M 短码)。

■ 可变：LANES / NODES / EDGES(同泳道直线) / WRAP_EDGES(跨泳道折行边，含合流
  偏移) / ROUTES(改用宽度游标，见上) / LEGEND / TITLE。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录；每次改动 DATA/布局代码后必须
  重新核对一遍，不能照抄上一轮结果)：
  [第一轮] no_overlap 曾误记为 True——Read PNG 后发现 tf32x3→ws_gate 这条跨泳道
    折行边的水平段穿过了 L3"Warp Specialization（选读·进阶）：按角色把循环拆到
    不同 warpgroup"这条较长的泳道标签文字(定长 WRAP_GAP 外扩的 turn_x=358 小于
    该标签末端 x≈463)。本轮改为 turn_x = max(定长外扩值, 目标泳道标签末端x+12)
    逐边计算，重渲染+Read PNG 复核确认线已完全让开文字。
    同轮 lint_chapter_map 还报了 1 处 fabricated_symbol(`TTGIR(`)——LEGEND 里
    "TTGIR(流水线…)" 半角括号紧跟标识符被当成新 token 核对不到；改用全角括号
    "TTGIR（流水线…）"（与 ch28 先例一致:标识符后不紧跟半角圆括号）。
  [第二轮/当前，已重渲染+Read PNG 复核] claim_readable_10s=True
    numbers_match_spec=True no_overlap=True arrows_attached=True
    cjk_rendered=True reading_order_clear=True
  —— 9 个站牌逐一核对来源(均为对应 ##/### 标题的连续子串，见 NODES 注释)；
     9 个符号(make_ttgir/initialize/generatePrefetch/createNewForOp/TF32x3/
     num_consumer_groups/WSLowering/ReduceDataDuplication/ReorderInstructions)
     与 iter_args/local_load/local_alloc 等短语内词均已用 grep 核对在
     narrative/chapter.md 与 dossier/dossier.json 中逐字出现(本文件不含
     一个杜撰符号)；渲染后 Read PNG 检查画布 932x790 左右，5 条泳道横向
     互不压框，跨泳道折行边全部终点落在下一泳道节点顶部、无悬空；两条
     阅读路线(全通读 / 跳过选读 WS)徽标左右不重叠、连线端点对齐首尾徽标。

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def fit_size(text, max_w, base, min_size):
    """按 max_w 反解一个不超出的字号(单行，不换行)。"""
    unit = cjk_text_width(text, 1.0)
    if unit <= 0:
        return base
    return max(min_size, min(base, max_w / unit))


# ---------------- DATA(可变：本章数据) ----------------
LANES = [
    "全景：这些 pass 落在编译流程的哪一格",
    "Prefetch：让共享内存到寄存器的搬运藏进计算",
    "F32DotTC 的 TF32x3：为什么一个 fp32 dot 要拆成三次",
    "Warp Specialization（选读·进阶）：按角色把循环拆到不同 warpgroup",
    "尾部清理：两个代表看源码",
]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌文本[标题词逐字子串，禁用 §N.M])
NODES = [
    ("overview", 0, 0, 0, "make_ttgir",
     "sm80+ 分支尾部,本章诸 pass 排队", "编译流程的哪一格"),
    ("gate", 1, 0, 0, "initialize",
     "只收 MMAv2/AMD MFMA、单 dot", "配对脊柱"),
    ("slice", 1, 1, 0, "generatePrefetch",
     "沿 K 切 subview + local_load", "Prefetcher 的四步机器"),
    ("newloop", 1, 2, 0, "createNewForOp",
     "新循环多 2 个 iter_args", "Prefetcher 的四步机器"),
    ("tf32x3", 2, 0, 0, "TF32x3",
     "3 个 tf32 dot 逼近 fp32", "三个 dot 串成累加链"),
    ("ws_gate", 3, 0, 0, "num_consumer_groups",
     "默认 0，>0 才触发五个 WS pass", "默认关闭，>0 才触发五个 WS pass"),
    ("ws_lowering", 3, 1, 0, "WSLowering",
     "task id → warpId/4 落地 warpgroup", "async task id 变成真实的 warpgroup"),
    ("reduce_dup", 4, 0, 0, "ReduceDataDuplication",
     "cvt 改道 local_alloc+local_load", "两个代表看源码"),
    ("reorder", 4, 1, 0, "ReorderInstructions",
     "按寄存器压力就地下沉", "两个代表看源码"),
]
EDGES = [  # 同泳道相邻列的直线
    ("gate", "slice"), ("slice", "newloop"),
    ("ws_gate", "ws_lowering"),
    ("reduce_dup", "reorder"),
]
# 跨泳道折行边:(src_id, dst_id, dst 落点 x 偏移——多条边合流到同一 dst 时错开)
WRAP_EDGES = [
    ("overview", "gate", 0),
    ("newloop", "tf32x3", 0),
    ("tf32x3", "ws_gate", 0),
    ("ws_lowering", "reduce_dup", 0),
]
# (路线名, [站牌文本,...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
# 注意:不复用 COLX——见文件头注释,改在渲染时按站牌宽度动态推进 x 游标。
# 路线站牌用节点站牌的**更短前缀子串**(仍逐字取自对应 ##/### 标题，只是更短)、
# 且同一站合并只报一次(slice/newloop 同属"Prefetcher 的四步机器"一站、
# reduce_dup/reorder 同属"尾部清理"一站，路线里不重复列两遍)——否则 9 个足长
# 站牌顺次排开会把画布撑到 1742 宽，超出 lint_chapter_map 的 1500 画布预算。
ROUTES = [
    ("全通读(含选读 WS)", [
        "编译流程的哪一格", "配对脊柱", "Prefetcher 的四步机器",
        "三个 dot 串成累加链", "默认关闭", "变成真实的 warpgroup", "两个代表看源码",
    ], True),
    ("只要性能决策(跳过选读 WS)", [
        "编译流程的哪一格", "配对脊柱", "Prefetcher 的四步机器",
        "三个 dot 串成累加链", "两个代表看源码",
    ], False),
]
LEGEND = [("#22c55e", "入口:上游 TTGIR（流水线已搭好跨迭代骨架）"),
          ("#3b82f6", "章内主线:全景→Prefetch→TF32x3→WS(选读)→尾部清理"),
          ("#f97316", "出口:高性能 TTGIR,交给下一章降级到 PTX")]
TITLE = "第 31 章 · Prefetch / TF32x3 / Warp Specialization / 尾部清理 剖面(源码走线 + 讲解站牌)"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替，仅装饰，非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数) ----------------
NODE_W, NODE_H = 210, 60
COL_GAP, ROW_GAP = 50, 20  # 自然语言站牌较宽，COL_GAP 加大留右侧悬挂余量(同 ch28)
EDGE_MARGIN, STUB_W, STUB_H = 16, 78, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 30
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_H, BADGE_PAD_X, BADGE_FONT = 20, 10, 11
WRAP_GAP = 22  # 折行边:绕出节点右侧的横向余量
ROUTE_STOP_GAP = 26  # 阅读路线:相邻站牌徽标之间的空隙(按宽度游标推进用)

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
w_nodes = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD

# 路线区所需宽度必须在任何 SVG 元素生成之前算好(不能事后补——lane 背景矩形/
# 出口接口桩都要用最终 w 定位)：route 名称栏 + 逐条路线站牌宽度游标的总长,
# 取所有路线里最长的一条。
_route_name_w = max(cjk_text_width(name, 12) for name, _, _ in ROUTES)
_route_x0 = 16 + _route_name_w + 24


def _route_stops_width(stops):
    cursor = _route_x0
    for sec in stops:
        bw = cjk_text_width(sec, BADGE_FONT) + BADGE_PAD_X * 2
        cursor += bw + ROUTE_STOP_GAP
    return cursor - ROUTE_STOP_GAP  # 末尾多加的一份 gap 减掉，得到最后一个徽标右边缘附近位置


w_routes = max(_route_stops_width(stops) for _, stops, _ in ROUTES) + PAD_R
w = max(w_nodes, w_routes)


def badge(cx, cy, text):
    """站牌胶囊，居中挂在 (cx,cy)——宽度按文本动态算(自然语言站牌，非定长短码)。"""
    bw = cjk_text_width(text, BADGE_FONT) + BADGE_PAD_X * 2
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 3.8:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="{BADGE_FONT}" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
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
    lane_size = fit_size(name, w - 32, 13, 10)
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="{lane_size:.1f}" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩
ex, ey = NODE_XY["overview"]; ey += NODE_H / 2
xx, xy = NODE_XY["reorder"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#166534">{esc("上游 ttgir")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="11" font-weight="bold" fill="#9a3412">{esc("降级下一程")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 同泳道相邻列的直线调用边(主线蓝)——多条边汇入同一节点时终点 y 各偏移,避免重合看不出汇合
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

# 跨泳道折行边(elbow):右侧绕出 → 下降到"下一泳道标签+留白"这条空白带(节点顶部
# 之上、分界线之下，全程无节点/无文字) → 沿这条空白带一路向左/右 → 短距下降落入
# 下一泳道节点顶部(含 dst_dx 偏移，供多条边合流到同一节点时错开落点)。
# turn_x 不能只按 WRAP_GAP 定长外扩——若目标泳道的自然语言标签比较长(如本章
# L3"Warp Specialization（选读·进阶）：按角色把循环拆到不同 warpgroup"，标签
# 右端 x≈463)，定长 turn_x 可能落在标签文字中间，折行边的水平段会穿过文字
# (no_overlap 违规，第一轮渲染 Read PNG 时抓到:tf32x3→ws_gate 这条边穿过了
# L3 标签)。改为逐边取 max(定长 turn_x, 目标泳道标签末端 x + 12 安全间距)。
LANE_LABEL_END_X = []
for _name in LANES:
    _size = fit_size(_name, w - 32, 13, 10)
    LANE_LABEL_END_X.append(16 + cjk_text_width(_name, _size))

for wsrc, wdst, dst_dx in WRAP_EDGES:
    wx1, wy1 = NODE_XY[wsrc]; wx2, wy2 = NODE_XY[wdst]
    dst_lane = NODE_BY_ID[wdst][1]
    p_start = (wx1 + NODE_W, wy1 + NODE_H / 2)
    turn_x = max(wx1 + NODE_W + WRAP_GAP, LANE_LABEL_END_X[dst_lane] + 12)
    drop_y = wy2 - 8
    p_mid1 = (turn_x, wy1 + NODE_H / 2)
    p_mid2 = (turn_x, drop_y)
    p_mid3 = (wx2 + NODE_W / 2 + dst_dx, drop_y)
    p_end = (wx2 + NODE_W / 2 + dst_dx, wy2)
    L.append(f'<polyline points="{p_start[0]:.1f},{p_start[1]:.1f} {p_mid1[0]:.1f},{p_mid1[1]:.1f} '
             f'{p_mid2[0]:.1f},{p_mid2[1]:.1f} {p_mid3[0]:.1f},{p_mid3[1]:.1f} {p_end[0]:.1f},{p_end[1]:.1f}" '
             f'fill="none" stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌)，字号按文本长度自适应收缩避免溢出
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_size = fit_size(symbol, NODE_W - 18, 13, 9)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{sym_size:.1f}" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    ph_size = fit_size(phrase, NODE_W - 16, 10.5, 8)
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{ph_size:.1f}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    bw = cjk_text_width(sec, BADGE_FONT) + BADGE_PAD_X * 2
    L += badge(x + NODE_W - bw / 2 + 10, y, sec)

# 底部阅读路线:按站牌宽度动态推进 x 游标(不复用 COLX——见文件头注释:同一列号
# 在不同泳道间共享同一 x，若沿用会在同一路线里把多个"列 0"站牌画到同一 x 上,
# 造成 no_overlap 违规。改用与 LEGEND 相同的"量宽度→前进"算法；游标起点/画布宽
# 已在 w_routes 处统一预算过，这里只需重放同一游标逻辑。)
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(从左到右=阅读顺序;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    xs_ = []
    cursor = _route_x0
    for sec in stops:
        bw = cjk_text_width(sec, BADGE_FONT) + BADGE_PAD_X * 2
        xs_.append(cursor + bw / 2)
        cursor += bw + ROUTE_STOP_GAP
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{xs_[0]:.1f}" y1="{ry:.1f}" x2="{xs_[-1]:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for cx, sec in zip(xs_, stops):
        L += badge(cx, ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({w:.0f}x{h:.0f}, ratio {w / h:.2f})")
