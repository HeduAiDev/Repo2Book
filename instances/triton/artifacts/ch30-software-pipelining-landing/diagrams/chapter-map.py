#!/usr/bin/env python3
"""ch30《软件流水线落地:MatmulLoopPipeline 建模与 PipelineExpander 展开》本章地图。

四条泳道 = 章节自己划的"两半"结构,每半再拆上/下两条泳道装 3 站:
  前半 · 建模(NV 语义,1/2):pipelineLoop 总入口 → createAsyncCopy(load 换 async copy)
    → scheduleLoads(CoarseSchedule 打 stage/cluster 标)
  前半 · 建模(NV 语义,2/2):scheduleRemainingToLastStage(漏网 op 兜底打标)
    → createAlloc(环形缓冲首维=numBuffers) → PipeliningOption(建模交付物,唯一接口桥)
  后半 · 展开(后端无关 SCF,1/2):pipelineForLoop 总控 → emitPrologue(阶梯填充)
    → analyzeCrossStageValues(跨 stage 活跃期)
  后半 · 展开(后端无关 SCF,2/2):createKernelLoop(模变量扩展补 iter_arg)
    → createKernel(稳态体重映射+谓词化收尾) → asyncLaunchDots(Hopper wgmma 尾声,出口)
中间那道 PipeliningOption → pipelineForLoop 的折行边就是全章反复点破的接缝:
建模端满是 cp.async/wgmma 的 NV 语义,展开端往后全是通用 scf.for 变换。

■ 本章特有(自然标题章,`## `/`### ` 标题无 `N.M` 编号,如
  "前半 · 建模:把顺序循环读成一张调度表"/"CoarseSchedule:给每个算子发两个号"——
  不匹配 lint_chapter_map 的 `^##\\s+\\d+\\.\\d+` 正则,heading_set 为空,判定为自然
  标题章,处理同 ch27/ch28/ch29 先例):
  - 节点右上角站牌**禁用带小数点的 §N.M 徽标**,改用真实标题词的逐字子串(如
    "换成异步预取"取自"### 喂 dot 的 load,就地换成异步预取"、"封装一座桥"取自
    "### 建模的交付物:四行代码封装一座桥");聚合站(第 11 站同时覆盖"稳态体改写"
    与"谓词化收尾"两个三级标题,因两者同出自 createKernel 一个函数)用
    "A + B"拼接两个真实子串,不假装是单一标题的逐字子串。
  - badge()/BADGE_W 按文本动态算宽(cjk_text_width + 内边距),站牌是完整词组
    而非定长短码,固定宽度会溢出。
  - 泳道名直接取真实二级标题"前半 · 建模:把顺序循环读成一张调度表"/"后半 · 展开:
    把一张调度表撑成流水线"逐字,只在末尾追加"(上)"/"(下)"标记同一标题下的
    两条泳道,不改动标题本身的字。
  - 标识符一律不紧跟半角圆括号(camelCase 符号如 pipelineLoop/createAsyncCopy/
    scheduleLoads/createAlloc/PipeliningOption/emitPrologue/createKernelLoop/
    createKernel/asyncLaunchDots 不含 `_`/`(`/内部 `.`,不入 lint 的杜撰符号核对,
    但均为正文逐字出现的真实符号,可直接用);scheduleRemainingToLastStage、
    analyzeCrossStageValues、pipelineForLoop 同理。

■ 不可变(全书统一视觉语言):站牌胶囊 / 入口绿#22c55e-出口橙#f97316-主线蓝#3b82f6 /
  高亮实线蓝-次要虚线灰 / cjk_text_width() 宽度估算——与 example-chapter-map.py /
  ch27 / ch28 / ch29 chapter-map.py 完全一致。

■ 可变:LANES / NODES / EDGES / WRAP_EDGES / ROUTES / LEGEND / TITLE。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录):
  [第一轮,已渲染+Read PNG 复核] claim_readable_10s=True numbers_match_spec=True
    no_overlap=True arrows_attached=True cjk_rendered=True reading_order_clear=True
    —— 12 个节点(pipelineLoop/createAsyncCopy/scheduleLoads/
       scheduleRemainingToLastStage/createAlloc/PipeliningOption/pipelineForLoop/
       emitPrologue/analyzeCrossStageValues/createKernelLoop/createKernel/
       asyncLaunchDots)均为 dossier.json code_spine/chapter.md 正文逐字出现的
       真实符号;站牌全部为真实标题词逐字子串(不含 §N.M);"建模半→展开半"折行边
       (options→pipelineForLoop)清楚标出全章的接缝论点;两条底部阅读路线对应
       正文明说的"通读"与"只看环形缓冲速览"两种读法;lint_chapter_map(无
       --require,试点期)与 lint_diagram_geometry 均核实通过(见文件末尾
       print 后 Bash 记录)。

用法:python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算:全角(ord>0x2E80)按 1.0×size,半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def fit_size(text, max_w, base, min_size):
    """按 max_w 反解一个不超出的字号(单行,不换行)。"""
    unit = cjk_text_width(text, 1.0)
    if unit <= 0:
        return base
    return max(min_size, min(base, max_w / unit))


# ---------------- DATA(可变:本章数据) ----------------
LANES = [
    "前半 · 建模：把顺序循环读成一张调度表（上）",
    "前半 · 建模：把顺序循环读成一张调度表（下）",
    "后半 · 展开：把一张调度表撑成流水线（上）",
    "后半 · 展开：把一张调度表撑成流水线（下）",
]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌文本[标题词逐字子串,禁用 §N.M])
NODES = [
    ("entry", 0, 0, 0, "pipelineLoop",
     "两半调度总控：建模→展开→MMAv3 后处理", "建模，然后展开"),
    ("async_copy", 0, 1, 0, "createAsyncCopy",
     "喂 dot 的 load 换 async copy 三件套（读写游标各定槽）", "换成异步预取"),
    ("coarse_sched", 0, 2, 0, "scheduleLoads",
     "root dot 钉最后 stage，load 逐层前移，算 distToUse", "给每个算子发两个号"),
    ("dep_sched", 1, 0, 0, "scheduleRemainingToLastStage",
     "anchor 依赖同 stage，distance-1 前移，剩余兜底最后 stage", "把漏网的 op 各归各位"),
    ("ring_buffer", 1, 1, 0, "createAlloc",
     "缓冲首维=numBuffers（max distToUse，+1 MMAv3）", "变成共享内存的那一维"),
    ("options", 1, 2, 0, "PipeliningOption",
     "调度表+peelEpilogue=false 等 3 旋钮，唯一接口", "封装一座桥"),
    ("expand_ctrl", 2, 0, 0, "pipelineForLoop",
     "initializeLoopInfo→emitPrologue→analyzeCrossStageValues→…", "五步总控"),
    ("prologue", 2, 1, 0, "emitPrologue",
     "maxStage 段阶梯克隆，动态循环时谓词裹住早段", "先空转几拍把流水灌满"),
    ("liverange", 2, 2, 0, "analyzeCrossStageValues",
     "defStage≠useStage 才登记，量活跃跨度", "一个值要在飞几拍"),
    ("modvar", 3, 0, 0, "createKernelLoop",
     "按跨度补 iter_arg，loopArgMap 记版本下标", "iter_args 从 3 撑到 7"),
    ("steady_body", 3, 1, 0, "createKernel",
     "操作数按 stage 差重映射；谓词化收尾内联进稳态体", "各操作数接各拍版本 + 谓词化收尾"),
    ("hopper", 3, 2, 0, "asyncLaunchDots",
     "wgmma 置 isAsync；三规则判省 wait，深度到 2", "两个 wgmma 真正流起来"),
]
EDGES = [  # 行内直线(同泳道相邻列)
    ("entry", "async_copy"), ("async_copy", "coarse_sched"),
    ("dep_sched", "ring_buffer"), ("ring_buffer", "options"),
    ("expand_ctrl", "prologue"), ("prologue", "liverange"),
    ("modvar", "steady_body"), ("steady_body", "hopper"),
]
# 跨泳道折行边:(src_id, dst_id, dst 落点 x 偏移)——第二条正是全章的接缝:
# options(建模的交付物)→ pipelineForLoop(展开总控),建模端 NV 语义在此收口,
# 展开端从这里往后全是后端无关的通用 SCF 变换。
WRAP_EDGES = [
    ("coarse_sched", "dep_sched", 0),
    ("options", "expand_ctrl", 0),
    ("liverange", "modvar", 0),
]
# (路线名, [(列, 站牌文本), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("① 完整链（推荐通读）",
     [(0, "建模，然后展开"), (1, "封装一座桥"), (2, "两个 wgmma 真正流起来")], True),
    ("② SRAM 账速览",
     [(0, "给每个算子发两个号"), (1, "变成共享内存的那一维")], False),
]
LEGEND = [("#22c55e", "入口：num_stages>1 的循环"),
          ("#3b82f6", "主线：建模 → 唯一接口 options → 展开 → Hopper 尾声"),
          ("#f97316", "出口：asyncLaunchDots 收尾")]
TITLE = "第 30 章 · 软件流水线落地剖面（建模半 → 展开半 + 讲解站牌）"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 214, 60
COL_GAP, ROW_GAP = 44, 20
EDGE_MARGIN, STUB_W, STUB_H = 16, 96, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 30
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_H, BADGE_PAD_X, BADGE_FONT = 20, 10, 11
WRAP_GAP = 22  # 折行边:绕出节点右侧的横向余量

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
w_grid = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
# 画布宽度须同时容得下图例一整行(légend 用不同字号横排,文字长度不由节点网格决定)——
# 取网格宽度与图例总宽度的较大者,否则长图例在窄章节网格下会被 viewBox 裁掉。
w_legend = PAD_L + sum(20 + cjk_text_width(lbl, 11.5) + 34 for _, lbl in LEGEND) + PAD_R
w = max(w_grid, w_legend)
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy)——宽度按文本动态算(自然语言站牌,非定长短码)。"""
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
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["hopper"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#166534">{esc("num_stages>1")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#9a3412">{esc("下一章预告")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 行内直线调用边(主线蓝)——多条边汇入同一节点时终点 y 各偏移
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

# 跨泳道折行边(elbow):右侧绕出 → 下降到"下一泳道标签+留白"空白带 → 沿空白带向左 →
# 短距下降落入下一泳道节点顶部(含 dst_dx 偏移)。
for wsrc, wdst, dst_dx in WRAP_EDGES:
    wx1, wy1 = NODE_XY[wsrc]; wx2, wy2 = NODE_XY[wdst]
    p_start = (wx1 + NODE_W, wy1 + NODE_H / 2)
    turn_x = wx1 + NODE_W + WRAP_GAP
    drop_y = wy2 - 8
    p_mid1 = (turn_x, wy1 + NODE_H / 2)
    p_mid2 = (turn_x, drop_y)
    p_mid3 = (wx2 + NODE_W / 2 + dst_dx, drop_y)
    p_end = (wx2 + NODE_W / 2 + dst_dx, wy2)
    L.append(f'<polyline points="{p_start[0]:.1f},{p_start[1]:.1f} {p_mid1[0]:.1f},{p_mid1[1]:.1f} '
             f'{p_mid2[0]:.1f},{p_mid2[1]:.1f} {p_mid3[0]:.1f},{p_mid3[1]:.1f} {p_end[0]:.1f},{p_end[1]:.1f}" '
             f'fill="none" stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 接缝标注:options → pipelineForLoop 这道折行边是全章的接缝论点,单独挂一个短标注
# (右对齐贴着折行边的竖段,避免文字伸出画布右边距——这里空间只够几个字,详细说明
# 交给图例第二条"唯一接口 options"与节点自身短语,不在此处堆长句)。
_seam_src = NODE_XY["options"]; _seam_dst = NODE_XY["expand_ctrl"]
_seam_turn_x = _seam_src[0] + NODE_W + WRAP_GAP
_seam_y = (_seam_src[1] + NODE_H / 2 + _seam_dst[1] - 8) / 2
L.append(f'<text x="{_seam_turn_x - 4:.1f}" y="{_seam_y:.1f}" text-anchor="end" '
         f'font-family="sans-serif" font-size="10" font-style="italic" '
         f'fill="{C_BADGE_TEXT}">{esc("接缝")}</text>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌),字号按文本长度自适应收缩避免溢出
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

# 底部阅读路线
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线（标号=图上讲解站牌；实线蓝=推荐 / 虚线灰=次要）")}</text>')
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
print(f"wrote {out} ({w:.0f}x{h:.0f}, ratio {w / h:.2f})")
