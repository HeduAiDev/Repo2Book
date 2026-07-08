#!/usr/bin/env python3
"""第 12 章「本章地图」——step_with_batch_queue() 填管道剖面。

改写自 .claude/skills/svg-diagram/references/example-chapter-map.py（几何/配色
不可变部分照抄），并参考已验收的 instances/vllm/artifacts/ch06-input-processor/
diagrams/chapter-map.py 的两处做法：
  - split_symbol()：真实符号名在节点宽度下装不下时,在离中点最近的下划线处
    拆两行(不加省略号,两段仍是原符号的连续子串,lint 子串核对仍能命中)。
  - 同列纵向堆叠(row)+ 同列竖直连接线公式(x1==x2 时画上一格底边中点→下一格
    顶边中点,而不是套用横向对角线公式)：把 9 个节点压进 6 列以内。

本章只有一个方法 step_with_batch_queue()，但它内部是"两态交替"的控制流——
上半段"填管道"(§12.3) 多数拍在 appendleft 后就 return(None, True) 提前返回，
不落到下半段；只有队满/尾批已完成/无新请求时才落到下半段"取结果"(§12.4)，
其中 deferred sampling(§12.5) 是取结果之后、真正 return 之前的一条支线。这
决定了本图的核心设计：entry→branch→enqueue 是所有拍都走的主干；enqueue→pop
是"落到下半段"的纵向掉落(同列，enqueue 在上半段泳道、pop 在下半段泳道)；
update→deferred→exit 与 update→exit 是两条汇入 exit 的路径(deferred 存在
与否)，复用 example 模板"两路径汇合"的几何写法(y 偏移防止终点重合看不出
汇合)。

■ 不可变(全书统一视觉语言,未改动):
  1. §徽标胶囊 badge()；2. 入口=绿#22c55e/出口=橙#f97316 接口桩；
  3. 章内主线调用边=蓝#3b82f6；4. 底部路线条(高亮=实线蓝/次要=虚线灰)；
  5. >2 种语义色画图例；6. cjk_text_width() 做宽度估算。

■ 本章数据设计要点:
  - 启动期绑定(§12.2)独占一条泳道、只用 1 列(col0)、纵向堆 2 行
    (max_concurrent_batches → step_fn)：这是"一次性"发生在 __init__ 尾段的
    静态绑定，不是每拍都走的运行时路径，所以不占运行时主干的列，只用一条
    从 stepfn 斜向落到 entry 的边表达"绑定完成后,后续每拍从这里进入"。
  - entry 节点的 phrase 把 schedule()+execute_model() 两个真实调用揉进一句
    短语而不单开节点——它们是上半段最前面严格顺序执行的两行,拆开成独立
    节点会把本就紧张的列预算(≤6 列才压得进画布宽 1500 的硬限)花在"发批"
    这个次要细节上,而不是花在真正的题眼(填管道判定)上。
  - enqueue(appendleft,§12.3)和 pop(batch_queue.pop(),§12.4)刻意同列
    (col3)、跨泳道纵向相邻：图上一眼能看出"入队"和"取结果"是同一个队列
    两端的操作,也呼应正文"appendleft 左进、pop 右出＝FIFO"的表述。
  - update(update_from_output,§12.4)之后分两路都汇入 exit：一路直接
    return(无 deferred 待办,多数情况)，一路先经 deferred_scheduler_output
    (§12.5,取草稿 token→补掩码→补采样→重新 appendleft)再 return——两路径
    在 exit 汇合,y 偏移让汇合可见(同 example 模板 fast_kernel/full_kernel
    →exit 的画法)。
  - 底部三条阅读路线对应本章三种拍：A 填管道快返回(多数拍,本章题眼,高亮)、
    B 取结果(队满/尾批完成/无新请求,下半段)、C deferred 支线(结构化输出+
    投机解码叠加时的小众路径)。A 路线止于 enqueue 列(不延伸到 exit)，是
    因为这条路径本来就在 appendleft 之后直接 return，不落到下半段——如实
    反映"只填不取"这句本章核心信条，不为了凑齐到 exit 的视觉对称而虚构一
    条它并不真的走的边。

用法: python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算:全角(ord>0x2E80)按 1.0×size,半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def split_symbol(text, max_w, size):
    """真实符号名在给定字号下装不下节点宽度时,在离中点最近的下划线处拆两行。
    两段各自仍是原符号的连续子串(不加省略号),lint_chapter_map 的子串核对
    对每段仍能命中——不会被判成杜撰符号。找不到下划线就原样返回单行。"""
    if cjk_text_width(text, size) <= max_w:
        return [text]
    positions = [i for i, c in enumerate(text) if c == '_' and i != 0]
    if not positions:
        return [text]
    mid = len(text) // 2
    split_at = min(positions, key=lambda p: abs(p - mid))
    return [text[:split_at], text[split_at:]]


_SOFT_BREAK = set('，；：、 ,;→')


def wrap_text(text, max_w, size):
    """一行短语按宽度贪心换行:逐字符累加,超宽时回溯当前行最近的软断点
    (中英文标点/箭头/空格)处折行,避免把节点框内长短语硬生生粘到相邻节点
    上(no_overlap)。找不到软断点才硬断(理论上仅极端场景触发)。"""
    lines, cur = [], ''
    for ch in text:
        trial = cur + ch
        if cjk_text_width(trial, size) <= max_w or not cur:
            cur = trial
            continue
        brk = -1
        for i in range(len(cur) - 1, -1, -1):
            if cur[i] in _SOFT_BREAK:
                brk = i
                break
        if 0 <= brk < len(cur) - 1:
            lines.append(cur[:brk + 1])
            cur = cur[brk + 1:] + ch
        else:
            lines.append(cur)
            cur = ch
    if cur:
        lines.append(cur)
    return lines


# ---------------- DATA(本章数据) ----------------
LANES = ["启动期绑定(一次性,§12.2)", "上半段 · 填管道(§12.3)", "下半段 · 取结果 / deferred (§12.4-12.5)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, [§编号,...])
NODES = [
    ("mcb",      0, 0, 0, "max_concurrent_batches",
     "PP→pp_size；单卡 async_scheduling→2", ["§12.2"]),
    ("stepfn",   0, 0, 1, "step_fn",
     "None→绑 step；否则绑 step_with_batch_queue", ["§12.2"]),
    ("entry",    1, 1, 0, "step_with_batch_queue()",
     "schedule()+execute_model()：non_block 不阻塞发批", ["§12.3"]),
    ("branch",   1, 2, 0, "pending_structured_output_tokens",
     "token 不够→存为 deferred；够→立即采样", ["§12.3"]),
    ("enqueue",  1, 3, 0, "appendleft",
     "三元组入队；队未满且队尾未 done()→return (None, True)", ["§12.3"]),
    ("pop",      2, 3, 0, "batch_queue.pop()",
     "队尾弹出，future.result() 阻塞取值", ["§12.4"]),
    ("update",   2, 4, 0, "update_from_output",
     "对回 SchedulerOutput，产出 engine_core_outputs", ["§12.4"]),
    ("deferred", 2, 4, 1, "deferred_scheduler_output",
     "取 draft token→补掩码→重新入队", ["§12.5"]),
    ("exit",     2, 5, 0, "engine_core_outputs",
     "return engine_core_outputs, model_executed", ["§12.4"]),
]
# (src_id, dst_id) —— 调用边,统一主线蓝
EDGES = [
    ("mcb", "stepfn"),
    ("stepfn", "entry"),
    ("entry", "branch"),
    ("branch", "enqueue"),
    ("enqueue", "pop"),
    ("pop", "update"),
    ("update", "deferred"),
    ("update", "exit"),
    ("deferred", "exit"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("填管道优先返回(题眼)",       [(0, "§12.2"), (1, "§12.3"), (2, "§12.3"), (3, "§12.3")], True),
    ("队满/尾批完成/无新请求→取结果",       [(1, "§12.3"), (3, "§12.4"), (4, "§12.4"), (5, "§12.4")], False),
    ("deferred 支线(结构化输出+投机解码)",  [(2, "§12.3"), (4, "§12.5"), (5, "§12.4")], False),
]
LEGEND = [
    ("#22c55e", "入口:忙循环每拍调用 step_with_batch_queue()"),
    ("#3b82f6", "章内主线调用边"),
    ("#f97316", "出口:return engine_core_outputs, model_executed"),
]
TITLE = "第 12 章 · step_with_batch_queue() 填管道剖面(源码走线 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
# NODE_H 按"最坏情形"(2 行符号名 + 2 行短语,本章多个节点都会触发)反推,保证
# 逐节点文字互不重叠;行数更少的节点顶部锚点固定、底部多留白,不居中但绝不压字。
NODE_W = 195
TITLE_SIZE, TITLE_LINE_H, SUB_SIZE, PHRASE_LINE_H = 12, 13, 10, 12
TITLE_TOP, PHRASE_GAP, NODE_BOTTOM_PAD = 22, 8, 10
MAX_TITLE_LINES, MAX_PHRASE_LINES = 2, 2
NODE_H = (TITLE_TOP + (MAX_TITLE_LINES - 1) * TITLE_LINE_H + PHRASE_GAP + PHRASE_LINE_H
          + (MAX_PHRASE_LINES - 1) * PHRASE_LINE_H + NODE_BOTTOM_PAD)
COL_GAP, ROW_GAP = 24, 22
EDGE_MARGIN, STUB_W, STUB_H = 16, 60, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 28  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 14
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
# 图例(3 种语义色画图例)
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

# 调用边(主线蓝)。多条边汇入同一节点时,终点 y 各偏移,否则重合的终点在视觉上
# 看不出"汇合"(本章 exit 有两条入边:update 直接来、deferred 转一手再来)。
# 同列纵向堆叠的节点之间的边是"从上一格掉到下一格",若仍套用"src 右边缘→
# dst 左边缘"的公式,会画出一条横贯节点宽度的对角线、倒穿几何上位于中间的
# 节点框。同列(x1==x2)时改画竖直连接线:上一格底边中点→下一格顶边中点。
_dst_total = {}
for _, dst in EDGES:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    if x1 == x2:
        cx = x1 + NODE_W / 2
        if y2 >= y1:
            p1, p2 = (cx, y1 + NODE_H), (cx, y2)
        else:
            p1, p2 = (cx, y1), (cx, y2 + NODE_H)
    else:
        y_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名[必要时拆两行] + 一行短语[必要时按软断点折两行] +
# 右上角 § 徽标)。符号名与短语各自独立按行显式垒高度(title_top 起,逐行
# TITLE_LINE_H/PHRASE_LINE_H 累加),不用固定比例锚点——避免"两行符号名+两行
# 短语"这种最坏情形在固定比例下互相压字或溢出节点框(no_overlap)。
for nid, lane, col, row, symbol, phrase, secs in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    title_lines = split_symbol(symbol, NODE_W - 26, TITLE_SIZE)
    title_base_y = y + TITLE_TOP
    for li, line in enumerate(title_lines):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{title_base_y + li * TITLE_LINE_H:.1f}" '
                  f'text-anchor="middle" font-family="sans-serif" font-size="{TITLE_SIZE}" '
                  f'font-weight="bold" fill="{C_NODE_TITLE}">{esc(line)}</text>')
    phrase_lines = wrap_text(phrase, NODE_W - 16, SUB_SIZE)
    phrase_base_y = title_base_y + (len(title_lines) - 1) * TITLE_LINE_H + PHRASE_GAP + PHRASE_LINE_H
    for pi, line in enumerate(phrase_lines):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{phrase_base_y + pi * PHRASE_LINE_H:.1f}" '
                  f'text-anchor="middle" font-family="sans-serif" font-size="{SUB_SIZE}" '
                  f'fill="{C_NODE_SUB}">{esc(line)}</text>')
    bcx = x + NODE_W - BADGE_W / 2 + 8
    for sec in secs:
        L += badge(bcx, y, sec)
        bcx -= (BADGE_W + 6)

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
print(f"wrote {out} ({w:.0f}x{h:.0f})")
