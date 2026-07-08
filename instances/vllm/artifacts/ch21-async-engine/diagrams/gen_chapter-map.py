#!/usr/bin/env python3
"""第 21 章(异步通信与数据并行:惰性 PP 同步 / DP wave 共识 / DP 负载均衡)
——本章地图:源码剖面图。

本章是**自然标题章**(正文标题为"惰性 PP 同步：把 wait 推迟到最后一刻"/"DP wave 共识：
让多个引擎齐步走"/"DP 协调进程与负载均衡"等,无 `## N.M` 编号)——按契约禁用 §N.M 徽标,
站牌改用标题词本身(取正文 `###` 小节标题的字面子串,不复述整句);badge 用自适应宽度
(同 ch08/ch10 的 badge_width() 写法,而非模板固定 46px 胶囊)。

[结构性必要泛化] 正文明确写道"这一章只讲三个**互相独立**的机制"(§这章要做什么)——三条
主线彼此没有调用关系(PP 收发在 gpu_worker.py 的 Worker.execute_model 一拍触发;DP wave
共识在 engine/core.py 的 DPEngineCoreProc.run_busy_loop 忙循环触发;DP 负载均衡横跨
core_client.py 前端与 coordinator.py 协调进程,由外部请求触发)。若像模板那样只画一对
入口/出口接口桩、把三条主线硬接成一条链,就是在画一条源码里不存在的调用边。因此本图把
"一对入口/出口"泛化成"每条独立主线各一对"(三对,颜色/形状/文案与模板的单对完全一致,
只是数量随本章真实结构从 1 变成 3)——配色规则(entry 绿 #22c55e / exit 橙 #f97316 /
主线蓝 #3b82f6)、§徽标→站牌胶囊的形状、图例规则均原样不变。

三条泳道(各自一行 4 个真实符号节点,行内箭头=章内主线调用边,泳道之间无边——因为源码
里确实无边):
  ① 惰性 PP 同步(vllm/v1/worker/gpu_worker.py):execute_model 触发→irecv_tensor_dict
     拿到非阻塞句柄→AsyncIntermediateTensors 包住句柄(__getattribute__ 拦 .tensors 首次
     访问才 wait)→isend_tensor_dict 非阻塞发送(句柄留到下一拍)。
  ② DP wave 共识(vllm/v1/engine/core.py 忙循环):run_busy_loop 每拍→
     _has_global_unfinished_reqs 节流(每 32 步一次)→sync_dp_state 单次 2 元素 SUM
     all-reduce→ignore_start_dp_wave 两阶段暂停收尾。
  ③ DP 协调进程与负载均衡(core_client.py + coordinator.py):make_async_mp_client 三选一
     →get_core_engine_for_request 评分选最空引擎→DPCoordinatorProc 居中聚合 stats/wave→
     add_request_async 盖 wave 并在暂停时抢发 FIRST_REQ。

节点预算 12 = 12(≤12,已到上限,故不再加卫星节点;"全局状态二元组"/"本地预增 100ms"/
三处"在 host 上跑通"验证小节未单独设站,由阅读路线的次要跳读行提及关键词覆盖)。

用法: python3 gen_chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):逐字符判定——
    全角(ord>0x2E80)按 1.0×size,半角(ASCII/拉丁等)按 0.58×size,求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(本章数据) ----------------
LANES = [
    "① 惰性 PP 同步 · vllm/v1/worker/gpu_worker.py",
    "② DP wave 共识 · vllm/v1/engine/core.py 忙循环",
    "③ DP 协调与负载均衡 · core_client.py / coordinator.py",
]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名(可含 "\n" 机械换行), 一行短语(可含 "\n"), 站牌(标题词字面子串))
NODES = [
    # ① 惰性 PP 同步
    ("pp_exec",  0, 0, 0, "execute_model",
     "非首 rank 收/非末 rank 发,\n每拍一次", "流水线的 bubble"),
    ("pp_irecv", 0, 1, 0, "irecv_tensor_dict",
     "先收元数据预分配缓冲,\n再发非阻塞 irecv", "irecv 为什么能不阻塞"),
    ("pp_async", 0, 2, 0, "AsyncIntermediateTensors",
     "__getattribute__ 拦 .tensors,\n首次访问才 wait_for_comm", "会自己 wait 的张量容器"),
    ("pp_isend", 0, 3, 0, "isend_tensor_dict",
     "非阻塞发送,\n句柄留到下一拍才 wait", "把一格流水线串起来"),
    # ② DP wave 共识
    ("dp_loop",  1, 0, 0, "run_busy_loop",
     "step→发布 counts→\n必要时跑 dummy batch", "忙循环"),
    ("dp_sync32", 1, 1, 0, "_has_global_unfinished\n_reqs",
     "step_counter%32≠0\n直接返回 True", "32 步才同步一次"),
    ("dp_ar",    1, 2, 0, "sync_dp_state",
     "单次 2 元素 SUM,\nOR 与 AND 一并解出", "解决两个共识"),
    ("dp_pause", 1, 3, 0, "ignore_start_dp_wave",
     "共识达成后丢弃\n迟到的 START_DP_WAVE", "两阶段暂停"),
    # ③ DP 协调与负载均衡
    ("lb_factory", 2, 0, 0, "make_async_mp_client",
     "按 dp_size /\nexternal_lb 三选一", "谁来当这个客户端"),
    ("lb_score", 2, 1, 0, "get_core_engine\n_for_request",
     "score=waiting*4+running,\n选中即本地预增", "评分选最空的引擎"),
    ("lb_coord", 2, 2, 0, "DPCoordinatorProc",
     "三 socket 聚合 stats,\n广播 (counts,wave,running)", "wave 状态机"),
    ("lb_wake",  2, 3, 0, "add_request_async",
     "盖 current_wave,\n暂停时先发 FIRST_REQ", "暂停时怎么唤醒"),
]
# 站牌→正文标题字面子串对照(供自查/盲审核对,不参与渲染;均为对应 `### ` 标题的原文子串):
#   流水线的 bubble        ⊂ "流水线的 bubble 从哪来"
#   irecv 为什么能不阻塞    = "irecv 为什么能不阻塞"(原题)
#   会自己 wait 的张量容器  ⊂ "一个会自己 wait 的张量容器"(去掉前缀"一个")
#   把一格流水线串起来      = "把一格流水线串起来"(原题)
#   忙循环                  ⊂ "忙循环：每拍都在维持对齐"
#   32 步才同步一次         = "32 步才同步一次"(原题)
#   解决两个共识            ⊂ "一次 all-reduce 解决两个共识"
#   两阶段暂停              ⊂ "两阶段暂停：别被迟到的唤醒拉起"
#   谁来当这个客户端        = "谁来当这个客户端"(原题)
#   评分选最空的引擎        = "评分选最空的引擎"(原题)
#   wave 状态机             ⊂ "协调进程的 wave 状态机"
#   暂停时怎么唤醒          = "暂停时怎么唤醒"(原题)
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝;三条泳道各自成链,泳道之间无边(源码里本无调用关系)
    ("pp_exec", "pp_irecv"), ("pp_irecv", "pp_async"), ("pp_async", "pp_isend"),
    ("dp_loop", "dp_sync32"), ("dp_sync32", "dp_ar"), ("dp_ar", "dp_pause"),
    ("lb_factory", "lb_score"), ("lb_score", "lb_coord"), ("lb_coord", "lb_wake"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序 —— 列号必须复用 NODES 里已出现的列, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("① 通读惰性 PP 同步",
     [(0, "流水线的 bubble"), (1, "irecv 为什么能不阻塞"),
      (2, "会自己 wait 的张量容器"), (3, "把一格流水线串起来")], True),
    ("② 通读 DP wave 共识",
     [(0, "忙循环"), (1, "32 步才同步一次"), (2, "解决两个共识"), (3, "两阶段暂停")], True),
    ("③ 通读负载均衡协调",
     [(0, "谁来当这个客户端"), (1, "评分选最空的引擎"),
      (2, "wave 状态机"), (3, "暂停时怎么唤醒")], True),
    ("跳读:只看路由怎么选/怎么唤醒",
     [(1, "评分选最空的引擎"), (3, "暂停时怎么唤醒")], False),
]
LEGEND = [("#22c55e", "入口:从上层调用进入"), ("#3b82f6", "章内主线调用边"), ("#f97316", "出口:返回上层")]
TITLE = "第 21 章 · 三条独立主线剖面(惰性 PP 收发 / DP wave 共识 / DP 负载均衡协调)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 210, 92  # 高度留 2 行符号 + 2 行短语(部分符号名较长,机械换行见 NODES)
COL_GAP, ROW_GAP = 30, 20
EDGE_MARGIN, STUB_W, STUB_H = 14, 64, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_H, BADGE_PAD_X, BADGE_MIN_W = 20, 14, 36  # 站牌高度固定,宽度按文字动态算

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


def _badge_width(text):
    """站牌宽度按文字动态算(自然标题站牌长短不一,不能像 §N.M 那样固定宽度)。"""
    return max(BADGE_MIN_W, cjk_text_width(text, 11) + BADGE_PAD_X)


def badge_topright(x, y, node_w, text):
    """站牌胶囊贴节点右上角。right_edge = x+node_w+8 恒定(与 width 无关),
    哪怕站牌变宽也不会撞到右侧下一列节点。"""
    width = _badge_width(text)
    cx = x + node_w - width / 2 + 8
    return _badge_rect_text(cx, y, width, text)


def badge_centered(cx, cy, text):
    """路线站牌:居中挂在 (cx,cy),宽度同样动态算。"""
    width = _badge_width(text)
    return _badge_rect_text(cx, cy, width, text)


def _badge_rect_text(cx, cy, width, text):
    bx, by = cx - width / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{width:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.1f} {h:.1f}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN))
) + '</defs>')
L.append(f'<rect width="{w:.1f}" height="{h:.1f}" fill="white"/>')

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
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w:.1f}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w:.1f}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w:.1f}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 三对入口/出口接口桩(每条独立主线各一对——三条主线源码里彼此无调用关系,
# 不能像单主线模板那样只画一对、硬接成一条链;颜色/形状/文案与模板单对完全一致)。
ENTRY_EXIT = [("pp_exec", "pp_isend"), ("dp_loop", "dp_pause"), ("lb_factory", "lb_wake")]
for entry_id, exit_id in ENTRY_EXIT:
    ex, ey = NODE_XY[entry_id]; ey += NODE_H / 2
    xx, xy = NODE_XY[exit_id]; xy += NODE_H / 2
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

# 调用边(主线蓝,先画边再画节点盖住端点毛刺)——三条泳道各自成链,同泳道内左→右,泳道间无边
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    p2 = (x2, y2 + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名(1~2 行) + 一行短语(1~2 行) + 右上角站牌)
SYMBOL_1LINE_Y, SYMBOL_2LINE_Y1, SYMBOL_2LINE_Y2 = 34, 26, 42
PHRASE_1LINE_Y, PHRASE_2LINE_Y1, PHRASE_2LINE_Y2 = 74, 68, 84
for nid, lane, col, row, symbol, phrase, tag in NODES:
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
    L += badge_topright(x, y, NODE_W, tag)

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌;实线蓝=按泳道通读 / 虚线灰=跨站跳读)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first = COLX[stops[0][0]] + NODE_W / 2
    x_last = COLX[stops[-1][0]] + NODE_W / 2
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for col, tag in stops:
        L += badge_centered(COLX[col] + NODE_W / 2, ry, tag)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}: {w:.0f}x{h:.0f}")

# [SELF-CHECK](渲染→Read PNG 亲眼看后如实记录,见 figure-manifest.json 对应条目)
