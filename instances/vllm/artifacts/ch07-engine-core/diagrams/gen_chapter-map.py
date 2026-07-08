#!/usr/bin/env python3
"""第 7 章「本章地图」——IPC 边界剖面(client 三层 -> ZMQ socket -> EngineCoreProc)。

改写自 .claude/skills/svg-diagram/references/example-chapter-map.py(直接沿用其
badge()/cjk_text_width()/入口出口接口桩/主线蓝边)，并参考同书 ch36 的两处扩展手法:
多个 § 徽标并排挂在同一节点(secs 为列表)、长标识符用 split_symbol() 拆两行。

■ 不可变(全书统一视觉语言，未改动): §徽标胶囊 badge()；入口=绿#22c55e/出口=橙#f97316
  接口桩；章内主线调用边=蓝#3b82f6；底部路线条(高亮=实线蓝/次要=虚线灰)；
  >2 种语义色画图例；cjk_text_width() 做宽度估算。

■ 本章新增(仅本章需要):
  - 三条泳道对应本章真实的三处代码位置：前端 Client 进程 / 序列化与张量旁路协议
    (客户端序列化 + torch.mp.Queue 共享内存旁路，横跨两进程但代码上是独立模块)/
    EngineCoreProc 进程。
  - 新增第四种边样式 "queue"(青绿虚线 #14b8a6，marker 单独一个)，专门表示
    "经内部 queue.Queue 解耦、不是直接函数调用"——这正是 §7.7 的核心论点
    (input_queue/output_queue 把 IO 线程和 busy loop 解耦)，用不同边样式如实
    区分"真调用"(蓝实线: send->recv 是 ROUTER->DEALER 真实过 socket;
    send_out->exit 是 PUSH->PULL 真实过 socket)和"队列解耦"(recv->busy,
    busy->send_out 中间隔着 input_queue/output_queue，busy loop 从不直接摸 socket)。
  - 一个节点可以挂多个 § 徽标(secs 是列表)：例如 process_input_sockets() 这同一个
    函数，§7.4 讲它开头发 ready 帧握手、§7.6(的 7.6.3 小节)讲它主循环怎么按字节
    标签选 decoder——徽标只能到 "## N.M" 整节粒度(lint_chapter_map 的 _HEADING_RE
    只认两级 "## 7.6"，认不出三级 "### 7.6.3")，所以两个徽标都挂在同一个节点上，
    如实反映"这一个真实符号，在两节里各被讲了一部分"。process_output_sockets()/
    process_outputs_socket() 同理挂 §7.9(故障哨兵先出现的位置)+§7.10(完整代码/
    buffer 复用出现的位置)。
  - TensorIpcSender 和 TensorIpcReceiver 合成一个节点(用 "/" 连接的双行标题)：
    两者是同一条共享内存旁路的发送端/接收端，本节点预算(9 个)下拆成两个独立节点
    不会带来更多可读信息，反而要多一列；发送端/接收端各自的行为差异在正文
    §7.12 讲得很细，图上只需指到这一节。
  - §7.2(InprocClient 对照组)和 §7.13/§7.14(端到端验证/小结)不设专属节点——
    前者不在真实 IPC 边界的主干路径上(是"没有 IPC 的对照"，不是这条剖面线的
    一部分)，后两者是验证与总结，没有新符号可画。§7.2 改在底部路线区用一条
    单站路线点出，指引读者"想看无 IPC 对照就翻那一节"，不占节点预算也不假称
    它在调用链上。

[FIX-ROUND-2](渲染+Read PNG 复核后发现并修正，替换第一轮列布局):
  - 第一轮 enc_tensor 和 recv 同挂第 3 列(col2)：send->recv 这条跨进程边界主线
    (516,147)->(542,483) 是一条几乎垂直的长对角线，量出它经过 enc_tensor 底边
    (y=396)时横向只剩约 6.7px 空隙——PNG 里肉眼看确实贴得很近，几乎擦着
    enc_tensor 左下角。改法：recv 单独占一列(col2)，enc_tensor/tensoripc 挪到
    col3/col4——send->recv 这条线所在列(col2)不再有其他节点，send->enc_tensor
    改成跨 2 列(col1->col3)，量出它在 recv 所在列的 y 值(<=361)全程低于 recv
    的 y 下界(448)，两者错开、不会互相穿框。
  - 第一轮 rpc_call->busy 是对角线 (516,239)->(753,483)，精确解得它在
    x∈[719,727]时 y∈[448,456]——恰好扎进 recv 节点框(x[542,727],y[448,518])
    的右上角一小块。这条边本身也是简化：call_utility() 真实内部是调
    _send_input()(和 send 同一条代码路径)才真正发到网络，"直接连到 busy"
    本就是压缩了中间的 send/recv 两跳。与其为了保留这条边而绕线(徒增复杂度
    和新的出错面)，不如去掉这条画不干净的边，把"call_utility 最终由 busy 的
    _handle_client_request 处理"这件事交给底部"跨进程 RPC"阅读路线的
    §7.1->§7.8->§7.8->§7.9 站牌走一遍——阅读路线本来就是为这种"隔着两跳"
    的关系设计的，不必每条关系都在主图里画一条线。

[FIX-ROUND-3](跑 lint_chapter_map.py 报错后修的，纯字符串层面，非视觉问题):
  - 节点符号名带空括号(如 "_send_input()")在 lint 的杜撰符号检查下判杜撰——正文
    里 _send_input 后面跟的是真参数列表，从没有 "_send_input()" 这个空括号子串
    连着出现过。改法：所有节点符号名去掉 "()" 后缀，对齐 dossier must_keep 里的
    裸写法(如 "_send_input")；顺带发现这样一来大多数符号在 NODE_W=185 下已能
    单行放下，不再需要 split_symbol() 拆行。
  - 泳道名"...+ torch.mp.Queue)"和路线名"...(call_utility)"/"IPC(InprocClient)"
    这几处，标识符和括号紧贴导致 lint 把"标识符+括号"当一个子串去核对，而这个
    带括号的组合从未在正文/dossier 原文出现过(只有不带括号的裸标识符出现过)。
    改法：在标识符和右括号之间垫一个词断开贴靠(如 "torch.mp.Queue 旁路)")，
    或干脆把路线名里的标识符去掉(路线名不必重复节点上已经写着的符号)。
  - 上一步顺手把"跨进程 RPC(以 call_utility 为例)"精简成"跨进程 RPC"时，
    另外量出它在原长度下(16px 起点、12px 字号，宽 211px)会盖住第 0 列 §7.1
    徽标(该徽标横跨 189.5-235.5px)——PNG 里确实能看到文字尾部压住徽标。
    精简后的短路线名(不到 90px)已退到安全宽度以内，"对照:无 IPC 的
    InprocClient"同理精简为"对照:无 IPC"。

用法: python3 gen_chapter-map.py -> 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0xsize，半角按 0.58xsize。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def split_symbol(text, max_w, size):
    """真实符号名装不下节点宽度时拆两行：优先在 "/" 处拆(两个独立符号并列时)，
    否则在离中点最近的下划线处拆——两段都还是原符号的连续子串，不加省略号，
    lint_chapter_map 的子串核对对每段仍能命中。"""
    if cjk_text_width(text, size) <= max_w:
        return [text]
    if " / " in text:
        return [p.strip() for p in text.split(" / ", 1)]
    positions = [i for i, c in enumerate(text) if c == '_' and i != 0]
    if not positions:
        return [text]
    mid = len(text) // 2
    split_at = min(positions, key=lambda p: abs(p - mid))
    return [text[:split_at], text[split_at:]]


# ---------------- DATA(本章数据) ----------------
LANES = [
    "前端 Client 进程",
    "序列化 / 张量旁路协议(msgpack 帧 + torch.mp.Queue 旁路)",
    "EngineCoreProc 进程",
]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, [§编号,...])
# 列号排布刻意让 recv 单独占第 2 列、enc_tensor/tensoripc 挪到第 3/4 列——
# 这样 send->recv 这条"请求跨进程"主线不会和同列的 enc_tensor 节点框贴太近
# (见 [FIX-ROUND-2])。真实符号名一律不带 "()" 后缀(对齐 dossier must_keep
# 里的裸符号写法，如 "_send_input" 不是 "_send_input()")——lint_chapter_map
# 的杜撰符号检查是逐字子串匹配，符号名一旦带上空括号反而在正文/dossier里找
# 不到这个精确子串(正文里 _send_input 后面跟的是真参数列表，不是空括号)，
# 见 [FIX-ROUND-3]。
NODES = [
    ("entry",      0, 0, 0, "EngineCoreClient",
     "两布尔选三层传输", ["§7.1", "§7.3"]),
    ("send",       0, 1, 0, "_send_input",
     "打包标签+多帧,copy=False发送", ["§7.6"]),
    ("rpc_call",   0, 1, 1, "call_utility",
     "call_id+Future,发起RPC", ["§7.8"]),
    ("recv",       2, 2, 0, "process_input_sockets",
     "先发ready,收帧按标签分派", ["§7.4", "§7.5", "§7.6"]),
    ("enc_tensor", 1, 3, 0, "_encode_tensor",
     "内联/aux_buffers/OOB三分流", ["§7.11"]),
    ("busy",       2, 3, 0, "run_busy_loop",
     "按标签分派,step_fn,不碰socket", ["§7.7", "§7.8"]),
    ("tensoripc",  1, 4, 0, "TensorIpcSender / TensorIpcReceiver",
     "共享内存旁路,drain-and-buffer", ["§7.12"]),
    ("send_out",   2, 4, 0, "process_output_sockets",
     "复用buffer,零拷贝,哨兵", ["§7.9", "§7.10"]),
    ("exit",       0, 5, 0, "process_outputs_socket",
     "验死讯,解RPC或投队列", ["§7.9", "§7.10"]),
]
# (src_id, dst_id, style) —— style 省略即 "main"(蓝实线,真实调用/真实过socket);
# "queue" = 青绿虚线,经内部 queue.Queue 解耦,不是直接函数调用。
# 注意:call_utility() 内部其实也是调 _send_input() 发送(和 send 同一份代码路径)，
# 但 call_utility 和 send 同列(见上)画不出这条边(见 [FIX-ROUND-2] 的取舍)——
# 该关系改在底部"跨进程 RPC"阅读路线里用 §7.8 站牌交代，不在主图勉强连线。
EDGES = [
    ("entry", "send"),
    ("entry", "rpc_call"),
    ("send", "recv"),
    ("send", "enc_tensor"),
    ("enc_tensor", "tensoripc"),
    ("recv", "busy", "queue"),
    ("busy", "send_out", "queue"),
    ("send_out", "exit"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("正常请求(ADD/ABORT)主线", [
        (0, "§7.1"), (1, "§7.6"), (2, "§7.6"), (3, "§7.7"), (4, "§7.10"), (5, "§7.9"),
    ], True),
    ("跨进程 RPC", [
        (0, "§7.1"), (1, "§7.8"), (3, "§7.8"), (5, "§7.9"),
    ], False),
    ("多模态张量共享内存旁路", [
        (3, "§7.11"), (4, "§7.12"),
    ], False),
    ("对照:无 IPC", [
        (0, "§7.2"),
    ], False),
]
LEGEND = [
    ("#22c55e", "入口:前端调用进入"),
    ("#3b82f6", "章内主线调用 / 真实过socket"),
    ("#14b8a6", "queue.Queue内部解耦(非直接调用)"),
    ("#f97316", "出口:返回上层"),
]
TITLE = "第 7 章 · IPC 边界剖面(client 三层 -> ZMQ socket -> EngineCoreProc,源码走线 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_QUEUE = "#8b5cf6"  # 紫色,和入口绿/主线蓝/出口橙都拉开,避免和绿色混淆
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 185, 70
TITLE_SIZE, TITLE_LINE_H, SUB_SIZE = 12, 13, 10
COL_GAP, ROW_GAP = 26, 22
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
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
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN), ("Queue", C_QUEUE))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例;本章 4 色)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 11) + 26

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="12.5" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩
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

# 调用边:main=蓝实线(真实调用/真实过socket)；queue=青绿虚线(经内部queue.Queue解耦)。
# 多条边汇入同一节点时,同一样式分组内的边各自 y 偏移(不同样式各自独立计数),
# 避免看不出"汇合"或不同样式的箭头头部彼此重叠。
def edge_style(e):
    return e[2] if len(e) > 2 else "main"


_dst_total = {}
for src, dst, *_ in EDGES:
    key = (dst, edge_style((src, dst, *_)))
    _dst_total[key] = _dst_total.get(key, 0) + 1
_dst_seen = {}
for e in EDGES:
    src, dst = e[0], e[1]
    style = edge_style(e)
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    key = (dst, style)
    n = _dst_total[key]
    i = _dst_seen.get(key, 0)
    _dst_seen[key] = i + 1
    y_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    color, marker, dash = (C_MAIN, "mMain", "") if style == "main" else (C_QUEUE, "mQueue", ' stroke-dasharray="7,5"')
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{color}" stroke-width="2"{dash} marker-end="url(#{marker})"/>')

# 节点(圆角框 + 真实符号名[必要时拆两行] + 一行短语 + 右上角 § 徽标[可多个并排])
for nid, lane, col, row, symbol, phrase, secs in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    title_lines = split_symbol(symbol, NODE_W - 22, TITLE_SIZE)
    if len(title_lines) == 1:
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.40:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{TITLE_SIZE}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(title_lines[0])}</text>')
    else:
        base_y = y + NODE_H * 0.32
        for li, line in enumerate(title_lines):
            L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{base_y + li * TITLE_LINE_H:.1f}" '
                      f'text-anchor="middle" font-family="sans-serif" font-size="{TITLE_SIZE}" '
                      f'font-weight="bold" fill="{C_NODE_TITLE}">{esc(line)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.86:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{SUB_SIZE}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    # 右上角 § 徽标:多个并排贴在上边框。secs 数据里从小到大写(如 §7.4,§7.5,§7.6);
    # 摆放从右边界开始往左退,所以反着摆(先摆最大的那个在最右边),这样最终左到右
    # 视觉顺序就是从小到大,不是反的。
    bcx = x + NODE_W - BADGE_W / 2 + 8
    for sec in reversed(secs):
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
