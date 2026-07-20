#!/usr/bin/env python3
"""第 36 章「本章地图」——worker 侧 KV 生命周期 + 三类传输后端的同一套契约。

改写自 .claude/skills/svg-diagram/references/example-chapter-map.py，布局思路
借鉴 ch39 的真实章节实践(gen_chapter-map.py：split_symbol/多 § 徽标/自定义边)。

本章的真实结构是一个「收敛-发散」的沙漏形状,不是简单的线性流水线：
  - 上半(lane0)：execute_model 的 KV 生命周期在 worker 侧一路线性推进——
    分流闸 → 异步发起 load → forward 内逐层 hook → 阻塞收齐 save。四步严格
    按时间顺序发生，画成同一泳道内从左到右的「main」蓝色实线调用边。
  - 中间(lane1)：上面四步里调的 start_load_kv / wait_for_layer_load /
    save_kv_layer / wait_for_save 四个抽象方法，加上默认实现的 get_finished，
    合计五个方法，全部由 KVConnectorBase_V1 这一份 worker 契约定义——这是
    全章「同一套契约」的收敛点。
  - 下半(lane2)：三类后端(P2P NCCL / NIXL RDMA / Offloading)各自把这份
    契约填实，但填法南辕北辙——这是发散点。

  lane0→lane1→lane2 之间的边不是「先调用 A 再调用 B」的时间顺序（三类后端
  不会在同一次请求里都被调用——每次部署只选一种传输），而是「这份抽象契约
  ↔ 具体实现」的结构关系。为避免和上面「main」蓝色实线（真实时间顺序调用）
  混淆，这类边单独用靛蓝虚线 + 独立图例（沿用 ch39 引入「非顺序调用边」
  单独配色的先例，这里颜色换成靛蓝而非灰色，因为契约↔实现比"章内换题"
  更接近"一份定义、多份实现"的静态关系，用胶囊徽标同色系靛蓝呼应）。
  cross-lane 边一律走「src 底边中点 → dst 顶边中点」的竖直/斜直线：因为
  lane0/lane1/lane2 之间是空白的泳道间隙(无节点)，直线不会穿过任何别的
  节点框，不需要 ch39 那种绕障折线。

■ 不可变(全书统一视觉语言，抄自模板，未改动):
  1. §徽标胶囊 badge()；2. 入口=绿#22c55e/出口=橙#f97316 接口桩；
  3. 章内主线调用边=蓝#3b82f6；4. 底部路线条(高亮=实线蓝/次要=虚线灰)；
  5. >2 种语义色画图例；6. cjk_text_width() 做宽度估算。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录):
    claim_readable_10s=True  numbers_match_spec=True(本图无数字,数字类
    自查项按"图上无杜撰数字"通过)  no_overlap=True  arrows_attached=True
    cjk_rendered=True  reading_order_clear=True

用法: python3 gen_chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def split_symbol(text, max_w, size):
    """真实符号名在给定字号下装不下节点宽度时，在离中点最近的下划线处拆两行。"""
    if cjk_text_width(text, size) <= max_w:
        return [text]
    positions = [i for i, c in enumerate(text) if c == '_' and i != 0]
    if not positions:
        return [text]
    mid = len(text) // 2
    split_at = min(positions, key=lambda p: abs(p - mid))
    return [text[:split_at], text[split_at:]]


# ---------------- DATA(本章数据) ----------------
LANES = ["worker 生命周期(夹住 forward)", "同一套 worker 契约", "三类后端各自的填法"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, [§编号,...])
NODES = [
    ("entry",      0, 0, 0, "maybe_get_kv_connector_output",
     "execute_model 的分流闸,无传输组则 nullcontext", ["§36.1"]),
    ("enter",      0, 1, 0, "start_load_kv",
     "bind 之后异步发起本步全部 load,forward 前置", ["§36.2"]),
    ("layer_hook", 0, 2, 0, "maybe_transfer_kv_layer",
     "进层等 wait_for_layer_load,出层发 save_kv_layer", ["§36.3"]),
    ("exit",       0, 3, 0, "wait_for_save",
     "阻塞收齐所有 save,防 paged buffer 被覆盖", ["§36.4"]),
    ("contract",   1, 1, 0, "KVConnectorBase_V1",
     "五个 worker 契约方法,三类后端各自填实", ["§36.6"]),
    ("p2p",        2, 0, 0, "P2pNcclConnector",
     "NCCL 点对点直发,is_producer 分角色", ["§36.7"]),
    ("nixl",       2, 2, 0, "NixlConnectorWorker",
     "RDMA 单边 READ,完成信号不对称", ["§36.8"]),
    ("offload",    2, 3, 0, "OffloadingConnectorWorker",
     "CPU/磁盘卸载,store 推迟到下一步", ["§36.9"]),
]
# main 边(同泳道,严格从左列到右列,时间顺序调用,蓝实线):(src_id, dst_id)
MAIN_EDGES = [
    ("entry", "enter"), ("enter", "layer_hook"), ("layer_hook", "exit"),
]
# impl 边(跨泳道,抽象契约↔具体实现,非时间顺序,靛蓝虚线):
# (src_id, dst_id, 落点在 dst 顶边/起点在 src 底边的横向比例 0~1)
IMPL_EDGES = [
    ("enter", "contract", 0.30, None),
    ("layer_hook", "contract", 0.50, None),
    ("exit", "contract", 0.70, None),
    ("contract", "p2p", None, 0.20),
    ("contract", "nixl", None, 0.55),
    ("contract", "offload", None, 0.80),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("worker 生命周期主线(推荐)",
     [(0, "§36.1"), (1, "§36.2"), (2, "§36.3"), (3, "§36.4")], True),
    ("P2P NCCL 填法", [(1, "§36.6"), (0, "§36.7")], False),
    ("NIXL RDMA 填法", [(1, "§36.6"), (2, "§36.8")], False),
    ("Offloading 分级缓存填法", [(1, "§36.6"), (3, "§36.9")], False),
]
LEGEND = [
    ("#22c55e", "入口:从上层调用进入"),
    ("#3b82f6", "worker 生命周期调用序列(时间顺序)"),
    ("#f97316", "出口:返回上层"),
    ("#6366f1", "抽象契约 ↔ 具体实现(非时间顺序)"),
]
TITLE = "第 36 章 · worker 侧 KV 生命周期与三类传输后端(源码走线 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_IMPL = "#6366f1"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 235, 74
TITLE_SIZE, TITLE_LINE_H, SUB_SIZE = 12, 13, 10
COL_GAP, ROW_GAP = 46, 22
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
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN), ("Impl", C_IMPL))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例;本章 4 色)
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

# 入口/出口接口桩(入口挂在 §36.1 的分流闸,出口挂在 §36.4 的 wait_for_save 围栏)
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

# main 边(同泳道,蓝实线,严格左列→右列的时间顺序调用)
_dst_total = {}
for src, dst in MAIN_EDGES:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst in MAIN_EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    y_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# impl 边(跨泳道,靛蓝虚线,抽象契约↔具体实现,非时间顺序):
# src 底边(按 launch_frac 定横向落点)→ dst 顶边(按 land_frac 定横向落点)的直线。
# lane 之间是空白泳道间隙(无节点),直线不会穿过任何别的节点框。
for src, dst, land_frac, launch_frac in IMPL_EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    lf = launch_frac if launch_frac is not None else 0.5
    p1 = (x1 + NODE_W * lf, y1 + NODE_H)
    df = land_frac if land_frac is not None else 0.5
    p2 = (x2 + NODE_W * df, y2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_IMPL}" stroke-width="1.8" stroke-dasharray="6,4" marker-end="url(#mImpl)"/>')

# 节点(圆角框 + 真实符号名[必要时拆两行] + 一行短语 + 右上角 § 徽标)
for nid, lane, col, row, symbol, phrase, secs in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    title_lines = split_symbol(symbol, NODE_W - 26, TITLE_SIZE)
    if len(title_lines) == 1:
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.36:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{TITLE_SIZE}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(title_lines[0])}</text>')
    else:
        base_y = y + NODE_H * 0.30
        for li, line in enumerate(title_lines):
            L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{base_y + li * TITLE_LINE_H:.1f}" '
                      f'text-anchor="middle" font-family="sans-serif" font-size="{TITLE_SIZE}" '
                      f'font-weight="bold" fill="{C_NODE_TITLE}">{esc(line)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.86:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{SUB_SIZE}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
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
