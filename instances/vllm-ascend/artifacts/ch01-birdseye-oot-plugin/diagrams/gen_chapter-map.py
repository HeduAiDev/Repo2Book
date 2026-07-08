#!/usr/bin/env python3
"""第 1 章「本章地图」——OOT 插件三支柱源码剖面图。

本章是全书鸟瞰章，主线不是单一 forward() 分流，而是「安装声明 → vLLM 发现/选定
→ NPUPlatform 接管分发 → 两段式打补丁」一条纵贯全链的路径，随后交棒给后续 35 章。
两段太长放不进一行(会超画布宽预算)，按契约「折成多行泳道」处理：LANES 在本章
语义化为「第一段」「第二段」两个先后接续的阅读行(而非并列架构层)，段尾用一条
自定义折线(不走模板默认的直线公式)从第一段末节点绕到第二段首节点，避免直线穿过
中间无关节点。第二段内部 NPUPlatform 之后是一个 fork-join：npuplat 同时长出
「两段式打补丁」与「工厂钩子分发」两支，最终汇入 exit——这是真实关系(两者都是
NPUPlatform 的职责切面)而非人为并列，故用分叉/汇合画法而非强行拉直成一条线。

■ 不可变(全书统一视觉语言,同 example-chapter-map.py 六条约定):
  §徽标胶囊样式 / 入口绿#22c55e-出口橙#f97316-主线蓝#3b82f6 / 路线条实线蓝-虚线灰 /
  legend 必须画(本章额外加了两种「符号所属方」色,一并入 legend) / cjk_text_width()。

■ 本章额外(可变,不与全书约定冲突,只是这章多用了两样):
  1. 每个节点左上角一个小圆点，标记这个真实符号住在哪一侧——
     indigo=vLLM 侧(扩展点)，teal=vllm-ascend 侧(登记实现)——呼应 §1.6 姊妹篇对照表，
     exit 节点(全书范式，不专属一侧)不点。
  2. 两段行之间留一条 LANE_GAP 空白带，专给"折行"折线走线，不与任何节点重叠。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录，见 figure-manifest.json)。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):逐字符判定——
    全角(ord>0x2E80)按 1.0×size,半角按 0.58×size,求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["第一段 · 从安装声明到选定平台（§1.2 → §1.3）", "第二段 · NPUPlatform 接管分发与打补丁（§1.3 → §1.5）"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, §编号, 所属方: "v"=vLLM侧/"a"=昇腾侧/None=不标)
NODES = [
    ("declare",  0, 0, 0, "entry_points",              "setup.py 声明 5 个回调进包元数据", "§1.2", "a"),
    ("discover", 0, 1, 0, "load_plugins_by_group()",   "按组名反查，plugin.load() 取回调", "§1.2", "v"),
    ("register", 0, 2, 0, "register()",                "只返回类名字符串，故意不 import",  "§1.3", "a"),
    ("curplat",  0, 3, 0, "current_platform",           "resolve 选定 + 懒加载唯一实例化",   "§1.3", "v"),
    ("npuplat",  1, 0, 0, "NPUPlatform",                "覆写身份属性 + 一整组工厂钩子",     "§1.3", "a"),
    ("patch",    1, 1, 0, "adapt_patch()",              "pre_register_and_update 触发",     "§1.4", "a"),
    ("hooks",    1, 1, 1, "get_attn_backend_cls 等",    "运行期被引擎按需问起，吐回类名",     "§1.3", "a"),
    ("exit",     1, 2, 0, "扩展点—登记实现",             "全书范式起点，后续 35 章逐站展开",   "§1.5", None),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝(curplat→npuplat 的折行连接单独手绘,不在这里)
    ("declare", "discover"), ("discover", "register"), ("register", "curplat"),
    ("npuplat", "patch"), ("npuplat", "hooks"),
    ("patch", "exit"), ("hooks", "exit"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("① 从声明到选定平台",        [(0, "§1.2"), (1, "§1.2"), (2, "§1.3"), (3, "§1.3")], True),
    ("②a 分发支线：工厂钩子",     [(0, "§1.3"), (1, "§1.3"), (2, "§1.5")], True),
    ("②b 分发支线：两段式打补丁", [(0, "§1.3"), (1, "§1.4"), (2, "§1.5")], False),
    ("选读 · 姊妹篇对照表",       [(3, "§1.6")], False),
]
LEGEND = [
    ("#22c55e", "入口：pip install / vLLM 启动触发"),
    ("#3b82f6", "章内主线走线"),
    ("#f97316", "出口：交给后续 35 章的登记实现"),
    ("#6366f1", "● vLLM 侧符号（扩展点）"),
    ("#0d9488", "● vllm-ascend 侧符号（登记实现）"),
]
TITLE = "第 1 章 · OOT 插件三支柱剖面（安装声明 → vLLM 选定 → NPUPlatform 分发 + 两段式打补丁）"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
C_SIDE = {"v": "#6366f1", "a": "#0d9488"}

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 225, 60
COL_GAP, ROW_GAP = 42, 22
EDGE_MARGIN, STUB_W, STUB_H = 16, 72, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
LANE_GAP = 40  # 两段行之间的空白带,专给折行连接线走,不与节点重叠
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
for i, bh in enumerate(band_h):
    if i > 0:
        _cum += LANE_GAP
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
         f'font-size="14.5" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例)
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
             f'font-size="13" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩(给入口/出口箭头一个可附着的框,兼表达"调用方在画布外")
ex, ey = NODE_XY["declare"]; ey += NODE_H / 2
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

# 调用边(主线蓝)。多条边汇入同一节点时,终点 y 各偏移(间距 16px 或按节点高度收窄),
# 避免重合的终点看不出"汇合"。
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
    if n > 1:
        spacing = min(16.0, (NODE_H - 14) / (n - 1))
        y_offset = (i - (n - 1) / 2) * spacing
    else:
        y_offset = 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 折行连接:第一段末节点(curplat,懒加载选定 NPUPlatform)→第二段首节点(npuplat)。
# 两段之间隔着 LANE_GAP 空白带,直线公式会穿过中间无关节点,故手绘一条走
# "右→下→左→下" 的折线,全程只经过空白带与左右边距,不与任何节点框相交
# (呼应契约"走线太长就折成多行泳道"的折行处理)。
cx1, cy1 = NODE_XY["curplat"]; cy1 += NODE_H / 2
cx2, cy2 = NODE_XY["npuplat"]; cy2 += NODE_H / 2
turn_x1 = cx1 + NODE_W + 22
turn_x2 = cx2 - 22
gap_y = (band_top[1] - LANE_GAP / 2)
wrap_path = (f'M {cx1 + NODE_W:.1f},{cy1:.1f} L {turn_x1:.1f},{cy1:.1f} '
             f'L {turn_x1:.1f},{gap_y:.1f} L {turn_x2:.1f},{gap_y:.1f} '
             f'L {turn_x2:.1f},{cy2:.1f} L {cx2:.1f},{cy2:.1f}')
L.append(f'<path d="{wrap_path}" fill="none" stroke="{C_MAIN}" stroke-width="2" '
         f'stroke-dasharray="2,3" marker-end="url(#mMain)"/>')
L.append(f'<text x="{(turn_x1 + turn_x2) / 2:.1f}" y="{gap_y - 6:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" fill="{C_NODE_SUB}">'
         f'{esc("懒加载后实例化 → 下一段登场")}</text>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角 § 徽标 + 左上角所属方圆点)
for nid, lane, col, row, symbol, phrase, sec, side in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    if side:
        L.append(f'<circle cx="{x + 14:.1f}" cy="{y + 14:.1f}" r="5" fill="{C_SIDE[side]}"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.42:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="13" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 8, y, sec)

# 底部阅读路线:复用列坐标 COLX,§ 徽标与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=推荐 / 虚线灰=次要或选读)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="12" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first = COLX[stops[0][0]] + NODE_W / 2
    x_last = COLX[stops[-1][0]] + NODE_W / 2
    if x_last != x_first:
        dash = '' if hi else ' stroke-dasharray="6,4"'
        L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
                  f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for col, sec in stops:
        L += badge(COLX[col] + NODE_W / 2, ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}: viewBox 0 0 {w:.0f} {h:.0f} (ratio {w/h:.2f}:1)")
