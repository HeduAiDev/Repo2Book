#!/usr/bin/env python3
"""第 19 章「本章地图」——OOT 后端接入 vLLM 的源码剖面图。
入口(绿,左侧接口桩)= 模型层向 vLLM 要一个注意力后端;主线蓝箭头沿「vLLM 选择
入口 → 昇腾三元 key 路由 → 解析成类 → 注册进 CUSTOM 槽 → 契约四方法(伪装/KV 形状/
运行期分流)」走线;出口(橙,右侧接口桩)= 选定的后端类连同 Impl/Builder 交还给
model-runner(续见第 20 章 MHA)。

■ 不可变(全书 72 章统一视觉语言,换章节数据时不要动这些,只改下面的 DATA):
  同 .claude/skills/svg-diagram/references/example-chapter-map.py 的六条约定
  (§徽标胶囊样式/入口绿#22c55e-出口橙#f97316-主线蓝#3b82f6/路线条实线蓝-虚线灰/
  legend 必须画/cjk_text_width() 估算)。

■ 可变:LANES/NODES/EDGES/ROUTES ——本章按「vLLM 框架层(selector.py 的入口与解析、
  backend.py 的契约基类)/ 昇腾路由层(platform.py 的三元 key 路由)/ 昇腾实现层
  (attention_v1.py 的 AscendAttentionBackend 及其四个契约方法)」三级铺开。
  AttentionBackend(契约基类,§19.5)与 resolve_obj_by_qualname(§19.1)同列不同行——
  两者都在「解析出类」之后、「类被注册/落地契约」之前完成,分别汇入
  AscendAttentionBackend(§19.3),对应正文「注册与解析是两条线,在这个类上汇合」。
  AscendAttentionBackend 之后扇出四个契约方法节点(get_name §19.4、
  get_kv_cache_shape §19.5、get_impl_cls/get_builder_cls §19.6),四者再汇入 exit——
  对应正文组 A 四个 @abstractmethod(get_name/get_impl_cls/get_builder_cls/
  get_kv_cache_shape)一一对账、最终交给 model-runner 的控制流。组 B(可覆写默认方法
  get_supported_kernel_block_sizes)与组 C(swap_blocks/copy_blocks,昇腾自带、非 v1
  契约的 v0 遗留)不单独设节点——本图只画组 A 四个硬性契约点以控制在 12 节点预算内、
  保持每列一条清晰主线,组 B/C 的细节留给正文 §19.5 展开,不在图上杜撰或暗示齐全。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录;每次改动 DATA/布局代码后必须
  重新核对一遍,不能照抄上一轮结果):见文件末尾运行时打印/figure-manifest.json
  对应条目。
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
LANES = ["vLLM 框架层", "昇腾路由层", "昇腾实现层"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, §编号或 "")
NODES = [
    ("entry",       0, 0, 0, "_cached_get_attn_backend",
     "问平台要哪个后端", "§19.1"),
    ("route",       1, 1, 0, "get_attn_backend_cls",
     "三元 key 查表选后端", "§19.2"),
    ("resolve",     0, 2, 0, "resolve_obj_by_qualname",
     "路径串解析成后端类", "§19.1"),
    ("base_cls",    0, 2, 1, "AttentionBackend",
     "4 个抽象方法定契约", "§19.5"),
    ("ascend_cls",  2, 3, 0, "AscendAttentionBackend",
     "占住 CUSTOM 槽", "§19.3"),
    ("fake_name",   2, 4, 0, "get_name",
     "V2 下伪装 FLASH_ATTN", "§19.4"),
    ("kv_shape",    2, 4, 1, "get_kv_cache_shape",
     "昇腾 KV 布局(2, …)", "§19.5"),
    ("impl_cls",    2, 4, 2, "get_impl_cls",
     "按 enable_cp() 二选一", "§19.6"),
    ("builder_cls", 2, 4, 3, "get_builder_cls",
     "同套路,延迟 import CP", "§19.6"),
    ("exit",        0, 5, 0, "后端类与 Impl/Builder",
     "交给 model-runner", ""),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝
    ("entry", "route"), ("route", "resolve"),
    ("resolve", "ascend_cls"), ("base_cls", "ascend_cls"),
    ("ascend_cls", "fake_name"), ("ascend_cls", "kv_shape"),
    ("ascend_cls", "impl_cls"), ("ascend_cls", "builder_cls"),
    ("fake_name", "exit"), ("kv_shape", "exit"),
    ("impl_cls", "exit"), ("builder_cls", "exit"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("完整读:五步选出后端", [(0, "§19.1"), (1, "§19.2"), (2, "§19.1"), (3, "§19.3"), (4, "§19.4")], True),
    ("只看契约:哪些方法是硬性要求", [(3, "§19.3"), (4, "§19.5")], False),
    ("只看运行期分流:CP 来不来走哪个实现", [(3, "§19.3"), (4, "§19.6")], False),
]
LEGEND = [("#22c55e", "入口:模型层发起后端请求"), ("#3b82f6", "章内主线调用/契约落地边"),
          ("#f97316", "出口:后端类交还 model-runner")]
TITLE = "第 19 章 · OOT 后端接入剖面(选择 → 注册 → 伪装 → 契约 → 分流)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 170, 58
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
# 多条边汇入同一节点时,终点 y 各偏移(间距 16px),否则重合的终点在视觉上看不出
# "汇合"、像一条线断头。
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
    if sec:
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
print(f"wrote {out}")
print(f"canvas: {w:.0f}x{h:.0f}  aspect={w / h:.2f}")
