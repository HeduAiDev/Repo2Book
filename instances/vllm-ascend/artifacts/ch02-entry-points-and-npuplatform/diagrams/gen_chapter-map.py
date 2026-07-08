#!/usr/bin/env python3
"""第 2 章「本章地图」——entry points 与 NPUPlatform 源码剖面图。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py 模板改写：
不可变机制(esc/cjk_text_width/badge/配色/图例规则)原样保留，只改 DATA 与「节点文字按
行数动态排布」的渲染逻辑(本章有一个 3 行符号名，模板原版只处理 1/2 行，这里泛化)。

节点预算 12(entry/load_plugins/register/resolve_qual/resolve_obj/getattr_cp/
npuplatform/hooks/worker_cls/is310p/exit/general_plugins) = 12 ≤ 12。
本章标题为编号标题(## 2.1 ... ## 2.12)，站牌用 §2.N。

设计要点(全部对应正文小节，符号取自 dossier.json / chapter.md 原文)：
- 主线(实边，蓝)是一条单入单出的脊柱，按「一根 qualname 字符串」的数据流顺序铺开：
  entry_points(装) → load_plugins_by_group(发现) → resolve_current_platform_cls_qualname
  (裁决) → resolve_obj_by_qualname(懒加载真正 import) → NPUPlatform(身份) →
  get_attn_backend_cls(运行期分发) → exit(交还 vLLM 引擎)。
- register()/__getattr__(current_platform)/worker_cls/is_310p() 四个是「卫星」节点——
  不挂调用边，只挂在主线邻近的同列不同行里，代表"随时可查的机制细节/横切关注点"：
    · register() 挂在 load_plugins_by_group 正下方(它是被 load 出来、又在 resolve 的
      chain 循环里被调用的那个 OOT 插件函数，紧邻其发现者最贴题)。
    · __getattr__(current_platform) 挂在 resolve_obj_by_qualname 正下方(它是真正调用
      resolve_obj_by_qualname 并把结果缓存成单例的那层——同一延迟绑定动作的两面)。
    · worker_cls / is_310p() 挂在 get_attn_backend_cls 同列下方两行(worker_cls 是"同招
      不同落点"的另一处工厂钩子；is_310p() 是两处工厂钩子都会读的横切分代信号)。
  四个卫星均不挂边，避免用"同列同框内的调用关系"这种不存在的边虚构因果——它们的
  真实关系已经写在短语文字里，读者一眼看列位置就懂"这是谁的附注"。
- general_plugins 分支(register_connector())独立成第 4 条泳道，由 entry 直接引一条边
  下来——因为它和 platform_plugins 共用同一份 setup.py entry_points 声明(§2.1)，但走的
  是另一个 group、在 engine-core 子进程里生效，与主链并行、不汇入 exit。
- exit 站牌节点没有 §(它是"控制返回 vLLM 引擎"这个抽象终点，不对应某一具体源码符号)。

六项自查记录(渲染→Read PNG 亲眼看后如实记录，见文件末尾 [SELF-CHECK] 注释)。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算——全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["安装期", "选择 · 懒加载期", "运行期分发", "子进程扩展(并行)"]  # 泳道,上→下

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名(可含 "\n" 机械换行,不改变拼写),
#  一行短语, §编号或 "")
NODES = [
    ("entry",        0, 0, 0, "entry_points",
     "两组入口写进包元数据", "§2.1"),
    ("load_plugins",  1, 1, 0, "load_plugins_\nby_group",
     "发现插件,只load不调用", "§2.2"),
    ("register",      1, 1, 1, "register()",
     "OOT 无条件返回字符串", "§2.4"),
    ("resolve_qual",  1, 2, 0, "resolve_current_\nplatform_cls_\nqualname()",
     "OOT优先builtin裁决", "§2.3"),
    ("resolve_obj",   1, 3, 0, "resolve_obj_\nby_qualname",
     "推迟的import在此发生", "§2.5"),
    ("getattr_cp",    1, 3, 1, "__getattr__(\ncurrent_platform)",
     "首访懒加载,缓存单例", "§2.6"),
    ("npuplatform",   2, 4, 0, "NPUPlatform",
     "身份替换类属性", "§2.7"),
    ("hooks",         2, 5, 0, "get_attn_\nbackend_cls",
     "查表选昇腾backend", "§2.8"),
    ("worker_cls",    2, 5, 1, "worker_cls",
     "auto 改写成 NPUWorker", "§2.8"),
    ("is310p",        2, 5, 2, "is_310p()",
     "设备分代:横切两处分流", "§2.10"),
    ("exit",          2, 6, 0, "vLLM 引擎",
     "拿到最终实现,继续执行", ""),
    ("general_plugins", 3, 0, 0, "register_\nconnector()",
     "子进程注册,与主链并行", "§2.9"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝;register/getattr_cp/worker_cls/
    # is310p 是卫星节点,不挂边(它们的关系已写在短语里,不虚构因果边)
    ("entry", "load_plugins"),
    ("load_plugins", "resolve_qual"),
    ("resolve_qual", "resolve_obj"),
    ("resolve_obj", "npuplatform"),
    ("npuplatform", "hooks"),
    ("hooks", "exit"),
    ("entry", "general_plugins"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("完整装配链(§2.1→§2.8)",
     [(0, "§2.1"), (1, "§2.2"), (2, "§2.3"), (3, "§2.5"), (4, "§2.7"), (5, "§2.8")], True),
    ("跳读:延迟绑定复用的三处细节",
     [(1, "§2.4"), (3, "§2.6"), (5, "§2.10")], False),
]
LEGEND = [("#22c55e", "入口:上层安装/首次访问触发"), ("#3b82f6", "章内主线:qualname 字符串流转"),
          ("#f97316", "出口:顶替生效,返回 vLLM 引擎")]
TITLE = "第 2 章 · entry points 与 NPUPlatform 剖面(一根 qualname 字符串的延迟绑定)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
# 本章有一个 3 行符号(resolve_current_platform_cls_qualname)，NODE_H 按「最多 3 行符号
# + 1 行短语」留够空间；文字改用「按行数动态居中排布」(见下方 render 段)，不再像模板
# 原版那样只支持写死的 1/2 行两套 Y 坐标。
NODE_W, NODE_H = 165, 100
COL_GAP, ROW_GAP = 22, 20
EDGE_MARGIN, STUB_W, STUB_H = 12, 50, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_W, BADGE_H = 46, 20
LINE_H_SYM, LINE_H_PHR, SYM_PHR_GAP = 15, 13, 6  # 符号行距/短语行距/两块之间的间隙

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

# 节点(圆角框 + 真实符号名(1~3 行,按行数动态居中) + 一行短语 + 右上角 § 徽标)
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_lines = symbol.split("\n")
    phr_lines = phrase.split("\n")
    block_h = len(sym_lines) * LINE_H_SYM + SYM_PHR_GAP + len(phr_lines) * LINE_H_PHR
    top = y + (NODE_H - block_h) / 2
    for i, line in enumerate(sym_lines):
        baseline = top + i * LINE_H_SYM + LINE_H_SYM * 0.8
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{baseline:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="12.5" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(line)}</text>')
    phr_top = top + len(sym_lines) * LINE_H_SYM + SYM_PHR_GAP
    for j, line in enumerate(phr_lines):
        baseline = phr_top + j * LINE_H_PHR + LINE_H_PHR * 0.8
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{baseline:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" fill="{C_NODE_SUB}">{esc(line)}</text>')
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
print(f"wrote {out}: {w:.0f}x{h:.0f}")

# [SELF-CHECK](渲染→Read PNG 亲眼看后如实记录，见下方注释——首轮记录见文件末尾追加)
