#!/usr/bin/env python3
"""第 5 章「本章地图」——check_and_update_config 的源码剖面图。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py 模板改写：
不可变机制(esc/cjk_text_width/badge 胶囊/入口绿-出口橙-主线蓝/图例/路线高亮规则)原样保留，
只改 DATA，以及一处必要的几何扩展——见下方"本章新增"说明。

节点预算：8 个(entry/fix/init_ascend/ascend_config/get_value/compile_rewrite/worker_cls/
env_exit) ≤ 12。本章标题为编号标题(## 5.1 ... ## 5.9)，站牌用 §5.N；entry 与 env_exit
两站各扛两个紧邻小节(§5.1+§5.2、§5.8+§5.9)，故徽标文字是 "§5.1 / §5.2" 这种组合形式。

设计要点：
- 主线(实线蓝，单入单出，无分叉)：check_and_update_config → _fix_incompatible_config
  → init_ascend_config →(下潜一层到 AscendConfig 强类型解析)→ 浮回主线 →
  compilation_config.mode 改写 → parallel_config.worker_cls 落定 →
  PYTORCH_NPU_ALLOC_CONF 写环境变量。这是本章"两条主线"里的第一条
  （平台=配置改写器），全流程严格顺序执行，没有 if/else 分叉，故只画一条脊柱。
- AscendConfig / _get_config_value 是本章第二条主线（无 schema 配置后门）的落地处，
  画成从主线"下潜"一层的两个纵向节点(同一列，AscendConfig 在上、_get_config_value
  在下)，用竖直箭头表示"调用陷入子过程、处理完再浮回主线"，而不是虚构一条平行分支。
- entry 节点是 NPUPlatform.check_and_update_config 本体：它既是 §5.2(入口编排骨架)的
  主角，也是 §5.1(vLLM 钩子契约)在昇腾侧的落地对象——两节共享同一个真实符号，
  故合并成一站，徽标挂 "§5.1 / §5.2"。env_exit 同理合并 §5.8(设环境变量)与
  §5.9(小结，配置改写完毕、原样返回上层继续构图)。

[本章新增，模板未覆盖的情形]：
1. init_ascend_config → AscendConfig 与 AscendConfig → _get_config_value 这两条边的
   起点/终点在同一列(同一 x)、只是跨泳道(纵向下潜再浮回)。模板自带的"右边→左边"
   横向连边公式对同列边会画出诡异的倒退线，故新增一段判定：起止点同列时改画
   "底边中点→顶边中点"的竖直箭头，其余(列号不同的)边仍走模板原有的横向公式。
2. get_value → compile_rewrite 这条"浮回主线"的斜向边，会从 AscendConfig/
   get_value 所在列的右侧穿出。若这两个下潜节点的 § 徽标也贴在右上角(模板默认位置)，
   徽标会正好卡在这条斜线的必经路径上(第一轮渲染 Read PNG 发现斜线贯穿了两个
   "§5.5"徽标)。这两个节点没有"同泳道右侧邻居"，徽标贴左上角同样合法且不丢语义，
   故只给这两个节点的徽标换到左上角，避开斜边——不改变徽标本身的样式/配色/字号。

六项自查记录见文件末尾 [SELF-CHECK] 注释(渲染→Read PNG 亲眼看后如实记录)。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算——全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["NPUPlatform.check_and_update_config 编排层", "AscendConfig 强类型解析层(无 schema 后门)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名(可含 "\n" 机械换行,不改变拼写), 一行短语(可含 "\n"), §编号)
NODES = [
    ("entry",           0, 0, 0, "check_and_update_\nconfig",
     "钩子契约:按引用接管完整config;\n两道守卫+编排七步", "§5.1 / §5.2"),
    ("fix",             0, 1, 0, "_fix_incompatible_\nconfig",
     "9段cascade reset:GPU/ROCm\n参数归零/改写(numa_bind 例外)", "§5.3"),
    ("init_ascend",     0, 2, 0, "init_ascend_config",
     "进程级懒加载单例;\n按同一config对象判存", "§5.4"),
    ("ascend_config",   1, 2, 0, "AscendConfig",
     "开放dict逐键解析成\n强类型子配置对象", "§5.5"),
    ("get_value",       1, 2, 1, "_get_config_value",
     "additional_config→env\n→default 三级取值(已塌缩)", "§5.5"),
    ("compile_rewrite", 0, 3, 0, "compilation_config.\nmode",
     "按enforce_eager/cudagraph\n能力收窄编译模式", "§5.6"),
    ("worker_cls",      0, 4, 0, "parallel_config.\nworker_cls",
     "'auto' 落成 NPUWorker/\nNPUWorker310/XliteWorker", "§5.7"),
    ("env_exit",        0, 5, 0, "PYTORCH_NPU_ALLOC_\nCONF",
     "追加expandable_segments;\n写回env,收尾返回构图", "§5.8 / §5.9"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝;init_ascend↔ascend_config↔get_value 同列竖直下潜/浮回
    ("entry", "fix"),
    ("fix", "init_ascend"),
    ("init_ascend", "ascend_config"),
    ("ascend_config", "get_value"),
    ("get_value", "compile_rewrite"),
    ("compile_rewrite", "worker_cls"),
    ("worker_cls", "env_exit"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("主线:全流程编排",
     [(0, "§5.1 / §5.2"), (1, "§5.3"), (2, "§5.4"), (3, "§5.6"), (4, "§5.7"), (5, "§5.8 / §5.9")], True),
    ("支线:无 schema 配置后门(跳读 AscendConfig 解析)",
     [(2, "§5.5"), (3, "§5.6")], False),
]
LEGEND = [("#22c55e", "入口:从上层调用进入"), ("#3b82f6", "章内主线调用边"), ("#f97316", "出口:返回上层")]
TITLE = "第 5 章 · check_and_update_config 配置改写剖面(平台钩子 + AscendConfig 解析 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 185, 90
COL_GAP, ROW_GAP = 30, 22
EDGE_MARGIN, STUB_W, STUB_H = 12, 60, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 32  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_H, BADGE_PAD_X, BADGE_MIN_W = 20, 8, 46  # 徽标高度固定;宽度按文字动态算,最小 46(与旧图一致)
LEFT_BADGE_IDS = {"ascend_config", "get_value"}  # 见文件头 [本章新增] 第2点:避开斜向浮回边

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
NODE_COL = {n[0]: n[2] for n in NODES}

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
    """§ 徽标胶囊,居中挂在 (cx,cy)。宽度按文字动态算(单 §N.M 与组合 "§N.M / §N.M"
    都要装得下、不溢出胶囊)，与旧图的固定 46 宽相比只是加了下限，视觉规格不变。"""
    bw = max(BADGE_MIN_W, cjk_text_width(text, 11) + BADGE_PAD_X * 2)
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
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
xx, xy = NODE_XY["env_exit"]; xy += NODE_H / 2
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

# 调用边(主线蓝)。多条边汇入同一节点时终点 y 各偏移,避免看不出"汇合"。
# [本章新增] 起止点同列(同一 x)——即"下潜/浮回"竖直边——改画底边中点→顶边中点，
# 不套用"右边→左边"的横向公式(否则会画出从右边倒退向左边的诡异线)。
_dst_total = {}
for _, dst in EDGES:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    same_col = NODE_COL[src] == NODE_COL[dst]
    if same_col:
        # 竖直下潜/浮回:谁在上谁的底边接谁的顶边
        if y1 < y2:
            p1 = (x1 + NODE_W / 2, y1 + NODE_H)
            p2 = (x2 + NODE_W / 2, y2)
        else:
            p1 = (x1 + NODE_W / 2, y1)
            p2 = (x2 + NODE_W / 2, y2 + NODE_H)
    else:
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        n = _dst_total[dst]
        i = _dst_seen.get(dst, 0)
        _dst_seen[dst] = i + 1
        y_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
        p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名(1~2 行) + 一行短语(1~2 行,始终锚在节点下半区) + 右上角 § 徽标)
SYMBOL_1LINE_Y, SYMBOL_2LINE_Y1, SYMBOL_2LINE_Y2 = 34, 24, 40
PHRASE_1LINE_Y, PHRASE_2LINE_Y1, PHRASE_2LINE_Y2 = 71, 66, 80
for nid, lane, col, row, symbol, phrase, sec in NODES:
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
    badge_cx = x + 4 if nid in LEFT_BADGE_IDS else x + NODE_W - 4
    L += badge(badge_cx, y, sec)

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
print(f"wrote {out}: {w:.0f}x{h:.0f} (ratio {w / h:.2f}:1, n_cols={n_cols})")

# [SELF-CHECK](渲染→Read PNG 亲眼看后如实记录，见下方 [ROUND-1]/[ROUND-2] 注释)
