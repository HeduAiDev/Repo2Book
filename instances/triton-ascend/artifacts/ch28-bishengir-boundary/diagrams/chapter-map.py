#!/usr/bin/env python3
"""ch28「本章地图」——linalg_to_bin 闭源边界剖面：装配层(add_stages 三选一，
回指 ch27) → 第①段抠元数据 + 第②段拼命令行/定位二进制(同一泳道，前后相接) →
闭源边界(subprocess.run，深色节点) → 第③段三通道回收(读回二进制/stdout 正则
UB-bits/dlopen 回调，扇出到三个下游用途)。底部另加一条"回顾对照"旁支(虚线灰，
非调用序)：cmdline_locate 站牌之下挂 910_95 vs A2_A3 差异(§28.7，f7 的答案)，
entry 之下挂 force_simt_only 快路径(§28.8)——两者都是 entry/cmdline 处三选一
决策的"另外两条没细追的路"，用回顾对照而非主线呈现，避免误导为调用序。

本章是**数字编号章**(`## 28.1`…`## 28.9`)，站牌用 §28.N 徽标；cmdline_locate
一站聚合 §28.3(拼命令行) + §28.4(定位/探测二进制)两节——两节实为同一段代码
准备工作的前后两半，聚合成一站，徽标写"28.3–28.4"(按契约"超长章聚合"处理，
仅前一半 §28.3 参与 lint_chapter_map 的徽标存在性核验，后半为范围说明文字)。

节点预算(9 个，≤12)：entry / parse / cmdline_locate / boundary / ch_binary /
ch_stdout / ch_dlopen / branch_divergence / fast_path。

主线(实线蓝，EDGES_MAIN)=真实调用序：entry→parse→cmdline_locate→boundary→
(ch_binary, ch_stdout, ch_dlopen)三扇出——扇出虽是"一处产生三处可读"，但确系
subprocess.run 一次调用后三条读取通道各自独立解析同一份返回结果，仍是因果
延续，不需要"无因果"注记。
旁支(虚线灰，EDGES_SIDE)=非调用序的回顾/对照关系：cmdline_locate→
branch_divergence(910_95 命令行拼装的"另一份实现"差异对照)、entry→fast_path
(entry 三选一里没往下追的第三条路，此处补全)。两条旁支都选在"该列/该泳道
纵向天然无遮挡"的坐标(fast_path 与 entry 同列 col0，branch_divergence 与
cmdline_locate 同列 col2)，故用直线纵向连接即可，不需要 ch27 那种绕列的
折线(draw_elbow)。

闭源边界给了第 4 种语义色(深色 #1e293b)——不是常规的入口/主线/出口，是
"这里往后书不追"的边界标记本身，故图例补第 4 条说明；旁支虚线灰也补第 5 条
图例说明其"回顾对照、非调用序"的语义(不同于 ch27 的省略，这里两条旁支更
显眼，宁可多写一行图例)。

不可变视觉语言(全书统一，来自 example-chapter-map.py 模板 + ch27 的动态换行/
自适应符号字号改法)：§徽标胶囊 fill #eef2ff / stroke #6366f1、入口绿 #22c55e /
出口橙 #f97316 / 主线蓝 #3b82f6 / 旁支虚线灰 #94a3b8、cjk_text_width() 逐字符
宽度估算。

六项自查(渲染→Read PNG 亲眼看后如实记录)：见同目录 figure-manifest.json 该图 selfcheck。

用法：python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算：全角(ord>0x2E80)按 1.0×size，半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


_BREAK_AFTER = set("，；：、/ ,;)")


def wrap_claim(text, max_w, size):
    """一句论点太长时换行——只在标点/斜杠/空格之后断行，不劈开一个标识符或中文词。
    贪心找"prefix 仍不超宽的最靠后一个合法断点"；找不到合法断点才整句照旧单行放行。"""
    breaks = [i for i, ch in enumerate(text) if ch in _BREAK_AFTER]
    best = None
    for i in breaks:
        if cjk_text_width(text[:i + 1], size) <= max_w:
            best = i
    if best is None:
        return [text]
    line1, line2 = text[:best + 1].rstrip(), text[best + 1:].lstrip()
    if cjk_text_width(line2, size) <= max_w:
        return [line1, line2]
    return [line1] + wrap_claim(line2, max_w, size)


# ---------------- DATA(可变：本章数据) ----------------
LANES = [
    "third_party/ascend/backend/compiler.py · 装配层：add_stages 三选一(回指 ch27)",
    "compiler.py 第①段抠元数据 + compiler.py/utils.py 第②段拼命令行/定位二进制",
    "third_party/ascend/backend/compiler.py · 闭源边界：subprocess.run",
    "third_party/ascend/backend/compiler.py · 第③段：三通道回收元数据",
    "回顾对照(旁支，非调用序)：两候选实现差异(§28.7) ／ 快路径命令行(§28.8)",
]

# (节点id, 泳道下标, 列, 泳道内行号, [符号行…], 一句论点, §编号)
NODES = [
    ("entry", 0, 0, 0,
     ["add_stages"],
     "force_simt_only、compile_on_910_95 两开关，从三候选选中本图解剖的 910_95 支",
     "28.1"),
    ("parse", 1, 1, 0,
     ["_parse_linalg_metadata"],
     "6 条正则一次扫描 IR 文本，抠出 mix_mode、kernel_name、tensor_kinds 等字段",
     "28.2"),
    ("cmdline_locate", 1, 2, 0,
     ["get_common_bishengir_compile_options", "_get_npucompiler_path"],
     "None 开关留白、显式设值才 append，约 30 个条件参数；--help 探版本、定位二进制",
     "28.3–28.4"),
    ("boundary", 2, 3, 0,
     ["subprocess.run"],
     "IR 文件与命令行交给闭源编译器，书读到这一行为止，内部不再猜测",
     "28.5"),
    ("ch_binary", 3, 4, 0,
     ["Path(bin_path).read_bytes()"],
     "编译产物整体读成字节，就是最终的 npubin，交给 driver 装载发射",
     "28.6"),
    ("ch_stdout", 3, 4, 1,
     ["required_ub_bits ← re.search"],
     "stdout 里一行文字，正则抠出位宽，回填 required_ub_bits 给自动调优",
     "28.6"),
    ("ch_dlopen", 3, 4, 2,
     ["__get_metadata_attr_by_callback"],
     "加载编译器顺带生成的动态库，靠几个回调函数取运行时同步参数",
     "28.6"),
    ("branch_divergence", 4, 2, 0,
     ["linalg_to_bin_enable_npu_compile_A2_A3"],
     "骨架完全同构，差异只在 target 取法、regbased 分叉与几个独有开关",
     "28.7"),
    ("fast_path", 4, 0, 0,
     ["ttir_to_npubin"],
     "绕开结构化下降，TTIR 直接编译，换一套 --pure-simt 专属参数",
     "28.8"),
]
NODE_BY_ID = {n[0]: n for n in NODES}
ENTRY_NODE = "entry"
EXIT_NODE = "ch_binary"        # 出口桩以此节点的行高对齐(三通道里的主产物)

EDGES_MAIN = [  # 实线蓝——真实调用序
    ("entry", "parse"), ("parse", "cmdline_locate"), ("cmdline_locate", "boundary"),
    ("boundary", "ch_binary"), ("boundary", "ch_stdout"), ("boundary", "ch_dlopen"),
]
EDGES_SIDE = [  # 虚线灰——非因果的回顾/对照关系，不是调用序。直连前提：源列与目标
    # 列相同、且途经泳道在该列均无节点/标签遮挡——cmdline_locate(col2)→
    # branch_divergence(col2)满足(泳道 2/3 的标签实测宽度都短于 col2 的 x)，
    # 故直连；entry(col0)→fast_path(col0)途中会穿过泳道 1/2/3 标签文字(泳道
    # 标签比 col0 宽得多)，改在下方用 BYPASS_X 绕道的折线单独画，不放进这里。
    ("cmdline_locate", "branch_divergence"),
]

ROUTES = [  # (路线名, [(列, §编号), ...]按阅读顺序, 是否高亮：True=实线蓝/False=虚线灰)
    ("910_95 主线(全通道解剖样本)", [(0, "28.1"), (1, "28.2"), (2, "28.3–28.4"), (3, "28.5"), (4, "28.6")], True),
    ("回顾对照：默认实现与快路径", [(0, "28.8"), (2, "28.7")], False),
]
LEGEND = [
    ("#22c55e", "入口(回指 ch27)：add_stages 已在上一章挂好三候选"),
    ("#3b82f6", "主线：910_95 解剖样本——抠元数据→拼命令行→闭源调用→三通道回收"),
    ("#1e293b", "深色节点＝闭源边界 subprocess.run 调用：往后是黑盒，书不再追"),
    ("#f97316", "出口(预告 ch29)：npubin 字节交 driver 装载发射"),
    ("#94a3b8", "虚线灰：回顾对照(§28.7／§28.8)，非调用序"),
]
TITLE = "第 28 章 · linalg_to_bin 闭源边界剖面(源码走线 + § 讲解站牌)"
SUBNOTE = "拼命令行/定位二进制一站聚合两个函数：命令行拼装取自 compiler.py，二进制定位/探测取自 third_party/ascend/backend/utils.py"

# ---------------- 不可变：配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BOUNDARY = "#1e293b"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算，零魔数) ----------------
NODE_W = 188
COL_GAP, ROW_GAP = 26, 18
EDGE_MARGIN, STUB_W, STUB_H = 12, 76, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 24
LANE_LABEL_H, BAND_PAD = 20, 10
TOP_PAD, TITLE_H, SUBNOTE_H, LEGEND_H, BOTTOM_PAD = 12, 24, 20, 5 * 13.5 + 8, 14
ROUTE_HEAD_H, ROUTE_ROW_H = 20, 36
BADGE_W, BADGE_H = 46, 20
SYM_FONT, SYM_LINE_H = 11.4, 14
CLAIM_FONT = 9.2

n_cols = max(n[2] for n in NODES) + 1
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

# 每个节点的论点先按 NODE_W 预算换行一遍，符号行数取全章最大值——统一定 NODE_H。
CLAIM_MAXW = NODE_W - 14
_CLAIM_LINES = {n[0]: wrap_claim(n[5], CLAIM_MAXW, CLAIM_FONT) for n in NODES}
_max_claim_lines = max(len(v) for v in _CLAIM_LINES.values())
_max_sym_lines = max(len(n[4]) for n in NODES)
SYM_TOP = 20
CLAIM_TOP = SYM_TOP + (_max_sym_lines - 1) * SYM_LINE_H + 16
NODE_H = CLAIM_TOP + (_max_claim_lines - 1) * 11.5 + 15

rows_per_lane = [0] * len(LANES)
for _id, lane, col, row, *_ in NODES:
    rows_per_lane[lane] = max(rows_per_lane[lane], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_lane]
band_top, _cum = [], TOP_PAD + TITLE_H + SUBNOTE_H + LEGEND_H
for bh in band_h:
    band_top.append(_cum)
    _cum += bh
lanes_bottom = _cum

NODE_XY = {}
for nid, lane, col, row, *_ in NODES:
    x = COLX[col]
    y = band_top[lane] + LANE_LABEL_H + BAND_PAD + row * (NODE_H + ROW_GAP)
    NODE_XY[nid] = (x, y)

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD
assert w <= 1500 and w / h <= 2.6, f"画布预算超标：{w}x{h}, {w / h:.2f}:1"


def badge(cx, cy, text):
    """§ 徽标胶囊，居中挂在 (cx,cy)。胶囊宽度按文字实际宽度自适应(取
    max(BADGE_W, 文字宽+14))——本章有"28.3–28.4"这种聚合区间号，比常规
    单号"28.1"长不少，固定宽度会把字挤出胶囊，故按内容动态放宽，不用
    再手写第二个魔数宽度。"""
    disp = "§" + text
    bw = max(BADGE_W, cjk_text_width(disp, 11) + 14)
    bx, by = cx - bw / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bw:.1f}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="11" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(disp)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.1f} {h:.1f}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN), ("Side", C_ROUTE_DIM))
) + '</defs>')
L.append(f'<rect width="{w:.1f}" height="{h:.1f}" fill="white"/>')

# 标题 + 副注
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 16}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
_subnote_lines = wrap_claim(SUBNOTE, w - 2 * PAD_L, 9.0)
for si, sline in enumerate(_subnote_lines[:2]):
    L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + TITLE_H + 7 + si * 10.5:.1f}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="9.0" fill="{C_NODE_SUB}">{esc(sline)}</text>')

# 图例(5 种语义色必须画图例；纵向列表，逐条文字较长，横向排会挤)
for li, (color, label) in enumerate(LEGEND):
    _row_y = TOP_PAD + TITLE_H + SUBNOTE_H + 11 + li * 13.5
    L.append(f'<rect x="{PAD_L}" y="{_row_y - 9.5}" width="12" height="12" rx="3" fill="{color}"/>')
    L.append(f'<text x="{PAD_L + 16}" y="{_row_y}" font-family="sans-serif" font-size="9.6" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w:.1f}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="14" y="{band_top[i] + LANE_LABEL_H - 5:.1f}" font-family="sans-serif" '
             f'font-size="10.2" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w:.1f}" y2="{band_top[i]:.1f}" '
                 f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w:.1f}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩(跨章标注：目标章号 > 本章号用「预告」，< 本章号用「回指」)
ex, ey = NODE_XY[ENTRY_NODE]; ey += NODE_H / 2
xx, xy = NODE_XY[EXIT_NODE]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 3.5:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.6" font-weight="bold" fill="#166534">{esc("回指 ch27")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 3.5:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.6" font-weight="bold" fill="#9a3412">{esc("预告 ch29")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 主线(实线蓝)——同列/相邻列直连即可，全部相邻列组合中间无遮挡节点。
# 多条边汇入同一节点时终点 y 各偏移，避免看不出"汇合"；本图仅 boundary 扇出
# 到三个不同行的 ch_* 节点(各自不同 row，天然不重合，无需偏移)。
_dst_total = {}
for _, dst in EDGES_MAIN:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst in EDGES_MAIN:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    y_offset = (i - (n - 1) / 2) * 14 if n > 1 else 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 旁支(虚线灰)——非因果的回顾/对照关系。cmdline_locate→branch_divergence 恰好
# 同列(col2)，且泳道 2/3 的标签实测宽度都短于 col2 的 x，纵向直连不会压标签。
for src, dst in EDGES_SIDE:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    cx_s = xs_ + NODE_W / 2
    cx_d = xd + NODE_W / 2
    assert abs(cx_s - cx_d) < 1e-6, f"{src}->{dst} 旁支未对齐同列，需改直连为折线"
    L.append(f'<line x1="{cx_s:.1f}" y1="{ys_ + NODE_H:.1f}" x2="{cx_d:.1f}" y2="{yd:.1f}" '
              f'stroke="{C_ROUTE_DIM}" stroke-width="1.6" stroke-dasharray="6,4" '
              f'marker-end="url(#mSide)"/>')

# entry→fast_path 单独处理：两者同列(col0)，但纵向直连会穿过泳道 1/2/3 的标签
# 文字、以及 lane1 里 parse/cmdline_locate 两个节点(它们分别占 col1/col2 的
# 整个宽度)。改走"泳道内横移→安全走廊纵向下探→回落到目标列"的折线：走廊选
# col1 与 col2 之间那条天生无节点的 COL_GAP 缝隙(该缝隙同时也早已越过全部
# 泳道标签的实测宽度，两条约束都满足，非拍脑袋取值)。
LANE_LABEL_MAXX = max(14 + cjk_text_width(lbl, 10.2) for lbl in LANES)
BYPASS_X = COLX[1] + NODE_W + COL_GAP / 2
assert BYPASS_X > LANE_LABEL_MAXX, "旁路走廊仍落在某条泳道标签文字范围内"
ent_x, ent_y = NODE_XY["entry"]
fp_x, fp_y = NODE_XY["fast_path"]
ent_cx, ent_bottom = ent_x + NODE_W / 2, ent_y + NODE_H
fp_cx, fp_top = fp_x + NODE_W / 2, fp_y
assert BYPASS_X > ent_cx and BYPASS_X < w - PAD_R, "旁路 x 超出画布可用范围"
_side_pts = [(ent_cx, ent_bottom), (BYPASS_X, ent_bottom), (BYPASS_X, fp_top), (fp_cx, fp_top)]
for i in range(len(_side_pts) - 1):
    (x1_, y1_), (x2_, y2_) = _side_pts[i], _side_pts[i + 1]
    is_last = i == len(_side_pts) - 2
    marker = ' marker-end="url(#mSide)"' if is_last else ''
    L.append(f'<line x1="{x1_:.1f}" y1="{y1_:.1f}" x2="{x2_:.1f}" y2="{y2_:.1f}" '
              f'stroke="{C_ROUTE_DIM}" stroke-width="1.6" stroke-dasharray="6,4"{marker}/>')

# 节点(圆角框 + 符号(自适应字号) + 论点(自适应换行) + 右上角 § 徽标)
# 闭源边界节点(boundary)用深色填充 + 白字，视觉上与其余白底节点区分开。
for nid, lane, col, row, syms, claim, sec in NODES:
    x, y = NODE_XY[nid]
    is_boundary = nid == "boundary"
    fill = C_BOUNDARY if is_boundary else C_NODE_FILL
    title_color = "#ffffff" if is_boundary else C_NODE_TITLE
    sub_color = "#cbd5e1" if is_boundary else C_NODE_SUB
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H:.1f}" rx="11" '
             f'fill="{fill}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    sym_w_budget = NODE_W - 16
    sym_size = SYM_FONT
    while max(cjk_text_width(s, sym_size) for s in syms) > sym_w_budget and sym_size > 7.0:
        sym_size -= 0.2
    for si, s in enumerate(syms):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + SYM_TOP + si * SYM_LINE_H:.1f}" '
                 f'text-anchor="middle" font-family="sans-serif" font-size="{sym_size:.1f}" '
                 f'font-weight="bold" fill="{title_color}">{esc(s)}</text>')
    for ci, cline in enumerate(_CLAIM_LINES[nid]):
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + CLAIM_TOP + ci * 11.5:.1f}" '
                 f'text-anchor="middle" font-family="sans-serif" font-size="{CLAIM_FONT}" '
                 f'fill="{sub_color}">{esc(cline)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 6, y, sec)

# 底部阅读路线：复用列坐标 COLX，§ 徽标与图上节点对齐成竖向落点
L.append(f'<text x="14" y="{routes_top + 13:.1f}" font-family="sans-serif" font-size="10.4" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌；实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (rname, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="14" y="{ry + 3.3:.1f}" font-family="sans-serif" font-size="9.8" '
             f'fill="{C_NODE_TITLE}">{esc(rname)}</text>')
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
print(f"wrote {out}  ({w:.0f}x{h:.0f}, aspect {w / h:.2f}:1)")
