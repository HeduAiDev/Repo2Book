#!/usr/bin/env python3
"""ch14「本章地图」——compile() 驱动主循环的源码剖面图
(python/triton/compiler/compiler.py + python/triton/backends/compiler.py
 + third_party/nvidia/backend/compiler.py)。

本章是自然标题章(chapter.md 无 `## N.M` 编号,标题用 §1..§8 单层记号 + 自然语言,
如"§3 内容寻址的编译器身份：triton_key（杠杆①）")——按契约禁用 §N.M 徽标(带小数点
的两级编号)。本章的 §N 是单层编号、字面就出现在标题文本里(不是虚构的子章节),
站牌沿用这套章节自己的记号(如"§3"),不额外发明新关键词摘要——与 ch11/ch12
(完全无编号,站牌改用关键词摘要)是同一条"自然标题"规则下的两种合法呈现。

剖面(三条泳道):
  ① 两扇入口(上):ASTSource(@jit 源码)/ IRSource(.ttgir 等文件)——两者都喂给
     主循环的"造起点"节点 src.make_ir(...)。
  ② 驱动主循环(中,主脊):compile(src, target, options) 入口 → make_backend
     选唯一后端 → triton_key() 拼内容寻址缓存键(杠杆①) →(未命中才展开:橙棕)
     add_stages 填 stages → first_stage 定起步级(杠杆②) → src.make_ir 造起点
     → compile_ir 循环逐级降级落盘→写回 CompiledKernel。缓存命中时从 triton_key
     节点直接弧形跳到出口,不展开橙棕四个节点(不跑一道 pass)。
  ③ 后端契约(下):BaseBackend 抽象契约 → CUDABackend.add_stages 具体样例,
     喂入②的 add_stages 节点(灰虚线,标"契约实现",非运行时数据流)。

配色:绿 #22c55e=入口(被 run() 调用,ch11 慢路径交棒);蓝 #3b82f6=主线调用边
(含命中跳过弧)/常规节点;橙棕 #b45309(填充 #fef3c7)=仅未命中(cache miss)才
展开的降级链节点;灰 #94a3b8 虚线=契约→具体实现的填充关系(非运行时数据流);
橙 #f97316=出口(返回 run(),交回 CompiledKernel)。

模板:.claude/skills/svg-diagram/references/example-chapter-map.py 的不可变视觉语言
(徽标胶囊/入口绿-出口橙-主线蓝/cjk_text_width)+ ch11/ch12 chapter-map.py 的
fit_size/wrap_symbol(长符号自适应换行)、多泳道 ARC_RISE 按泳道独立取值、
skip/branch 边画法照搬;新增 down/up 两种跨泳道竖向边(entries→make_ir,
contract→add_stages)。

六项自查(渲染→Read PNG 亲眼看后如实记录):见 figure-manifest.json 该图 selfcheck。

用法:python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算:全角(ord>0x2E80)按 1.0×size,半角按 0.58×size。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def wrap_symbol(text, max_w, sizes):
    """符号名/短语较长时的通用换行:先试单行从大到小的字号;仍塞不下,在
    ' '/'_'/'('/',' 边界二分成两行(挑一个让两行里"更长的那行"最短的切点),
    用最小字号。返回 (lines, size)。"""
    for size in sizes:
        if cjk_text_width(text, size) <= max_w:
            return [text], size
    size = sizes[-1]
    candidates = ([i + 1 for i, c in enumerate(text) if c == ' ']
                  + [i + 1 for i, c in enumerate(text) if c == '_']
                  + [i for i, c in enumerate(text) if c == '(']
                  # 中文标点后一般不跟空格("下标,IRSource"),切点取逗号后一位
                  # (紧跟逗号),不能像英文习惯("a, b")那样再 +1——否则会切进
                  # 紧邻逗号的下一个词内部(如切进 "IRSource" 中间变 "I"/"RSource")
                  # [FIX-ROUND-2:原 +2 撞见无空格中文逗号导致英文标识符被腰斩]
                  + [i + 1 for i, c in enumerate(text) if c == ','])
    if not candidates:
        candidates = [len(text) // 2]
    best = None
    for idx in candidates:
        if idx <= 0 or idx >= len(text):
            continue
        a, b = text[:idx], text[idx:]
        w = max(cjk_text_width(a, size), cjk_text_width(b, size))
        if best is None or w < best[0]:
            best = (w, a, b)
    if best is None:
        return [text], size
    return [best[1], best[2]], size


# ---------------- DATA(可变:本章数据) ----------------
LANES = [
    "两扇入口:ASTSource(源码) / IRSource(IR 文件,杠杆②)",
    "驱动主循环:compile()——选后端→缓存键→填 stages→造起点→逐级降级落盘",
    "后端契约:CUDABackend 具体样例,紧邻 add_stages / BaseBackend 抽象基类",
]

FONT_SIZES = (12.5, 11.5, 10.5, 9.5, 8.5)

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌(本章自带的 §N,
#  非虚构 §N.M), is_branch: 是否"仅未命中(cache miss)才展开")
NODES = [
    # ② 驱动主循环(主脊,行 0)
    ("entry", 1, 0, 0, "compile(src, target, options)",
     "定 target,make_backend 选唯一后端", "§2", False),
    ("key", 1, 1, 0, "triton_key()",
     "拼缓存键:五段乘性组成,杠杆①", "§3", False),
    ("stages", 1, 2, 0, "add_stages",
     "填出 ir_name→pass 有序字典", "§4", True),
    ("first", 1, 3, 0, "first_stage",
     "= src.ext 下标,IRSource 再 +1,杠杆②", "§7", True),
    ("makeir", 1, 4, 0, "src.make_ir(...)",
     "两扇门在此合流,造起点 module", "§6", True),
    ("loop", 1, 5, 0, "compile_ir(module, metadata)",
     "逐级降级+落盘,返回 CompiledKernel(...)", "§4", True),
    # ① 两扇入口(上,同一行并排:IRSource 对齐 first_stage 所在列 3——
    #   呼应它俩的 +1 关系;ASTSource 对齐 make_ir 所在列 4——直下不绕行。
    #   并排而非上下堆叠,是为了不让上面那个节点挡住下面那个节点通向
    #   make_ir 的竖直边[FIX-ROUND-2:原堆叠布局导致 ASTSource→make_ir 的边
    #   被 IRSource 的框挡住,边标签"跑前端..."被截断——已改并排])
    ("ir", 0, 3, 0, "IRSource",
     "从 IR 文件进,绕过前端(杠杆②)", "§6", False),
    ("ast", 0, 4, 0, "ASTSource",
     "从 @jit 源码进,ext 固定 ttir", "§6", False),
    # ③ 后端契约(下,列对齐 stages 所在列 2;CUDABackend 紧邻 add_stages 在
    #   row0、BaseBackend 在 row1——同样是为了不让 row1 的节点挡住 row0
    #   通向 add_stages 的边[FIX-ROUND-2:原顺序 base=row0/cuda=row1 时,
    #   cuda→add_stages 的边要穿过 BaseBackend 的框,箭头视觉上像是从
    #   BaseBackend 发出而非 CUDABackend——已互换行序])
    ("cuda", 2, 2, 0, "CUDABackend.add_stages",
     "五行填满 ttir→ttgir→llir→ptx→cubin", "§5", False),
    ("base", 2, 2, 1, "BaseBackend",
     "抽象契约:supports_target/hash/add_stages/...", "§5", False),
]
NODE_BY_ID = {n[0]: n for n in NODES}

# (src_id, dst_id, 边样式: "main"=蓝实线常规调用 / "skip"=蓝实线命中直达(跨过
#  未命中才展开的节点) / "branch"=橙虚线仅未命中才执行 / "down"=蓝实线自上方
#  泳道垂直喂入(真实数据流) / "up"=灰虚线自下方泳道垂直喂入(契约→实现,
#  非运行时数据流), 边上小字标注或 None, x_off:多条边汇入同列时的水平错开量)
EDGES = [
    ("entry", "key", "main", None, 0),
    ("key", "stages", "branch", "未命中", 0),
    ("stages", "first", "branch", None, 0),
    ("first", "makeir", "branch", None, 0),
    ("makeir", "loop", "branch", None, 0),
    ("key", "loop", "skip", "命中直达,不跑一道 pass", 0),
    ("ast", "makeir", "down", "跑前端 ast_to_ttir", 15),
    ("ir", "makeir", "down", "parse_mlir_module,绕过前端", -15),
    ("cuda", "stages", "up", "填出契约", 0),
    ("base", "cuda", "up", "NVIDIA 实现", 0),
]

# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("主干:按 §1→§4 顺序读完整驱动循环",
     [(0, "§2"), (1, "§3"), (2, "§4"), (4, "§6"), (5, "§4")], True),
    ("杠杆①:只想知道缓存何时失效,跳 §3",
     [(1, "§3")], True),
    ("杠杆②:只想拿一份 .ttgir 做 IR 级实验,跳 §7",
     [(3, "§7")], True),
    ("后端接缝:新卡怎么接,看 §5",
     [(2, "§5")], False),
]
LEGEND = [
    ("#22c55e", "入口:被 run() 调用(ch11 慢路径交棒)"),
    ("#3b82f6", "章内主线调用边(含命中跳过弧) / 两扇入口的真实数据流"),
    ("#b45309", "橙棕:仅未命中(cache miss)才展开的降级链节点"),
    ("#94a3b8", "灰虚线:契约→具体后端实现的填充关系(非运行时数据流)"),
    ("#f97316", "出口:返回 run(),交回 CompiledKernel"),
]
TITLE = "第 14 章 · compile() 驱动壳剖面——选后端→缓存键→填 stages→造起点→逐级降级"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc"]
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"
# 本章沿用 ch11/ch12 已建立的 miss-only 橙棕色;新增一条灰虚线契约边(复用 C_ROUTE_DIM)
C_BRANCH_FILL, C_BRANCH_STROKE = "#fef3c7", "#b45309"
C_CONTRACT_EDGE = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 168, 92
COL_GAP, ROW_GAP = 30, 20
EDGE_MARGIN, STUB_W, STUB_H = 10, 54, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 24
LANE_LABEL_H, BAND_PAD = 22, 14
ARC_RISE = [0, 46, 0]  # 每条泳道各自的跳过弧顶部预留空间;仅②(key→loop)需要
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 46, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 42
BADGE_W, BADGE_H = 40, 20

n_cols = max(n[2] for n in NODES) + 1
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_lane = [0] * len(LANES)
for _id, lane, col, row, *_ in NODES:
    rows_per_lane[lane] = max(rows_per_lane[lane], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + ARC_RISE[i] + r * NODE_H + max(0, r - 1) * ROW_GAP
          for i, r in enumerate(rows_per_lane)]
band_top, _cum = [], TOP_PAD + TITLE_H + LEGEND_H
for bh in band_h:
    band_top.append(_cum)
    _cum += bh
lanes_bottom = _cum

NODE_XY = {}
for nid, lane, col, row, *_ in NODES:
    x = COLX[col]
    y = band_top[lane] + LANE_LABEL_H + BAND_PAD + ARC_RISE[lane] + row * (NODE_H + ROW_GAP)
    NODE_XY[nid] = (x, y)

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge_w(text):
    """站牌胶囊宽度——按文字自适应,不用固定 BADGE_W 截断。"""
    return max(BADGE_W, cjk_text_width(text, 11) + 14)


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy)——本章自带的 §N 单层记号(非虚构 §N.M)。"""
    bw = badge_w(text)
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
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN),
                        ("Branch", C_BRANCH_STROKE), ("Contract", C_CONTRACT_EDGE))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="15" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(5 色,两行摆放避免单行过宽)
_legend_rows = [LEGEND[:3], LEGEND[3:]]
_ly = TOP_PAD + TITLE_H + 14
for row_items in _legend_rows:
    _lx = PAD_L
    for color, label in row_items:
        L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
        L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="11" '
                 f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
        _lx += 20 + cjk_text_width(label, 11) + 26
    _ly += 18

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="16" y="{band_top[i] + LANE_LABEL_H - 4:.1f}" font-family="sans-serif" '
             f'font-size="12" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩:入口挂主脊最左(compile 被 run() 调用),出口挂主脊最右(返回 run())
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["loop"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#166534">{esc("run()")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10" font-weight="bold" fill="#9a3412">{esc("run()")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边:main/branch/skip 走主脊同泳道(沿用 ch11/ch12 画法);down/up 是本章新增
# 的跨泳道竖向边(入口→造起点 是真实数据流用主线蓝;契约→add_stages 是"这是个
# 样例"的说明关系,用灰虚线,不与运行时数据流同色以免误读)。
for src, dst, kind, label, x_off in EDGES:
    xs_, ys_ = NODE_XY[src]; xd, yd = NODE_XY[dst]
    if kind == "skip":
        src_lane = NODE_BY_ID[src][1]
        rise = ARC_RISE[src_lane]
        p1 = (xs_ + NODE_W / 2, ys_)
        p2 = (xd + NODE_W / 2, yd)
        arc_y = ys_ - rise + 8
        path = f'M {p1[0]:.1f},{p1[1]:.1f} C {p1[0]:.1f},{arc_y:.1f} {p2[0]:.1f},{arc_y:.1f} {p2[0]:.1f},{p2[1]:.1f}'
        L.append(f'<path d="{path}" fill="none" stroke="{C_MAIN}" stroke-width="2.2" '
                  f'marker-end="url(#mMain)"/>')
        if label:
            L.append(f'<text x="{(p1[0] + p2[0]) / 2:.1f}" y="{arc_y - 6:.1f}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="10" font-weight="bold" '
                     f'fill="{C_MAIN}">{esc(label)}</text>')
    elif kind == "branch":
        p1 = (xs_ + NODE_W, ys_ + NODE_H / 2 - 10)
        p2 = (xd, yd + NODE_H / 2 - 10)
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_BRANCH_STROKE}" stroke-width="2" stroke-dasharray="6,4" '
                  f'marker-end="url(#mBranch)"/>')
        if label:
            L.append(f'<text x="{(p1[0] + p2[0]) / 2:.1f}" y="{p1[1] - 6:.1f}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="9" fill="{C_BRANCH_STROKE}">{esc(label)}</text>')
    elif kind == "down":
        p1 = (xs_ + NODE_W / 2 + x_off, ys_ + NODE_H)
        p2 = (xd + NODE_W / 2 + x_off, yd)
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
        if label:
            # 标在源节点正下方(泳道①自己的空当里),避开泳道②顶部命中跳过弧
            # 的标签区[FIX-ROUND-2:原取竖直中点与跳过弧标签同高,两行字互相
            # 压叠——已挪到紧贴源节点下沿]
            L.append(f'<text x="{p1[0] + (10 if x_off >= 0 else -10):.1f}" '
                     f'y="{p1[1] + 14:.1f}" text-anchor="{"start" if x_off >= 0 else "end"}" '
                     f'font-family="sans-serif" font-size="9" fill="{C_MAIN}">{esc(label)}</text>')
    elif kind == "up":
        p1 = (xs_ + NODE_W / 2 + x_off, ys_)
        p2 = (xd + NODE_W / 2 + x_off, yd + NODE_H)
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_CONTRACT_EDGE}" stroke-width="1.8" stroke-dasharray="5,4" '
                  f'marker-end="url(#mContract)"/>')
        if label:
            L.append(f'<text x="{p1[0] + 10:.1f}" y="{(p1[1] + p2[1]) / 2:.1f}" '
                     f'font-family="sans-serif" font-size="9" fill="{C_CONTRACT_EDGE}">{esc(label)}</text>')
    else:  # main
        p1 = (xs_ + NODE_W, ys_ + NODE_H / 2 + 10)
        p2 = (xd, yd + NODE_H / 2 + 10)
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')
        if label:
            L.append(f'<text x="{(p1[0] + p2[0]) / 2:.1f}" y="{p1[1] - 6:.1f}" text-anchor="middle" '
                     f'font-family="sans-serif" font-size="9.5" fill="{C_MAIN}">{esc(label)}</text>')

# 节点(圆角框 + 真实符号名(必要时自动换行/缩字号) + 一行短语 + 右上角 §N 站牌)
SYM_MAXW = NODE_W - 16
PHR_MAXW = NODE_W - 14
for nid, lane, col, row, symbol, phrase, sec, is_branch in NODES:
    x, y = NODE_XY[nid]
    fill, stroke = C_NODE_FILL, C_NODE_STROKE
    if is_branch:
        fill, stroke = C_BRANCH_FILL, C_BRANCH_STROKE
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    sym_lines, sym_size = wrap_symbol(symbol, SYM_MAXW, FONT_SIZES)
    cx = x + NODE_W / 2
    if len(sym_lines) == 1:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.30:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
    else:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.22:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.22 + sym_size + 2:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{sym_size}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[1])}</text>')
    phr_lines, phr_size = wrap_symbol(phrase, PHR_MAXW, (9.5, 9, 8.5, 8))
    py0 = y + NODE_H * 0.56
    for k, pl in enumerate(phr_lines):
        L.append(f'<text x="{cx:.1f}" y="{py0 + k * (phr_size + 3):.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{phr_size}" fill="{C_NODE_SUB}">{esc(pl)}</text>')
    bw = badge_w(sec)
    L += badge(x + NODE_W - bw / 2 + 8, y, sec)

# 底部阅读路线:复用列坐标 COLX,§N 站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌,对应正文同号小节;实线蓝=推荐 / 虚线灰=次要跳读)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    row_top = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H
    name_y = row_top + 13
    ry = row_top + ROUTE_ROW_H - 16
    L.append(f'<text x="16" y="{name_y:.1f}" font-family="sans-serif" font-size="11" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first = COLX[stops[0][0]] + NODE_W / 2
    x_last = COLX[stops[-1][0]] + NODE_W / 2
    dash = '' if hi else ' stroke-dasharray="6,4"'
    if x_first != x_last:
        L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
                  f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for col, sec in stops:
        L += badge(COLX[col] + NODE_W / 2, ry, sec)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  ({w}x{h}, aspect {w / h:.2f}:1)")
