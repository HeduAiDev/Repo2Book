#!/usr/bin/env python3
"""第 35 章「本章地图」——DFlash 块扩散并行起草剖面图(primer 原理章)。

基于 .claude/skills/svg-diagram/references/example-chapter-map.py 模板改写(同
ch37-primer-dspark 手法):不可变机制(esc/cjk_text_width/badge 胶囊样式/入口绿-
出口橙-主线蓝配色/路线条实线蓝-虚线灰/legend 必须画)原样保留,只改 DATA 与
(为适配自然标题站牌文字长度不一)badge() 的宽度算法——按 cjk_text_width() 逐条
计算宽度,不用定长 BADGE_W 常量;另新增 symbol_lines() 换行辅助(仅两个超长
triton/precompute 符号需要 2 行,其余符号单行足够,不引入新颜色/新语义)。

本章为**自然标题章**(chapter.md 只有"一、二、三、四、五"与"小结",无 `## N.M`
编号)——按契约禁用 §N.M 徽标,站牌改用标题词本身:"块扩散并行起草"(对应
"## 一、块扩散并行起草：延迟为什么不随块变大")、"KV注入"(对应
"## 二、KV 注入：目标特征如何进入 draft 的每一层")、"交叉注意力"(对应
"## 三、交叉注意力：Q 来自 draft、K/V 由 [H_t; H_d] 拼接")、"训练:位置加权损失"
(对应"## 四、训练：随机锚点掩码与位置加权损失")、"接受率:树验证"(对应
"## 五、接受率与加速比：从单轨迹到树验证")。

本章额外的诚实约束(primer 章的落地边界):§一/§二/§三是**已落地的昇腾推理代码**
(AscendDflashProposer/precompute_and_store_context_kv/DFlashQwen3Attention 等
真实符号,均可在 narrative/chapter.md 正文逐字核到);§四(训练)与§五的 DDTree
树验证**无对应昇腾代码**——训练机制正文明写"训练机制无对应昇腾推理代码"
(dossier mechanisms[8].anchor_note),DDTree 是"DFlash 之上的后续论文工作…
树验证无昇腾代码,仅作延伸对照"(dossier mechanisms[10].anchor_note),昇腾当前
落地的是靠 max_query_tokens 钉死的单轨迹 vanilla 版。为让"落地到哪"这条落差
一眼可辨,复用 ch37 手法:在既有三色语义(绿入口/蓝主线/橙出口)之外,第四条
"仅论文侧延伸"语义**复用同一个 C_ROUTE_DIM 灰色**——节点虚线边框 + 调用边虚线
+ 路线虚线,三处统一,不引入模板未定义的新颜色。

节点预算 10(combine/set_inputs/build_fused_kv/kernel_expand/precompute/
draft_attn/weighted_loss/exit_node/ddtree = 9 主节点,鉴于 combine 为入口、
exit_node 为出口,共 9 个数据节点,＋隐含的调用方/返回上层接口桩)≤ 12。

设计要点(节点顺序按真实调用顺序摆列,不是按章节顺序硬凑):
- combine_hidden_states 在起草器调 set_inputs_first_pass **之前**由基座
  llm_base_proposer 调用(正文 L649-L654 摘录),是本图入口——它虽在正文里排在
  §二讲解,但代码时序上确实最先跑,站牌仍标"KV注入"(内容归属),不是"块扩散
  并行起草"(避免张冠李戴)。
- kernel_expand(copy_and_expand_dflash_inputs_kernel_single_grid)由
  set_inputs_first_pass 内部调用,同时写出 context 与 query 两段的
  positions/slot_mapping——它的输出(context_positions_buffer/
  context_slot_mapping_buffer)正是下一步 precompute 的入参,故
  kernel_expand→precompute 有真实数据依赖边(不是虚构的调用关系)。
- _build_fused_kv_buffers 是 precompute 内部"首次调用时"才会触发的懒初始化
  (`if not hasattr(self, "_num_attn_layers"): self._build_fused_kv_buffers()`),
  与 set_inputs_first_pass 并列在同一列(不同泳道)、边直接指向 precompute,
  不虚构一条它并不存在的"先于 set_inputs"的顺序关系。
- 非因果元数据改写(cad.causal=False 等)是 set_inputs_first_pass **同一个
  函数体的尾部**(正文 L129-L148,与头部 L63-L120 同一份 embed_excerpt 的两段),
  为免拆成两个几乎贴在一起的节点、也为控制节点预算,并入 set_inputs_first_pass
  节点的短语里说明,不单独占一个节点/一列。
- 仅论文侧延伸(训练/DDTree)分别用虚线灰边挂在它们真实共享的上游节点后面:
  weighted_loss 挂在 precompute 之后(训练与推理共用同一套 KV 注入通路,正文
  "训练与推理共用一套注入通路"一节原话),ddtree 挂在 draft_attn 之后(DDTree
  用的正是块扩散一次前向给出的逐位边际分布,该分布正来自 draft 各层前向)。

六项自查记录见文件末尾 [SELF-CHECK] 注释(渲染→Read PNG 亲眼看后如实记录)。
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版)——全角(ord>0x2E80)按
    1.0×size,半角按 0.58×size,求和。中英混排的图例/标签/站牌必须用这个,
    不能直接 0.58 * size * len(s)。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = [
    "①起草准备:块扩散并行起草(proposer,§一)",
    "②KV 注入融合(precompute,§二)",
    "③交叉注意力 + 出口(§三/§五)",
    "④仅论文侧延伸(训练/树验证,§四·§五,无昇腾代码)",
]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名(str 或 2 行 tuple),
#  一行短语, 站牌(自然标题词,禁用 §N.M), landed:True=已落地实线/False=仅论文侧虚线)
NODES = [
    ("combine",      1, 0, 0, "combine_hidden_states",
     "5层target隐藏态拼接→Wc投影出context特征", "KV注入", True),
    ("set_inputs",   0, 1, 0, "set_inputs_first_pass",
     "存context;摆考卷;改cad非因果", "块扩散并行起草", True),
    ("build_fused",  1, 1, 0, "_build_fused_kv_buffers",
     "懒初始化:堆叠K/V权重成_fused_kv_weight", "KV注入", True),
    ("kernel_expand", 0, 2, 0, ("copy_and_expand_dflash_inputs", "_kernel_single_grid"),
     "摆考卷:context+query(1bonus+N mask)", "块扩散并行起草", True),
    ("precompute",   1, 3, 0, ("precompute_and_store", "_context_kv"),
     "hidden_norm→融合GEMM出K/V→写cache", "KV注入", True),
    ("draft_attn",   2, 4, 0, "DFlashQwen3Attention.forward",
     "Q来自draft;K/V=cache context+query", "交叉注意力", True),
    ("weighted_loss", 3, 4, 0, "Eq.(4)",
     "随机锚点=bonus;w_k指数衰减,早位置更贵", "训练:位置加权损失", False),
    ("exit_node",    2, 5, 0, "max_query_tokens",
     "单轨迹长度=1+num_speculative_tokens", "接受率:树验证", True),
    ("ddtree",       3, 5, 0, "DDTree best-first",
     "同一次前向边际分布建树,预算B内验证前缀", "接受率:树验证", False),
]
NODE_BY_ID = {n[0]: n for n in NODES}

EDGES = [  # (src_id, dst_id, style) —— style: "solid"=已落地主线蓝, "dashed"=仅论文侧灰虚线
    ("combine", "set_inputs", "solid"),
    ("set_inputs", "kernel_expand", "solid"),
    ("build_fused", "precompute", "solid"),
    ("kernel_expand", "precompute", "solid"),
    ("precompute", "draft_attn", "solid"),
    ("precompute", "weighted_loss", "dashed"),
    ("draft_attn", "exit_node", "solid"),
    ("draft_attn", "ddtree", "dashed"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序,列号须严格递增, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("主干(§一+§二):块扩散并行起草→KV注入", [(1, "块扩散并行起草"), (3, "KV注入")], True),
    ("细节延伸(§三+§四+§五):交叉注意力→接受率/树验证", [(4, "交叉注意力"), (5, "接受率:树验证")], False),
]
LEGEND = [("#22c55e", "入口:上一轮验证/起草回合触发"), ("#3b82f6", "章内主线:已落地调用/数据流"),
          ("#f97316", "出口:draft tokens 交给 target 验证"), ("#94a3b8", "虚线:仅论文侧延伸,无对应昇腾代码")]
TITLE = "第 35 章 · DFlash 块扩散起草 + KV 注入剖面(推理主链 §一-§三 · 训练/树验证仅论文侧延伸 §四·§五)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff", "#f8fafc", "#fef9f2"]  # 泳道背景交替,仅装饰,非语义色(第四道浅暖色呼应"仅论文侧")
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"  # 复用同一灰色表达"仅论文侧/次要"语义(节点虚线边、调用边虚线、路线虚线三处统一)

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 210, 68
COL_GAP, ROW_GAP = 14, 20
EDGE_MARGIN, STUB_W, STUB_H = 10, 50, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 16  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 24, 12
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 34, 26, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 44
BADGE_H = 20  # 宽度改按文字动态算(见 badge_w),站牌是变长中文词,不能用定长常量

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

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge_w(text):
    """站牌是变长中文词(如"训练:位置加权损失"9 字 vs "KV注入"混排),不能用
    定长 BADGE_W——按 cjk_text_width 估算文字宽度再加左右各 8px 内边距。"""
    return cjk_text_width(text, 11) + 16


def badge(cx, cy, text):
    """站牌胶囊,居中挂在 (cx,cy) —— 节点用它贴右上角,路线图例用它居中挂线上。
    宽度按文字动态算,但形状/配色/挂法与全书统一模板一致。"""
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
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN), ("Dim", C_ROUTE_DIM))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13.5" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 14
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 11}" width="14" height="14" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 20}" y="{_ly}" font-family="sans-serif" font-size="10.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 20 + cjk_text_width(label, 10.5) + 22

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

# 入口/出口接口桩(给入口/出口箭头一个可附着的框,兼表达"调用方在画布外")
ex, ey = NODE_XY["combine"]; ey += NODE_H / 2
xx, xy = NODE_XY["exit_node"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.3"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.3"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 4:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="10.5" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(先画边再画节点盖住端点毛刺):实线主线蓝 / 虚线仅论文侧灰
# 多条边汇入同一节点时,终点 y 各偏移(间距 16px),否则重合的终点看不出"汇合"。
_dst_total = {}
for _, dst, _style in EDGES:
    _dst_total[dst] = _dst_total.get(dst, 0) + 1
_dst_seen = {}
for src, dst, style in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    n = _dst_total[dst]
    i = _dst_seen.get(dst, 0)
    _dst_seen[dst] = i + 1
    y_offset = (i - (n - 1) / 2) * 16 if n > 1 else 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    if style == "solid":
        stroke, marker, dash = C_MAIN, "mMain", ""
    else:
        stroke, marker, dash = C_ROUTE_DIM, "mDim", ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{stroke}" stroke-width="2" marker-end="url(#{marker})"{dash}/>')

# 节点(圆角框 + 真实符号名(单行或 2 行)+ 一行短语 + 右上角站牌):
# 已落地=实线深色边框,仅论文侧=虚线灰边框
for nid, lane, col, row, symbol, phrase, station, landed in NODES:
    x, y = NODE_XY[nid]
    stroke = C_NODE_STROKE if landed else C_ROUTE_DIM
    dash = '' if landed else ' stroke-dasharray="5,3"'
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{stroke}" stroke-width="1.5"{dash}/>')
    sym_lines = symbol if isinstance(symbol, tuple) else (symbol,)
    if len(sym_lines) == 1:
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.36:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="11" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
        phrase_y = y + NODE_H * 0.66
    else:
        # 超长符号(如融合 triton kernel 名)换成 2 行,字号降到 9.5 避免溢出节点框。
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.28:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="9.5" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.46:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="9.5" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[1])}</text>')
        phrase_y = y + NODE_H * 0.72
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{phrase_y:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="9" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    bw = badge_w(station)
    L += badge(x + NODE_W - bw / 2 + 8, y, station)

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌;实线蓝=推荐 / 虚线灰=仅论文侧次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="11.5" '
              f'fill="{C_NODE_TITLE}">{esc(name)}</text>')
    x_first = COLX[stops[0][0]] + NODE_W / 2
    x_last = COLX[stops[-1][0]] + NODE_W / 2
    dash = '' if hi else ' stroke-dasharray="6,4"'
    L.append(f'<line x1="{x_first:.1f}" y1="{ry:.1f}" x2="{x_last:.1f}" y2="{ry:.1f}" '
              f'stroke="{C_MAIN if hi else C_ROUTE_DIM}" stroke-width="{3 if hi else 1.5}"{dash}/>')
    for col, station in stops:
        L += badge(COLX[col] + NODE_W / 2, ry, station)

L.append('</svg>')
out = Path(__file__).with_name("chapter-map.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}: {w:.0f}x{h:.0f}")

# [SELF-CHECK](渲染→Read PNG 亲眼看后如实记录)
#   见渲染后同轮记录(下方由脚本外的 illustrator 流程回填,首轮如有问题会在此追加
#   [FIX-ROUND-2] 段落)。
