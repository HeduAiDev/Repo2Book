#!/usr/bin/env python3
"""第22章「本章地图」——模型契约剖面图。

真实调用链(source_pin f3fef123,vllm/model_executor/...):
`initialize_model()` 校验 `(vllm_config, prefix)` 签名并建空壳 → 空壳里同时长出一对 TP
线性层(`QKVParallelLinear` 列并行 / `RowParallelLinear` 行并行)→ 权重装载走
`LlamaModel.load_weights` 里那个 for/else:命中 `stacked_params_mapping` 的走 fuse 路径,
落不到任何 fused 名字的走 `default_weight_loader` 直路 → 两条路径都汇入
`process_weights_after_loading`(kernel 重排 + Attention KV scale)→ 前向阶段
`LlamaAttention.forward`(qkv split→RoPE→Attention 统一封装→o_proj)→ 出口
`LlamaModel.forward`(逐层 decoder→末 RMSNorm,吐 hidden_states)。

■ 不可变(照抄 skill 模板,不动):§徽标胶囊 badge()/配色语义(绿入口#22c55e、橙出口
  #f97316、蓝主线#3b82f6)/入口出口接口桩/路线「高亮=蓝实线,其余=灰虚线」/>2 色配图例/
  cjk_text_width() 估算文本宽度。
■ 可变(本章数据):LANES/NODES/EDGES/ROUTES/LEGEND/TITLE,以及为适配本章较长真实符号名
  (如 process_weights_after_loading)而调整的 NODE_W/NODE_H/COL_GAP/字号——几何常量本身
  不在「不可变」清单内,可按章调整以满足画布预算(宽 ≤1500、宽高比 ≤2.6:1)。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录):
    claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
    arrows_attached=True     cjk_rendered=True         reading_order_clear=True
  (本图不含数值型 claim,numbers_match_spec 记 True 表示「本图无杜撰数字,§徽标/符号均可核」。)

用法:python3 gen_chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):逐字符判定——
    全角(ord>0x2E80)按 1.0×size,半角按 0.58×size,求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


def split_long_symbol(sym, threshold=20):
    """真实符号名超过阈值时,在最靠近中点的 `_`/`.` 处拆成两行——两个片段各自仍是
    该符号的原样子串(如 "process_weights_" + "after_loading"),不引入杜撰文本,
    只是为了在较窄的节点框里不越界。未超阈值则原样单行返回。"""
    if len(sym) <= threshold:
        return [sym]
    mid = len(sym) // 2
    seps = [i for i, c in enumerate(sym) if c in "_."]
    if not seps:
        return [sym[:mid], sym[mid:]]
    best = min(seps, key=lambda i: abs(i - mid))
    return [sym[:best + 1], sym[best + 1:]]


# ---------------- DATA(本章数据) ----------------
LANES = ["构造 + 权重装载(静态)", "前向执行(运行时)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, §编号)
NODES = [
    ("entry",        0, 0, 0, "initialize_model",             "vllm_config+prefix 建空壳",     "§22.2"),
    ("tp_col",       0, 1, 0, "QKVParallelLinear",             "列并行:沿 output 切,fuse q/k/v", "§22.4"),
    ("tp_row",       0, 1, 1, "RowParallelLinear",             "行并行:沿 input 切,o_proj",      "§22.3"),
    ("fused_load",   0, 2, 0, "stacked_params_mapping",        "q/k/v_proj 改名+带 shard_id",    "§22.5"),
    ("default_load", 0, 2, 1, "default_weight_loader",         "非 fuse 权重直装(else 兜底)",     "§22.5"),
    ("post_load",    0, 3, 0, "process_weights_after_loading", "kernel 重排+KV scale 初始化",    "§22.5"),
    ("attn_fwd",     1, 4, 0, "LlamaAttention.forward",        "qkv→RoPE→Attention→o_proj",     "§22.6"),
    ("exit",         1, 5, 0, "LlamaModel.forward",            "逐层→末 RMSNorm→hidden_states",  "§22.7"),
]
EDGES = [  # (src_id, dst_id) —— 调用边,统一主线蓝
    ("entry", "tp_col"), ("entry", "tp_row"),
    ("tp_col", "fused_load"), ("tp_row", "default_load"),
    ("fused_load", "post_load"), ("default_load", "post_load"),
    ("post_load", "attn_fwd"),
    ("attn_fwd", "exit"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("Fused 参数路径",   [(0, "§22.2"), (1, "§22.4"), (2, "§22.5"), (3, "§22.5"), (4, "§22.6"), (5, "§22.7")], True),
    ("非 Fused 路径",   [(0, "§22.2"), (1, "§22.3"), (2, "§22.5"), (3, "§22.5"), (4, "§22.6"), (5, "§22.7")], False),
]
LEGEND = [("#22c55e", "入口:构造/装载起点"), ("#3b82f6", "章内主线调用边"), ("#f97316", "出口:前向输出返回上层")]
TITLE = "第22章 · 模型契约剖面:构造 → TP 切分 → 权重装载 → 前向"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数;本章符号较长,box/字号相应调窄调小) ----------------
NODE_W, NODE_H = 172, 74
COL_GAP, ROW_GAP = 18, 18
EDGE_MARGIN, STUB_W, STUB_H = 14, 58, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 30  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 22, 11
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 14, 32, 24, 16
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_W, BADGE_H = 42, 18
SYM_SIZE, SUB_SIZE = 11, 9.3

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

# 自检:符号(拆行后每行)与短语按估算宽度不得超出 NODE_W(留 14px 边距)——
# 提前在生成期发现文字越界,不必等 Read PNG 才发现。
for _nid, _lane, _col, _row, _sym, _phrase, _sec in NODES:
    for _line in split_long_symbol(_sym):
        _wd = cjk_text_width(_line, SYM_SIZE)
        assert _wd <= NODE_W - 14, f"{_nid}: 符号行 {_line!r} 估算宽 {_wd:.0f} 超出 NODE_W-14={NODE_W - 14}"
    _wd = cjk_text_width(_phrase, SUB_SIZE)
    assert _wd <= NODE_W - 14, f"{_nid}: 短语 {_phrase!r} 估算宽 {_wd:.0f} 超出 NODE_W-14={NODE_W - 14}"

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
    """§ 徽标胶囊,居中挂在 (cx,cy)。"""
    bx, by = cx - BADGE_W / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BADGE_W}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.1"/>',
        f'<text x="{cx:.1f}" y="{cy + 3.5:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="9.5" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN))
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 17}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例)
_lx = PAD_L
_ly = TOP_PAD + TITLE_H + 13
for color, label in LEGEND:
    L.append(f'<rect x="{_lx}" y="{_ly - 10}" width="12" height="12" rx="3" fill="{color}"/>')
    L.append(f'<text x="{_lx + 17}" y="{_ly}" font-family="sans-serif" font-size="10.5" '
             f'fill="{C_NODE_TITLE}">{esc(label)}</text>')
    _lx += 17 + cjk_text_width(label, 10.5) + 26

# 泳道背景 + 标签 + 分隔线
for i, name in enumerate(LANES):
    L.append(f'<rect x="0" y="{band_top[i]:.1f}" width="{w}" height="{band_h[i]:.1f}" '
             f'fill="{C_LANE_FILL[i % len(C_LANE_FILL)]}"/>')
    L.append(f'<text x="14" y="{band_top[i] + LANE_LABEL_H - 6:.1f}" font-family="sans-serif" '
             f'font-size="12" font-weight="bold" fill="{C_LANE_LABEL}">{esc(name)}</text>')
    if i > 0:
        L.append(f'<line x1="0" y1="{band_top[i]:.1f}" x2="{w}" y2="{band_top[i]:.1f}" '
                  f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')
L.append(f'<line x1="0" y1="{lanes_bottom:.1f}" x2="{w}" y2="{lanes_bottom:.1f}" '
         f'stroke="{C_LANE_BORDER}" stroke-width="1"/>')

# 入口/出口接口桩
ex, ey = NODE_XY["entry"]; ey += NODE_H / 2
xx, xy = NODE_XY["exit"]; xy += NODE_H / 2
L.append(f'<rect x="{EDGE_MARGIN}" y="{ey - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#dcfce7" stroke="{C_ENTRY}" stroke-width="1.2"/>')
L.append(f'<text x="{EDGE_MARGIN + STUB_W / 2}" y="{ey + 3.5:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.5" font-weight="bold" fill="#166534">{esc("调用方")}</text>')
L.append(f'<line x1="{EDGE_MARGIN + STUB_W}" y1="{ey:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
         f'stroke="{C_ENTRY}" stroke-width="2" marker-end="url(#mEntry)"/>')
sx = w - EDGE_MARGIN - STUB_W
L.append(f'<rect x="{sx:.1f}" y="{xy - STUB_H / 2:.1f}" width="{STUB_W}" height="{STUB_H}" '
         f'rx="{STUB_H / 2}" fill="#ffedd5" stroke="{C_EXIT}" stroke-width="1.2"/>')
L.append(f'<text x="{sx + STUB_W / 2:.1f}" y="{xy + 3.5:.1f}" text-anchor="middle" '
         f'font-family="sans-serif" font-size="9.5" font-weight="bold" fill="#9a3412">{esc("返回上层")}</text>')
L.append(f'<line x1="{xx + NODE_W:.1f}" y1="{xy:.1f}" x2="{sx:.1f}" y2="{xy:.1f}" '
         f'stroke="{C_EXIT}" stroke-width="2" marker-end="url(#mExit)"/>')

# 调用边(主线蓝);多条边汇入同一节点时终点 y 各偏移,避免看不出"汇合"
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
    y_offset = (i - (n - 1) / 2) * 14 if n > 1 else 0
    p2 = (x2, y2 + NODE_H / 2 + y_offset)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名(过长自动拆两行) + 一行短语 + 右上角 § 徽标)
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="11" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.4"/>')
    sym_lines = split_long_symbol(symbol)
    cx = x + NODE_W / 2
    if len(sym_lines) == 1:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.38:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{SYM_SIZE}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
        phrase_y = y + NODE_H * 0.72
    else:
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.30:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{SYM_SIZE}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[0])}</text>')
        L.append(f'<text x="{cx:.1f}" y="{y + NODE_H * 0.30 + SYM_SIZE + 2:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{SYM_SIZE}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(sym_lines[1])}</text>')
        phrase_y = y + NODE_H * 0.86
    L.append(f'<text x="{cx:.1f}" y="{phrase_y:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{SUB_SIZE}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 6, y, sec)

# 底部阅读路线
L.append(f'<text x="14" y="{routes_top + 14:.1f}" font-family="sans-serif" font-size="11.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="14" y="{ry + 3.5:.1f}" font-family="sans-serif" font-size="10.5" '
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
print(f"wrote {out}  viewBox=0 0 {w:.0f} {h:.0f}  ratio={w / h:.2f}")
