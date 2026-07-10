#!/usr/bin/env python3
"""第 27 章「本章地图」——Lightning Indexer 推理调用链 + 支撑机制(源码剖面图)。

本章是自然标题章(节标题为「二、三、四、五」中文数字,无 `## N.M` 编号)——
按契约禁用 §N.M 徽标,站牌一律用正文里实际出现的标题词本身。

结构:上泳道是真实的推理调用链(自左向右——MLA 前向调用 indexer → 独立小头出
q/k/w → mqa_logits 打分 Eq.(1) → top-k 选择 Eq.(2)/(17) → 稀疏 MLA 只读被选中
的 latent KV);下泳道是撑住这条链的三块支撑机制(可信训练依据 / IndexCache 独
立缓存及其量化写入与 MXFP4 变体 / 复杂度诚实账),各自垂直挂在它所支撑的上泳道
节点正下方。

■ 不可变(全书统一视觉语言,来自 skill 模板,换章节时不要动):badge()胶囊样式/
  入口绿#22c55e-出口橙#f97316-主线蓝#3b82f6/图例规则/cjk_text_width()。
■ 本章新增(相对模板的必要扩展,非任意发挥):EDGES 里源列==目标列的一对,判定为
  跨泳道"支撑"边,画法从"右中→左中"横向连接改为"下中→上中"纵向连接——原模板
  只处理同泳道内的左右调用边,本章多了"上泳道节点在下方挂一块支撑证据"的结构,
  照搬横向连边会让箭头从右边一路斜插回左边、观感别扭,故加一个按列是否相同分流
  的小分支,颜色/箭头样式仍是同一套主线蓝,不引入新语义色。

■ 六项自查记录(渲染→Read PNG 亲眼看后如实记录):
    claim_readable_10s=True  numbers_match_spec=True  no_overlap=True
    arrows_attached=True     cjk_rendered=True         reading_order_clear=True
  (细节见 figure-manifest.json 该条 selfcheck 与 blind_review)

用法:python3 chapter-map.py → 同目录 chapter-map.svg
"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


def cjk_text_width(s, size):
    """CJK 感知的文本宽度估算(布局用,非精确排版):全角(ord>0x2E80)按 1.0×size,
    半角按 0.58×size,求和。"""
    return sum(size * (1.0 if ord(ch) > 0x2E80 else 0.58) for ch in s)


# ---------------- DATA(可变:本章数据) ----------------
LANES = ["推理调用链(entry → 独立小头 → 打分 → top-k → exit)", "支撑机制(可信依据 / IndexCache 与量化 / 复杂度账)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, 站牌——自然标题词,禁用 §N.M)
NODES = [
    ("entry",       0, 0, 0, "self.indexer(...)",        "MLA 前向触发打分(纯副作用调用)", "接线"),
    ("indep_head",  0, 1, 0, "wq_b · wk_weights_proj",   "index_* 专属头维,与主注意力解耦", "独立小头"),
    ("score",       0, 2, 0, "fp8_fp4_mqa_logits",       "ReLU 截负+头权重加权求和",        "打分函数"),
    ("topk",        0, 3, 0, "top_k_per_row_prefill",    "选 k 个索引,写入共享 buffer",      "top-k 选择"),
    ("exit",        0, 4, 0, "topk_indices_buffer",      "稀疏 MLA 只算被选中的 latent KV", "接线"),

    ("kl",          1, 1, 0, "detach + KL 对齐",          "单独训练,逼近主注意力真实分布",  "两阶段 KL 对齐"),
    ("cache",       1, 2, 0, "DeepseekV32IndexerCache",  "132B/条,与主 KV cache 分开分配", "IndexCache"),
    ("quant",       1, 2, 1, "quant_block_size",         "k 量化与缓存插入融合为一步",      "量化写入"),
    ("mxfp4",       1, 2, 2, "MXFP4_BLOCK_SIZE",         "2 值/字节打包,top-k 2× 提速",     "量化变体"),
    ("complexity",  1, 3, 0, "O(L²) → O(Lk)",  "indexer 打分仍 O(L²),常数远小", "复杂度诚实账"),
]
EDGES = [  # (src_id, dst_id) —— 同列 = 纵向支撑边,异列 = 横向调用边(见文件头说明)
    ("entry", "indep_head"), ("indep_head", "score"), ("score", "topk"), ("topk", "exit"),
    ("indep_head", "kl"),
    ("score", "cache"), ("cache", "quant"), ("quant", "mxfp4"),
    ("topk", "complexity"),
]
# (路线名, [(列, 站牌), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("数学优先(只读二·三两节)", [(1, "独立小头"), (2, "打分函数"), (3, "复杂度诚实账")], True),
    ("直接跳落地(五节)",       [(0, "接线"), (2, "IndexCache"), (4, "接线")], False),
]
LEGEND = [("#22c55e", "入口:MLA 前向调用进入"), ("#3b82f6", "调用边 / 支撑关系"), ("#f97316", "出口:交回稀疏 MLA 数值计算")]
TITLE = "第 27 章 · Lightning Indexer 推理调用链 + 支撑机制(源码剖面图)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 190, 58
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
NODE_COL = {}
for nid, lane, col, row, *_ in NODES:
    x = COLX[col]
    y = band_top[lane] + LANE_LABEL_H + BAND_PAD + row * (NODE_H + ROW_GAP)
    NODE_XY[nid] = (x, y)
    NODE_COL[nid] = col
NODE_BY_ID = {n[0]: n for n in NODES}

routes_top = lanes_bottom + 8
w = PAD_L + n_cols * NODE_W + (n_cols - 1) * COL_GAP + PAD_R
h = routes_top + ROUTE_HEAD_H + len(ROUTES) * ROUTE_ROW_H + BOTTOM_PAD


def badge(cx, cy, text):
    """§ 徽标胶囊(本章为自然标题,文字是标题词而非 §N.M),居中挂在 (cx,cy)。"""
    bx, by = cx - BADGE_W / 2, cy - BADGE_H / 2
    return [
        f'<rect x="{bx:.1f}" y="{by:.1f}" width="{BADGE_W}" height="{BADGE_H}" rx="{BADGE_H / 2}" '
        f'fill="{C_BADGE_FILL}" stroke="{C_BADGE_STROKE}" stroke-width="1.2"/>',
        f'<text x="{cx:.1f}" y="{cy + 4:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="10" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
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

# 调用边/支撑边(统一主线蓝):列相同 → 本章新增的纵向支撑边(上泳道节点正下方
# 挂一块支撑证据);列不同 → 模板原有的横向调用边(右中→左中)。
for src, dst in EDGES:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    if NODE_COL[src] == NODE_COL[dst]:
        p1 = (x1 + NODE_W / 2, y1 + NODE_H)
        p2 = (x2 + NODE_W / 2, y2)
    else:
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        p2 = (x2, y2 + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# 节点(圆角框 + 真实符号名 + 一行短语 + 右上角站牌)
for nid, lane, col, row, symbol, phrase, sec in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.4:.1f}" text-anchor="middle" '
              f'font-family="monospace" font-size="12" font-weight="bold" '
              f'fill="{C_NODE_TITLE}">{esc(symbol)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.72:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="10" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    L += badge(x + NODE_W - BADGE_W / 2 + 8, y, sec)

# 底部阅读路线:复用列坐标 COLX,站牌与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12.5" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
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
print(f"wrote {out}  size={w:.0f}x{h:.0f}  ratio={w / h:.2f}")
