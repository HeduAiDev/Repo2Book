#!/usr/bin/env python3
"""第 28 章「本章地图」—— DeepSeek-V4 前向剖面:Llama 骨架上叠的四摞 delta。

改写自 .claude/skills/svg-diagram/references/example-chapter-map.py（多徽标节点/
split_symbol/transition 边则参考 ch39 的 gen_chapter-map.py 先例扩展）。

本章主线是"读一个真实大模型":DeepseekV4DecoderLayer.forward 每层内 hc_pre 包住
attn(MLA)、hc_post 收尾,再 hc_pre 包住 ffn(MoE)、hc_post 收尾——这段代码同时是
§28.1(骨架对照 Llama)和 §28.2/§28.3(各自的低秩/路由细节)的对象,所以 attn/ffn
两个节点各挂两块 § 徽标(复用 ch39 的多徽标节点写法,不是模板之外的新样式)。

三类特殊边(均非模板默认,均如实标注语义,不与"章内主线调用边=蓝实线"混淆):
  - "loop"(ffn→attn,虚线蓝,同色不同款):forward 里 `for layer in islice(...)`
    真实存在的调用重复,不是新的语义类别,只是同一主线边的"重复"变体,故不入图例、
    直接在弧线旁写"× N 层"自解释。
  - "bridge"(mtp_buf→mtp_block,虚线红 #ef4444):`_mtp_hidden_buffer` 是运行时
    真实存在的数据依赖(主模型 copy_ 写入,MTP draft 之后读出),但两次调用之间没有
    直接函数调用关系——用与本章另一张图(ch25-hc-residual-and-mtp.png)相同的红色
    表达同一座"桥",全书内保持这一处的色彩一致性。
  - "transition"(mtp_block→load_w,虚线灰 #94a3b8):MTP 与权重装载是两个完全不
    同调用时机(前者在 decode 循环里,后者在模型构造时一次性跑),之间不存在调用
    关系,只是 §28.5→§28.6 的行文顺序衔接,标法照抄 ch39 的"章内换题"先例。

■ 不可变(全书 72 章统一视觉语言，未改动):
  1. §徽标胶囊 badge()；2. 入口=绿#22c55e/出口=橙#f97316 接口桩；
  3. 章内主线调用边=蓝#3b82f6；4. 底部路线条(高亮=实线蓝/次要=虚线灰)；
  5. >2 种语义色画图例；6. cjk_text_width() 做宽度估算。

■ 本章新增(仅本章需要，未改动上面的不可变部分):
  - 双徽标节点(attn/ffn)：同一段真实代码被 §28.1(骨架层面)和 §28.2/§28.3(delta
    细节层面)分别讲过，两块徽标并排贴节点右上角，复用 ch39 的 NODES 徽标列表字段。
  - 出口只挂在 norm_out(真正 `return hidden_states` 的那个位置)——load_w(§28.6,
    权重装载)是全章最后讨论的真实机制，但它发生在模型构造期而非前向期，不是
    "返回上层"的语义，所以不占用出口桩，作为旁路的自然收尾悬空即可(全书首次
    出现"末节点不接出口桩"的布局，因为本章恰好是唯一一章末尾话题与前向返回值
    时序不同源)。

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
    """真实符号名在给定字号下装不下节点宽度时，在离中点最近的下划线处拆两行。
    两段各自仍是原符号的连续子串(不加省略号)，lint_chapter_map 的子串核对
    对每段仍能命中——不会被判成杜撰符号。找不到下划线就原样返回单行。"""
    if cjk_text_width(text, size) <= max_w:
        return [text]
    positions = [i for i, c in enumerate(text) if c == '_' and i != 0]
    if not positions:
        return [text]
    mid = len(text) // 2
    split_at = min(positions, key=lambda p: abs(p - mid))
    return [text[:split_at], text[split_at:]]


# ---------------- DATA(本章数据) ----------------
LANES = ["主干:DeepseekV4Model 前向(骨架 §28.1 + 三摞 delta)", "旁路:第四摞 delta(MTP draft)与收尾(权重装载)"]

# (节点id, 泳道下标, 列, 泳道内行号, 真实符号名, 一行短语, [§编号,...])
NODES = [
    ("embed",      0, 0, 0, "embed_input_ids",
     "embedding → repeat 成 hc_mult 条流", ["§28.4"]),
    ("attn",       0, 1, 0, "fused_wqa_wkv",
     "hc_pre 包裹→MLA 低秩+解耦 RoPE→hc_post", ["§28.1", "§28.2"]),
    ("ffn",        0, 2, 0, "fused_topk_bias",
     "hc_pre 包裹→MoE 路由+共享专家→hc_post", ["§28.1", "§28.3"]),
    ("mtp_buf",    0, 3, 0, "_mtp_hidden_buffer",
     "N 层后 copy_ 暂存 pre-hc_head 残差", ["§28.5"]),
    ("hc_head",    0, 4, 0, "hc_head",
     "sigmoid 门控,hc_mult 条流压回单流", ["§28.4"]),
    ("norm_out",   0, 5, 0, "self.norm",
     "末层归一,返回上层 logits", ["§28.4"]),
    ("mtp_block",  1, 4, 0, "mtp_block",
     "融合下一 token embed 与目标残差", ["§28.5"]),
    ("load_w",     1, 5, 0, "load_weights",
     "e8m0fnu 须 view(uint8) 装载", ["§28.6"]),
]
# (src_id, dst_id, style) —— style 省略即 "main"(蓝实线,真实调用主线);
# "loop" = 虚线蓝,同一主线边的循环重复(× N 层),旁边直接写文字自解释,不入图例;
# "bridge" = 虚线红#ef4444,真实数据依赖但非直接调用(与 ch25-hc-residual-and-mtp
#            图同色,同指 _mtp_hidden_buffer 这座桥);
# "transition" = 虚线灰,章内换题,非函数调用(抄 ch39 先例)。
EDGES = [
    ("embed", "attn"),
    ("attn", "ffn"),
    ("ffn", "attn", "loop"),
    ("ffn", "mtp_buf"),
    ("mtp_buf", "hc_head"),
    ("hc_head", "norm_out"),
    ("mtp_buf", "mtp_block", "bridge"),
    ("mtp_block", "load_w", "transition"),
]
# (路线名, [(列, §编号), ...] 按阅读顺序, 是否高亮:True=实线蓝/False=虚线灰)
ROUTES = [
    ("顺读全线",
     [(0, "§28.4"), (1, "§28.2"), (2, "§28.3"), (3, "§28.5"), (4, "§28.4"), (5, "§28.6")], True),
    ("只读 MLA", [(0, "§28.4"), (1, "§28.2")], False),
    ("只读 MoE", [(0, "§28.4"), (2, "§28.3")], False),
    ("只读 MTP+装载", [(3, "§28.5"), (4, "§28.5"), (5, "§28.6")], False),
]
LEGEND = [
    ("#22c55e", "入口:从上层调用进入"),
    ("#3b82f6", "章内主线调用边(真实调用,含循环)"),
    ("#f97316", "出口:返回上层(logits)"),
    ("#ef4444", "数据桥(非直接调用):_mtp_hidden_buffer 跨时刻传递"),
    ("#94a3b8", "章内换题(非调用):话题转向权重装载"),
]
TITLE = "第 28 章 · DeepSeek-V4 前向剖面:Llama 骨架上叠的四摞 delta(源码走线 + § 讲解站牌)"

# ---------------- 不可变:配色 ----------------
C_ENTRY, C_EXIT, C_MAIN = "#22c55e", "#f97316", "#3b82f6"
C_BRIDGE, C_TRANSITION = "#ef4444", "#94a3b8"
C_BADGE_FILL, C_BADGE_STROKE, C_BADGE_TEXT = "#eef2ff", "#6366f1", "#4338ca"
C_NODE_FILL, C_NODE_STROKE, C_NODE_TITLE, C_NODE_SUB = "#ffffff", "#475569", "#0f172a", "#64748b"
C_LANE_FILL = ["#f8fafc", "#eef2ff"]  # 泳道背景交替,仅装饰,非语义色
C_LANE_BORDER, C_LANE_LABEL = "#e2e8f0", "#334155"
C_ROUTE_DIM = "#94a3b8"

# ---------------- 几何常量(全计算,零魔数) ----------------
NODE_W, NODE_H = 168, 70
TITLE_SIZE, TITLE_LINE_H, SUB_SIZE = 12, 13, 9
COL_GAP, ROW_GAP = 32, 22
EDGE_MARGIN, STUB_W, STUB_H = 12, 58, 26
PAD_L = PAD_R = EDGE_MARGIN + STUB_W + 22  # 左右各留:接口桩 + 一段箭头
LANE_LABEL_H, BAND_PAD = 22, 16
TOP_PAD, TITLE_H, LEGEND_H, BOTTOM_PAD = 12, 32, 26, 14
ROUTE_HEAD_H, ROUTE_ROW_H = 22, 40
BADGE_W, BADGE_H = 44, 19
LOOP_DIP = 34  # 循环弧线在行下方探出的深度

n_cols = max(n[2] for n in NODES) + 1
COLX = [PAD_L + c * (NODE_W + COL_GAP) for c in range(n_cols)]

rows_per_lane = [0] * len(LANES)
for _id, lane, col, row, *_ in NODES:
    rows_per_lane[lane] = max(rows_per_lane[lane], row + 1)
band_h = [LANE_LABEL_H + BAND_PAD * 2 + r * NODE_H + max(0, r - 1) * ROW_GAP for r in rows_per_lane]
# 第一条泳道(主干)下方要给循环弧线留出额外空间
band_h[0] += LOOP_DIP
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
        f'<text x="{cx:.1f}" y="{cy + 3.6:.1f}" text-anchor="middle" font-family="sans-serif" '
        f'font-size="10.5" font-weight="bold" fill="{C_BADGE_TEXT}">{esc(text)}</text>',
    ]


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">']
L.append('<defs>' + ''.join(
    f'<marker id="m{name}" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
    f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{color}"/></marker>'
    for name, color in (
        ("Entry", C_ENTRY), ("Exit", C_EXIT), ("Main", C_MAIN),
        ("Bridge", C_BRIDGE), ("Trans", C_TRANSITION),
    )
) + '</defs>')
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

# 标题
L.append(f'<text x="{w / 2:.1f}" y="{TOP_PAD + 18}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="14.5" font-weight="bold" fill="{C_NODE_TITLE}">{esc(TITLE)}</text>')
# 图例(>2 种语义色必须画图例;本章 5 色)
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

# 入口/出口接口桩(入口挂 embed;出口只挂 norm_out——load_w 是构造期机制,见文件头说明)
ex, ey = NODE_XY["embed"]; ey += NODE_H / 2
xx, xy = NODE_XY["norm_out"]; xy += NODE_H / 2
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

# ── 主线边(main,蓝实线)────────────────────────────────────────────────
main_edges = [e for e in EDGES if (e[2] if len(e) > 2 else "main") == "main"]
for src, dst in main_edges:
    x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
    p1 = (x1 + NODE_W, y1 + NODE_H / 2)
    p2 = (x2, y2 + NODE_H / 2)
    L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
              f'stroke="{C_MAIN}" stroke-width="2" marker-end="url(#mMain)"/>')

# ── loop 边(ffn→attn,虚线蓝,行下方走弧线,旁写"× N 层"自解释)───────────
for e in EDGES:
    if len(e) > 2 and e[2] == "loop":
        src, dst, _ = e
        sx0, sy0 = NODE_XY[src]; dx0, dy0 = NODE_XY[dst]
        p_start = (sx0 + NODE_W * 0.3, sy0 + NODE_H)
        p_end = (dx0 + NODE_W * 0.7, dy0 + NODE_H)
        dip_y = sy0 + NODE_H + LOOP_DIP
        path_d = (f"M {p_start[0]:.1f},{p_start[1]:.1f} "
                  f"C {p_start[0]:.1f},{dip_y:.1f} {p_end[0]:.1f},{dip_y:.1f} {p_end[0]:.1f},{p_end[1]:.1f}")
        L.append(f'<path d="{path_d}" fill="none" stroke="{C_MAIN}" stroke-width="2" '
                  f'stroke-dasharray="6,4" marker-end="url(#mMain)"/>')
        mid_x = (p_start[0] + p_end[0]) / 2
        L.append(f'<text x="{mid_x:.1f}" y="{dip_y - 4:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="10.5" font-weight="bold" '
                  f'fill="{C_MAIN}">{esc("× N 层")}</text>')

# ── bridge 边(mtp_buf→mtp_block,虚线红,跨泳道折线,绕开 hc_head)────────
for e in EDGES:
    if len(e) > 2 and e[2] == "bridge":
        src, dst, _ = e
        bx, by = NODE_XY[src]; rx, ry = NODE_XY[dst]
        drop_x = bx + NODE_W / 2
        land_x = rx + NODE_W / 2
        turn_y = ry - 14
        pts = [(drop_x, by + NODE_H), (drop_x, turn_y), (land_x, turn_y), (land_x, ry)]
        path_d = "M " + " L ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
        L.append(f'<path d="{path_d}" fill="none" stroke="{C_BRIDGE}" stroke-width="2" '
                  f'stroke-dasharray="7,5" marker-end="url(#mBridge)"/>')

# ── transition 边(mtp_block→load_w,虚线灰,同泳道直连)──────────────────
for e in EDGES:
    if len(e) > 2 and e[2] == "transition":
        src, dst, _ = e
        x1, y1 = NODE_XY[src]; x2, y2 = NODE_XY[dst]
        p1 = (x1 + NODE_W, y1 + NODE_H / 2)
        p2 = (x2, y2 + NODE_H / 2)
        L.append(f'<line x1="{p1[0]:.1f}" y1="{p1[1]:.1f}" x2="{p2[0]:.1f}" y2="{p2[1]:.1f}" '
                  f'stroke="{C_TRANSITION}" stroke-width="2" stroke-dasharray="7,5" '
                  f'marker-end="url(#mTrans)"/>')

# 节点(圆角框 + 真实符号名[必要时拆两行] + 一行短语 + 右上角 § 徽标[可多个并排])
for nid, lane, col, row, symbol, phrase, secs in NODES:
    x, y = NODE_XY[nid]
    L.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" rx="12" '
              f'fill="{C_NODE_FILL}" stroke="{C_NODE_STROKE}" stroke-width="1.5"/>')
    title_lines = split_symbol(symbol, NODE_W - 26, TITLE_SIZE)
    if len(title_lines) == 1:
        L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.38:.1f}" text-anchor="middle" '
                  f'font-family="sans-serif" font-size="{TITLE_SIZE}" font-weight="bold" '
                  f'fill="{C_NODE_TITLE}">{esc(title_lines[0])}</text>')
    else:
        base_y = y + NODE_H * 0.30
        for li, line in enumerate(title_lines):
            L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{base_y + li * TITLE_LINE_H:.1f}" '
                      f'text-anchor="middle" font-family="sans-serif" font-size="{TITLE_SIZE}" '
                      f'font-weight="bold" fill="{C_NODE_TITLE}">{esc(line)}</text>')
    L.append(f'<text x="{x + NODE_W / 2:.1f}" y="{y + NODE_H * 0.84:.1f}" text-anchor="middle" '
              f'font-family="sans-serif" font-size="{SUB_SIZE}" fill="{C_NODE_SUB}">{esc(phrase)}</text>')
    # 右上角 § 徽标:多个并排贴在上边框,不下探进文字区
    bcx = x + NODE_W - BADGE_W / 2 + 6
    for sec in secs:
        L += badge(bcx, y, sec)
        bcx -= (BADGE_W + 5)

# 底部阅读路线:复用列坐标 COLX,§ 徽标与图上节点对齐成竖向落点
L.append(f'<text x="16" y="{routes_top + 15:.1f}" font-family="sans-serif" font-size="12" '
         f'font-weight="bold" fill="{C_LANE_LABEL}">'
         f'{esc("阅读路线(标号=图上 § 站牌;实线蓝=推荐 / 虚线灰=次要)")}</text>')
for ri, (name, stops, hi) in enumerate(ROUTES):
    ry = routes_top + ROUTE_HEAD_H + ri * ROUTE_ROW_H + ROUTE_ROW_H / 2
    L.append(f'<text x="16" y="{ry + 4:.1f}" font-family="sans-serif" font-size="11.5" '
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
print(f"wrote {out} ({w:.0f}x{h:.0f}, aspect={w/h:.2f}:1)")
