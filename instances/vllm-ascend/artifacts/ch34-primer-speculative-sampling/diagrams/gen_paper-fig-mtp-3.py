#!/usr/bin/env python3
"""论文精髓图重绘 —— arXiv:2412.19437 Figure 3(§2.2 Multi-Token Prediction 实现示意图)。
DeepSeek-V3 官方 MTP 模块示意:主模型 + D 个 MTP 模块顺序排列,隐藏状态跨深度传递、
Embedding/Output Head 跨深度共享 —— 正是本章 Eq.21-23 的图形化(与本章自绘的
fig33-mtp-causal-chain 互补:那张图讲"因果链为什么收缩",这张是论文本身最广泛引用的标准图,
用来认这套模块结构长什么样)。
信息结构核对原图(arxiv.org/html/2412.19437v2/x3.png):3 个模块左右排列,每个模块内
Input Tokens -> Embedding Layer(共享) -> [MTP 模块专属:RMSNorm×2 拼接 -> Linear
Projection -> Transformer Block / 主模型专属:Transformer Block × L] -> Output Head(共享)
-> Cross-Entropy Loss <- Target Tokens；模块间以一条水平+竖直折线传递隐藏状态
(从上一深度 Transformer 输出、Output Head 之前的位置引出,喂给下一深度左侧 RMSNorm)。
非像素复制:折线路径走向对齐原图信息结构,坐标改由代码整齐计算。
provenance = 论文原图本身(key_figure 重绘,豁免 explainer/spec.numbers 通道)。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


GREEN = "#bbf0c4"
GREEN_STROKE = "#15803d"
CREAM = "#fde9b8"
CREAM_STROKE = "#92400e"
GRAY = "#e2e8f0"
GRAY_STROKE = "#64748b"
DASH_BLUE = "#3b82f6"
INK = "#0f172a"
ARROW = "#1e293b"

COLUMNS = [
    {"name": "主模型", "sub": "(预测下 1 个 token)", "kind": "main",
     "input": ["t1", "t2", "t3", "t4"], "target": ["t2", "t3", "t4", "t5"], "loss": "L_Main"},
    {"name": "MTP 模块 1", "sub": "(预测下 2 个 token)", "kind": "mtp",
     "input": ["t2", "t3", "t4", "t5"], "target": ["t3", "t4", "t5", "t6"], "loss": "L¹_MTP"},
    {"name": "MTP 模块 2", "sub": "(预测下 3 个 token)", "kind": "mtp",
     "input": ["t3", "t4", "t5", "t6"], "target": ["t4", "t5", "t6", "t7"], "loss": "L²_MTP"},
]

COL_W = 300
COL_GAP = 56
ELLIPSIS_W = 60
PAD = 30
N = len(COLUMNS)

TITLE_H = 96
TOP = TITLE_H + 60          # 目标 token 行顶部

CHIP_W, CHIP_H, CHIP_GAP = 44, 26, 6
ARR_TOK_CE = 32
CE_H = 44
ARR_CE_OH = 68
OH_H = 40
ARR_OH_BODY = 34
TB_H = 52          # Transformer Block(MTP)/顶部子段
ARR_TB_LP = 18
LP_H = 40          # Linear Projection
ARR_LP_RMS = 22     # 含 concatenation 标注
RMS_H = 34
BODY_H = TB_H + ARR_TB_LP + LP_H + ARR_LP_RMS + RMS_H   # MTP 与主模型公用同一段总高
ARR_BODY_EMB = 30
EMB_H = 40
ARR_EMB_INPUT = 30

y_target_top = TOP
y_target_bot = y_target_top + CHIP_H
y_ce_top = y_target_bot + ARR_TOK_CE
y_ce_bot = y_ce_top + CE_H
y_oh_top = y_ce_bot + ARR_CE_OH
y_oh_bot = y_oh_top + OH_H
y_body_top = y_oh_bot + ARR_OH_BODY
y_body_bot = y_body_top + BODY_H
y_emb_top = y_body_bot + ARR_BODY_EMB
y_emb_bot = y_emb_top + EMB_H
y_input_top = y_emb_bot + ARR_EMB_INPUT
y_input_bot = y_input_top + CHIP_H

w = PAD * 2 + N * COL_W + (N - 1) * COL_GAP + ELLIPSIS_W
h = y_input_bot + 70

col_x = [PAD + i * (COL_W + COL_GAP) for i in range(N)]


def esc_mid(x, y, s, size=13, color=INK, weight="normal", anchor="middle"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="sans-serif" '
            f'font-size="{size}" font-weight="{weight}" fill="{color}">{esc(s)}</text>')


def box(x, y, wd, ht, fill, stroke, rx=8, sw=1.6):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{wd:.1f}" height="{ht:.1f}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'


def varrow(x, y1, y2, color=ARROW, sw=1.6, marker="mA"):
    return (f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{sw}" marker-end="url(#{marker})"/>')


L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w:.0f} {h:.0f}">']
L.append('<defs>'
          '<marker id="mA" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#1e293b"/></marker>'
          '<marker id="mB" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>'
          '</defs>')
L.append(f'<rect width="{w:.0f}" height="{h:.0f}" fill="white"/>')

TITLE = "arXiv:2412.19437 Fig.3 重绘：DeepSeek-V3 MTP 模块实现示意"
SUBTITLE = "主模型 + D 个 MTP 模块顺序排列；隐藏状态跨深度串行传递、Embedding/Output Head 跨深度共享 —— Eq.21-23 的图形化"
L.append(esc_mid(PAD, 34, TITLE, size=18, color="#1e40af", weight="bold", anchor="start"))
L.append(esc_mid(PAD, 56, SUBTITLE, size=12.5, color="#64748b", anchor="start"))

# 图例
legend_y = 80
legend_items = [("Embedding / Output Head（跨深度共享）", GREEN, GREEN_STROKE),
                ("计算子模块", CREAM, CREAM_STROKE),
                ("token", GRAY, GRAY_STROKE)]
lx = PAD
for label, fill, stroke in legend_items:
    L.append(box(lx, legend_y - 12, 16, 14, fill, stroke, rx=3, sw=1.2))
    tx = lx + 22
    L.append(esc_mid(tx, legend_y, label, size=12, color="#334155", anchor="start"))
    lx = tx + 7.2 * len(label) + 30
L.append(esc_mid(lx + 6, legend_y, "紫色折线 = 上一深度隐藏状态传给下一深度", size=12, color="#7c3aed", anchor="start"))

# ---- 逐列绘制 ----
body_top_wire_x = []   # 每列 transformer 输出wire 的 x(供跨列折线使用)
rms_left_x = []        # 每列(若为 mtp)左侧 RMSNorm 的中心 x
for i, col in enumerate(COLUMNS):
    cx0 = col_x[i]
    ccx = cx0 + COL_W / 2
    body_top_wire_x.append(ccx)

    # 虚线模块边框(从 CE Loss 底部往下一点开始,到 Embedding Layer 底部)
    dash_top = y_ce_bot + 10
    dash_bot = y_emb_bot + 14
    L.append(f'<rect x="{cx0-8:.1f}" y="{dash_top:.1f}" width="{COL_W+16:.1f}" height="{dash_bot-dash_top:.1f}" '
              f'rx="10" fill="none" stroke="{DASH_BLUE}" stroke-width="2.2" stroke-dasharray="9,6"/>')
    L.append(esc_mid(cx0 + 4, dash_top + 18, col["name"], size=14.5, color="#1e3a8a", weight="bold", anchor="start"))
    L.append(f'<text x="{cx0+4:.1f}" y="{dash_top+35:.1f}" font-family="sans-serif" font-size="11.5" '
              f'font-style="italic" fill="#1e3a8a">{esc(col["sub"])}</text>')

    # Target tokens(顶部)+ 向下箭头进 CE Loss
    n_tok = len(col["target"])
    total_chip_w = n_tok * CHIP_W + (n_tok - 1) * CHIP_GAP
    tok_x0 = ccx - total_chip_w / 2
    chip_centers = []
    for k, tok in enumerate(col["target"]):
        tx = tok_x0 + k * (CHIP_W + CHIP_GAP)
        chip_centers.append(tx + CHIP_W / 2)
        L.append(box(tx, y_target_top, CHIP_W, CHIP_H, GRAY, GRAY_STROKE, rx=4, sw=1.2))
        L.append(esc_mid(tx + CHIP_W / 2, y_target_top + CHIP_H / 2 + 5, tok, size=12.5, color="#334155"))
    for tcx in chip_centers:
        L.append(varrow(tcx, y_target_bot, y_ce_top))

    # Cross-Entropy Loss(loss 符号放在box内右上角,避免伸进相邻列被裁)
    L.append(box(cx0, y_ce_top, COL_W, CE_H, CREAM, CREAM_STROKE))
    L.append(esc_mid(ccx - 14, y_ce_top + CE_H / 2 + 9, "交叉熵损失", size=13.5))
    L.append(esc_mid(cx0 + COL_W - 10, y_ce_top + 15, f"→ {col['loss']}", size=11.5,
                      color="#b45309", weight="bold", anchor="end"))

    # arrow: Output Head -> CE Loss
    L.append(varrow(ccx, y_oh_bot, y_ce_top - 2))
    # Output Head(共享)
    L.append(box(cx0, y_oh_top, COL_W, OH_H, GREEN, GREEN_STROKE))
    L.append(esc_mid(ccx, y_oh_top + OH_H / 2 + 5, "Output Head", size=14, weight="bold", color="#14532d"))
    if i > 0:
        L.append(f'<line x1="{col_x[i-1]+COL_W:.1f}" y1="{y_oh_top+OH_H/2:.1f}" x2="{cx0:.1f}" y2="{y_oh_top+OH_H/2:.1f}" '
                  f'stroke="#16a34a" stroke-width="1.4" stroke-dasharray="3,4"/>')
        L.append(esc_mid((col_x[i-1]+COL_W+cx0)/2, y_oh_top + OH_H / 2 - 6, "共享", size=10.5, color="#16a34a"))

    # arrow: body top -> Output Head
    L.append(varrow(ccx, y_body_top, y_oh_top - 2))

    if col["kind"] == "main":
        # 堆叠卡片效果(Transformer Block x L)
        for off in (10, 5):
            L.append(box(cx0 + off, y_body_top + off, COL_W - 2 * off, BODY_H - off, CREAM, CREAM_STROKE, sw=1.2))
        L.append(box(cx0, y_body_top, COL_W, BODY_H, CREAM, CREAM_STROKE))
        L.append(esc_mid(ccx, y_body_top + BODY_H / 2 + 2, "Transformer Block", size=14))
        L.append(esc_mid(ccx, y_body_top + BODY_H / 2 + 20, "× L", size=13, color="#92400e"))
        # arrow: Embedding -> body
        L.append(varrow(ccx, y_emb_top, y_body_bot - 2))
    else:
        # Transformer Block(单个)
        tb_y = y_body_top
        L.append(box(cx0, tb_y, COL_W, TB_H, CREAM, CREAM_STROKE))
        L.append(esc_mid(ccx, tb_y + TB_H / 2 + 5, "Transformer Block", size=13.5))
        # arrow: Linear Projection -> Transformer Block
        lp_y = tb_y + TB_H + ARR_TB_LP
        L.append(varrow(ccx, lp_y, tb_y + TB_H))
        # Linear Projection
        L.append(box(cx0, lp_y, COL_W, LP_H, CREAM, CREAM_STROKE))
        L.append(esc_mid(ccx, lp_y + LP_H / 2 + 5, "线性投影 Linear Projection", size=12.5))
        # concatenation + arrow up from RMSNorm pair
        rms_y = lp_y + LP_H + ARR_LP_RMS
        rw = (COL_W - 14) / 2
        rms_left_cx = cx0 + rw / 2
        rms_right_cx = cx0 + rw + 14 + rw / 2
        rms_left_x.append(rms_left_cx)
        # 汇合折线(两个 RMSNorm -> Linear Projection 中点)
        merge_y = rms_y - ARR_LP_RMS / 2
        L.append(f'<polyline points="{rms_left_cx:.1f},{rms_y:.1f} {rms_left_cx:.1f},{merge_y:.1f} '
                  f'{rms_right_cx:.1f},{merge_y:.1f} {rms_right_cx:.1f},{rms_y:.1f}" '
                  f'fill="none" stroke="{ARROW}" stroke-width="1.4"/>')
        L.append(varrow(ccx, merge_y, lp_y, marker="mA"))
        L.append(esc_mid(ccx + 8, merge_y - 6, "拼接", size=11, color="#475569", anchor="start"))
        L.append(box(cx0, rms_y, rw, RMS_H, CREAM, CREAM_STROKE))
        L.append(esc_mid(rms_left_cx, rms_y + RMS_H / 2 + 5, "RMSNorm", size=12.5))
        L.append(box(cx0 + rw + 14, rms_y, rw, RMS_H, CREAM, CREAM_STROKE))
        L.append(esc_mid(rms_right_cx, rms_y + RMS_H / 2 + 5, "RMSNorm", size=12.5))
        # arrow: Embedding -> 右 RMSNorm(本深度自己的 token embedding)
        L.append(varrow(rms_right_cx, y_emb_top, rms_y + RMS_H))

    # Embedding Layer(共享)
    L.append(box(cx0, y_emb_top, COL_W, EMB_H, GREEN, GREEN_STROKE))
    L.append(esc_mid(ccx, y_emb_top + EMB_H / 2 + 5, "Embedding 层", size=13.5, weight="bold", color="#14532d"))
    if i > 0:
        L.append(f'<line x1="{col_x[i-1]+COL_W:.1f}" y1="{y_emb_top+EMB_H/2:.1f}" x2="{cx0:.1f}" y2="{y_emb_top+EMB_H/2:.1f}" '
                  f'stroke="#16a34a" stroke-width="1.4" stroke-dasharray="3,4"/>')
        L.append(esc_mid((col_x[i-1]+COL_W+cx0)/2, y_emb_top + EMB_H / 2 - 6, "共享", size=10.5, color="#16a34a"))

    # arrow: Input tokens -> Embedding
    L.append(varrow(ccx, y_input_top, y_emb_bot - 2))
    # Input tokens(底部)
    tok_x0i = ccx - total_chip_w / 2
    for k, tok in enumerate(col["input"]):
        tx = tok_x0i + k * (CHIP_W + CHIP_GAP)
        L.append(box(tx, y_input_top, CHIP_W, CHIP_H, GRAY, GRAY_STROKE, rx=4, sw=1.2))
        L.append(esc_mid(tx + CHIP_W / 2, y_input_top + CHIP_H / 2 + 5, tok, size=12.5, color="#334155"))

# ---- 跨深度隐藏状态折线(紫色,从上一列 body 顶部wire 引到下一列左侧 RMSNorm 底部) ----
for i in range(1, N):
    src_x = body_top_wire_x[i - 1]
    dst_x = rms_left_x[i - 1]
    gap_mid = (col_x[i - 1] + COL_W + col_x[i]) / 2
    tap_y = y_body_top
    drop_y = y_body_bot
    up_y = drop_y - 14  # 最后一段留 14px 竖直冲刺,箭头朝上顶进 RMSNorm 底边
    L.append(f'<polyline points="{src_x:.1f},{tap_y:.1f} {gap_mid:.1f},{tap_y:.1f} '
              f'{gap_mid:.1f},{up_y:.1f} {dst_x:.1f},{up_y:.1f}" '
              f'fill="none" stroke="#7c3aed" stroke-width="1.8"/>')
    L.append(varrow(dst_x, up_y, drop_y, color="#7c3aed", marker="mB"))
    L.append(f'<circle cx="{src_x:.1f}" cy="{tap_y:.1f}" r="3.2" fill="#7c3aed"/>')

# ---- 省略号(表示后续还有更多 MTP 深度) ----
ell_x = col_x[-1] + COL_W + 34
ell_y = (y_oh_top + y_body_bot) / 2
L.append(esc_mid(ell_x, ell_y, "⋯", size=28, color="#64748b", weight="bold"))
L.append(f'<line x1="{col_x[-1]+COL_W:.1f}" y1="{y_body_top:.1f}" x2="{ell_x-14:.1f}" y2="{y_body_top:.1f}" '
          f'stroke="#7c3aed" stroke-width="1.8" stroke-dasharray="4,4"/>')

foot_y = h - 26
L.append(esc_mid(PAD, foot_y, "主模型的隐藏状态经拼接(RMSNorm 后)喂给 MTP 模块 1，MTP 模块 1 再喂给模块 2 —— "
                  "深度间只能顺序算完,这正是本章因果链示意图讲的有效窗口逐层收缩的源头", size=12, color="#64748b", anchor="start"))
L.append('</svg>')

out = Path(__file__).with_name("paper-fig-mtp-3.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}  size={w:.0f}x{h:.0f}  aspect={w/h:.2f}")
