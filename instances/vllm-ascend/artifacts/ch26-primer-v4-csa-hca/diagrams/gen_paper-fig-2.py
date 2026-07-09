#!/usr/bin/env python3
"""paper-fig-2: 重绘自 arXiv:2606.19348 Fig.2——DeepSeek-V4 整体架构(原图已抓到:
https://arxiv.org/html/2606.19348v1/x2.png)。信息结构对齐原图:一个 Transformer 块
内堆叠两个子层(先注意力子层用 CSA/HCA、再 FFN 子层用 DeepSeekMoE),每个子层都是
「块前混合(mHC)→子层→块后混合(mHC)」与「残差混合(mHC)」两路在 ⊕ 处相加——
三处改动(混合注意力/DeepSeekMoE/mHC)拼进同一层。顶部接预测头与 MTP 模块各自
的损失,底部由词嵌入接入输入词元。配色:CSA/HCA=绿(承本章既有配色),
DeepSeekMoE=蓝,残差混合(mHC)=紫,嵌入/预测头/MTP=暖黄(承原图配色),
文字译中,非逐字复刻原图像素,provenance=原论文本身。全坐标由上而下顺序累加
计算(y 单调递增,零手写魔数、零负坐标平移)。"""
import xml.sax.saxutils as xs
from pathlib import Path


def esc(s):
    return xs.escape(s)


INK, SUB = "#0f172a", "#64748b"
WARM_FILL, WARM_STROKE = "#fef3c7", "#d97706"
GRAY_FILL, GRAY_STROKE = "#e2e8f0", "#475569"
ATTN, ATTN_DARK = "#059669", "#065f46"
FFN, FFN_DARK = "#2563eb", "#1e3a8a"
MHC, MHC_DARK = "#7c3aed", "#5b21b6"
ARROW = "#64748b"

W = 780
CX = 320        # 主干(残差流)列
SIDE_X = 560    # 侧支(子层实体框 / 残差混合框)列
PAD = 40

STREAM_W, STREAM_H = 210, 26
SUB_W, SUB_H = 190, 40
SIDE_W, SIDE_H = 210, 46
ADD_R = 15
GAP = 34

L = []


def stream_icon(cx, y_top):
    for i in range(3):
        dx, dy = i * 5, i * 4
        L.append(f'<rect x="{cx-STREAM_W/2+dx:.1f}" y="{y_top+dy:.1f}" width="{STREAM_W}" height="{STREAM_H}" '
                  f'rx="6" fill="#f8fafc" stroke="{GRAY_STROKE}" stroke-width="1.2"/>')


def box(cx, y_top, w, h, fill, stroke, lines, font_size=12.5, text_color="white", bold=True, rx=8):
    L.append(f'<rect x="{cx-w/2:.1f}" y="{y_top:.1f}" width="{w}" height="{h}" rx="{rx}" '
              f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    n = len(lines)
    for i, line in enumerate(lines):
        ly = y_top + h / 2 - (n - 1) * 8 + i * 16 + 5
        fw = 'font-weight="bold" ' if bold else ''
        L.append(f'<text x="{cx:.1f}" y="{ly:.1f}" text-anchor="middle" font-family="sans-serif" '
                  f'font-size="{font_size}" {fw}fill="{text_color}">{esc(line)}</text>')


def varrow(x, y1, y2, color=ARROW, width=2):
    L.append(f'<line x1="{x:.1f}" y1="{y1:.1f}" x2="{x:.1f}" y2="{y2:.1f}" '
              f'stroke="{color}" stroke-width="{width}" marker-end="url(#a)"/>')


def harrow(x1, y, x2, color=ARROW, width=2):
    L.append(f'<line x1="{x1:.1f}" y1="{y:.1f}" x2="{x2:.1f}" y2="{y:.1f}" '
              f'stroke="{color}" stroke-width="{width}" marker-end="url(#a)"/>')


def sublayer_group(y_top, sub_name, sub_color, sub_dark):
    """自上而下画一个子层单元:y_top = 该组「输出多流残差」stack 的顶边。
    结构(自上而下):h_out stack -> ⊕(汇合 Post-Mix 输出与 Residual Mixing 输出)
    -> [Post-Block Mixing(主干) / 子层实体框(侧支,由 Pre-Mix 供给)]
    -> Pre-Block Mixing(主干) -> h_in stack(该组输入,同时也是下一组/下一级的连接点)。
    返回 h_in stack 的底边 y(供上一级继续往下接)。"""
    y_out_top = y_top
    stream_icon(CX, y_out_top)
    y_add = y_out_top + STREAM_H + GAP
    L.append(f'<circle cx="{CX}" cy="{y_add}" r="{ADD_R}" fill="white" stroke="{INK}" stroke-width="2"/>')
    L.append(f'<line x1="{CX-ADD_R+4:.1f}" y1="{y_add}" x2="{CX+ADD_R-4:.1f}" y2="{y_add}" stroke="{INK}" stroke-width="2"/>')
    L.append(f'<line x1="{CX:.1f}" y1="{y_add-ADD_R+4:.1f}" x2="{CX:.1f}" y2="{y_add+ADD_R-4:.1f}" stroke="{INK}" stroke-width="2"/>')
    varrow(CX, y_add + ADD_R, y_out_top)  # ⊕ -> h_out stack

    y_row1_top = y_add + ADD_R + GAP  # Post-Block Mixing / 子层实体框所在行
    box(CX, y_row1_top, SUB_W, SUB_H, GRAY_FILL, GRAY_STROKE,
        ["块后混合 (Post-Block Mixing)"], font_size=11.5, text_color=INK)
    box(SIDE_X, y_row1_top, SIDE_W, SIDE_H, sub_color, sub_dark, [sub_name], font_size=13)
    # 子层实体框(侧支) -> Post-Block Mixing(主干,横向)
    harrow(SIDE_X - SIDE_W / 2, y_row1_top + SIDE_H / 2, CX + SUB_W / 2, color=sub_color)
    # Post-Block Mixing -> ⊕(竖直向上)
    varrow(CX, y_row1_top, y_add - ADD_R, color=sub_color)

    # Residual Mixing 侧支框(与 ⊕ 同一带,略靠下,横向接入 ⊕)
    y_res = y_add + GAP * 0.15
    box(SIDE_X, y_res - SIDE_H / 2, SIDE_W, SIDE_H, "#ede9fe", MHC,
        ["残差混合 (Residual Mixing, mHC)"], font_size=10.5, text_color=MHC_DARK, bold=True)
    harrow(SIDE_X - SIDE_W / 2, y_res, CX + ADD_R, color=MHC)

    y_row2_top = y_row1_top + SUB_H + GAP  # Pre-Block Mixing 行
    box(CX, y_row2_top, SUB_W, SUB_H, GRAY_FILL, GRAY_STROKE,
        ["块前混合 (Pre-Block Mixing)"], font_size=11.5, text_color=INK)
    # Pre-Block Mixing -> 子层实体框(横向)
    harrow(CX + SUB_W / 2, y_row2_top + SUB_H / 2, SIDE_X - SIDE_W / 2, color=sub_color)

    y_in_top = y_row2_top + SUB_H + GAP
    stream_icon(CX, y_in_top)
    # Pre-Block Mixing <- h_in stack(竖直向上箭头指向 Pre-Block Mixing)
    varrow(CX, y_in_top, y_row2_top + SUB_H)
    # h_in stack -> 残差混合(侧支,竖直转横向,跳过子层直连 Residual Mixing)
    jog_y = y_in_top - GAP * 0.4
    L.append(f'<path d="M {CX+STREAM_W/2-8:.1f} {y_in_top:.1f} L {CX+STREAM_W/2-8:.1f} {jog_y:.1f} '
              f'L {SIDE_X:.1f} {jog_y:.1f} L {SIDE_X:.1f} {y_res+SIDE_H/2:.1f}" '
              f'fill="none" stroke="{MHC}" stroke-width="2" marker-end="url(#a)"/>')

    return y_in_top + STREAM_H


L.append(f'<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          f'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="{ARROW}"/></marker></defs>')

CONTENT_TOP = 96
HEAD_H = 42

# ---- MTP 模块(最顶) ----
box(CX, CONTENT_TOP, 170, HEAD_H, WARM_FILL, WARM_STROKE, ["MTP 模块 (MTP Modules)"],
    font_size=11.5, text_color=INK)
L.append(f'<line x1="{CX+85:.1f}" y1="{CONTENT_TOP+HEAD_H/2:.1f}" x2="{CX+220:.1f}" y2="{CONTENT_TOP+HEAD_H/2:.1f}" '
          f'stroke="{INK}" stroke-width="1.6" stroke-dasharray="5,4" marker-end="url(#a)"/>')
L.append(f'<text x="{CX+228:.1f}" y="{CONTENT_TOP+HEAD_H/2+5:.1f}" font-family="sans-serif" font-size="12" '
          f'fill="{INK}">MTP 损失</text>')

y = CONTENT_TOP + HEAD_H + 44
box(CX, y, 170, HEAD_H, WARM_FILL, WARM_STROKE, ["预测头 (Prediction Head)"],
    font_size=11.5, text_color=INK)
varrow(CX, CONTENT_TOP + HEAD_H, y)
L.append(f'<line x1="{CX+85:.1f}" y1="{y+HEAD_H/2:.1f}" x2="{CX+220:.1f}" y2="{y+HEAD_H/2:.1f}" '
          f'stroke="{INK}" stroke-width="1.6" stroke-dasharray="5,4" marker-end="url(#a)"/>')
L.append(f'<text x="{CX+228:.1f}" y="{y+HEAD_H/2+5:.1f}" font-family="sans-serif" font-size="12" '
          f'fill="{INK}">LM 损失</text>')
head_bottom = y + HEAD_H

dash_top = head_bottom + 40
varrow(CX, head_bottom, dash_top)

ffn_bottom = sublayer_group(dash_top + 14, "DeepSeekMoE", FFN, FFN_DARK)
attn_bottom = sublayer_group(ffn_bottom + 20, "CSA / HCA(本章)", ATTN, ATTN_DARK)

dash_bottom = attn_bottom + 20

emb_top = dash_bottom + 44
varrow(CX, dash_bottom, emb_top)
EMB_H = 44
box(CX, emb_top, 170, EMB_H, WARM_FILL, WARM_STROKE, ["词嵌入 (Embedding)"], font_size=13, text_color=INK)
input_y = emb_top + EMB_H + 34
varrow(CX, emb_top + EMB_H, input_y - 8)
L.append(f'<text x="{CX:.1f}" y="{input_y:.1f}" text-anchor="middle" font-family="sans-serif" '
          f'font-size="12.5" fill="{SUB}">输入词元 (Input Tokens)</text>')

# ---- 虚线框:Transformer 块 × L(包住两个子层组) ----
dash_left = CX - SUB_W / 2 - 60
dash_right = SIDE_X + SIDE_W / 2 + 30
L.insert(1, f'<rect x="{dash_left:.1f}" y="{dash_top:.1f}" width="{dash_right-dash_left:.1f}" '
          f'height="{dash_bottom-dash_top:.1f}" rx="12" fill="none" stroke="#94a3b8" stroke-width="2" '
          f'stroke-dasharray="9,6"/>')
L.insert(2, f'<text x="{dash_right-8:.1f}" y="{dash_top+20:.1f}" text-anchor="end" font-family="sans-serif" '
          f'font-size="12.5" font-weight="bold" fill="{SUB}">Transformer 块 × L</text>')

H = int(input_y + 30)

svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
svg.append(f'<rect width="{W}" height="{H}" fill="white"/>')
svg.append(f'<text x="{PAD}" y="34" font-family="sans-serif" font-size="16" '
            f'font-weight="bold" fill="{INK}">DeepSeek-V4 一层长什么样:CSA/HCA、DeepSeekMoE、mHC 三处改动拼进同一层</text>')
svg.append(f'<text x="{PAD}" y="56" font-family="sans-serif" font-size="12" '
            f'fill="{SUB}">注意力子层(绿)与 FFN 子层(蓝)结构同构:块前混合 → 子层 → 块后混合,与残差混合(紫,mHC)在 ⊕ 处相加</text>')
svg.append("\n".join(L))
svg.append('</svg>')

out = Path(__file__).with_name("paper-fig-2.svg")
out.write_text('\n'.join(svg), encoding="utf-8")
print(f"wrote {out}  H={H}")
