#!/usr/bin/env python3
"""paper-fig-3: 重绘自 arXiv:2512.02556 Fig.2——DeepSeek-V3.2 注意力架构:DSA 在 MLA
之下如何运作(原图已抓到:https://arxiv.org/html/2512.02556v1/x2.png)。信息结构对齐
原图自底向上的流水线:Input Hidden h_t → 主分支(MLA 下投影 c_t^Q/c_t^KV,拼接
[q^A;q^R]/[c^KV;k^R]) 并行 indexer 分支(q^I/k^I/w^I 部分 RoPE → 点积+ReLU 打分 →
Lightning Indexer) → 两分支汇入 Top-k Selector(绿色,对应原图"绿色标出被 top-k
选中的 latent KV 条目") → Multi-Query Attention(Core Attention) → Output Hidden
u_t。记号与本章正文 §三/§四 一致(I_{t,s}/w^I_{t,j}/q^I_{t,j}/k_s^I/Top-k(I_{t,:}));
配色套本章语言(indexer/top-k 绿沿用原图绿色语义),文字译中,provenance=原论文本身。
全坐标由循环/常量计算。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s): return xs.escape(s)

INK, SUB = "#0f172a", "#64748b"
MLA_FILL, MLA_STROKE = "#3b82f6", "#1e3a8a"
IDX_FILL, IDX_STROKE = "#dcfce7", "#16a34a"
CORE_FILL, CORE_STROKE = "#94a3b8", "#334155"
OUT_FILL = "#0f172a"

W = 1150
PAD = 40
TITLE_TOP, SUBTITLE_TOP = 34, 56

CX_CENTER = 600              # 全图共用中心(Input Hidden / Selector / Core Attn / Output)
MAIN_CX = 380                # 主分支(MLA)中心 x
IDX_CX = 820                 # indexer 分支中心 x

ROW_H = 56
GAP = 42

rows_y = {}
y = 620  # Row A(最底) 起点,自底向上排布,后续统一翻转坐标(直接用绝对 y,从下往上递减)
rows_y["A"] = 620   # Input Hidden h_t
rows_y["B"] = rows_y["A"] - GAP - ROW_H   # 522: 下投影 / indexer 输入
rows_y["C"] = rows_y["B"] - GAP - ROW_H   # 424: concat / Lightning Indexer
rows_y["D"] = rows_y["C"] - GAP - ROW_H   # 326: Top-k Selector
rows_y["E"] = rows_y["D"] - GAP - ROW_H   # 228: Multi-Query Attention
rows_y["F"] = rows_y["E"] - GAP - ROW_H   # 130: Output Hidden u_t

H = rows_y["A"] + 70

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {int(H)}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
          '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" '
          'orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker></defs>')
L.append(f'<rect width="{W}" height="{int(H)}" fill="white"/>')
L.append(f'<text x="{PAD}" y="{TITLE_TOP}" font-family="sans-serif" font-size="18" '
         f'font-weight="bold" fill="{INK}">DSA 在 MLA 之下如何运作:indexer 打分 → Top-k 选出跨头一致的 latent KV → 核心注意力</text>')
L.append(f'<text x="{PAD}" y="{SUBTITLE_TOP}" font-family="sans-serif" font-size="12.5" '
         f'fill="{SUB}">自底向上:Input Hidden 并行喂入主分支(MLA 投影)与 indexer 分支(绿);两分支汇入 Top-k Selector(绿=被选中);最终喂入 Multi-Query 核心注意力</text>')


def box(x, y_top, w, h, fill, stroke, lines, text_color="white", fs=12.5, fw="bold"):
    out = [f'<rect x="{x-w/2:.1f}" y="{y_top:.1f}" width="{w}" height="{h}" rx="7" '
           f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>']
    n = len(lines)
    line_h = 16
    start_y = y_top + h / 2 - (n - 1) * line_h / 2 + 5
    for i, (txt, weight, size) in enumerate(lines):
        out.append(f'<text x="{x:.1f}" y="{start_y+i*line_h:.1f}" text-anchor="middle" '
                   f'font-family="sans-serif" font-size="{size}" font-weight="{weight}" '
                   f'fill="{text_color}">{esc(txt)}</text>')
    return out


def arrow(x1, y1, x2, y2, color="#64748b", marker="a", dash=None, width=1.6):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="{width}" marker-end="url(#{marker})"{d}/>')


# ---- Row A: Input Hidden h_t(贯通主+indexer 两分支起点,宽条覆盖两分支正下方) ----
ay = rows_y["A"]
L += box(CX_CENTER, ay, 900, ROW_H, "#e2e8f0", "#94a3b8",
         [("Input Hidden h_t", "bold", 13.5)], text_color=INK, fs=13.5)
L.append(arrow(MAIN_CX, ay, MAIN_CX, rows_y["B"] + ROW_H, MLA_STROKE))
L.append(arrow(IDX_CX, ay, IDX_CX, rows_y["B"] + ROW_H, IDX_STROKE, marker="ag"))

# ---- Row B: 主分支下投影 / indexer 分支输入 ----
by = rows_y["B"]
L += box(MAIN_CX, by, 300, ROW_H, MLA_FILL, MLA_STROKE,
         [("MLA 下投影", "bold", 13), ("c_t^Q, c_t^KV, k_t^R(部分 apply RoPE)", "normal", 11.5)])
L += box(IDX_CX, by, 300, ROW_H, IDX_FILL, IDX_STROKE,
         [("indexer 投影", "bold", 13), ("q_{t,j}^I, k_t^I, w_{t,j}^I(部分 RoPE)", "normal", 11)],
         text_color="#166534")
L.append(arrow(MAIN_CX, by, MAIN_CX, rows_y["C"] + ROW_H, MLA_STROKE))
L.append(arrow(IDX_CX, by, IDX_CX, rows_y["C"] + ROW_H, IDX_STROKE, marker="ag"))

# ---- Row C: concat([q^A;q^R] / [c^KV;k^R]) / Lightning Indexer 打分 ----
cy = rows_y["C"]
concat_w = 145
L += box(MAIN_CX - concat_w/2 - 8, cy, concat_w, ROW_H, "#dbeafe", MLA_STROKE,
         [("concat", "bold", 11.5), ("[q_{t,i}^A; q_{t,i}^R]", "normal", 11)], text_color="#1e3a8a")
L += box(MAIN_CX + concat_w/2 + 8, cy, concat_w, ROW_H, "#dbeafe", MLA_STROKE,
         [("concat", "bold", 11.5), ("[c_t^KV; k_t^R]", "normal", 11)], text_color="#1e3a8a")
L += box(IDX_CX, cy, 300, ROW_H, "#bbf7d0", IDX_STROKE,
         [("Lightning Indexer", "bold", 13),
          ("I_{t,s} = Σ_j w^I·ReLU(q^I·k^I)", "normal", 11)], text_color="#166534")
main_left_x = MAIN_CX - concat_w/2 - 8
main_right_x = MAIN_CX + concat_w/2 + 8
L.append(arrow(main_left_x, cy, main_left_x, rows_y["D"] + ROW_H, MLA_STROKE))
L.append(arrow(main_right_x, cy, main_right_x, rows_y["D"] + ROW_H, MLA_STROKE))
L.append(arrow(IDX_CX, cy, IDX_CX, rows_y["D"] + ROW_H, IDX_STROKE, marker="ag"))

# ---- Row D: Top-k Selector(绿,汇合主分支 KV 与 indexer 分数,梯形收窄喂入核心注意力) ----
dy = rows_y["D"]
sel_bottom_w, sel_top_w = 860, 620
L.append(f'<polygon points="{CX_CENTER-sel_bottom_w/2:.1f},{dy+ROW_H:.1f} {CX_CENTER+sel_bottom_w/2:.1f},{dy+ROW_H:.1f} '
         f'{CX_CENTER+sel_top_w/2:.1f},{dy:.1f} {CX_CENTER-sel_top_w/2:.1f},{dy:.1f}" '
         f'fill="#86efac" stroke="{IDX_STROKE}" stroke-width="1.8"/>')
L.append(f'<text x="{CX_CENTER:.1f}" y="{dy+22:.1f}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="13.5" font-weight="bold" fill="#14532d">Top-k Selector</text>')
L.append(f'<text x="{CX_CENTER:.1f}" y="{dy+40:.1f}" text-anchor="middle" font-family="sans-serif" '
         f'font-size="11" fill="#14532d">只留 I_{{t,s}} ∈ Top-k(I_{{t,:}}) 的 latent KV 条目 c_s(绿=被选中)</text>')
L.append(arrow(CX_CENTER, dy, CX_CENTER, rows_y["E"] + ROW_H, IDX_STROKE, marker="ag"))

# ---- Row E: Multi-Query Attention(Core Attention) ----
ey = rows_y["E"]
L += box(CX_CENTER, ey, 650, ROW_H, CORE_FILL, CORE_STROKE,
         [("Multi-Query Attention (Core Attention)", "bold", 13.5),
          ("u_t = Attn(h_t, {c_s | 被选中})", "normal", 11.5)])
L.append(arrow(CX_CENTER, ey, CX_CENTER, rows_y["F"] + ROW_H, CORE_STROKE))

# ---- Row F: Output Hidden u_t ----
fy = rows_y["F"]
L += box(CX_CENTER, fy, 300, ROW_H, OUT_FILL, OUT_FILL,
         [("Output Hidden u_t", "bold", 14)])

# ---- 说明角标 ----
note_x = PAD
note_y = rows_y["D"] - 8
L.append(f'<text x="{note_x}" y="{note_y:.1f}" font-family="sans-serif" font-size="11" '
         f'fill="{SUB}">← 主分支携带的是 latent KV [c_t^KV;k_t^R](MQA 下全部 query 头共享)</text>')

L.append('</svg>')
out = Path(__file__).with_name("paper-fig-3.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out}")
