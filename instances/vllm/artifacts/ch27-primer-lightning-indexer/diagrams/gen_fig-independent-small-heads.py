#!/usr/bin/env python3
"""fig-independent-small-heads: layout 模板,论文精髓图重绘。
重绘自 arXiv:2512.02556 Figure 2(DeepSeek-V3.2 注意力架构图),已用 curl 下载原图
(2512.02556v1/x2.png)并用 Read 工具亲眼查看。布局与信息结构对齐原图(输入 h_t 在底部同源分叉
出 MLA 主路径与独立索引器路径、indexer 打分只喂给 Top-k Selector、Top-k Selector 与 MLA
Query/KV 一起汇入 Multi-Query Attention),配色套用原图的"主路径灰蓝 / 索引器绿色高亮"语义,
文字全部译中并换成 vLLM 里的真实符号名与真实 config 数值,非像素复制。"""
import xml.sax.saxutils as xs
from pathlib import Path

def esc(s):
    return xs.escape(s)

def text(x, y, s, size=13, anchor="middle", weight="normal", fill="#0f172a", style=None):
    fw = f' font-weight="{weight}"' if weight != "normal" else ""
    fs = f' font-style="{style}"' if style else ""
    return (f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-family="sans-serif" '
            f'font-size="{size}"{fw}{fs} fill="{fill}">{esc(s)}</text>')

def lines(x, y, items, size=12, anchor="middle", fill="#334155", lh=16):
    out = []
    for i, s in enumerate(items):
        out.append(text(x, y + i * lh, s, size=size, anchor=anchor, fill=fill))
    return out

def box(x, y, w, h, fill, stroke, rx=10, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')

def varrow(x, y1, y2, color="#64748b", sw=1.8):
    return (f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{sw}" marker-end="url(#a)"/>')

# ---- 版式常量 ----
W = 1180
COL_W = 330
GAP = 30
PAD = 40
COL_X = [PAD, PAD + COL_W + GAP, PAD + 2 * (COL_W + GAP)]
TOP = 90
TIER_H = [70, 132, 96, 60, 44]  # h_t / 三列投影 / topk selector / MQA / output
TIER_GAP = 34

y0 = TOP
y_input = y0
y_proj = y_input + TIER_H[0] + TIER_GAP
y_topk = y_proj + TIER_H[1] + TIER_GAP
y_mqa = y_topk + TIER_H[2] + TIER_GAP
y_out = y_mqa + TIER_H[3] + TIER_GAP
H = y_out + TIER_H[4] + 60

BLUE_FILL, BLUE_STROKE = "#dbeafe", "#3b82f6"
GREEN_FILL, GREEN_STROKE = "#dcfce7", "#16a34a"
GRAY_FILL, GRAY_STROKE = "#e2e8f0", "#64748b"

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
          '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
          'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#16a34a"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

L.append(text(W / 2, 32, "lightning indexer 是挂在 MLA 之下的独立小头模块", size=16.5, weight="bold"))
L.append(text(W / 2, 54, "重绘自 arXiv:2512.02556 Figure 2:h_t 同源分叉出 MLA 主路径与索引器路径,索引器打分只喂给 Top-k Selector",
               size=11.5, fill="#475569"))

# ---- Tier 0: Input Hidden h_t (共享输入,跨三列) ----
inp_w = 3 * COL_W + 2 * GAP
L.append(box(PAD, y_input, inp_w, TIER_H[0], GRAY_FILL, GRAY_STROKE))
L.append(text(PAD + inp_w / 2, y_input + 28, "Input Hidden h_t / h_s", size=13.5, weight="bold"))
L.append(text(PAD + inp_w / 2, y_input + 50, "同一份隐藏向量,三条投影各自独立读取,互不共享参数", size=11, fill="#64748b"))

# ---- Tier 1: 三列投影 ----
col_titles = ["MLA Query 投影", "MLA KV 压缩", "独立索引器(index_*)"]
col_fills = [(BLUE_FILL, BLUE_STROKE), (BLUE_FILL, BLUE_STROKE), (GREEN_FILL, GREEN_STROKE)]
col_bodies = [
    ["c_t^Q 低秩压缩(q_lora_rank=1536)", "apply RoPE(qk_rope_head_dim=64)", "拼接 [q_t^A ; q_t^R]"],
    ["c_t^KV 压缩 KV / k_t^R", "apply RoPE", "拼接 [c_t^KV ; k_t^R]"],
    ["wq_b: 1536 → head_dim×n_head=128×64", "wk_weights_proj(融合 GEMM): 输出 [128, 64]", "k_norm + 部分 apply RoPE(rope_dim=64)"],
]
for c in range(3):
    fill, stroke = col_fills[c]
    x = COL_X[c]
    L.append(box(x, y_proj, COL_W, TIER_H[1], fill, stroke))
    L.append(text(x + COL_W / 2, y_proj + 24, col_titles[c], size=13, weight="bold",
                   fill=("#166534" if c == 2 else "#1e3a8a")))
    L.extend(lines(x + COL_W / 2, y_proj + 48, col_bodies[c], size=11, fill="#334155", lh=20))
    L.append(varrow(x + COL_W / 2, y_input + TIER_H[0], y_proj))

# ---- Tier 2: 主注意力 Query(左,由 col0 汇入) / Lightning Indexer 打分(绿,由 col2 汇入) ----
q_w = COL_W
L.append(box(COL_X[0], y_topk, q_w, TIER_H[2], BLUE_FILL, BLUE_STROKE))
L.append(text(COL_X[0] + q_w / 2, y_topk + 26, "主注意力 Query", size=13, weight="bold", fill="#1e3a8a"))
L.append(text(COL_X[0] + q_w / 2, y_topk + 50, "[q_t^A ; q_t^R]", size=11.5, fill="#334155"))
L.append(text(COL_X[0] + q_w / 2, y_topk + 74, "与主注意力头数/头维绑定", size=10.5, fill="#64748b"))
L.append(varrow(COL_X[0] + q_w / 2, y_proj + TIER_H[1], y_topk))

idx_w = COL_W
L.append(box(COL_X[2], y_topk, idx_w, TIER_H[2], GREEN_FILL, GREEN_STROKE))
L.append(text(COL_X[2] + idx_w / 2, y_topk + 24, "Lightning Indexer 打分", size=13, weight="bold", fill="#166534"))
L.append(text(COL_X[2] + idx_w / 2, y_topk + 46, "n_head=64, head_dim=128, rope_dim=64", size=10.8, fill="#166534"))
L.append(text(COL_X[2] + idx_w / 2, y_topk + 64, "softmax_scale = head_dim^-0.5", size=10.8, fill="#166534"))
L.append(text(COL_X[2] + idx_w / 2, y_topk + 84, "→ I_{t,s}(全部历史打分,可 FP8)", size=10.8, fill="#334155"))
L.append(varrow(COL_X[2] + idx_w / 2, y_proj + TIER_H[1], y_topk, color="#16a34a"))

# Top-k Selector(中,读取 col1 的 KV + indexer 打分)
tk_w = COL_W
tk_x = COL_X[1]
L.append(box(tk_x, y_topk, tk_w, TIER_H[2], "#fef3c7", "#d97706"))
L.append(text(tk_x + tk_w / 2, y_topk + 26, "Top-k Selector", size=13.5, weight="bold", fill="#92400e"))
L.append(text(tk_x + tk_w / 2, y_topk + 50, "按 I_{t,:} 挑选压缩 KV 条目 {c_s}", size=11, fill="#334155"))
L.append(text(tk_x + tk_w / 2, y_topk + 72, "只消费打分结果,不关心怎么打的分", size=10.5, fill="#64748b"))
L.append(varrow(tk_x + tk_w / 2, y_proj + TIER_H[1], y_topk))
# indexer -> topk selector 的横向绿色箭头(核心:indexer 只喂给 selector)
mid_y = y_topk + TIER_H[2] / 2
L.append(f'<line x1="{COL_X[2]}" y1="{mid_y}" x2="{tk_x+tk_w}" y2="{mid_y}" '
         f'stroke="#16a34a" stroke-width="2.2" marker-end="url(#ag)"/>')
L.append(text((COL_X[2] + tk_x + tk_w) / 2, mid_y - 10, "index scores", size=10.5, fill="#166534"))

# ---- Tier 3: Multi-Query Attention(Core Attention) ----
mqa_w = 3 * COL_W + 2 * GAP
L.append(box(PAD, y_mqa, mqa_w, TIER_H[3], GRAY_FILL, GRAY_STROKE))
L.append(text(PAD + mqa_w / 2, y_mqa + 26, "Multi-Query Attention (Core Attention)", size=13.5, weight="bold"))
L.append(text(PAD + mqa_w / 2, y_mqa + 46, "只对 Top-k 选中的共享 latent KV 计算,Query 来自 MLA 主路径", size=10.8, fill="#64748b"))
for c, x_center in enumerate([COL_X[0] + q_w / 2, tk_x + tk_w / 2, COL_X[2] + idx_w / 2]):
    color = "#16a34a" if c == 2 else "#64748b"
    if c == 2:
        # indexer 本身不直接进 MQA,只经 Top-k Selector,这里不画第三条线
        continue
    L.append(varrow(x_center, y_topk + TIER_H[2], y_mqa, color=color))
L.append(varrow(tk_x + tk_w / 2, y_topk + TIER_H[2], y_mqa, color="#d97706"))

# ---- Tier 4: Output Hidden u_t ----
L.append(box(PAD, y_out, mqa_w, TIER_H[4], GRAY_FILL, GRAY_STROKE))
L.append(text(PAD + mqa_w / 2, y_out + 28, "Output Hidden u_t", size=13.5, weight="bold"))
L.append(varrow(PAD + mqa_w / 2, y_mqa + TIER_H[3], y_out))

# ---- 图例 ----
ly = H - 26
L.append(box(PAD, ly - 12, 16, 16, BLUE_FILL, BLUE_STROKE, rx=3))
L.append(text(PAD + 24, ly + 2, "MLA 主路径(主注意力头/维)", size=11, anchor="start", fill="#334155"))
L.append(box(PAD + 260, ly - 12, 16, 16, GREEN_FILL, GREEN_STROKE, rx=3))
L.append(text(PAD + 284, ly + 2, "独立索引器路径(index_* 专属参数)", size=11, anchor="start", fill="#334155"))

L.append('</svg>')
out = Path(__file__).with_name("fig-independent-small-heads.svg")
out.write_text('\n'.join(L), encoding="utf-8")
print(f"wrote {out} ({W}x{H})")
