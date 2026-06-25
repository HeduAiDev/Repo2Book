#!/usr/bin/env python3
"""Generate ch25 DeepSeek-V4 diagrams (delta-over-Llama)."""
import xml.sax.saxutils as xs

def esc(s):
    return xs.escape(str(s))

DEFS = (
    '<defs>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#b45309"/></marker>'
    '<marker id="ag" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
    '</defs>'
)

def box(x, y, w, h, fill, stroke, rx=8):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'

def text(x, y, s, size=13, anchor="middle", fill="#1e293b", weight="normal", family="sans-serif"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
            f'fill="{fill}" font-weight="{weight}" font-family="{family}">{esc(s)}</text>')

def line(x1, y1, x2, y2, stroke="#64748b", marker="a", width="1.5", dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{stroke}" '
            f'stroke-width="{width}"{d} marker-end="url(#{marker})"/>')

def save(name, parts, w, h):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}">'
           + DEFS + f'<rect width="{w}" height="{h}" fill="white"/>'
           + ''.join(parts) + '</svg>')
    open(name, 'w').write(svg)
    print("wrote", name)


# ============================================================
# Diagram 1: delta-stack (Llama 基线 | V4 delta), 4 行
# ============================================================
def delta_stack():
    W, H = 940, 500
    P = []
    P.append(text(W/2, 32, "DeepSeek-V4 = Llama 基线 + 一叠 delta", 19, weight="bold"))
    colL_x, colR_x = 60, 540
    colW = 340
    P.append(text(colL_x + colW/2, 64, "Llama 基线（第 22 章）", 15, weight="bold", fill="#334155"))
    P.append(text(colR_x + colW/2, 64, "DeepSeek-V4 delta", 15, weight="bold", fill="#b45309"))

    rows = [
        ("注意力", ["LlamaAttention", "qkv_proj 全量 Q/K/V", "o_proj 直出"],
         ["DeepseekV4Attention", "fused_wqa_wkv 低秩 + wq_b 升维", "wo_a/wo_b 输出也低秩 · 解耦 RoPE"], "MHA → MLA"),
        ("FFN", ["LlamaMLP", "单条 dense SwiGLU", "gate_up_proj / down_proj"],
         ["DeepseekV4MoE", "gate 路由 top-k routed 专家", "+ 始终走的 shared_experts"], "dense → MoE"),
        ("残差", ["LlamaDecoderLayer", "input/post_attention_layernorm", "固定 add-norm 一条流"],
         ["hc_pre / hc_post", "hc_mult 条残差流学习式混合", "末尾 hc_head 压回单流"], "add-norm → hc 多流"),
        ("解码头", ["单 lm_head", "compute_logits 出 1 个 token", ""],
         ["lm_head + MTP draft", "消费 _mtp_hidden_buffer 残差", "多 token 预测（投机解码）"], "1 头 → +MTP"),
    ]
    y0, rh, gap = 84, 84, 14
    for i, (label, lll, rrr, delta) in enumerate(rows):
        y = y0 + i * (rh + gap)
        # row label
        P.append(text(28, y + rh/2, label, 13, anchor="middle", fill="#475569", weight="bold"))
        # left box
        P.append(box(colL_x, y, colW, rh, "#f1f5f9", "#94a3b8"))
        P.append(text(colL_x + 14, y + 24, lll[0], 13.5, anchor="start", weight="bold", fill="#334155"))
        for j, ln in enumerate(lll[1:]):
            if ln:
                P.append(text(colL_x + 14, y + 44 + j*19, ln, 11.5, anchor="start", fill="#475569"))
        # right box
        P.append(box(colR_x, y, colW, rh, "#fff7ed", "#fb923c"))
        P.append(text(colR_x + 14, y + 24, rrr[0], 13.5, anchor="start", weight="bold", fill="#b45309"))
        for j, ln in enumerate(rrr[1:]):
            if ln:
                P.append(text(colR_x + 14, y + 44 + j*19, ln, 11.5, anchor="start", fill="#9a3412"))
        # arrow + delta label
        ax1, ax2 = colL_x + colW + 6, colR_x - 6
        P.append(line(ax1, y + rh/2, ax2, y + rh/2, stroke="#b45309", marker="ar", width="2"))
        P.append(text((ax1+ax2)/2, y + rh/2 - 8, delta, 11, fill="#b45309", weight="bold"))
    P.append(text(W/2, H-15, "四类 delta：低秩压缩 · 稀疏路由 · 多流残差 · 多 token 预测 —— 都叠在同一份 decoder-only 骨架上", 12, fill="#64748b"))
    save("ch25-delta-stack.svg", P, W, H)


# ============================================================
# Diagram 2: MLA projection flow
# ============================================================
def mla_flow():
    W, H = 1000, 540
    P = []
    P.append(text(W/2, 30, "MLA 投影数据流：低秩压缩 → 升维 → 解耦 RoPE → 低秩输出", 18, weight="bold"))

    def node(x, y, w, h, title, sub, fill="#fff7ed", stroke="#fb923c"):
        P.append(box(x, y, w, h, fill, stroke))
        P.append(text(x + w/2, y + (20 if sub else h/2+5), title, 13, weight="bold", fill="#9a3412"))
        if sub:
            P.append(text(x + w/2, y + 38, sub, 10.5, fill="#b45309"))

    # input
    node(40, 100, 130, 56, "hidden", "[T, hidden]", "#eef2ff", "#6366f1")
    # fused_wqa_wkv
    node(210, 100, 175, 56, "fused_wqa_wkv", "压成低秩 latent", "#fef3c7", "#d97706")
    P.append(line(170, 128, 208, 128))
    # multi-stream note
    P.append(box(210, 168, 175, 40, "#f8fafc", "#cbd5e1", rx=6))
    P.append(text(297, 184, "默认流：fused_wqa_wkv（重）", 9.5, fill="#475569"))
    P.append(text(297, 199, "aux 流：compressor / indexer（轻）", 9.5, fill="#475569"))
    P.append(line(297, 156, 297, 166, stroke="#94a3b8", marker="a", width="1"))

    # split
    P.append(text(425, 60, "split [q_lora_rank, head_dim]", 11, anchor="start", fill="#64748b"))
    P.append(line(385, 128, 470, 100, stroke="#64748b"))
    P.append(line(385, 128, 470, 250, stroke="#64748b"))

    # q branch (top)
    node(470, 74, 100, 50, "q latent", "q_lora_rank", "#fef3c7", "#d97706")
    node(600, 74, 100, 50, "q_norm", "RMSNorm", "#fff7ed", "#fb923c")
    node(730, 74, 100, 50, "wq_b", "升回 full Q", "#fef3c7", "#d97706")
    P.append(line(570, 99, 598, 99))
    P.append(line(700, 99, 728, 99))

    # kv branch (bottom)
    node(470, 226, 100, 50, "kv latent", "head_dim", "#fef3c7", "#d97706")
    node(600, 226, 100, 50, "kv_norm", "RMSNorm", "#fff7ed", "#fb923c")
    P.append(line(570, 251, 598, 251))

    # nope/rope split note
    P.append(box(730, 150, 230, 96, "#f0fdf4", "#86efac", rx=8))
    P.append(text(845, 172, "head 拆两段", 12.5, weight="bold", fill="#15803d"))
    P.append(text(845, 192, "nope 段：被低秩吸收，不旋转", 10.5, fill="#166534"))
    P.append(text(845, 210, "rope 段：解耦 RoPE 只旋这里", 10.5, fill="#166534"))
    P.append(text(845, 230, "head_dim = nope + rope", 10.5, fill="#166534"))
    P.append(line(780, 124, 800, 148, stroke="#15803d", marker="ag"))
    P.append(line(650, 276, 760, 244, stroke="#15803d", marker="ag"))

    # attn kernel (ch24 placeholder)
    node(390, 330, 220, 60, "注意力内核", "（后端 / 第 24 章）", "#e2e8f0", "#64748b")
    P.append(line(780, 124, 600, 330, stroke="#64748b"))   # q -> kernel
    P.append(line(650, 276, 500, 330, stroke="#64748b"))   # kv -> kernel

    # output low-rank
    node(390, 420, 145, 56, "wo_a", "低秩 · FP8 einsum", "#fef3c7", "#d97706")
    node(575, 420, 110, 56, "wo_b", "回 hidden", "#fef3c7", "#d97706")
    P.append(line(500, 390, 462, 418))
    P.append(line(535, 448, 573, 448))
    node(725, 420, 130, 56, "hidden out", "[T, hidden]", "#eef2ff", "#6366f1")
    P.append(line(685, 448, 723, 448))

    # Llama contrast sidenote
    P.append(box(40, 420, 320, 80, "#f8fafc", "#94a3b8", rx=8))
    P.append(text(200, 440, "对照 Llama（第 22 章）", 12, weight="bold", fill="#475569"))
    P.append(text(200, 460, "qkv_proj 一把出全量 Q/K/V", 11, fill="#475569"))
    P.append(text(200, 478, "o_proj 一把回 hidden —— 没有低秩", 11, fill="#475569"))
    P.append(text(200, 496, "也没有 nope/rope 拆分", 11, fill="#475569"))

    save("ch25-mla-projection-flow.svg", P, W, H)


# ============================================================
# Diagram 3: MoE dual backend
# ============================================================
def moe_dual():
    W, H = 960, 540
    P = []
    P.append(text(W/2, 30, "MoE：top-k 路由专家 + 共享专家 + 双后端", 18, weight="bold"))

    def node(x, y, w, h, title, sub, fill="#fff7ed", stroke="#fb923c"):
        P.append(box(x, y, w, h, fill, stroke))
        P.append(text(x + w/2, y + (20 if sub else h/2+5), title, 13, weight="bold", fill="#9a3412"))
        if sub:
            P.append(text(x + w/2, y + 38, sub, 10.5, fill="#b45309"))

    node(60, 230, 120, 56, "hidden", "[T, hidden]", "#eef2ff", "#6366f1")
    node(220, 230, 110, 56, "gate", "router logits", "#fef3c7", "#d97706")
    P.append(line(180, 258, 218, 258))
    node(360, 230, 170, 56, "fused_topk_bias", "sqrtsoftplus top-k", "#fef3c7", "#d97706")
    P.append(line(330, 258, 358, 258))

    # branch to two backends
    P.append(text(620, 150, "use_mega_moe ?", 12, weight="bold", fill="#64748b"))
    node(580, 175, 300, 56, "MegaMoE（EP / DeepGEMM）", "单算子跑全部专家 · FP4/FP8 · SM100", "#fef3c7", "#d97706")
    node(580, 285, 300, 56, "FusedMoE（TP）", "张量并行 · 内部已聚合 shared_experts", "#fef3c7", "#d97706")
    P.append(line(530, 248, 578, 203, stroke="#64748b"))
    P.append(line(530, 268, 578, 313, stroke="#64748b"))

    # shared experts (always-on dense)
    node(360, 400, 200, 56, "shared_experts", "每 token 必走的 dense", "#dcfce7", "#16a34a")
    P.append(line(445, 286, 445, 398, stroke="#15803d", marker="ag", dash="5,4"))
    P.append(text(560, 360, "并行 dense 残留", 11, anchor="start", fill="#15803d"))

    # sum (mega path)
    P.append(box(720, 400, 130, 56, "#eef2ff", "#6366f1", rx=8))
    P.append(text(785, 422, "相加", 13, weight="bold", fill="#3730a3"))
    P.append(text(785, 440, "routed + shared", 10, fill="#4338ca"))
    P.append(line(880, 203, 880, 428, stroke="#b45309", marker="ar"))
    P.append(line(720, 428, 562, 428, stroke="#15803d", marker="ag"))
    P.append(text(660, 392, "mega：外部相加", 10, fill="#64748b"))

    # contrast note
    P.append(box(60, 400, 270, 90, "#f8fafc", "#94a3b8", rx=8))
    P.append(text(195, 422, "对照 LlamaMLP（第 22 章）", 12, weight="bold", fill="#475569"))
    P.append(text(195, 442, "只有一条 dense SwiGLU", 11, fill="#475569"))
    P.append(text(195, 460, "每 token 全部参数都算", 11, fill="#475569"))
    P.append(text(195, 478, "MoE = 稀疏路由 + 一条共享 dense", 11, fill="#475569"))

    P.append(text(W/2, H-16, "两路后端聚合 shared_experts 的位置不同：mega 在外相加，TP 在 FusedMoE 内部聚合", 11.5, fill="#64748b"))
    save("ch25-moe-dual-backend.svg", P, W, H)


# ============================================================
# Diagram 4: hc residual + MTP bridge
# ============================================================
def hc_mtp():
    W, H = 980, 580
    P = []
    P.append(text(W/2, 30, "hc 多流残差贯穿全模型 · _mtp_hidden_buffer 桥到 MTP draft", 17, weight="bold"))

    def node(x, y, w, h, title, sub, fill="#fff7ed", stroke="#fb923c", tcol="#9a3412", scol="#b45309"):
        P.append(box(x, y, w, h, fill, stroke))
        P.append(text(x + w/2, y + (19 if sub else h/2+5), title, 12.5, weight="bold", fill=tcol))
        if sub:
            P.append(text(x + w/2, y + 36, sub, 10, fill=scol))

    # === target model (top) ===
    P.append(text(70, 62, "目标模型", 13, anchor="start", weight="bold", fill="#334155"))
    node(60, 76, 110, 52, "embed", "[T, hidden]", "#eef2ff", "#6366f1", "#3730a3", "#4338ca")
    node(200, 76, 150, 52, "repeat(hc_mult)", "→ hc_mult 条残差流", "#ede9fe", "#8b5cf6", "#5b21b6", "#6d28d9")
    P.append(line(170, 102, 198, 102))

    # layer stack (multi-stream)
    lx, ly, lw, lh = 380, 70, 240, 64
    P.append(box(lx, ly, lw, lh, "#fff7ed", "#fb923c"))
    P.append(text(lx + lw/2, ly + 20, "× N 层 DeepseekV4DecoderLayer", 12, weight="bold", fill="#9a3412"))
    P.append(text(lx + lw/2, ly + 38, "hc_pre→attn→hc_post", 10, fill="#b45309"))
    P.append(text(lx + lw/2, ly + 53, "hc_pre→ffn→hc_post（多流并行）", 10, fill="#b45309"))
    P.append(line(350, 102, 378, 102))

    # branch: stash + hc_head
    node(680, 70, 130, 52, "_mtp_hidden_buffer", "copy_ pre-hc_head", "#fee2e2", "#ef4444", "#991b1b", "#b91c1c")
    P.append(line(620, 92, 678, 88, stroke="#b91c1c", marker="ar"))
    node(680, 150, 130, 52, "hc_head", "门控加权压回单流", "#fef3c7", "#d97706")
    P.append(line(620, 112, 678, 168))
    node(840, 150, 100, 52, "norm → lm_head", "", "#eef2ff", "#6366f1", "#3730a3", "#4338ca")
    P.append(line(810, 176, 838, 176))

    # divider
    P.append(line(40, 250, W-40, 250, stroke="#cbd5e1", marker="", width="1", dash="6,5"))

    # === MTP draft (bottom) ===
    P.append(text(70, 290, "MTP draft（投机解码旁路）", 13, anchor="start", weight="bold", fill="#334155"))
    node(60, 310, 150, 52, "_mtp_hidden_buffer", "target 的 pre-hc_head", "#fee2e2", "#ef4444", "#991b1b", "#b91c1c")
    node(60, 392, 150, 52, "下一 token embed", "inputs_embeds", "#eef2ff", "#6366f1", "#3730a3", "#4338ca")
    P.append(line(680, 96, 135, 308, stroke="#b91c1c", marker="ar", dash="5,4"))
    P.append(text(420, 270, "桥：target 残差喂给 draft", 11, fill="#b91c1c"))

    node(250, 310, 120, 52, "hnorm + h_proj", "归一 + 投影", "#fff7ed", "#fb923c")
    node(250, 392, 120, 52, "enorm + e_proj", "归一 + 投影", "#fff7ed", "#fb923c")
    P.append(line(210, 336, 248, 336))
    P.append(line(210, 418, 248, 418))

    P.append(box(410, 350, 70, 52, "#ede9fe", "#8b5cf6", rx=8))
    P.append(text(445, 380, "融合 +", 13, weight="bold", fill="#5b21b6"))
    P.append(line(370, 336, 408, 366))
    P.append(line(370, 418, 408, 388))

    node(520, 350, 150, 52, "mtp_block", "复用 DecoderLayer", "#fff7ed", "#fb923c")
    P.append(line(480, 376, 518, 376))
    node(710, 350, 150, 52, "新 pre-hc_head 残差", "可多步串联", "#fee2e2", "#ef4444", "#991b1b", "#b91c1c")
    P.append(line(670, 376, 708, 376))
    node(710, 432, 150, 52, "compute_logits", "补 hc_head → draft", "#fef3c7", "#d97706")
    P.append(line(785, 402, 785, 430))

    P.append(text(W/2, H-18, "draft 把「目标模型的隐状态」与「下一 token 的 embedding」两路信号融合后，只跑一层就出预测", 11.5, fill="#64748b"))
    save("ch25-hc-residual-and-mtp.svg", P, W, H)


delta_stack()
mla_flow()
moe_dual()
hc_mtp()
