#!/usr/bin/env python3
"""fig28-2: 两条采样路径 —— torch.multinomial 触发 CPU-NPU 同步 vs Gumbel-max 全程 NPU 无同步。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1320, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="9" markerHeight="6" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '<marker id="arg" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="9" markerHeight="6" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#15803d"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, rx=10, sw=1.6):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def text(x, y, s, size=14, anchor="start", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


def arrow(x1, y, x2, mk="ar", color="#475569"):
    L.append(f'<line x1="{x1}" y1="{y}" x2="{x2}" y2="{y}" stroke="{color}" stroke-width="2" marker-end="url(#{mk})"/>')


text(W / 2, 40, "同一分布，两条路：multinomial 卡同步 vs Gumbel-max 全程在 NPU", 23, "middle", "#0f172a", "bold")

# 上路：multinomial（红 / 卡）
text(70, 96, "上路 · torch.multinomial（基类未走）", 15, "start", "#b91c1c", "bold")
mb = [(70, "probs", "#eef2ff", "#6366f1"),
      (270, "构建 CDF\n累积分布", "#eef2ff", "#6366f1"),
      (470, "CPU-NPU\n同步 ⚠", "#fee2e2", "#ef4444"),
      (700, "host 取样\n回拷设备", "#fee2e2", "#ef4444"),
      (930, "token id", "#eef2ff", "#6366f1")]
for x, label, fill, stroke in mb:
    box(x, 116, 150, 64, fill, stroke, 10, 1.8)
    lines = label.split("\n")
    for j, ln in enumerate(lines):
        text(x + 75, 116 + 32 + (j - (len(lines) - 1) / 2) * 18 + 5, ln, 13, "middle", "#1e293b")
for x in (220, 420, 650, 880):
    arrow(x, 148, x + 48)
text(550, 210, "流水线停顿：等 host 算完才能继续 —— 这正是要避开的同步点。", 13.5, "middle", "#b91c1c")

# 下路：Gumbel-max（绿 / 顺）
text(70, 282, "下路 · probs.div_(q).argmax,  q ~ Exp(1)（昇腾与基类都走这条）", 15, "start", "#15803d", "bold")
gb = [(70, "probs", "#ecfdf5", "#10b981"),
      (270, "q.exponential_()\nq ~ Exp(1)", "#ecfdf5", "#10b981"),
      (490, "probs.div_(q)\n逐元素", "#ecfdf5", "#10b981"),
      (710, "argmax(dim=-1)", "#ecfdf5", "#10b981"),
      (930, "token id", "#ecfdf5", "#10b981")]
for x, label, fill, stroke in gb:
    box(x, 302, 160, 64, fill, stroke, 10, 1.8)
    lines = label.split("\n")
    for j, ln in enumerate(lines):
        text(x + 80, 302 + 32 + (j - (len(lines) - 1) / 2) * 18 + 5, ln, 13, "middle", "#1e293b",
             mono=("(" in ln or "_" in ln))
for x in (230, 450, 670, 890):
    arrow(x, 334, x + 38, "arg", "#15803d")

# async 旁注
box(70, 402, 1180, 56, "#f0fdf4", "#15803d", 10, 1.6)
text(90, 426, "昇腾的 NPU delta（两行）：", 13.5, "start", "#15803d", "bold")
text(290, 426, "with npu_stream_switch(global_stream()):  q.exponential_()   →   wait_stream() 汇流",
     13, "start", "#166534", mono=True)
text(90, 448, "把指数随机切到独立 stream 异步发起，与模型计算重叠；Gumbel 数学来自上游，昇腾只加这层异步包裹。",
     12.5, "start", "#166534")

text(W / 2, 506, "同分布证明：q_i/p_i ~ Exp(p_i)，独立指数取最小落在 i 的概率 = p_i / Σp_j = p_i。故 argmax(p/q) ~ Categorical(p)。",
     13, "middle", "#475569")

L.append('</svg>')
open("fig28-2-gumbel.svg", "w").write('\n'.join(L))
print("wrote fig28-2-gumbel.svg")
