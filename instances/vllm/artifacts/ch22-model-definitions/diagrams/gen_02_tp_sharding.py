#!/usr/bin/env python3
"""TP 列并行接行并行：一个 attention 子块内的切分与归约（tp=2）。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1180, 640
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" '
    'markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

R0 = "#dbeafe"  # rank0 blue
R0E = "#2563eb"
R1 = "#fde68a"  # rank1 amber
R1E = "#d97706"
GRAY = "#f1f5f9"
GRAY_E = "#94a3b8"
GREEN = "#dcfce7"
GREEN_E = "#16a34a"


def rect(x, y, w, h, fill, edge, sw=1.5):
    L.append(
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
        f'fill="{fill}" stroke="{edge}" stroke-width="{sw}"/>'
    )


def txt(x, y, s, size=13, anchor="middle", fill="#111827", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


def arrow(x1, y1, x2, y2, color="#64748b"):
    L.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="1.6" marker-end="url(#a)"/>'
    )


# Lane labels
txt(70, 105, "rank 0", 13, "middle", R0E, "bold")
txt(70, 300, "rank 1", 13, "middle", R1E, "bold")

# 1. hidden_states (full, replicated)
rect(40, 250, 120, 60, GRAY, GRAY_E)
txt(100, 275, "hidden_states", 12, "middle", "#111827", "bold")
txt(100, 293, "(全量，两 rank 同)", 10.5, "middle", "#475569")

# 2. QKVParallelLinear column-shard
qx = 215
rect(qx, 70, 150, 200, "#eff6ff", "#3b82f6", 1.4)
txt(qx + 75, 60, "QKVParallelLinear（列并行）", 12.5, "middle", "#1e3a8a", "bold")
# weight split into two column blocks
rect(qx + 18, 95, 50, 150, R0, R0E)
txt(qx + 43, 175, "W[:, 列0]", 10.5, "middle", "#1e3a8a")
rect(qx + 82, 95, 50, 150, R1, R1E)
txt(qx + 107, 175, "W[:, 列1]", 10.5, "middle", "#92400e")
arrow(160, 280, qx + 18, 200, R1E)
arrow(160, 280, qx + 18, 150, R0E)

# 3. q_i/k_i/v_i per rank
def qkv(x, y, fill, edge):
    rect(x, y, 95, 56, fill, edge)
    txt(x + 47, y + 22, "q_i k_i v_i", 11.5, "middle", "#111827", "bold")
    txt(x + 47, y + 40, "(本 rank 的头)", 9.5, "middle", "#475569")


qkv(420, 80, R0, R0E)
qkv(420, 255, R1, R1E)
arrow(365, 130, 420, 108, R0E)
arrow(365, 210, 420, 283, R1E)

# 4. rope + Attention
def attn(x, y, fill, edge):
    rect(x, y, 90, 56, fill, edge)
    txt(x + 45, y + 22, "rope →", 11, "middle", "#111827")
    txt(x + 45, y + 40, "Attention", 11, "middle", "#111827", "bold")


attn(555, 80, R0, R0E)
attn(555, 255, R1, R1E)
arrow(515, 108, 555, 108, R0E)
arrow(515, 283, 555, 283, R1E)

# 5. RowParallelLinear o_proj (row-shard)
ox = 700
rect(ox, 70, 150, 270, "#fef2f2", "#ef4444", 1.4)
txt(ox + 75, 60, "RowParallelLinear o_proj（行并行）", 11.5, "middle", "#991b1b", "bold")
rect(ox + 30, 95, 90, 55, R0, R0E)
txt(ox + 75, 127, "W[行0, :] · q0", 9.5, "middle", "#1e3a8a")
rect(ox + 30, 255, 90, 55, R1, R1E)
txt(ox + 75, 287, "W[行1, :] · q1", 9.5, "middle", "#92400e")
arrow(645, 108, ox + 30, 122, R0E)
arrow(645, 283, ox + 30, 283, R1E)
txt(ox + 75, 175, "partial_0", 10.5, "middle", "#1e3a8a", "bold")
txt(ox + 75, 235, "partial_1", 10.5, "middle", "#92400e", "bold")

# 6. all_reduce
rx = 905
rect(rx, 170, 110, 100, GREEN, GREEN_E, 1.8)
txt(rx + 55, 200, "all_reduce", 13, "middle", "#14532d", "bold")
txt(rx + 55, 222, "partial_0", 10, "middle", "#14532d")
txt(rx + 55, 238, "+ partial_1", 10, "middle", "#14532d")
arrow(ox + 120, 150, rx, 200, R0E)
arrow(ox + 120, 282, rx, 240, R1E)

# 7. output full
rect(1055, 195, 100, 50, GRAY, GRAY_E)
txt(1105, 217, "output", 12.5, "middle", "#111827", "bold")
txt(1105, 234, "(全量)", 10.5, "middle", "#475569")
arrow(rx + 110, 220, 1055, 220)

# banner
rect(40, 400, 1115, 200, "#f8fafc", "#cbd5e1", 1.2)
txt(60, 428, "为什么整个子块只有一次通信？", 14.5, "start", "#0f172a", "bold")
lines = [
    "列并行（QKV / gate_up）：权重沿 output 维切。每个 rank 拿一段「列」，算出来的 q_i/k_i/v_i 已经是沿头维切好的——forward 里零通信。",
    "行并行（o_proj / down_proj）：权重沿 input 维切。它的输入正是列并行切好的那段（input_is_parallel=True），不用再切；",
    "    各 rank 算出来的是「部分和」partial_i，末尾一次 all_reduce 把 Σ partial_i 汇成完整结果。",
    "于是「列并行 → 行并行」成对出现：中间衔接零通信，整个 attention / MLP 子块的跨卡通信被压到末尾这一次 all_reduce。",
    "通信量 ≈ O(batch × seq × hidden)，与切多少卡无关——这是 TP 划算的关键。",
]
for i, ln in enumerate(lines):
    txt(60, 456 + i * 26, ln, 12.5, "start", "#334155")

L.append("</svg>")
open(
    "/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch22-model-definitions/diagrams/02-tp-sharding-flow.svg",
    "w",
    encoding="utf-8",
).write("\n".join(L))
print("wrote 02-tp-sharding-flow.svg")
