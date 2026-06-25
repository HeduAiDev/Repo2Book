#!/usr/bin/env python3
"""权重装载三段式与 qkv fuse。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1180, 660
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" '
    'markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker></defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

Q = "#dbeafe"
QE = "#2563eb"
K = "#dcfce7"
KE = "#16a34a"
V = "#fde68a"
VE = "#d97706"
GRAY = "#f1f5f9"
GRAY_E = "#94a3b8"


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


def arrow(x1, y1, x2, y2, color="#64748b", sw=1.6):
    L.append(
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="{color}" stroke-width="{sw}" marker-end="url(#a)"/>'
    )


# three stage headers
stages = [
    (40, "① initialize_model", "建空壳：形状已按 tp 切好，参数尚未填值"),
    (410, "② load_weights", "流式喂 checkpoint：边读边重命名、边切、边填"),
    (820, "③ process_weights_after_loading", "kernel 重排 + Attention kv scale → eval()"),
]
for x, t, sub in stages:
    rect(x, 30, 320 if x < 800 else 320, 40, "#eef2ff", "#6366f1", 1.4)
    txt(x + 160, 55, t, 14.5, "middle", "#3730a3", "bold")
    txt(x + 160, 90, sub, 11.5, "middle", "#475569")

# Stage 1: empty fused qkv param
rect(95, 130, 210, 60, "#ffffff", "#94a3b8", 1.6)
# three empty segments
for i, (lab, edge) in enumerate([("q 段", QE), ("k 段", KE), ("v 段", VE)]):
    sw = [90, 60, 60][i]
    off = [0, 90, 150][i]
    L.append(
        f'<rect x="{95 + off}" y="130" width="{sw}" height="60" '
        f'fill="none" stroke="{edge}" stroke-width="1.4" stroke-dasharray="4 3"/>'
    )
    txt(95 + off + sw / 2, 165, lab, 11, "middle", edge, "bold")
txt(200, 210, "qkv_proj.weight（空，[q|k|v] 三段）", 11, "middle", "#475569")

# Stage 2: checkpoint tensors (independent q/k/v)
cx = 410
txt(cx + 120, 130, "checkpoint 里是分开的三张：", 12, "middle", "#0f172a", "bold")
ck = [
    ("q_proj.weight", Q, QE, 150),
    ("k_proj.weight", K, KE, 230),
    ("v_proj.weight", V, VE, 310),
]
for name, fill, edge, y in ck:
    rect(cx, y, 150, 50, fill, edge)
    txt(cx + 75, y + 22, name, 11.5, "middle", "#111827", "bold")
    txt(cx + 75, y + 40, ".replace(.q/k/v_proj → .qkv_proj)", 9, "middle", "#475569", mono=True)

# stacked_params_mapping label
rect(cx + 175, 220, 175, 70, "#fffbeb", "#d97706", 1.3)
txt(cx + 262, 245, "stacked_params_mapping", 11, "middle", "#92400e", "bold")
txt(cx + 262, 263, "重命名 + 带 shard_id", 10.5, "middle", "#78350f")
txt(cx + 262, 280, "(q / k / v)", 10.5, "middle", "#78350f", mono=True)
for y in (175, 255, 335):
    arrow(cx + 150, y, cx + 175, 255, "#d97706", 1.3)

# Stage 3: filled fused param (target segments)
fx = 830
txt(fx + 130, 130, "weight_loader：按 shard_id 算 offset，再按 rank narrow", 11, "middle", "#0f172a", "bold")
rect(fx, 160, 260, 70, "#ffffff", "#475569", 1.8)
seg = [("q", Q, QE, 110, 0), ("k", K, KE, 75, 110), ("v", V, VE, 75, 185)]
for lab, fill, edge, sw, off in seg:
    rect(fx + off, 160, sw, 70, fill, edge)
    txt(fx + off + sw / 2, 198, lab, 13, "middle", "#111827", "bold")
txt(fx + 130, 250, "qkv_proj.weight（已填，本 rank 的切片）", 10.5, "middle", "#475569")
# offset annotations
txt(fx + 55, 145, "offset 0", 9.5, "middle", QE)
txt(fx + 147, 145, "+q宽", 9.5, "middle", KE)
txt(fx + 222, 145, "+q+k宽", 9.5, "middle", VE)
# arrows from stacked_params_mapping right edge to fused segments
arrow(cx + 350, 235, fx + 55, 195, QE, 1.6)
arrow(cx + 350, 255, fx + 147, 195, KE, 1.6)
arrow(cx + 350, 272, fx + 222, 195, VE, 1.6)

# big arrows between stages
arrow(305, 160, 405, 160, "#1e293b", 2.2)
arrow(770, 195, 825, 195, "#1e293b", 2.2)

# banner
rect(40, 340, 1115, 290, "#f8fafc", "#cbd5e1", 1.2)
txt(60, 370, "一次 weight_loader 同时干两件事：fuse + TP 切分", 14.5, "start", "#0f172a", "bold")
lines = [
    "offset（fuse 定位）：q 段从 0 起、宽 num_heads×head_size；k 段紧跟其后；v 段再往后。三张独立 checkpoint 各自落进对应段。",
    "narrow（TP 切分）：从磁盘全量里按 start_idx = shard_rank × shard_size 切出本 rank 那一份，copy_ 进去。",
    "",
    "q 与 k/v 的 shard_rank 不一样——这是 GQA 的关键：",
    "    q：shard_rank = tp_rank          （每张卡拿自己那批 query 头）",
    "    k/v：shard_rank = tp_rank // num_kv_head_replicas   （KV 头比卡少时，相邻几张卡复制同一份 K/V）",
    "",
    "三段式为何分开：先建空壳（峰值显存可控），再流式边读边切边填，最后才做 kernel 重排——三段职责清晰，与量化/TP/PP 正交。",
]
for i, ln in enumerate(lines):
    txt(60, 398 + i * 26, ln, 12.5, "start", "#334155", mono=("shard_rank =" in ln))

L.append("</svg>")
open(
    "/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch22-model-definitions/diagrams/03-weight-load-pipeline.svg",
    "w",
    encoding="utf-8",
).write("\n".join(L))
print("wrote 03-weight-load-pipeline.svg")
