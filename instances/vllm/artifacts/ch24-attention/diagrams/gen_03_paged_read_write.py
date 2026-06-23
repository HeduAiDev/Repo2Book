#!/usr/bin/env python3
"""03-paged-attention-read-write: slot_mapping 写 / block_table 读 两级页表。"""
import xml.sax.saxutils as xs

OUT = "/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch24-attention/diagrams/03-paged-attention-read-write.svg"


def esc(s):
    return xs.escape(str(s))


W, H = 1080, 660
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs>'
    '<marker id="aw" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" '
    'markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" '
    'markerHeight="6" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#2563eb"/></marker>'
    '<marker id="a" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" '
    'markerHeight="5" orient="auto"><path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, lines, fs=12.5, tcol="#0f172a", weight="normal"):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.4"/>')
    n = len(lines)
    cy = y + h / 2 - (n - 1) * (fs + 3) / 2 + fs * 0.35
    for i, t in enumerate(lines):
        L.append(f'<text x="{x + w/2}" y="{cy + i*(fs+3)}" font-family="sans-serif" '
                 f'font-size="{fs}" font-weight="{weight}" text-anchor="middle" fill="{tcol}">{esc(t)}</text>')


def label(x, y, t, fs=12, col="#475569", weight="normal", anchor="middle"):
    L.append(f'<text x="{x}" y="{y}" font-family="sans-serif" font-size="{fs}" '
             f'font-weight="{weight}" text-anchor="{anchor}" fill="{col}">{esc(t)}</text>')


def arrow(x1, y1, x2, y2, marker="a", col="#475569"):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{col}" '
             f'stroke-width="2" marker-end="url(#{marker})"/>')


label(W/2, 30, "PagedAttention：slot_mapping 写 · block_table 读", 18, "#0f172a", "bold")

# 中心：KV cache 物理块（4 个 block，每块 4 槽简画）
BX, BY = 380, 200
BLOCK_W, SLOT_H = 320, 32
label(BX + BLOCK_W/2, BY - 56, "KV cache 张量", 13, "#0f172a", "bold")
label(BX + BLOCK_W/2, BY - 38, "shape = (2, num_blocks, block_size,", 11, "#64748b")
label(BX + BLOCK_W/2, BY - 22, "num_kv_heads, head_size)", 11, "#64748b")

# 画 4 个物理块，每块 4 个槽位
blocks = 4
slots_per = 4
# 哪些 slot 已被本 step 写入（演示 slot 0→blk0off0, 17→blk1off1）
written = {(0, 0): "key[0]", (1, 1): "key[1]"}
for b in range(blocks):
    by = BY + b * (slots_per * SLOT_H + 14)
    label(BX - 30, by + slots_per*SLOT_H/2 + 4, f"block {b}", 11.5, "#475569", "bold", "end")
    for s in range(slots_per):
        sy = by + s * SLOT_H
        key = (b, s)
        if key in written:
            box(BX, sy, BLOCK_W, SLOT_H, "#fecaca", "#dc2626", [f"off {s}  ← {written[key]}"], fs=11.5, weight="bold", tcol="#991b1b")
        else:
            box(BX, sy, BLOCK_W, SLOT_H, "#f1f5f9", "#cbd5e1", [f"off {s}"], fs=11.5, tcol="#94a3b8")

# 左：写
label(150, BY - 30, "写（当前 step）", 14, "#dc2626", "bold")
box(40, BY, 230, 80, "#fee2e2", "#ef4444", [
    "key/value[i]", "= [num_kv_heads, head_size]", "(本 step 新算出的 K/V)"], fs=11.5)
box(40, BY + 110, 230, 96, "#fef2f2", "#dc2626", [
    "slot = slot_mapping[i]",
    "block_idx = slot // block_size",
    "block_offset = slot % block_size",
    "slot == -1 → 跳过（padding）"], fs=11, weight="normal", tcol="#991b1b")
box(40, BY + 230, 230, 50, "#fee2e2", "#ef4444",
    ["reshape_and_cache_flash", "（PagedAttention 写半边）"], fs=11.5, weight="bold")
arrow(150, BY + 80, 150, BY + 110, marker="aw", col="#dc2626")
arrow(150, BY + 206, 150, BY + 230, marker="aw", col="#dc2626")
# 写箭头落入块格子
arrow(270, BY + 255, BX, BY + SLOT_H/2, marker="aw", col="#dc2626")   # →blk0 off0
arrow(270, BY + 262, BX, BY + (slots_per*SLOT_H+14) + SLOT_H + SLOT_H/2, marker="aw", col="#dc2626")  # →blk1 off1

# 右：读
RX = BX + BLOCK_W + 110
label(RX + 90, BY - 30, "读（历史 KV）", 14, "#2563eb", "bold")
box(RX, BY, 230, 80, "#dbeafe", "#3b82f6", [
    "block_table[req]", "= [逻辑块号...]", "如 [0, 1, 2, 3]"], fs=11.5, weight="bold")
box(RX, BY + 110, 230, 96, "#eff6ff", "#2563eb", [
    "顺着块号列出该请求的",
    "历史 KV 块；再按 seq_lens",
    "决定读多少个块",
    "与 query 算 causal 注意力"], fs=11, tcol="#1e40af")
box(RX, BY + 230, 230, 50, "#dbeafe", "#3b82f6",
    ["flash_attn_varlen_func", "（PagedAttention 读半边）"], fs=11.5, weight="bold")
arrow(RX + 115, BY + 80, RX + 115, BY + 110, marker="ar", col="#2563eb")
arrow(RX + 115, BY + 206, RX + 115, BY + 230, marker="ar", col="#2563eb")
# 读箭头从块格子取出
arrow(BX + BLOCK_W, BY + SLOT_H/2, RX, BY + 30, marker="ar", col="#2563eb")
arrow(BX + BLOCK_W, BY + (slots_per*SLOT_H+14) + slots_per*SLOT_H/2, RX, BY + 60, marker="ar", col="#2563eb")

# 底注：两级页表
L.append(f'<rect x="160" y="{H-78}" width="760" height="56" rx="8" fill="#f8fafc" stroke="#cbd5e1"/>')
label(W/2, H - 52, "两级页表：slot_mapping = token → 物理槽（管『新 token 写哪』）", 12.5, "#991b1b", "bold")
label(W/2, H - 32, "block_table = 请求 → 逻辑块号列表（管『历史 KV 读哪』）", 12.5, "#1e40af", "bold")

L.append('</svg>')
open(OUT, "w").write('\n'.join(L))
print("ok", OUT)
