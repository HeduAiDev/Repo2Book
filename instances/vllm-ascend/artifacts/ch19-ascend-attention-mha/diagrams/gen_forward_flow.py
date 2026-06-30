#!/usr/bin/env python3
"""ch19 前向数据流：forward → reshape_and_cache → forward_impl 菱形分流 → paged / fused → output。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(str(s))


W, H = 1180, 760
TXT = "#1e293b"
BLUE = "#1e3a8a"
ORANGE = "#b45309"
GREEN = "#15803d"
GREY = "#64748b"

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append('<defs><marker id="ar" viewBox="0 0 10 8" refX="9" refY="4" markerWidth="9" markerHeight="7" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="#475569"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="{W/2}" y="36" text-anchor="middle" font-size="24" font-weight="bold" fill="{TXT}">AscendAttentionBackendImpl.forward 的数据流</text>')
L.append(f'<text x="{W/2}" y="54" text-anchor="middle" font-size="13" fill="{GREY}">两条算子路径在图捕获下都先预取 workspace 再录图——workspace 预取是 NPU 算子的共性特有节拍，非 paged 专属</text>')

cx = W / 2


def box(x, y, w, h, lines, fill, stroke, fs=15, rx=10, sub=None):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')
    n = len(lines)
    for i, ln in enumerate(lines):
        ty = y + h / 2 + (i - (n - 1) / 2) * (fs + 4) + 5
        fw = "bold" if i == 0 else "normal"
        fsz = fs if i == 0 else fs - 2
        L.append(f'<text x="{x + w/2}" y="{ty}" text-anchor="middle" font-size="{fsz}" font-weight="{fw}" fill="{stroke}">{esc(ln)}</text>')


def varrow(x, y1, y2, label=None):
    L.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2 - 2}" stroke="#475569" stroke-width="2.2" marker-end="url(#ar)"/>')
    if label:
        L.append(f'<text x="{x + 10}" y="{(y1+y2)/2 + 4}" font-size="12.5" fill="{GREY}">{esc(label)}</text>')


# 1. 输入
box(cx - 270, 60, 540, 56, ["query / key / value  [num_tokens, num_heads, head_size]"], "#eef2ff", BLUE, fs=14)
varrow(cx, 116, 150)
# 2. reshape_and_cache
box(cx - 250, 150, 500, 70,
    ["reshape_and_cache → DeviceOperator → _npu_reshape_and_cache",
     "按 slot_mapping 把本步 K/V 散写回分页 cache"], "#fff7ed", ORANGE, fs=14.5)
varrow(cx, 220, 256)
# 3. 菱形判定
dy = 256
dh = 132
dw = 560
# 画菱形
mx, my = cx, dy + dh / 2
L.append(f'<polygon points="{mx},{dy} {mx+dw/2},{my} {mx},{dy+dh} {mx-dw/2},{my}" fill="#f1f5f9" stroke="{BLUE}" stroke-width="2"/>')
L.append(f'<text x="{mx}" y="{my - 22}" text-anchor="middle" font-size="15" font-weight="bold" fill="{BLUE}">forward_impl 分流判定</text>')
L.append(f'<text x="{mx}" y="{my + 2}" text-anchor="middle" font-size="13.5" fill="{TXT}">attn_state == DecodeOnly</text>')
L.append(f'<text x="{mx}" y="{my + 22}" text-anchor="middle" font-size="13.5" fill="{TXT}">且 using_paged_attention(num_tokens) 且 sliding_window is None ?</text>')

# 左分支：是 → paged
lbx = 40
rbx = 720
branch_y = dy + dh + 60
# 是（左）
L.append(f'<line x1="{mx - dw/2}" y1="{my}" x2="{lbx + 200}" y2="{my}" stroke="#475569" stroke-width="2"/>')
L.append(f'<line x1="{lbx + 200}" y1="{my}" x2="{lbx + 200}" y2="{branch_y - 2}" stroke="#475569" stroke-width="2.2" marker-end="url(#ar)"/>')
L.append(f'<text x="{mx - dw/2 - 18}" y="{my - 8}" text-anchor="middle" font-size="14" font-weight="bold" fill="{GREEN}">是</text>')
box(lbx, branch_y, 400, 78,
    ["forward_paged_attention",
     "torch_npu._npu_paged_attention",
     "吃 block_table + context_lens(=seq_lens)"], "#f0fdf4", GREEN, fs=14.5)
# 否（右）
L.append(f'<line x1="{mx + dw/2}" y1="{my}" x2="{rbx + 200}" y2="{my}" stroke="#475569" stroke-width="2"/>')
L.append(f'<line x1="{rbx + 200}" y1="{my}" x2="{rbx + 200}" y2="{branch_y - 2}" stroke="#475569" stroke-width="2.2" marker-end="url(#ar)"/>')
L.append(f'<text x="{mx + dw/2 + 18}" y="{my - 8}" text-anchor="middle" font-size="14" font-weight="bold" fill="{ORANGE}">否</text>')
box(rbx, branch_y, 400, 78,
    ["forward_fused_infer_attention",
     "_get_fia_params 按五态整理 KV → ",
     "torch_npu.npu_fused_infer_attention_score (TND)"], "#fff7ed", ORANGE, fs=13.5)

# workspace 侧注（两条算子路径在图捕获下都先预取 workspace 再录图——非 paged 专属）
wy = branch_y + 96
# 左：paged 的 workspace 预取
L.append(f'<rect x="{lbx}" y="{wy}" width="400" height="44" rx="8" fill="#fffbeb" stroke="#d97706" stroke-dasharray="5,4"/>')
L.append(f'<text x="{lbx + 200}" y="{wy + 27}" text-anchor="middle" font-size="12.5" fill="#92400e">图捕获：_npu_paged_attention_get_workspace 预取</text>')
L.append(f'<line x1="{lbx+200}" y1="{branch_y+78}" x2="{lbx+200}" y2="{wy-2}" stroke="#d97706" stroke-width="1.6" stroke-dasharray="4,3" marker-end="url(#ar)"/>')
# 右：fused 的 workspace 预取（对称——同样的特有节拍）
L.append(f'<rect x="{rbx}" y="{wy}" width="400" height="44" rx="8" fill="#fffbeb" stroke="#d97706" stroke-dasharray="5,4"/>')
L.append(f'<text x="{rbx + 200}" y="{wy + 27}" text-anchor="middle" font-size="12.5" fill="#92400e">图捕获：*_get_max_workspace 预取</text>')
L.append(f'<line x1="{rbx+200}" y1="{branch_y+78}" x2="{rbx+200}" y2="{wy-2}" stroke="#d97706" stroke-width="1.6" stroke-dasharray="4,3" marker-end="url(#ar)"/>')

# 汇合 → output（两路都经各自 workspace 预取后汇合）
out_y = wy + 90
ocx = cx
L.append(f'<line x1="{lbx+200}" y1="{wy+44}" x2="{lbx+200}" y2="{out_y - 22}" stroke="#475569" stroke-width="2"/>')
L.append(f'<line x1="{lbx+200}" y1="{out_y - 22}" x2="{ocx}" y2="{out_y - 22}" stroke="#475569" stroke-width="2"/>')
L.append(f'<line x1="{rbx+200}" y1="{wy+44}" x2="{rbx+200}" y2="{out_y - 22}" stroke="#475569" stroke-width="2"/>')
L.append(f'<line x1="{rbx+200}" y1="{out_y - 22}" x2="{ocx}" y2="{out_y - 22}" stroke="#475569" stroke-width="2"/>')
L.append(f'<line x1="{ocx}" y1="{out_y - 22}" x2="{ocx}" y2="{out_y - 2}" stroke="#475569" stroke-width="2.2" marker-end="url(#ar)"/>')
box(ocx - 230, out_y, 460, 54, ["output[:num_tokens]  →  [num_tokens, num_heads * head_size]"], "#eef2ff", BLUE, fs=14)

L.append('</svg>')
open("ch19-forward-flow.svg", "w").write('\n'.join(L))
print("wrote ch19-forward-flow.svg", W, H)
