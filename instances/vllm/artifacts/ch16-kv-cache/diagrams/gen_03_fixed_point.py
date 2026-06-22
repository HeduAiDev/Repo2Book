#!/usr/bin/env python3
"""03-fixed-point-iteration-trace: state-evolution table of the fixed-point loop."""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

w, h = 1000, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" font-family="sans-serif">']
L.append(f'<rect width="{w}" height="{h}" fill="white"/>')

def txt(x, y, t, fs=14, anchor="start", fill="#334155", tw="normal"):
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" font-weight="{tw}" fill="{fill}">{esc(t)}</text>')

def cell(x, y, cw, ch, fill, stroke, lines, fs=14, tc="#0f172a", tw="normal"):
    L.append(f'<rect x="{x}" y="{y}" width="{cw}" height="{ch}" fill="{fill}" stroke="{stroke}" stroke-width="1.2"/>')
    n = len(lines)
    cyy = y + ch/2 - (n-1)*(fs+3)/2 + fs/2 - 2
    for i, t in enumerate(lines):
        L.append(f'<text x="{x+cw/2}" y="{cyy+i*(fs+3)}" text-anchor="middle" font-size="{fs}" font-weight="{tw}" fill="{tc}">{esc(t)}</text>')

txt(40, 40, "Hybrid 不动点迭代：每类型对候选命中长度「接受或缩短」", 20, "start", "#0f172a", "bold")
txt(40, 66, "示例：max_cache_hit_length = 512，block_size = 128，full + sliding window 两组", 14, "start", "#64748b")

# table grid
x0 = 60
y0 = 100
col_w = [150, 270, 270, 200]
headers = ["迭代轮次", "full attn（排首）", "sliding window", "curr_hit_length"]
rows = [
    ("第 1 轮", "首次查表 → 接受 512", "窗内连续 → 缩短到 384", "512 → 384（变了，重启）", "#fef9c3"),
    ("第 2 轮", "下闭包：截到 384", "在 384 处接受 384", "384 ≥ 384（收敛退出）", "#dcfce7"),
]
rh = 80
hh = 50

# header
cx = x0
for i, head in enumerate(headers):
    cell(cx, y0, col_w[i], hh, "#1e293b", "#0f172a", [head], 14.5, "white", "bold")
    cx += col_w[i]

# body
ry = y0 + hh
for label, full_c, sw_c, res_c, resfill in rows:
    cx = x0
    cell(cx, ry, col_w[0], rh, "#f1f5f9", "#94a3b8", [label], 14, "#0f172a", "bold"); cx += col_w[0]
    cell(cx, ry, col_w[1], rh, "#eff6ff", "#94a3b8", [full_c], 13.5); cx += col_w[1]
    cell(cx, ry, col_w[2], rh, "#fef2f2", "#94a3b8", [sw_c], 13.5); cx += col_w[2]
    cell(cx, ry, col_w[3], rh, resfill, "#94a3b8", [res_c], 13, "#0f172a", "bold")
    ry += rh

# convergence note box
ny = ry + 36
L.append(f'<rect x="{x0}" y="{ny}" width="{sum(col_w)}" height="120" rx="8" fill="#f8fafc" stroke="#cbd5e1" stroke-width="1.5"/>')
txt(x0+20, ny+32, "为什么必收敛：", 15, "start", "#0f172a", "bold")
txt(x0+20, ny+58, "• 每类型把候选长度 L 映射为 L′ ≤ L（接受则 L′ = L，缩短则 L′ < L）→ 单调不增", 13.5, "start", "#334155")
txt(x0+20, ny+82, "• L 取值落在 {0, lcm, 2·lcm, …} 的有限集合，下界为 0 → 至多 L / lcm 轮触底", 13.5, "start", "#334155")
txt(x0+20, ny+106, "• simple hybrid（1 full + 1 other，full 排首）：full 给上界后 other 至多缩一次 → 一轮足够", 13.5, "start", "#334155")

svg = '\n'.join(L) + '\n</svg>\n'
open("/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch16-kv-cache/diagrams/03-fixed-point-iteration-trace.svg", "w", encoding="utf-8").write(svg)
print("wrote 03")
