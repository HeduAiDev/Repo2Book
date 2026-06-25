#!/usr/bin/env python3
"""ch17-output-rank-grid: TP×PP rank grid showing which global rank is selected as output_rank.
Example: TP=4, PP=3 → world_size=12; output_rank = 12 - 4*1 = 8 (first TP worker of last PP stage).
"""
import xml.sax.saxutils as xs

def esc(s): return xs.escape(str(s))

W, H = 1100, 620
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

def txt(x, y, t, fs=13, anchor="middle", fill="#1e293b", fw="normal"):
    L.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" font-size="{fs}" font-weight="{fw}" fill="{fill}">{esc(t)}</text>')

# Title
txt(W//2, 36, "_get_output_rank — TP×PP rank 布局（TP=4, PP=3, world_size=12）", 17, "middle", "#0f172a", "bold")
txt(W//2, 58, "output_rank = world_size − tp_size × pcp_size  =  12 − 4 × 1  =  8", 14, "middle", "#334155")

# Grid parameters
TP = 4   # columns
PP = 3   # rows
cell_w = 130
cell_h = 68
margin_left = 220  # left margin for PP stage labels
margin_top = 100   # top margin for TP header

colors = {
    "output": "#fef08a",   # yellow highlight: output_rank
    "last_pp": "#dbeafe",  # blue tint: last PP stage (other TP workers)
    "other": "#f8fafc",    # default
    "output_stroke": "#b45309",
    "last_stroke": "#1d4ed8",
    "other_stroke": "#94a3b8",
}

# Column headers (TP rank)
for tp in range(TP):
    cx = margin_left + tp * cell_w + cell_w // 2
    L.append(f'<rect x="{margin_left + tp*cell_w}" y="{margin_top - 30}" width="{cell_w}" height="26" rx="4" fill="#e0e7ff" stroke="#6366f1" stroke-width="1"/>')
    txt(cx, margin_top - 11, f"TP rank {tp}", 12, "middle", "#3730a3", "bold")

# Row headers (PP stage) + cells
for pp in range(PP):
    ry = margin_top + pp * cell_h
    cy = ry + cell_h // 2

    # PP stage label box
    L.append(f'<rect x="20" y="{ry}" width="180" height="{cell_h}" rx="4" fill="#f0fdf4" stroke="#16a34a" stroke-width="1"/>')
    txt(110, cy - 6, f"PP stage {pp}", 12, "middle", "#15803d", "bold")
    txt(110, cy + 10, f"global ranks {pp*TP}–{pp*TP+TP-1}", 11, "middle", "#166534")

    for tp in range(TP):
        global_rank = pp * TP + tp
        rx = margin_left + tp * cell_w
        is_output = (global_rank == (PP * TP - TP))  # = world_size - tp_size
        is_last_pp = (pp == PP - 1)

        if is_output:
            fill = colors["output"]
            stroke = colors["output_stroke"]
            sw = 2.5
        elif is_last_pp:
            fill = colors["last_pp"]
            stroke = colors["last_stroke"]
            sw = 1.5
        else:
            fill = colors["other"]
            stroke = colors["other_stroke"]
            sw = 1

        L.append(f'<rect x="{rx}" y="{ry}" width="{cell_w}" height="{cell_h}" rx="4" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
        rank_color = "#7c2d12" if is_output else ("#1e40af" if is_last_pp else "#334155")
        txt(rx + cell_w//2, cy - 4, f"rank {global_rank}", 16, "middle", rank_color, "bold")
        if is_output:
            txt(rx + cell_w//2, cy + 14, "★ output_rank", 10.5, "middle", "#92400e", "bold")
        elif is_last_pp and tp == 0:
            pass  # no extra label

# Annotation box below grid (comes first, placed at top of bottom section)
ax = margin_left
ay = margin_top + PP * cell_h + 22
note_w = TP * cell_w
L.append(f'<rect x="{ax}" y="{ay}" width="{note_w}" height="50" rx="5" fill="#fffbeb" stroke="#f59e0b" stroke-width="1.2"/>')
txt(ax + note_w//2, ay + 18, "广播仍是 1 次 enqueue（O(1) 次发送）", 12, "middle", "#92400e")
txt(ax + note_w//2, ay + 36, "应答从 O(world_size=12) 次 dequeue 降到 O(1) 次，省掉 11 次跨进程搬运", 11.5, "middle", "#78350f")

# Legend placed below annotation box (annotation box ends at ay+50)
lx, ly = 30, ay + 60
txt(lx, ly + 14, "图例：", 12, "start", "#0f172a", "bold")
items = [
    (colors["output"], colors["output_stroke"], "output_rank（最后一段 PP 的第一个 TP worker）返回 ModelRunnerOutput"),
    (colors["last_pp"], colors["last_stroke"],  "最后一段 PP 的其他 TP worker（结果冗余，不需收回）"),
    (colors["other"],   colors["other_stroke"],  "其他 PP 段（中间层激活，无最终 token）"),
]
for i, (f, s, label) in enumerate(items):
    bx = lx + 54
    by = ly + 32 + i * 26
    L.append(f'<rect x="{bx}" y="{by - 12}" width="20" height="16" rx="3" fill="{f}" stroke="{s}" stroke-width="1.5"/>')
    txt(bx + 28, by, label, 11.5, "start", "#334155")

svg = '\n'.join(L) + '\n</svg>\n'
out = "/mnt/e/Laboratory/Repo2Book/instances/vllm/artifacts/ch17-worker-and-executor/diagrams/ch17-output-rank-grid.svg"
open(out, "w", encoding="utf-8").write(svg)
print(f"wrote {out}")
