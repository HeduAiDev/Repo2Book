#!/usr/bin/env python3
"""ch03 figure: async_scheduling tri-state decision tree."""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1280, 620
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append('<defs><marker id="a" viewBox="0 0 10 6" refX="9" refY="3" '
         'markerWidth="7" markerHeight="5" orient="auto">'
         '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker></defs>')
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(f'<text x="40" y="38" font-size="20" font-weight="bold" fill="#0f172a">'
         f'async_scheduling 三态决策（VllmConfig.__post_init__）</text>')


def box(x, y, w, h, text_lines, fill, stroke, fs=13):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="7" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.6"/>')
    n = len(text_lines)
    for i, t in enumerate(text_lines):
        ty = y + h / 2 + (i - (n - 1) / 2) * (fs + 4) + fs / 3
        L.append(f'<text x="{x + w / 2}" y="{ty:.0f}" text-anchor="middle" '
                 f'font-size="{fs}" fill="#0f172a">{esc(t)}</text>')


def arrow(x1, y1, x2, y2, label=None, lx=None, ly=None):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
             f'stroke="#475569" stroke-width="1.8" marker-end="url(#a)"/>')
    if label:
        L.append(f'<text x="{lx}" y="{ly}" text-anchor="middle" font-size="12" '
                 f'font-weight="bold" fill="#334155">{esc(label)}</text>')


# Root
rootx, rooty, rw, rh = 440, 60, 230, 56
box(rootx, rooty, rw, rh, ["scheduler_config", ".async_scheduling 的值？"], "#e0e7ff", "#4338ca", 13)
root_cx = rootx + rw / 2
root_by = rooty + rh

# Three branch heads
b_true_x, b_false_x, b_none_x = 130, 470, 810
branch_y = 200
bw, bh = 240, 50
# True branch
arrow(root_cx, root_by, b_true_x + bw / 2, branch_y, "True (显式开)", (root_cx + b_true_x + bw / 2) / 2 - 40, 160)
box(b_true_x, branch_y, bw, bh, ["硬校验：执行器", "supports_async_scheduling()?"], "#fef9c3", "#ca8a04")
# False branch
arrow(root_cx, root_by, b_false_x + bw / 2, branch_y, "False (显式关)", root_cx, 160)
box(b_false_x, branch_y, bw, bh, ["直接采纳"], "#f1f5f9", "#64748b")
# None branch
arrow(root_cx, root_by, b_none_x + bw / 2, branch_y, "None (自动)", (root_cx + b_none_x + bw / 2) / 2 + 40, 160)
box(b_none_x, branch_y, bw, bh, ["逐条排除不兼容项"], "#dcfce7", "#16a34a")

# True branch leaves
ty = 320
arrow(b_true_x + bw / 2, branch_y + bh, b_true_x - 10 + bw / 4, ty, "否", b_true_x + 20, ty - 8)
box(b_true_x - 100, ty, 180, 46, ["raise ValueError"], "#fee2e2", "#dc2626")
arrow(b_true_x + bw / 2, branch_y + bh, b_true_x + bw - 20, ty, "是", b_true_x + bw - 20, ty - 8)
box(b_true_x + bw - 100, ty, 170, 46, ["enabled"], "#dcfce7", "#16a34a")

# False branch leaf
arrow(b_false_x + bw / 2, branch_y + bh, b_false_x + bw / 2, ty)
box(b_false_x + bw / 2 - 90, ty, 180, 46, ["disabled"], "#f1f5f9", "#64748b")

# None branch chained checks
checks = [
    "runner_type == pooling?",
    "有不兼容的投机方法?",
    "执行器不支持 async?",
]
ny = 320
cx = b_none_x + bw / 2
prev_y = branch_y + bh
for i, c in enumerate(checks):
    cy = ny + i * 70
    box(b_none_x, cy, bw, 46, [c], "#ecfeff", "#0891b2")
    arrow(cx, prev_y, cx, cy)
    # "yes -> disabled" to the right
    arrow(b_none_x + bw, cy + 23, b_none_x + bw + 70, cy + 23, "是", b_none_x + bw + 30, cy + 16)
    box(b_none_x + bw + 70, cy, 150, 40, ["disabled"], "#f1f5f9", "#64748b", 12)
    prev_y = cy + 46
# all-no -> enabled
arrow(cx, prev_y, cx, prev_y + 28)
box(b_none_x, prev_y + 28, bw, 46, ["全否 → enabled"], "#dcfce7", "#16a34a")

L.append(f'<text x="40" y="{H - 16}" font-size="12" fill="#64748b">'
         f'默认 None 表示「自动」：默认想开 async，但遇到 pooling / 不兼容投机 / 执行器不支持就安全退化为 disabled。</text>')
L.append('</svg>')

with open("async-scheduling-decision.svg", "w", encoding="utf-8") as f:
    f.write("\n".join(L))
print("wrote async-scheduling-decision.svg")
