#!/usr/bin/env python3
"""fig24-3: 图捕获期形状传播链 —— 一处缺 meta 即断链。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1320, 520
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#3b82f6"/></marker>'
    '<marker id="arX" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#dc2626"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, rx=10, sw=1.6):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def text(x, y, s, size=15, anchor="start", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


def arrow(x1, y1, x2, y2, color="#3b82f6", marker="ar"):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.4" marker-end="url(#{marker})"/>')


def tensor(cx, cy, label):
    r = 46
    L.append(f'<rect x="{cx-r}" y="{cy-26}" width="{2*r}" height="52" rx="26" fill="#eff6ff" stroke="#3b82f6" stroke-width="1.8"/>')
    text(cx, cy + 5, label, 13, "middle", "#1d4ed8", "bold", mono=True)


def op(cx, cy, name, ok=True):
    w, h = 150, 60
    fill = "#f0fdf4" if ok else "#fef2f2"
    stroke = "#22c55e" if ok else "#dc2626"
    box(cx - w / 2, cy - h / 2, w, h, fill, stroke, 10, 2.2)
    text(cx, cy - 4, name, 12.5, "middle", "#166534" if ok else "#b91c1c", "bold", mono=True)
    text(cx, cy + 16, "meta ✓" if ok else "no meta ✗", 12, "middle", "#15803d" if ok else "#dc2626", "bold")


text(W / 2, 40, "图捕获期：meta 把形状一站一站推下去", 24, "middle", "#0f172a", "bold")
text(W / 2, 68, "trace 用 Meta 张量（只有 shape/dtype），每遇算子调它的 meta 求输出形状，喂给下一个", 14, "middle", "#64748b")

yc = 200
xs_t = [120, 430, 740]
xs_o = [275, 585]
tensor(xs_t[0], yc, "[B,S]")
op(xs_o[0], yc, "rms_norm")
tensor(xs_t[1], yc, "[B,S]")
op(xs_o[1], yc, "lightning_idx")
tensor(xs_t[2], yc, "[B,S,N,sparse_count]")
arrow(xs_t[0] + 48, yc, xs_o[0] - 78, yc)
arrow(xs_o[0] + 78, yc, xs_t[1] - 48, yc)
arrow(xs_t[1] + 48, yc, xs_o[1] - 78, yc)
arrow(xs_o[1] + 78, yc, xs_t[2] - 48, yc)

# continue to a broken op
op4x = 1000
op(op4x, yc, "apply_top_k", ok=False)
arrow(xs_t[2] + 48, yc, op4x - 78, yc)
# broken arrow out
L.append(f'<line x1="{op4x+78}" y1="{yc}" x2="{op4x+150}" y2="{yc}" stroke="#dc2626" stroke-width="2.4" stroke-dasharray="7 6" marker-end="url(#arX)"/>')
text(op4x + 175, yc - 12, "？", 26, "start", "#dc2626", "bold")
text(op4x + 120, yc + 38, "推不出 shape", 13, "middle", "#dc2626", "bold")
text(op4x + 120, yc + 58, "→ 链断，图捕获中止", 13, "middle", "#dc2626", "bold")

# shape inference callout for lightning_idx
cox, coy, cow, coh = 430, 320, 470, 150
box(cox, coy, cow, coh, "#f8fafc", "#94a3b8", 12, 1.6)
text(cox + cow / 2, coy + 30, "复杂 meta 怎么推形状（npu_lightning_indexer）", 14, "middle", "#334155", "bold")
text(cox + 24, coy + 62, "output_size = {query.size(0), query.size(1),", 13, "start", "#475569", "normal", mono=True)
text(cox + 24, coy + 86, "               key.size(2), sparse_count}", 13, "start", "#475569", "normal", mono=True)
text(cox + 24, coy + 116, "at::empty(output_size, ...dtype(kInt))", 13, "start", "#475569", "normal", mono=True)
text(cox + 24, coy + 140, "trace 器自己不会算这套逻辑 → 必须靠 meta 告诉它", 12.5, "start", "#b45309", "bold")
arrow(xs_o[1], yc + 32, cox + 100, coy - 4, "#94a3b8")

L.append('</svg>')
open("fig24-3-shape-chain.svg", "w").write('\n'.join(L))
print("wrote fig24-3-shape-chain.svg")
