#!/usr/bin/env python3
"""fig28-4: HAS_TRITON 优雅回退 —— 每个热点都是「有 Triton 走昇腾内核，否则走基类/纯 torch」。加速非依赖。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1280, 560
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#475569"/></marker>'
    '</defs>'
)


def box(x, y, w, h, fill, stroke, rx=10, sw=1.6):
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def diamond(cx, cy, w, h, fill, stroke):
    L.append(f'<polygon points="{cx},{cy - h/2} {cx + w/2},{cy} {cx},{cy + h/2} {cx - w/2},{cy}" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.8"/>')


def text(x, y, s, size=14, anchor="start", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


text(W / 2, 42, "加速，不是依赖：HAS_TRITON 不可用时优雅回退", 24, "middle", "#0f172a", "bold")
text(W / 2, 70, "penalties / top-k-top-p / 拒绝采样 —— 每个热点都长这一个样", 14, "middle", "#64748b")

# 判定菱形
diamond(W / 2, 150, 240, 96, "#fef9c3", "#ca8a04")
text(W / 2, 146, "if HAS_TRITON ?", 16, "middle", "#854d0e", "bold", mono=True)
text(W / 2, 168, "（昇腾 Triton 是否可用）", 12, "middle", "#a16207")

# 左：是 → Triton
box(150, 270, 440, 220, "#ecfdf5", "#10b981", 12, 1.8)
text(370, 300, "是 → 走昇腾 Triton-Ascend 内核（加速）", 15, "middle", "#15803d", "bold")
triton = [
    "apply_penalties_triton(...)",
    "rejection_greedy_sample_with_triton(...)",
    "rejection_random_sample_kernel[...](...)",
    "sample_recovered_tokens_kernel[...](...)",
    "npu_apply_top_k_top_p（A2/A3 AscendC 算子）",
]
for i, t in enumerate(triton):
    text(180, 340 + i * 30, "• " + t, 12.5, "start", "#166534", mono=True)

# 右：否 → 回退
box(690, 270, 440, 220, "#fef2f2", "#ef4444", 12, 1.8)
text(910, 300, "否 → 回退基类 / 纯 torch（仍正确）", 15, "middle", "#b91c1c", "bold")
fb = [
    "Sampler.apply_penalties(...)  基类原版",
    "rejection_greedy_sample_pytorch(...)",
    "rejection_random_sample_pytorch(...)",
    "sample_recovered_tokens_pytorch(...)",
    "_apply_top_k_top_p_pytorch（sort/cumsum）",
]
for i, t in enumerate(fb):
    text(720, 340 + i * 30, "• " + t, 12.5, "start", "#7f1d1d", mono=True)

# 箭头
L.append(f'<line x1="{W/2 - 90}" y1="186" x2="400" y2="270" stroke="#15803d" stroke-width="2" marker-end="url(#ar)"/>')
L.append(f'<line x1="{W/2 + 90}" y1="186" x2="880" y2="270" stroke="#b91c1c" stroke-width="2" marker-end="url(#ar)"/>')
text(330, 240, "是", 15, "middle", "#15803d", "bold")
text(950, 240, "否", 15, "middle", "#b91c1c", "bold")

text(W / 2, 528, "两条路结果同分布：Triton 只换更快的内核，不改采样语义。无 NPU Triton 也能跑，只是慢一点。",
     13, "middle", "#475569")

L.append('</svg>')
open("fig28-4-fallback.svg", "w").write('\n'.join(L))
print("wrote fig28-4-fallback.svg")
