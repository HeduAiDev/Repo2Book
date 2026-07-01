#!/usr/bin/env python3
"""ch30 收束图：vLLM 处处留扩展点，昇腾往每个扩展点登记一个实现。
左=对应章，中=vLLM 扩展点，右=昇腾挂进去的实现。本章三行高亮。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


rows = [
    # chapter, vllm extension point, ascend impl, highlight
    ("ch02", "Platform 注册", "NPUPlatform", False),
    ("ch23", "CustomOp 注册替换", "Ascend* CustomOp 子类", False),
    ("ch27", "Quantization 注册", "AscendQuantConfig", False),
    ("ch29", "Proposer 工厂分发", "Ascend* Proposer", False),
    ("ch30", "ModelRegistry.register_model", "AscendDeepseekV4ForCausalLM", True),
    ("ch30", "_all_lora_classes 元组", "4 × Ascend*LinearWithLoRA", True),
    ("ch30", "model-loader registry", "ModelNetLoaderElastic", True),
]

# geometry
W = 1180
top = 130
row_h = 74
gap = 16
n = len(rows)
H = top + n * (row_h + gap) + 40

ch_x, ch_w = 40, 120
mid_x, mid_w = 360, 420
imp_x, imp_w = 850, 300

L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">']
L.append(
    '<defs><marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" '
    'markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#7c3aed"/></marker>'
    '<marker id="arg" viewBox="0 0 10 6" refX="9" refY="3" '
    'markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#94a3b8"/></marker></defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# title
L.append(f'<text x="{W/2}" y="46" text-anchor="middle" font-family="sans-serif" '
         f'font-size="30" font-weight="bold" fill="#1e293b">'
         f'{esc("vLLM 处处留扩展点 · 昇腾往每个扩展点登记一个实现")}</text>')
L.append(f'<text x="{W/2}" y="78" text-anchor="middle" font-family="sans-serif" '
         f'font-size="17" fill="#64748b">'
         f'{esc("OOT 插件 = 注册 + 薄壳继承 + 必要时特化（紫色为本章收口的最后三个扩展点）")}</text>')

# column headers
heads = [(ch_x + ch_w / 2, "对应章"),
         (mid_x + mid_w / 2, "vLLM 留出的扩展点"),
         (imp_x + imp_w / 2, "昇腾挂进去的实现")]
for hx, ht in heads:
    L.append(f'<text x="{hx}" y="{top-14}" text-anchor="middle" font-family="sans-serif" '
             f'font-size="16" font-weight="bold" fill="#475569">{esc(ht)}</text>')

for i, (ch, mid, imp, hl) in enumerate(rows):
    y = top + i * (row_h + gap)
    cy = y + row_h / 2
    if hl:
        mid_fill, mid_stroke, mid_tc = "#f3e8ff", "#7c3aed", "#6b21a8"
        imp_fill, imp_stroke, imp_tc = "#ede9fe", "#7c3aed", "#5b21b6"
        ch_fill, ch_stroke, ch_tc = "#7c3aed", "#7c3aed", "#ffffff"
        ar = "ar"
    else:
        mid_fill, mid_stroke, mid_tc = "#f1f5f9", "#cbd5e1", "#334155"
        imp_fill, imp_stroke, imp_tc = "#f8fafc", "#cbd5e1", "#475569"
        ch_fill, ch_stroke, ch_tc = "#e2e8f0", "#cbd5e1", "#475569"
        ar = "arg"

    # chapter chip
    L.append(f'<rect x="{ch_x}" y="{cy-18}" width="{ch_w}" height="36" rx="18" '
             f'fill="{ch_fill}" stroke="{ch_stroke}" stroke-width="1.5"/>')
    L.append(f'<text x="{ch_x+ch_w/2}" y="{cy+6}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="17" font-weight="bold" '
             f'fill="{ch_tc}">{esc(ch)}</text>')

    # arrow chapter -> mid
    L.append(f'<line x1="{ch_x+ch_w+6}" y1="{cy}" x2="{mid_x-6}" y2="{cy}" '
             f'stroke="{mid_stroke}" stroke-width="2" marker-end="url(#{ar})"/>')

    # mid box (extension point)
    L.append(f'<rect x="{mid_x}" y="{y}" width="{mid_w}" height="{row_h}" rx="10" '
             f'fill="{mid_fill}" stroke="{mid_stroke}" stroke-width="2"/>')
    L.append(f'<text x="{mid_x+mid_w/2}" y="{cy+7}" text-anchor="middle" '
             f'font-family="monospace" font-size="20" font-weight="bold" '
             f'fill="{mid_tc}">{esc(mid)}</text>')

    # arrow mid -> imp
    L.append(f'<line x1="{mid_x+mid_w+6}" y1="{cy}" x2="{imp_x-6}" y2="{cy}" '
             f'stroke="{imp_stroke}" stroke-width="2" marker-end="url(#{ar})"/>')

    # imp box
    L.append(f'<rect x="{imp_x}" y="{y}" width="{imp_w}" height="{row_h}" rx="10" '
             f'fill="{imp_fill}" stroke="{imp_stroke}" stroke-width="2"/>')
    L.append(f'<text x="{imp_x+imp_w/2}" y="{cy+6}" text-anchor="middle" '
             f'font-family="sans-serif" font-size="16" font-weight="bold" '
             f'fill="{imp_tc}">{esc(imp)}</text>')

L.append('</svg>')
svg = '\n'.join(L)
with open('fig30-1-extension-points.svg', 'w', encoding='utf-8') as f:
    f.write(svg)
print("wrote fig30-1-extension-points.svg", W, H)
