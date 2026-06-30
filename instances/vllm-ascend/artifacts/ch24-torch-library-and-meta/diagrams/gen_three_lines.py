#!/usr/bin/env python3
"""fig24-2: 三条注册线汇入 torch 算子注册表（PrivateUse1 列 / Meta 列）。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1340, 720
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="7" markerHeight="5" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#64748b"/></marker>'
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


def arrow(x1, y1, x2, y2, color="#64748b"):
    L.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="2.2" marker-end="url(#ar)"/>')


text(W / 2, 40, "三条注册线，汇入同一张 torch 算子注册表", 24, "middle", "#0f172a", "bold")

# three swimlanes on the left
lane_x, lane_w = 60, 620
rows = [
    (90, "①  C++ 真实现", "#fff7ed", "#f97316", "#c2410c", [
        "import vllm_ascend.vllm_ascend_C （加载 .so）",
        "csrc/torch_binding.cpp",
        "TORCH_LIBRARY_EXPAND(_C_ascend, ops)",
        "63 个 ops.def + ops.impl(name, kPrivateUse1, &真实现)",
    ]),
    (250, "②  C++ meta + Python 兜底", "#eff6ff", "#3b82f6", "#1d4ed8", [
        "同一个 .so 内 / import meta_registration",
        "csrc/torch_binding_meta.cpp",
        "TORCH_LIBRARY_IMPL_EXPAND(_C_ascend, Meta, ops)",
        "57 个 meta（缺口 6 个无 meta）+ Python 补 3 个",
    ]),
    (410, "③  纯 Python op", "#f0fdf4", "#22c55e", "#15803d", [
        "import vllm_ascend.ops.register_custom_ops",
        "vllm_ascend/ops/register_custom_ops.py",
        "direct_register_custom_op × 10",
        "每个 = impl(PrivateUse1) + _register_fake(fake)",
    ]),
]
for y0, title, fill, stroke, tcol, lines in rows:
    box(lane_x, y0, lane_w, 130, fill, stroke, 12, 2.0)
    text(lane_x + 18, y0 + 30, title, 17, "start", tcol, "bold")
    for i, ln in enumerate(lines):
        mono = i >= 1
        text(lane_x + 18, y0 + 56 + i * 22, ln, 12.5, "start", "#334155", "normal", mono)

# right: dispatcher registry table
tx, ty, tw, th = 780, 110, 500, 430
box(tx, ty, tw, th, "#f8fafc", "#475569", 12, 2.0)
text(tx + tw / 2, ty + 34, "torch Dispatcher 注册表", 19, "middle", "#0f172a", "bold")
text(tx + tw / 2, ty + 58, "算子全名 → 按 dispatch key 索引的实现", 13, "middle", "#64748b")

# two columns
col1 = tx + 30
col2 = tx + 270
text(col1, ty + 96, "PrivateUse1（真算）", 15, "start", "#c2410c", "bold")
text(col2, ty + 96, "Meta（推形状）", 15, "start", "#1d4ed8", "bold")
L.append(f'<line x1="{tx+250}" y1="{ty+76}" x2="{tx+250}" y2="{ty+th-90}" stroke="#cbd5e1" stroke-width="1.4"/>')
L.append(f'<line x1="{tx+20}" y1="{ty+108}" x2="{tx+tw-20}" y2="{ty+108}" stroke="#cbd5e1" stroke-width="1.4"/>')

ns_rows = [
    (ty + 138, "_C_ascend::*", "63 真实现", "57 + 3"),
    (ty + 180, "  其中缺口 6 个", "有真实现", "—  无 meta"),
    (ty + 222, "vllm::*（10 个）", "10 真实现", "10 fake"),
]
for gi, (yy, name, c1, c2) in enumerate(ns_rows):
    # faint band separator above each group (except the first) to lock left/right rows together
    if gi > 0:
        sep_y = yy - 18
        L.append(f'<line x1="{tx+20}" y1="{sep_y}" x2="{tx+tw-20}" y2="{sep_y}" stroke="#e2e8f0" stroke-width="1.2"/>')
    col = "#dc2626" if "缺口" in name else "#334155"
    text(col1, yy, name, 13.5, "start", col, "bold", mono=("::" in name))
    text(col1 + 12, yy + 22, c1, 13, "start", "#9a3412")
    text(col2, yy + 22, c2, 13, "start", ("#dc2626" if "无" in c2 else "#1e40af"))

# product line
box(tx + 24, ty + 300, tw - 48, 100, "#eef2ff", "#6366f1", 10, 1.6)
text(tx + tw / 2, ty + 328, "注册产物（Python 侧可调用）", 14, "middle", "#4338ca", "bold")
text(tx + tw / 2, ty + 356, "torch.ops._C_ascend.<op>", 14, "middle", "#334155", "normal", mono=True)
text(tx + tw / 2, ty + 382, "torch.ops.vllm.<op>", 14, "middle", "#334155", "normal", mono=True)

# arrows lanes -> table
for y0, *_ in rows:
    arrow(lane_x + lane_w + 4, y0 + 65, tx - 4, ty + 220)

text(W / 2, 690, "推理走 PrivateUse1 列真算；图捕获走 Meta 列只推形状 —— 缺口 6 个 op 进不了捕获图", 14, "middle", "#475569", "bold")

L.append('</svg>')
open("fig24-2-three-lines.svg", "w").write('\n'.join(L))
print("wrote fig24-2-three-lines.svg")
