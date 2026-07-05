#!/usr/bin/env python3
"""fig-head-not-body: 换头不换身 —— 身(灰，继承不变) vs 头(红，forward 实现被换)。"""
import xml.sax.saxutils as xs


def esc(s):
    return xs.escape(s)


W, H = 1280, 700
L = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="sans-serif">']
L.append(
    '<defs>'
    '<marker id="ar" viewBox="0 0 10 6" refX="9" refY="3" markerWidth="8" markerHeight="6" orient="auto">'
    '<path d="M0,0 L10,3 L0,6 Z" fill="#b91c1c"/></marker>'
    '</defs>'
)
L.append(f'<rect width="{W}" height="{H}" fill="white"/>')


def box(x, y, w, h, fill, stroke, rx=10, sw=1.6, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    L.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def text(x, y, s, size=15, anchor="start", fill="#1e293b", weight="normal", mono=False):
    fam = "monospace" if mono else "sans-serif"
    L.append(
        f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" '
        f'fill="{fill}" font-weight="{weight}" font-family="{fam}">{esc(s)}</text>'
    )


text(W / 2, 42, "换头不换身：同一个算子，只替 forward 实现", 24, "middle", "#0f172a", "bold")
text(W / 2, 72, "以 SiluAndMul 为例 —— 身（接口/权重/注册位置）继承自基座，头（forward）从 CUDA 换成 NPU", 14, "middle", "#64748b")

col_w = 480
gap = 80
lx = (W - 2 * col_w - gap) / 2
rx = lx + col_w + gap
top = 110

for cx, title, tcolor, head_lines, head_fill, head_stroke in [
    (lx, "基座 SiluAndMul（CUDA）", "#475569",
     ["forward_cuda(self, x):", "    self.op(out, x)", "    # torch.ops._C.silu_and_mul"], "#f1f5f9", "#94a3b8"),
    (rx, "AscendSiluAndMul（NPU）", "#b91c1c",
     ["forward_oot(self, x):", "    out = torch_npu.npu_swiglu(x)", "    return out"], "#fef2f2", "#ef4444"),
]:
    box(cx, top, col_w, 470, "white", tcolor, 14, 2)
    text(cx + col_w / 2, top + 32, title, 17, "middle", tcolor, "bold")

    # 头 (head)
    hx, hy, hw, hh = cx + 30, top + 56, col_w - 60, 150
    box(hx, hy, hw, hh, head_fill, head_stroke, 10, 2)
    text(hx + 16, hy + 28, "头 · forward 实现", 15, "start", tcolor, "bold")
    for i, ln in enumerate(head_lines):
        text(hx + 16, hy + 60 + i * 26, ln, 13, "start", "#334155", mono=True)

    # 身 (body)
    bx, by, bw, bh = cx + 30, top + 230, col_w - 60, 210
    box(bx, by, bw, bh, "#f8fafc", "#cbd5e1", 10, 2, dash="5 4")
    text(bx + 16, by + 28, "身 · 继承自基座，一行不改", 15, "start", "#64748b", "bold")
    body_lines = [
        "class : 接口/签名 forward(x)",
        "__init__ / 权重 / 形状约定",
        "注册键 name（写进 op_registry_oot）",
        "forward_native（PyTorch 原生对照）",
    ]
    for i, ln in enumerate(body_lines):
        text(bx + 16, by + 58 + i * 34, "• " + ln, 13, "start", "#475569")

# swap arrow between heads
ay = top + 56 + 75
L.append(f'<line x1="{lx + col_w}" y1="{ay}" x2="{rx}" y2="{ay}" stroke="#b91c1c" stroke-width="2.5" marker-end="url(#ar)"/>')
text((lx + col_w + rx) / 2, ay - 14, "换头", 15, "middle", "#b91c1c", "bold")
text((lx + col_w + rx) / 2, ay + 22, "覆写", 12, "middle", "#b91c1c")

# body identity link
by_mid = top + 230 + 105
L.append(f'<line x1="{lx + col_w}" y1="{by_mid}" x2="{rx}" y2="{by_mid}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="6 5"/>')
text((lx + col_w + rx) / 2, by_mid - 12, "身不变", 14, "middle", "#64748b", "bold")
text((lx + col_w + rx) / 2, by_mid + 22, "= 继承", 12, "middle", "#64748b")

L.append('</svg>')
open("fig-head-body.svg", "w").write('\n'.join(L))
print("wrote fig-head-body.svg")
